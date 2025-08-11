#!/usr/bin/env python3
"""
Reachy Mini — speech-driven head motion:
- Robust accent detection (F0 OR energy) inside VAD
- Continuous 6-DoF "speech sway" while speaking
- Beat-like pitch pulses on accents (apex near accent time)
- Plots: dBFS, F0 with accent stems, commanded angles (deg)
- Realtime monitor

Install:
  pip install soundcard numpy matplotlib
"""

import time, math, threading
from collections import deque
from itertools import islice
import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ============================== TUNABLES =======================================
# Analysis cadence
SR = 16_000
FRAME_MS = 20
HOP_MS   = 10

# VAD thresholds (durable lines on plot)
VAD_DB_ON  = -35.0
VAD_DB_OFF = -45.0
VAD_ATTACK_MS  = 120
VAD_RELEASE_MS = 250

# F0 estimation + smoothing
FMIN, FMAX = 70, 380     # Hz
DBFS_SILENCE = -55.0
F0_MEDIAN_MS = 50

# Accent detection (inside VAD)
PEAK_LOOKAHEAD_MS   = 20     # confirm local max
VOICED_NEIGHBOR_MS  = 80     # bridge short unvoiced gaps
PROMINENCE_HZ       = 8.0    # modest pitch prominence
PROMINENCE_DB       = 3.0    # energy prominence for energy peaks
RISE_MIN_HZ         = 3.0    # min rise vs nearest left voiced
FALL_MIN_HZ         = 3.0    # min fall vs nearest right voiced
MIN_PEAK_SEP_MS     = 160    # refractory period
MAX_NODS_PER_SEC    = 5.0    # global safety rate limit

# Amplitude mapping for accent pulses (dBFS → degrees)
# Your trace lives ~[-20, -40] dBFS.
NOD_MIN_DEG = 4.0
NOD_MAX_DEG = 20.0
AMP_MAP_DB_LOW  = -44.0
AMP_MAP_DB_HIGH = -18.0

# Motion timing for accent pulses
NOD_DUR   = 0.20          # s, half-sine pulse (snappy)
APEX_LEAD = 0.04          # s, start before event so apex lands on it

# Continuous "speech sway" while VAD=ON (all 6 DoF, subtle)
# Frequencies (Hz) and peak amplitudes (deg or mm). Amplitudes are scaled by a
# loudness envelope (see SWAY_DB_* below) and an overall envelope that fades in/out with VAD.
SWAY_F_PITCH = 2.2        # small quick bob on vowels
SWAY_A_PITCH_DEG = 3.0
SWAY_F_YAW   = 0.6        # side-to-side discourse sway
SWAY_A_YAW_DEG = 5.0
SWAY_F_ROLL  = 1.3        # tiny roll to break symmetry
SWAY_A_ROLL_DEG = 1.5
SWAY_F_X     = 0.35       # forward/back drift (mm)
SWAY_A_X_MM  = 3.0
SWAY_F_Y     = 0.45       # left/right drift (mm)
SWAY_A_Y_MM  = 2.5
SWAY_F_Z     = 0.25       # subtle vertical "breathing" (mm)
SWAY_A_Z_MM  = 1.5

# Loudness mapping for sway (how much sway at given dBFS)
SWAY_DB_LOW  = -46.0   # below -> almost no sway
SWAY_DB_HIGH = -18.0   # above -> full sway

# Sway envelope (attack/release around VAD edges)
SWAY_ATTACK_MS  = 120
SWAY_RELEASE_MS = 250

# Plot and monitoring
PLOT_HZ   = 12
WINDOW_S  = 10.0
RT_PRINT_EVERY = 2.0
RT_TOL = 0.06

# Device selection
SPEAKER_SUBSTR = "Headphones"  # "" for default device
# ==============================================================================

# Derived
FRAME = int(SR * FRAME_MS / 1000)
HOP   = int(SR * HOP_MS / 1000)
LOOKAHEAD_FR       = max(1, int(PEAK_LOOKAHEAD_MS  / HOP_MS))
ATTACK_FRAMES      = max(1, int(VAD_ATTACK_MS      / HOP_MS))
RELEASE_FRAMES     = max(1, int(VAD_RELEASE_MS     / HOP_MS))
F0_MEDIAN_FR       = max(1, int(F0_MEDIAN_MS       / HOP_MS))
MIN_PEAK_SEP_FR    = max(1, int(MIN_PEAK_SEP_MS    / HOP_MS))
VOICED_NEIGHBOR_FR = max(1, int(VOICED_NEIGHBOR_MS / HOP_MS))
SWAY_ATTACK_FR     = max(1, int(SWAY_ATTACK_MS     / HOP_MS))
SWAY_RELEASE_FR    = max(1, int(SWAY_RELEASE_MS    / HOP_MS))

# ------------------ DSP helpers ------------------
def rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def autocorr_f0(frame: np.ndarray, sr=SR) -> float:
    x = frame.astype(np.float64)
    x -= np.mean(x)
    m = np.max(np.abs(x))
    if m > 0: x /= m
    x *= np.hanning(len(x))
    acf = np.correlate(x, x, mode='full')[len(x)-1:]
    if acf[0] <= 0: return 0.0
    acf /= (acf[0] + 1e-12)
    minlag = int(sr / FMAX)
    maxlag = min(int(sr / FMIN), len(acf)-1)
    if minlag >= maxlag: return 0.0
    region = acf[minlag:maxlag]
    i = int(np.argmax(region)) + minlag
    if acf[i] < 0.3: return 0.0
    if 1 <= i < len(acf)-1:
        y0,y1,y2 = acf[i-1],acf[i],acf[i+1]
        delta = (y0 - y2) / (2*(2*y1 - y0 - y2) + 1e-12)
        i = i + delta
    f0 = sr / i if i > 0 else 0.0
    return float(f0) if FMIN <= f0 <= FMAX else 0.0

def amp_from_db(db):
    # For pulses (deg → rad)
    t = (db - AMP_MAP_DB_LOW) / (AMP_MAP_DB_HIGH - AMP_MAP_DB_LOW)
    t = max(0.0, min(1.0, t))
    deg = NOD_MIN_DEG + t * (NOD_MAX_DEG - NOD_MIN_DEG)
    return math.radians(deg)

def sway_gain_from_db(db):
    # 0..1 loudness mapping for continuous sway
    t = (db - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    return max(0.0, min(1.0, t))

# ------------------ Analyzer ------------------
class Analyzer:
    """Loopback audio → dBFS, smoothed F0, VAD, accents, loudness envelope."""
    def __init__(self, speaker_substr=SPEAKER_SUBSTR):
        self.speaker_substr = speaker_substr
        self.stop = False

        self.samples = deque(maxlen=10*SR)
        self.t_hist, self.db_hist, self.f0_hist = [], [], []
        self.accent_times = []

        self.frame_idx = 0
        self.start_wall = None

        # VAD state with attack/release
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0

        # Speech sway envelope (separate from VAD for smoother fades)
        self.sway_env = 0.0  # 0..1
        self.sway_up = 0
        self.sway_down = 0

        # F0 smoothing
        self._f0_win = deque(maxlen=F0_MEDIAN_FR)
        self.last_peak_frame = -10**9

        # Events
        self.accents = deque()    # (t_wall, amp)
        self.recent_nods = deque(maxlen=40)  # timestamps for rate limit

    def _choose_loopback(self):
        spk = None
        for s in sc.all_speakers():
            if self.speaker_substr and self.speaker_substr.lower() in s.name.lower():
                spk = s; break
        if spk is None: spk = sc.default_speaker()
        try:
            mic = sc.get_microphone(id=spk.name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            for m in sc.all_microphones(include_loopback=True):
                nm = m.name.lower()
                if (self.speaker_substr and self.speaker_substr.lower() in nm) or "monitor" in nm or "loopback" in nm:
                    mic = m; break
        if mic is None:
            raise RuntimeError("No loopback microphone found")
        return mic, spk

    def _median(self, q):
        if not q: return 0.0
        a = np.fromiter(q, dtype=np.float32, count=len(q))
        return float(np.median(a))

    def _nearest_voiced(self, arr, i, direction, max_steps):
        step = -1 if direction < 0 else 1
        for k in range(1, max_steps + 1):
            j = i + step * k
            if 0 <= j < len(arr) and arr[j] > 0:
                return arr[j], j
        return 0.0, None

    def _rate_limited(self, t_wall):
        while self.recent_nods and (t_wall - self.recent_nods[0]) > 1.0:
            self.recent_nods.popleft()
        return len(self.recent_nods) >= MAX_NODS_PER_SEC

    def run(self):
        mic, spk = self._choose_loopback()
        block = max(HOP, 512)
        print(f"[audio] Loopback of: {spk.name}")
        with mic.recorder(samplerate=SR, channels=1, blocksize=block) as r:
            carry = np.zeros(0, dtype=np.float32)
            self.start_wall = time.time()
            t0 = self.start_wall
            while not self.stop:
                data = r.record(numframes=block).astype(np.float32).flatten()
                if data.size == 0: continue
                carry = np.concatenate([carry, data])

                while carry.size >= HOP:
                    # advance one hop
                    self.samples.extend(carry[:HOP].tolist())
                    carry = carry[HOP:]
                    if len(self.samples) < FRAME: continue

                    # frame
                    frame = np.fromiter(islice(self.samples, len(self.samples)-FRAME, len(self.samples)),
                                        dtype=np.float32, count=FRAME)
                    db = rms_dbfs(frame)
                    f0 = 0.0 if db < DBFS_SILENCE else autocorr_f0(frame)

                    # VAD (hard gate)
                    if db >= VAD_DB_ON:
                        self.vad_above += 1; self.vad_below = 0
                        if not self.vad_on and self.vad_above >= ATTACK_FRAMES:
                            self.vad_on = True
                    elif db <= VAD_DB_OFF:
                        self.vad_below += 1; self.vad_above = 0
                        if self.vad_on and self.vad_below >= RELEASE_FRAMES:
                            self.vad_on = False

                    # Sway envelope follows VAD with separate attack/release
                    if self.vad_on:
                        self.sway_up = min(SWAY_ATTACK_FR, self.sway_up + 1)
                        self.sway_down = 0
                    else:
                        self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1)
                        self.sway_up = 0
                    up_gain = self.sway_up / SWAY_ATTACK_FR
                    down_gain = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
                    target_env = up_gain if self.vad_on else down_gain
                    # critically damped-ish 1st order follow
                    self.sway_env += 0.3 * (target_env - self.sway_env)
                    self.sway_env = max(0.0, min(1.0, self.sway_env))

                    # Smooth F0
                    self._f0_win.append(f0 if f0 > 0 else 0.0)
                    f0_med = self._median(self._f0_win)

                    # histories for plot
                    t_rel = self.frame_idx * (HOP_MS / 1000.0)
                    self.t_hist.append(t_rel)
                    self.db_hist.append(db)
                    self.f0_hist.append(f0_med)

                    # ---------------- Accent detection (inside VAD) ----------------
                    i = len(self.f0_hist) - LOOKAHEAD_FR - 1
                    if self.vad_on and i > 2:
                        fired = False

                        # F0 route
                        if self.f0_hist[i] > 0:
                            lo = max(0, i - 2); hi = min(len(self.f0_hist), i + 3)
                            window = [v for v in self.f0_hist[lo:hi] if v > 0]
                            is_local_max = (len(window) > 0) and (self.f0_hist[i] >= max(window))
                            if is_local_max:
                                left_val, left_idx   = self._nearest_voiced(self.f0_hist, i, -1, VOICED_NEIGHBOR_FR)
                                right_val, right_idx = self._nearest_voiced(self.f0_hist, i, +1, VOICED_NEIGHBOR_FR)
                                if left_idx is not None and right_idx is not None:
                                    rise_ok = (self.f0_hist[i] - left_val)  >= RISE_MIN_HZ
                                    fall_ok = (self.f0_hist[i] - right_val) >= FALL_MIN_HZ
                                    prom_ok = (self.f0_hist[i] - max(left_val, right_val)) >= PROMINENCE_HZ
                                    sep_ok  = (i - self.last_peak_frame) >= MIN_PEAK_SEP_FR
                                    if rise_ok and fall_ok and prom_ok and sep_ok:
                                        fired = True

                        # Energy route (helps when F0 drops on unvoiced syllables)
                        if not fired:
                            d_im1 = self.db_hist[i-1] if i-1 >= 0 else -90
                            d_i   = self.db_hist[i]
                            d_ip1 = self.db_hist[i+1] if i+1 < len(self.db_hist) else -90
                            is_energy_peak = (d_i > d_im1) and (d_i > d_ip1) and (d_i - max(d_im1, d_ip1)) >= PROMINENCE_DB
                            sep_ok = (i - self.last_peak_frame) >= MIN_PEAK_SEP_FR
                            if is_energy_peak and sep_ok:
                                fired = True

                        if fired:
                            t_wall = t0 + i * (HOP_MS / 1000.0)
                            # global rate limiting
                            while self.recent_nods and (t_wall - self.recent_nods[0]) > 1.0:
                                self.recent_nods.popleft()
                            if len(self.recent_nods) < MAX_NODS_PER_SEC:
                                self.last_peak_frame = i
                                amp = amp_from_db(self.db_hist[i])
                                self.accents.append((t_wall, amp))
                                self.accent_times.append(self.t_hist[i])
                                self.recent_nods.append(t_wall)

                    self.frame_idx += 1

# ------------------ Main control & plotting ------------------
def run():
    analyzer = Analyzer()
    th = threading.Thread(target=analyzer.run, daemon=True)
    th.start()

    # Plot setup: 3 panels
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    # Top: dBFS + thresholds
    ax1.set_title("Energy dBFS with VAD thresholds"); ax1.set_ylim(-80, 0); ax1.set_xlim(0, WINDOW_S)
    line_db, = ax1.plot([], [], label="dBFS")
    ax1.axhline(VAD_DB_ON,  color="tab:green", linestyle="--", linewidth=1.0, label=f"VAD ON {VAD_DB_ON} dB")
    ax1.axhline(VAD_DB_OFF, color="tab:red",   linestyle="--", linewidth=1.0, label=f"VAD OFF {VAD_DB_OFF} dB")
    vad_poly = None
    rt_text = ax1.text(0.01, 0.05, "", transform=ax1.transAxes)
    ax1.legend(loc="lower right")

    # Middle: F0 + accents
    ax2.set_title("F0 Hz with accent markers"); ax2.set_ylim(50, 450); ax2.set_xlim(0, WINDOW_S)
    line_f0, = ax2.plot([], [], label="F0")
    acc_lines = None
    ax2.legend(loc="lower right")

    # Bottom: commanded angles (deg)
    ax3.set_title("Commanded angles (deg)"); ax3.set_ylim(-25, 25); ax3.set_xlim(0, WINDOW_S)
    cmd_pitch_line, = ax3.plot([], [], linestyle="--", label="pitch cmd")
    cmd_yaw_line,   = ax3.plot([], [], linestyle=":",  label="yaw cmd")
    cmd_roll_line,  = ax3.plot([], [], linestyle="-.", label="roll cmd")
    ax3.legend(loc="lower right")

    last_draw = 0.0
    last_rt_print = 0.0

    # Command histories (audio-time base)
    cmd_t_hist, cmd_pitch_deg_hist, cmd_yaw_deg_hist, cmd_roll_deg_hist = [], [], [], []

    with ReachyMini() as mini:
        print("[robot] Ready. 6-DoF sway + accent pulses (pitch).")
        nod_active = False
        nod_t0 = 0.0
        nod_amp = 0.0

        # random phases so sway isn't perfectly aligned axes
        rng = np.random.default_rng(7)
        phase_pitch = rng.random() * 2*math.pi
        phase_yaw   = rng.random() * 2*math.pi
        phase_roll  = rng.random() * 2*math.pi
        phase_x     = rng.random() * 2*math.pi
        phase_y     = rng.random() * 2*math.pi
        phase_z     = rng.random() * 2*math.pi

        try:
            dt = 0.01  # 100 Hz control
            while True:
                now = time.time()
                # audio time for plotting alignment
                audio_time_now = analyzer.frame_idx * (HOP_MS / 1000.0) if analyzer.start_wall else 0.0

                # schedule accent pulse
                if not nod_active and analyzer.accents:
                    _, amp = analyzer.accents.popleft()
                    nod_t0  = now + APEX_LEAD
                    nod_amp = amp
                    nod_active = True

                # continuous sway while speaking (scaled by loudness & envelope)
                # Use most recent dB; if empty, 0.
                db_now = analyzer.db_hist[-1] if analyzer.db_hist else -80.0
                loud = sway_gain_from_db(db_now)
                env  = analyzer.sway_env  # 0..1 envelope following VAD

                # angular sways (radians)
                pitch_sway = math.radians(SWAY_A_PITCH_DEG) * loud * env * math.sin(2*math.pi*SWAY_F_PITCH*now + phase_pitch)
                yaw_sway   = math.radians(SWAY_A_YAW_DEG)   * loud * env * math.sin(2*math.pi*SWAY_F_YAW*now   + phase_yaw)
                roll_sway  = math.radians(SWAY_A_ROLL_DEG)  * loud * env * math.sin(2*math.pi*SWAY_F_ROLL*now  + phase_roll)

                # translational sways (mm)
                x_sway = SWAY_A_X_MM * loud * env * math.sin(2*math.pi*SWAY_F_X*now + phase_x)
                y_sway = SWAY_A_Y_MM * loud * env * math.sin(2*math.pi*SWAY_F_Y*now + phase_y)
                z_sway = SWAY_A_Z_MM * loud * env * math.sin(2*math.pi*SWAY_F_Z*now + phase_z)

                # accent pulse in pitch (half-sine)
                pitch_pulse = 0.0
                if nod_active:
                    phi = (now - nod_t0) / NOD_DUR
                    if phi <= 1.0:
                        pitch_pulse = nod_amp * math.sin(math.pi * max(0.0, phi))
                    else:
                        nod_active = False

                # final commands (radians for angles, mm for x/y/z)
                pitch_cmd = pitch_sway + pitch_pulse
                yaw_cmd   = yaw_sway
                roll_cmd  = roll_sway

                head_pose = create_head_pose(
                    x=x_sway, y=y_sway, z=z_sway,
                    roll=roll_cmd, pitch=pitch_cmd, yaw=yaw_cmd,
                    degrees=False, mm=True
                )
                mini.set_target(head=head_pose, antennas=(0.0, 0.0))

                # record commands in deg for bottom plot
                cmd_t_hist.append(audio_time_now)
                cmd_pitch_deg_hist.append(math.degrees(pitch_cmd))
                cmd_yaw_deg_hist.append(math.degrees(yaw_cmd))
                cmd_roll_deg_hist.append(math.degrees(roll_cmd))

                # realtime monitor
                if analyzer.start_wall is not None:
                    wall_time = now - analyzer.start_wall
                    audio_time = analyzer.frame_idx * (HOP_MS / 1000.0)
                    rt_ratio = (audio_time / wall_time) if wall_time > 0 else 0.0
                    if (now - last_rt_print) >= RT_PRINT_EVERY:
                        status = "OK" if abs(rt_ratio - 1.0) <= RT_TOL else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                        print(f"[rt] audio={audio_time:.2f}s wall={wall_time:.2f}s ratio={rt_ratio:.3f} [{status}]")
                        last_rt_print = now

                # plots (throttled)
                if (now - last_draw) >= (1.0 / PLOT_HZ) and analyzer.t_hist:
                    tmax = analyzer.t_hist[-1]
                    tmin = max(0.0, tmax - WINDOW_S)

                    def last_window(t_all, y_all):
                        if not t_all: return [], []
                        i0 = 0
                        for i in range(len(t_all)-1, -1, -1):
                            if t_all[i] < tmin:
                                i0 = i + 1; break
                        return t_all[i0:], y_all[i0:]

                    tx, dbx   = last_window(analyzer.t_hist, analyzer.db_hist)
                    tx2, f0x  = last_window(analyzer.t_hist, analyzer.f0_hist)
                    tc, pcmd  = last_window(cmd_t_hist,   cmd_pitch_deg_hist)
                    _,  ycmd  = last_window(cmd_t_hist,   cmd_yaw_deg_hist)
                    _,  rcmd  = last_window(cmd_t_hist,   cmd_roll_deg_hist)

                    line_db.set_data(tx, dbx)
                    line_f0.set_data(tx2, f0x)
                    cmd_pitch_line.set_data(tc, pcmd)
                    cmd_yaw_line.set_data(tc, ycmd)
                    cmd_roll_line.set_data(tc, rcmd)

                    for ax in (ax1, ax2, ax3):
                        ax.set_xlim(tmin, max(tmin + 0.5, tmax))

                    # VAD shading (current state)
                    if vad_poly is not None: vad_poly.remove(); vad_poly = None
                    if tx:
                        y0 = [-80]*len(tx); y1 = [0 if analyzer.vad_on else -80]*len(tx)
                        vad_poly = ax1.fill_between(tx, y0, y1, alpha=0.10, step="pre", color="gray")

                    # Accent stems
                    if acc_lines is not None: acc_lines.remove(); acc_lines = None
                    inwin = [t for t in analyzer.accent_times if tmin <= t <= tmax]
                    if inwin: acc_lines = ax2.vlines(inwin, 60, 440, linewidth=1.0, alpha=0.6)

                    # RT overlay
                    if analyzer.start_wall is not None:
                        wall_time = now - analyzer.start_wall
                        audio_time = analyzer.frame_idx * (HOP_MS / 1000.0)
                        rt_ratio = (audio_time / wall_time) if wall_time > 0 else 0.0
                        status = "OK" if abs(rt_ratio - 1.0) <= RT_TOL else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                        rt_text.set_text(f"RT ratio: {rt_ratio:.3f} [{status}]")

                    plt.pause(0.001); last_draw = now

                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nCtrl-C. Shutting down...")
        finally:
            analyzer.stop = True

if __name__ == "__main__":
    run()
