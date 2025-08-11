#!/usr/bin/env python3
"""
Reachy Mini - speech-reactive nods with explained accent detection and commanded-pitch plot.

What you get:
- Loopback of system output (no mic) with soundcard.
- Energy (dBFS), F0 with median smoothing, VAD with attack/release.
- Accent detection from F0 peaks with derivative + prominence + cooldown.
- Pitch-only nods scheduled so the apex lands near the accent.
- Plots: dBFS with durable VAD thresholds, F0 with accent markers, and the actual
  commanded pitch in degrees. RT ratio overlay to verify realtime.

Install:
  pip install soundcard numpy matplotlib
"""

import time
import math
import threading
from collections import deque
from itertools import islice
import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ============================== TUNABLES =======================================
# Analysis cadence
SR = 16_000          # Hz
FRAME_MS = 20        # ms
HOP_MS = 10          # ms

# VAD thresholds (durable horizontal lines on plot)
VAD_DB_ON  = -35.0   # raise toward -32 if it still triggers in silence
VAD_DB_OFF = -45.0   # lower toward -48 to leave VAD sooner when quiet
VAD_ATTACK_MS  = 120 # must stay above ON this long to turn on
VAD_RELEASE_MS = 250 # must stay below OFF this long to turn off

# F0 estimation and smoothing
FMIN, FMAX = 70, 380     # Hz
DBFS_SILENCE = -55.0     # skip F0 below this
F0_MEDIAN_MS = 50        # median window in ms to stabilize F0

# Accent detection from F0
PEAK_LOOKAHEAD_MS = 30   # small lookahead to confirm local max
PROMINENCE_HZ = 18.0     # how much higher than neighbors to count as accent
MIN_PEAK_SEP_MS = 220    # cooldown between accents

# Amplitude mapping from level to degrees
# Your screenshot hovered around -20 to -40 dBFS, so map that to a wide range.
NOD_MIN_DEG = 4.0        # visible even on soft syllables
NOD_MAX_DEG = 20.0       # emphatic syllables
AMP_MAP_DB_LOW  = -44.0  # <= this -> ~NOD_MIN_DEG
AMP_MAP_DB_HIGH = -18.0  # >= this -> ~NOD_MAX_DEG

# Motion timing
NOD_DUR   = 0.22         # seconds for half-sine pulse
APEX_LEAD = 0.04         # start slightly before accent so apex lands on it

# Plot and monitoring
PLOT_HZ   = 12
WINDOW_S  = 10.0
RT_PRINT_EVERY = 2.0
RT_TOL = 0.06

# Device selection
SPEAKER_SUBSTR = "Headphones"  # "" for default speaker
# ==============================================================================

# Derived
FRAME = int(SR * FRAME_MS / 1000)
HOP   = int(SR * HOP_MS / 1000)
LOOKAHEAD_FR   = max(1, int(PEAK_LOOKAHEAD_MS / HOP_MS))
ATTACK_FRAMES  = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FRAMES = max(1, int(VAD_RELEASE_MS / HOP_MS))
F0_MEDIAN_FR   = max(1, int(F0_MEDIAN_MS / HOP_MS))
MIN_PEAK_SEP_FR = max(1, int(MIN_PEAK_SEP_MS / HOP_MS))

def rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def autocorr_f0(frame: np.ndarray, sr=SR) -> float:
    x = frame.astype(np.float64)
    x -= np.mean(x)
    m = np.max(np.abs(x))
    if m > 0:
        x /= m
    x *= np.hanning(len(x))
    acf = np.correlate(x, x, mode='full')[len(x)-1:]
    if acf[0] <= 0:
        return 0.0
    acf /= (acf[0] + 1e-12)
    minlag = int(sr / FMAX)
    maxlag = min(int(sr / FMIN), len(acf)-1)
    if minlag >= maxlag:
        return 0.0
    region = acf[minlag:maxlag]
    i = int(np.argmax(region)) + minlag
    if acf[i] < 0.3:
        return 0.0
    if 1 <= i < len(acf)-1:
        y0, y1, y2 = acf[i-1], acf[i], acf[i+1]
        denom = 2*(2*y1 - y0 - y2) + 1e-12
        delta = (y0 - y2) / denom
        i = i + delta
    f0 = sr / i if i > 0 else 0.0
    return float(f0) if FMIN <= f0 <= FMAX else 0.0

def amp_from_db(db):
    db = max(-80.0, min(0.0, db))
    t = (db - AMP_MAP_DB_LOW) / (AMP_MAP_DB_HIGH - AMP_MAP_DB_LOW)
    t = max(0.0, min(1.0, t))
    deg = NOD_MIN_DEG + t * (NOD_MAX_DEG - NOD_MIN_DEG)
    return math.radians(deg)

class Analyzer:
    """Loopback audio -> dBFS, F0, VAD, accent events."""
    def __init__(self, speaker_substr=SPEAKER_SUBSTR):
        self.speaker_substr = speaker_substr
        self.stop = False

        self.samples = deque(maxlen=10*SR)
        self.t_hist, self.db_hist, self.f0_hist = [], [], []
        self.accent_times = []

        self.frame_idx = 0
        self.start_wall = None

        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self._f0_win = deque(maxlen=F0_MEDIAN_FR)
        self.last_peak_frame = -10**9

        self.accents = deque()    # (t_wall, amp)

    def _choose_loopback(self):
        spk = None
        for s in sc.all_speakers():
            if self.speaker_substr and self.speaker_substr.lower() in s.name.lower():
                spk = s
                break
        if spk is None:
            spk = sc.default_speaker()
        try:
            mic = sc.get_microphone(id=spk.name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            for m in sc.all_microphones(include_loopback=True):
                nm = m.name.lower()
                if (self.speaker_substr and self.speaker_substr.lower() in nm) or "monitor" in nm or "loopback" in nm:
                    mic = m
                    break
        if mic is None:
            raise RuntimeError("No loopback microphone found")
        return mic, spk

    def _median(self, q):
        if not q:
            return 0.0
        a = np.fromiter(q, dtype=np.float32, count=len(q))
        return float(np.median(a))

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
                if data.size == 0:
                    continue
                carry = np.concatenate([carry, data])

                while carry.size >= HOP:
                    # advance one hop
                    self.samples.extend(carry[:HOP].tolist())
                    carry = carry[HOP:]
                    if len(self.samples) < FRAME:
                        continue

                    # latest FRAME
                    frame = np.fromiter(islice(self.samples, len(self.samples)-FRAME, len(self.samples)),
                                        dtype=np.float32, count=FRAME)

                    db = rms_dbfs(frame)
                    f0 = 0.0 if db < DBFS_SILENCE else autocorr_f0(frame)

                    # VAD with hysteresis and attack-release
                    if db >= VAD_DB_ON:
                        self.vad_above += 1
                        self.vad_below = 0
                        if not self.vad_on and self.vad_above >= ATTACK_FRAMES:
                            self.vad_on = True
                    elif db <= VAD_DB_OFF:
                        self.vad_below += 1
                        self.vad_above = 0
                        if self.vad_on and self.vad_below >= RELEASE_FRAMES:
                            self.vad_on = False

                    # F0 smoothing
                    self._f0_win.append(f0 if f0 > 0 else 0.0)
                    f0_med = self._median(self._f0_win)

                    # history for plots
                    t_rel = self.frame_idx * (HOP_MS / 1000.0)
                    self.t_hist.append(t_rel)
                    self.db_hist.append(db)
                    self.f0_hist.append(f0_med)

                    # Accent detection inside VAD
                    i = len(self.f0_hist) - LOOKAHEAD_FR - 1
                    if self.vad_on and i > 2:
                        f_im1 = self.f0_hist[i-1]
                        f_i   = self.f0_hist[i]
                        f_ip1 = self.f0_hist[i+1] if i+1 < len(self.f0_hist) else 0.0
                        # local max
                        if f_i > 0 and f_im1 < f_i and f_ip1 < f_i:
                            # derivative windows: ensure a rise then a fall in a small neighborhood
                            rise_ok = (f_im1 > 0) and (f_i - f_im1) >= 1.0
                            fall_ok = (f_ip1 > 0) and (f_i - f_ip1) >= 1.0
                            prom_ok = (f_i - max(f_im1, f_ip1)) >= PROMINENCE_HZ
                            sep_ok  = (i - self.last_peak_frame) >= MIN_PEAK_SEP_FR
                            if rise_ok and fall_ok and prom_ok and sep_ok:
                                self.last_peak_frame = i
                                t_wall = t0 + i * (HOP_MS / 1000.0)
                                amp = amp_from_db(self.db_hist[i])
                                self.accents.append((t_wall, amp))
                                self.accent_times.append(self.t_hist[i])

                    self.frame_idx += 1

def run():
    analyzer = Analyzer()
    th = threading.Thread(target=analyzer.run, daemon=True)
    th.start()

    # Plot setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    ax1.set_title("Energy dBFS with VAD thresholds")
    ax1.set_ylim(-80, 0)
    ax1.set_xlim(0, WINDOW_S)
    ax2.set_title("F0 Hz, accents, and commanded pitch (deg)")
    ax2.set_ylim(50, 450)
    ax2.set_xlim(0, WINDOW_S)

    line_db, = ax1.plot([], [], label="dBFS")
    line_f0, = ax2.plot([], [], label="F0")
    # commanded pitch line
    cmd_line, = ax2.plot([], [], linestyle="--", alpha=0.85, label="cmd pitch (deg)")

    # durable threshold lines
    ax1.axhline(VAD_DB_ON,  color="tab:green", linestyle="--", linewidth=1.0, label=f"VAD ON {VAD_DB_ON} dB")
    ax1.axhline(VAD_DB_OFF, color="tab:red",   linestyle="--", linewidth=1.0, label=f"VAD OFF {VAD_DB_OFF} dB")

    vad_poly = None
    acc_lines = None
    rt_text = ax1.text(0.01, 0.05, "", transform=ax1.transAxes)

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    last_draw = 0.0
    last_rt_print = 0.0

    # histories of commanded pitch in the same audio-time base
    cmd_t_hist, cmd_pitch_deg_hist = [], []

    with ReachyMini() as mini:
        print("[robot] Ready. Pitch only.")
        nod_active = False
        nod_t0 = 0.0
        nod_amp = 0.0

        try:
            dt = 0.01  # 100 Hz
            while True:
                now = time.time()

                # compute audio timeline now so cmd history aligns with analyzer plots
                if analyzer.start_wall is not None:
                    audio_time_now = analyzer.frame_idx * (HOP_MS / 1000.0)
                else:
                    audio_time_now = 0.0

                # schedule nod if available
                if not nod_active and analyzer.accents:
                    t_evt, amp = analyzer.accents.popleft()
                    nod_t0 = now + APEX_LEAD
                    nod_amp = amp
                    nod_active = True
                    print(f"[nod] amp={math.degrees(amp):.1f}° lead={int(APEX_LEAD*1000)}ms dur={int(NOD_DUR*1000)}ms")

                # generate commanded pitch
                pitch = 0.0
                if nod_active:
                    phi = (now - nod_t0) / NOD_DUR
                    if phi <= 1.0:
                        pitch = nod_amp * math.sin(math.pi * max(0.0, phi))
                    else:
                        nod_active = False
                        pitch = 0.0

                # send to robot
                head_pose = create_head_pose(
                    x=0.0, y=0.0, z=0.0,
                    roll=0.0, pitch=pitch, yaw=0.0,
                    degrees=False, mm=False
                )
                mini.set_target(head=head_pose, antennas=(0.0, 0.0))

                # record commanded pitch history in degrees for plotting
                cmd_t_hist.append(audio_time_now)
                cmd_pitch_deg_hist.append(math.degrees(pitch))

                # RT monitor
                if analyzer.start_wall is not None:
                    wall_time = now - analyzer.start_wall
                    audio_time = analyzer.frame_idx * (HOP_MS / 1000.0)
                    rt_ratio = (audio_time / wall_time) if wall_time > 0 else 0.0
                    if (now - last_rt_print) >= RT_PRINT_EVERY:
                        status = "OK" if abs(rt_ratio - 1.0) <= RT_TOL else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                        print(f"[rt] audio={audio_time:.2f}s wall={wall_time:.2f}s ratio={rt_ratio:.3f} [{status}]")
                        last_rt_print = now

                # plots
                if (now - last_draw) >= (1.0 / PLOT_HZ) and analyzer.t_hist:
                    tmax = analyzer.t_hist[-1]
                    tmin = max(0.0, tmax - WINDOW_S)

                    def last_window(t_all, y_all):
                        if not t_all:
                            return [], []
                        i0 = 0
                        for i in range(len(t_all)-1, -1, -1):
                            if t_all[i] < tmin:
                                i0 = i + 1
                                break
                        return t_all[i0:], y_all[i0:]

                    tx, dbx   = last_window(analyzer.t_hist, analyzer.db_hist)
                    tx2, f0x  = last_window(analyzer.t_hist, analyzer.f0_hist)
                    tc, cmdx  = last_window(cmd_t_hist,   cmd_pitch_deg_hist)

                    line_db.set_data(tx, dbx)
                    line_f0.set_data(tx2, f0x)
                    cmd_line.set_data(tc, cmdx)

                    ax1.set_xlim(tmin, max(tmin + 0.5, tmax))
                    ax2.set_xlim(tmin, max(tmin + 0.5, tmax))

                    # VAD shading - current state only
                    if vad_poly is not None:
                        vad_poly.remove(); vad_poly = None
                    if tx:
                        y0 = [-80] * len(tx)
                        y1 = [0 if analyzer.vad_on else -80] * len(tx)
                        vad_poly = ax1.fill_between(tx, y0, y1, alpha=0.10, step="pre", color="gray")

                    # Accent markers
                    if acc_lines is not None:
                        acc_lines.remove(); acc_lines = None
                    inwin = [t for t in analyzer.accent_times if tmin <= t <= tmax]
                    if inwin:
                        acc_lines = ax2.vlines(inwin, 60, 440, linewidth=1.0, alpha=0.6)

                    # RT overlay
                    if analyzer.start_wall is not None:
                        wall_time = now - analyzer.start_wall
                        audio_time = analyzer.frame_idx * (HOP_MS / 1000.0)
                        rt_ratio = (audio_time / wall_time) if wall_time > 0 else 0.0
                        status = "OK" if abs(rt_ratio - 1.0) <= RT_TOL else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                        rt_text.set_text(f"RT ratio: {rt_ratio:.3f} [{status}]")

                    plt.pause(0.001)
                    last_draw = now

                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nCtrl-C. Shutting down...")
        finally:
            analyzer.stop = True

if __name__ == "__main__":
    run()
