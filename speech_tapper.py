#!/usr/bin/env python3
"""
speech_tapper.py — tap system output (no mic) and compute metrics for head motion.

Backends:
  - soundcard (default): robust loopback of system output on Linux/PulseAudio/PipeWire.
  - sounddevice: optional; may or may not expose "monitor" on your system.
Also supports --file path.wav.

Metrics per 10 ms hop:
  - dBFS, F0 (autocorr), voiced, ACCENT with ~200 ms lookahead.

Flags:
  --list            list available devices for the chosen backend
  --plot            small live plot of dBFS and F0
  --speaker STR     substring to choose speaker (soundcard)
  --device STR      substring to choose input device (sounddevice)
  --device-index N  explicit device index (sounddevice)
  --file WAV        analyze a WAV file instead of live loopback
  
  
  
True documentation:
46226  [2025-08-08 21:24:54] pip install soundcard
46227  [2025-08-08 21:24:58] python3 speech_tapper.py --list
46228  [2025-08-08 21:25:50] python3 speech_tapper.py --backend soundcard --plot
46229  [2025-08-08 21:26:38] python3 speech_tapper.py --backend soundcard --speaker "Headphones" --plot
46230  [2025-08-08 21:29:07] python3 -m pip install -U soundcard
46231  [2025-08-08 21:29:37] python3 speech_tapper.py --backend soundcard --speaker "Headphones" --plot
46232  [2025-08-08 21:36:46] python3 speech_tapper.py --backend soundcard --speaker "Headphones"
46233  [2025-08-08 21:37:02] python3 speech_tapper.py --backend soundcard --speaker "Headphones" --plot

TODO : we're back at not real time. Needs tuning.
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

# =====================================================================================
# TUNABLES - READ THIS
# =====================================================================================
# Audio analysis rate and framing. 16 kHz, 20 ms frames, 10 ms hop are a good start.
SR = 16_000              # sample rate used for analysis. Raise to 24 k if your CPU is fine.
FRAME_MS = 20            # frame length in ms for F0 and energy
HOP_MS = 10              # hop size in ms for processing

# Voice Activity Detection
# - VAD_DB_ON: speech "turns on" once dBFS is above this value for ATTACK_FRAMES hops
# - VAD_DB_OFF: speech "turns off" once it drops below this value for RELEASE_FRAMES hops
# - Set VAD_DB_ON around -35 dBFS for normal TTS volume. Lower if you want more sensitivity.
VAD_DB_ON = -35.0        # gate-on threshold in dBFS
VAD_DB_OFF = -42.0       # gate-off threshold in dBFS (hysteresis below ON)
VAD_ATTACK_MS = 120      # must stay above ON for this long to enter VAD=ON
VAD_RELEASE_MS = 250     # must stay below OFF for this long to go back to VAD=OFF

# Pitch estimation
FMIN, FMAX = 70, 380     # search band for F0
DBFS_SILENCE = -55.0     # below this we skip F0 entirely
F0_MEDIAN_MS = 50        # median smoothing window for F0 in ms. Increase to be more stable, but a bit laggier.

# Accent detection
# - We detect a local F0 maximum with small lookahead and require a prominence in Hz.
# - MIN_PEAK_SEP_MS avoids machine-gun nods.
# - PROMINENCE_HZ should be 12 to 20 for clean TTS. Raise if you still see false hits.
PEAK_LOOKAHEAD_MS = 30   # tiny lookahead to confirm a local max
PROMINENCE_HZ = 15.0     # how much above neighbors the peak must be to count as accent
MIN_PEAK_SEP_MS = 250    # cooldown between accents to avoid over-nodding

# Amplitude mapping
# - Map level to nod size. Keep it modest to avoid cartoonish motion.
NOD_MIN_DEG = 2.0        # very soft syllables
NOD_MAX_DEG = 8.0        # emphatic syllables
AMP_MAP_DB_LOW = -42.0   # below this, nod amplitude is near NOD_MIN_DEG
AMP_MAP_DB_HIGH = -12.0  # above this, nod amplitude reaches NOD_MAX_DEG

# Motion timing
# - NOD_DUR controls how snappy nods are. 0.18 to 0.26 s is typical.
# - APEX_LEAD is the delay before starting the stroke so apex lands visually on the accent.
NOD_DUR = 0.22           # seconds, half-sine duration
APEX_LEAD = 0.06         # seconds, start nod slightly before accent apex for perceived sync

# Plotting
# - Drop PLOT_HZ if you see lag. Disable the plot to validate realtime first if needed.
PLOT_HZ = 12             # draw at most this many frames per second

# Device selection
# - If default speaker is not the one producing sound, set a substring here.
SPEAKER_SUBSTR = "Headphones"  # e.g. "Headphones", "HDMI", or "" for default speaker
# =====================================================================================


# ---------- derived constants ----------
FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
LOOKAHEAD_FR = max(1, int(PEAK_LOOKAHEAD_MS / HOP_MS))
ATTACK_FRAMES = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FRAMES = max(1, int(VAD_RELEASE_MS / HOP_MS))
F0_MEDIAN_FR = max(1, int(F0_MEDIAN_MS / HOP_MS))
MIN_PEAK_SEP_FR = max(1, int(MIN_PEAK_SEP_MS / HOP_MS))


# ---------- helpers ----------
def rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def autocorr_f0(frame: np.ndarray, sr=SR) -> float:
    # very small, stable enough for TTS
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
    # parabolic refine
    if 1 <= i < len(acf)-1:
        y0, y1, y2 = acf[i-1], acf[i], acf[i+1]
        denom = 2*(2*y1 - y0 - y2) + 1e-12
        delta = (y0 - y2) / denom
        i = i + delta
    f0 = sr / i if i > 0 else 0.0
    return float(f0) if FMIN <= f0 <= FMAX else 0.0

def amp_from_db(db):
    # maps dBFS to radians between NOD_MIN_DEG and NOD_MAX_DEG, with clamping
    db = max(-80.0, min(0.0, db))
    t = (db - AMP_MAP_DB_LOW) / (AMP_MAP_DB_HIGH - AMP_MAP_DB_LOW)
    t = max(0.0, min(1.0, t))
    deg = NOD_MIN_DEG + t * (NOD_MAX_DEG - NOD_MIN_DEG)
    return math.radians(deg)


# ---------- Audio analyzer thread ----------
class Analyzer:
    def __init__(self, speaker_substr=SPEAKER_SUBSTR):
        self.speaker_substr = speaker_substr
        self.stop = False

        # ring buffer for raw samples
        self.samples = deque(maxlen=10*SR)
        # histories for plotting
        self.t_hist, self.db_hist, self.f0_hist = [], [], []
        self.accent_times = []

        # realtime state
        self.frame_idx = 0
        self.last_peak_frame = -10**9
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0

        # smoothed F0 for stability
        self._f0_window = deque(maxlen=F0_MEDIAN_FR)

        # accents queue for robot loop: tuples (t_wall, amp)
        self.accents = deque()

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
        a = np.fromiter(q, dtype=np.float32, count=len(q))
        return float(np.median(a)) if len(a) else 0.0

    def run(self):
        mic, spk = self._choose_loopback()
        block = max(HOP, 512)
        print(f"[audio] Loopback of: {spk.name}")
        with mic.recorder(samplerate=SR, channels=1, blocksize=block) as r:
            carry = np.zeros(0, dtype=np.float32)
            t0 = time.time()
            while not self.stop:
                data = r.record(numframes=block).astype(np.float32).flatten()
                if data.size == 0:
                    continue
                carry = np.concatenate([carry, data])

                while carry.size >= HOP:
                    # push one hop
                    self.samples.extend(carry[:HOP].tolist())
                    carry = carry[HOP:]
                    if len(self.samples) < FRAME:
                        continue

                    # take last FRAME efficiently
                    frame = np.fromiter(islice(self.samples, len(self.samples)-FRAME, len(self.samples)),
                                        dtype=np.float32, count=FRAME)
                    db = rms_dbfs(frame)
                    f0 = 0.0 if db < DBFS_SILENCE else autocorr_f0(frame)

                    # VAD state machine with hysteresis and attack/release
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
                    else:
                        # between thresholds, keep counters but do not flip
                        pass

                    # F0 median smoothing to kill jitter
                    if f0 > 0:
                        self._f0_window.append(f0)
                    else:
                        self._f0_window.append(0.0)
                    f0_med = self._median(self._f0_window)

                    # store for plot
                    t_rel = self.frame_idx * (HOP_MS / 1000.0)
                    self.t_hist.append(t_rel)
                    self.db_hist.append(db)
                    self.f0_hist.append(f0_med)

                    # Accent detection only when VAD is ON
                    # local max with tiny lookahead and prominence check
                    i = len(self.f0_hist) - LOOKAHEAD_FR - 1
                    if self.vad_on and i > 2:
                        lo = max(0, i - 2)
                        hi = min(len(self.f0_hist), i + 3)
                        center = self.f0_hist[i]
                        if center > 0:
                            nbhd = self.f0_hist[lo:hi]
                            if center == max(nbhd):
                                left = self.f0_hist[i-1] if i-1 >= 0 else 0
                                right = self.f0_hist[i+1] if i+1 < len(self.f0_hist) else 0
                                if (center - max(left, right)) >= PROMINENCE_HZ:
                                    if (i - self.last_peak_frame) >= MIN_PEAK_SEP_FR:
                                        # trigger
                                        self.last_peak_frame = i
                                        t_wall = t0 + i * (HOP_MS / 1000.0)
                                        amp = amp_from_db(self.db_hist[i])
                                        self.accents.append((t_wall, amp))
                                        self.accent_times.append(self.t_hist[i])

                    self.frame_idx += 1


# ---------- Robot and plotting loop ----------
def run():
    analyzer = Analyzer()
    th = threading.Thread(target=analyzer.run, daemon=True)
    th.start()

    # plotting setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

    ax1.set_title("Energy dBFS and VAD")
    ax1.set_ylim(-80, 0)
    ax1.set_xlim(0, 10)
    ax2.set_title("F0 Hz with accent markers")
    ax2.set_ylim(50, 450)
    ax2.set_xlim(0, 10)

    # Lines we update in place (fast)
    line_db, = ax1.plot([], [], label="dBFS")
    line_f0, = ax2.plot([], [], label="F0")

    # We RECREATE these artists each draw (keep handles so we can .remove())
    vad_poly = None           # shaded area indicating VAD state
    acc_lines = None          # vertical lines for accents

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")


    last_draw = 0.0

    with ReachyMini() as mini:
        print("[robot] Ready. Pitch only. No antennas.")
        nod_active = False
        nod_t0 = 0.0
        nod_amp = 0.0

        try:
            dt = 0.01
            while True:
                now = time.time()

                # start a nod if idle and accent queued
                if not nod_active and analyzer.accents:
                    t_evt, amp = analyzer.accents.popleft()
                    nod_t0 = now + APEX_LEAD
                    nod_amp = amp
                    nod_active = True

                # compute pitch as half-sine pulse
                pitch = 0.0
                if nod_active:
                    phi = (now - nod_t0) / NOD_DUR
                    if phi <= 1.0:
                        pitch = nod_amp * math.sin(math.pi * max(0.0, phi))
                    else:
                        nod_active = False

                # send command
                head_pose = create_head_pose(
                    x=0.0, y=0.0, z=0.0,
                    roll=0.0, pitch=pitch, yaw=0.0,
                    degrees=False, mm=False
                )
                mini.set_target(head=head_pose, antennas=(0.0, 0.0))

                # update chart at most PLOT_HZ
                if (now - last_draw) >= (1.0 / PLOT_HZ) and len(analyzer.t_hist) > 5:
                    tmax = analyzer.t_hist[-1]
                    tmin = max(0.0, tmax - 10.0)

                    def last10(arr):
                        i0 = 0
                        for i in range(len(arr)-1, -1, -1):
                            if analyzer.t_hist[i] < tmin:
                                i0 = i + 1
                                break
                        return analyzer.t_hist[i0:], arr[i0:]

                    tx, dbx = last10(analyzer.db_hist)
                    tx2, f0x = last10(analyzer.f0_hist)

                    # update lines
                    line_db.set_data(tx, dbx)
                    line_f0.set_data(tx2, f0x)
                    ax1.set_xlim(tmin, tmax)
                    ax2.set_xlim(tmin, tmax)

                    # VAD shading: remove previous fill, then draw new one
                    if vad_poly is not None:
                        vad_poly.remove()
                        vad_poly = None
                    if tx:
                        # simple: shade the whole window if VAD currently ON
                        y0 = [-80] * len(tx)
                        y1 = [0 if analyzer.vad_on else -80] * len(tx)
                        vad_poly = ax1.fill_between(tx, y0, y1, alpha=0.10, step="pre", color="gray")

                    # Accent stems: remove previous collection, draw new ones within window
                    if acc_lines is not None:
                        acc_lines.remove()
                        acc_lines = None
                    inwin = [t for t in analyzer.accent_times if tmin <= t <= tmax]
                    if inwin:
                        acc_lines = ax2.vlines(inwin, 60, 440, linewidth=1.0, alpha=0.6)

                    plt.pause(0.001)
                    last_draw = now


                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nCtrl-C. Shutting down...")
        finally:
            analyzer.stop = True


if __name__ == "__main__":
    run()
