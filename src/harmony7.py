#!/usr/bin/env python3
"""
Reachy‑Mini sonic game – v9
• A reggaetón kick‑snare loop plays at a fixed baseline volume so you
  always hear the beat, no matter how far you start.
• l2 distance (0‑40 mm)
      – adds a HAT+CLAVE layer whose loudness grows steeply below 10 mm
        (easy to tell 10 ⇢ 7 ⇢ 5 mm).
      – also raises an overall “thump gain” so kicks feel heavy near 0 mm.
• angle error (0‑40 °)
      – smooth consonant pad: root ➜ triad ➜ add6/9 ➜ sparkle octave.
      – warm triangle timbre, gentle vibrato that fades as you align.
No external samples, mono safe, single file.
"""

import time, threading, queue
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ─────────── Audio constants ───────────
RATE = 44100
SLICE_SEC = 0.05

VOL_BEAT_BASE = 0.30  # always on
VOL_BEAT_GAIN = 0.45  # extra when close
VOL_HATCLAVE = 0.45
VOL_PAD = 0.28

BPM = 96
BAR_SEC = 60 / BPM * 4  # 4‑beat reggaetón bar ≈2.5 s

state_q = queue.Queue()  # (l2_mm, ang_deg)


# ─────────── Helper: envelope & drum snippets ───────────
def exp_env(n, tau):
    return np.exp(-np.arange(n) / RATE / tau, dtype=np.float32)


def kick():
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * exp_env(n, 0.15)
    click = 0.12 * np.sin(2 * np.pi * 180 * t) * exp_env(n, 0.005)
    return body + click


def snare():
    n = int(RATE * 0.18)
    noise = np.random.randn(n).astype(np.float32) * exp_env(n, 0.04)
    body = 0.25 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * exp_env(n, 0.12)
    return noise + body


def hat():
    n = int(RATE * 0.05)
    return np.random.randn(n).astype(np.float32) * exp_env(n, 0.008)


def clave():
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    return 0.5 * np.sin(2 * np.pi * 1000 * t) * exp_env(n, 0.01)


def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)


# ─────────── Pre‑render loops (1 bar) ───────────
def build_loop(base_only=False):
    buf = np.zeros(int(RATE * BAR_SEC), np.float32)
    q = 60 / BPM  # quarter length

    def put(s, pos_s):
        i = int(pos_s * RATE)
        buf[i : i + s.size] += s[: max(0, buf.size - i)]

    if base_only or True:
        put(kick(), 0)
        put(kick(), 1.5 * q)
        put(kick(), 2.5 * q)
        put(snare(), 1 * q)
        put(snare(), 3 * q)
    if not base_only:
        # hats 8th notes
        for off in np.arange(0, 4 * q, q / 2):
            put(hat(), off)
        # clave on beat 1 & 3
        put(clave(), 0)
        put(clave(), 2 * q)
    buf /= np.max(np.abs(buf))
    return buf


LOOP_BASE = build_loop(base_only=True)
LOOP_HC = build_loop(base_only=False) - LOOP_BASE  # hats+clave diff
BLEN = LOOP_BASE.size


# ─────────── Pad oscillator ───────────
class Pad:
    def __init__(self):
        self.phase = {}

    def tri(self, f, t, phi):
        sig = np.zeros_like(t)
        for k in range(1, 9, 2):  # odd harmonics
            sig += (-1) ** ((k - 1) // 2) * np.sin(2 * np.pi * k * f * t + phi) / k**2
        return sig * 8 / np.pi**2

    def step(self, nsamples, ang_deg):
        beauty = max(0, 1 - ang_deg / 40)
        root = midi2hz(60)  # C4
        notes = [root]
        if beauty > 0.25:
            notes += [root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
        if beauty > 0.60:
            notes += [root * 2 ** (9 / 12), root * 2 ** (14 / 12)]
        if beauty > 0.90:
            notes += [root * 2]
        vib = 0.004 * beauty
        t = np.arange(nsamples) / RATE
        buf = np.zeros(nsamples, np.float32)
        for f in notes:
            phi = self.phase.get(f, 0.0)
            buf += self.tri(f * (1 + vib * np.sin(2 * np.pi * 4 * t)), t, phi)
            self.phase[f] = (phi + 2 * np.pi * f * nsamples / RATE) % (2 * np.pi)
        buf /= len(notes)
        return buf * VOL_PAD


PAD = Pad()


# ─────────── Audio thread ───────────
def audio_loop():
    # grab first state so variables defined
    l2_mm, ang_deg = state_q.get()
    base_idx = hc_idx = 0
    while True:
        try:
            while True:
                l2_mm, ang_deg = state_q.get_nowait()
        except queue.Empty:
            pass

        ns = int(RATE * SLICE_SEC)

        # --- beat baseline (always audible) ---
        seg = (
            lambda loop, idx: loop[idx : idx + ns]
            if idx + ns < BLEN
            else np.concatenate((loop[idx:], loop[: (idx + ns) % BLEN]))
        )
        beat_base = seg(LOOP_BASE, base_idx) * VOL_BEAT_BASE
        base_idx = (base_idx + ns) % BLEN

        # --- hats+clave layer loudness below 10 mm ---
        closeness = max(0, 1 - l2_mm / 10)  # 0..1
        hc_gain = (closeness**2) * VOL_HATCLAVE  # quadratic fade‑in
        beat_hc = seg(LOOP_HC, hc_idx) * hc_gain
        hc_idx = (hc_idx + ns) % BLEN

        # --- extra kick “thump” gain below 10 mm ---
        beat_total = (
            beat_base * (1 + closeness * VOL_BEAT_GAIN / VOL_BEAT_BASE) + beat_hc
        )

        # --- pad from angle ---
        pad_slice = PAD.step(ns, ang_deg)

        mix = beat_total + pad_slice
        peak = np.max(np.abs(mix))
        if peak > 1:
            mix /= peak
        sa.play_buffer((mix * 32767).astype(np.int16), 1, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


threading.Thread(target=audio_loop, daemon=True).start()


# ─────────── Robot loop ───────────
def main():
    with ReachyMini() as reachy:
        reachy.disable_motors()
        target = np.eye(4)
        try:
            while True:
                joints, _ = reachy._get_current_joint_positions()
                pose = reachy.head_kinematics.fk(joints)
                l2, ang, _ = distance_between_poses(target, pose)
                l2_mm = l2 * 1000
                ang_deg = np.degrees(ang)

                state_q.put((l2_mm, ang_deg))

                print(f"\rl2 {l2_mm:6.2f} mm | ang {ang_deg:5.2f}°", end="")
                if l2_mm < 5 and ang_deg < 5:
                    print("\nTODO success – new pose.")
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
