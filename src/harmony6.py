#!/usr/bin/env python3
"""
Reachy‑Mini sonic game – raggaetón beat (loudness = proximity) +
pad that blossoms with alignment.  Bug‑fixed version.
"""

import time, threading, queue, math, collections
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# -------------------- global audio constants --------------------
RATE = 44100
SLICE = 0.05  # seconds per buffer
VOL_BEAT_MAX = 0.5
VOL_PAD = 0.35
BEAT_LEN_BAR = 4  # quarter notes / bar
BPM = 96
BAR_SEC = 60 / BPM * BEAT_LEN_BAR

state_q = queue.Queue()  # (l2_mm, ang_deg)


##################### 1 · helper synthesis functions ########################
def exp_env(n, tau):
    t = np.arange(n) / RATE
    return np.exp(-t / tau)


def kick(dur=0.30, f0=80):
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    env = exp_env(n, 0.18)
    tone = np.sin(2 * np.pi * f0 * t * (1 - (t / dur) * 0.5))
    click = 0.2 * np.sin(2 * np.pi * 180 * t) * np.exp(-200 * t)
    return (tone * env + click).astype(np.float32)


def snare(dur=0.20):
    n = int(RATE * dur)
    noise = np.random.randn(n) * exp_env(n, 0.05)
    tone = 0.3 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * exp_env(n, 0.15)
    return (noise + tone).astype(np.float32)


def hat(dur=0.05):
    n = int(RATE * dur)
    return (np.random.randn(n) * exp_env(n, 0.01)).astype(np.float32)


def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)


##################### 2 · pre‑render raggaetón loop #########################
def build_beat_loop():
    n_bar = int(RATE * BAR_SEC)
    beat = np.zeros(n_bar, np.float32)

    def put(sample, at_sec):
        i = int(at_sec * RATE)
        beat[i : i + sample.size] += sample[: max(0, n_bar - i)]

    q = 60 / BPM
    put(kick(), 0 * q)
    put(kick(), 1.5 * q)
    put(kick(), 2.5 * q)
    put(snare(), 1 * q)
    put(snare(), 3 * q)
    for off in np.arange(0.0, BEAT_LEN_BAR * q, q / 2):
        put(hat(), off)

    beat /= np.max(np.abs(beat))
    return beat


BEAT_LOOP = build_beat_loop()
BLEN = BEAT_LOOP.size


##################### 3 · pad oscillator ##############################
class PadOsc:
    def __init__(self):
        self.phases = {}

    def step(self, nsamples, ang_deg):
        beauty = max(0.0, 1 - ang_deg / 40)
        root = midi2hz(60)
        freqs = [root]
        if beauty > 0.2:
            freqs += [root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
        if beauty > 0.5:
            freqs += [root * 2 ** (9 / 12), root * 2 ** (14 / 12)]
        if beauty > 0.8:
            freqs += [root * 2]
        buf = np.zeros(nsamples, np.float32)
        t = np.arange(nsamples) / RATE
        for f in freqs:
            ph = self.phases.get(f, 0.0)
            wave = np.sin(2 * np.pi * f * t + ph)
            self.phases[f] = (ph + 2 * np.pi * f * nsamples / RATE) % (2 * np.pi)
            buf += wave
        buf /= len(freqs)
        return buf


PAD = PadOsc()


##################### 4 · audio thread ######################################
def audio_loop():
    # --- wait for first state so vars are defined ---
    l2_mm, ang_deg = state_q.get()  # blocking
    beat_idx = 0
    while True:
        # non‑blocking updates afterwards
        try:
            while True:
                l2_mm, ang_deg = state_q.get_nowait()
        except queue.Empty:
            pass

        ns = int(RATE * SLICE)

        vol = max(0.0, 1 - (l2_mm / 40)) ** 0.7
        start = beat_idx
        end = beat_idx + ns
        if end < BLEN:
            beat_slice = BEAT_LOOP[start:end]
        else:
            beat_slice = np.concatenate((BEAT_LOOP[start:], BEAT_LOOP[: end % BLEN]))
        beat_idx = end % BLEN
        beat_slice = beat_slice * (vol * VOL_BEAT_MAX)

        pad_slice = PAD.step(ns, ang_deg) * VOL_PAD

        mix = beat_slice + pad_slice
        peak = np.max(np.abs(mix))
        if peak > 1:
            mix /= peak
        sa.play_buffer((mix * 32767).astype(np.int16), 1, 2, RATE)
        time.sleep(SLICE * 0.9)


threading.Thread(target=audio_loop, daemon=True).start()


##################### 5 · robot loop ########################################
def main():
    with ReachyMini() as r:
        r.disable_motors()
        target = np.eye(4)
        try:
            while True:
                joints, _ = r._get_current_joint_positions()
                pose = r.head_kinematics.fk(joints)
                l2, ang, score = distance_between_poses(target, pose)
                l2_mm = l2 * 1000
                ang_deg = np.degrees(ang)

                state_q.put((l2_mm, ang_deg))

                print(
                    f"\rl2 {l2_mm:6.2f} mm | ang {ang_deg:5.2f}° | score {score:7.4f}",
                    end="",
                )
                if l2_mm < 5 and ang_deg < 5:
                    print("\nTODO success! new pose.")
                    target = r.head_kinematics.random_pose()
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nExiting…")


if __name__ == "__main__":
    main()
