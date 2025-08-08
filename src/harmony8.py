#!/usr/bin/env python3
"""
Reachy‑Mini sonic game v10
  • Distance scaling reverted to v7 curve.
  • Three pad modes switchable at runtime (press 1/2/3).
  • Prints: l2 mm | angle ° | magic | best score.
"""

import time, threading, queue, sys, select, termios, tty
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ─────────── audio constants ───────────
RATE = 44100
SLICE_SEC = 0.05
VOL_BEAT_BASE = 0.30
VOL_BEAT_GAIN = 0.45
VOL_HC = 0.45  # hats+clave
VOL_PAD = 0.30

BPM = 96
BAR_SEC = 60 / BPM * 4

state_q = queue.Queue()  # (l2_mm, ang_deg)
mode_q = queue.Queue()  # angle‑mode index (1‑3)


# ─────────── helper: drums ───────────
def exp_env(n, tau):
    return np.exp(-np.arange(n) / RATE / tau, dtype=np.float32)


def kick():
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * exp_env(n, 0.15)
    click = 0.15 * np.sin(2 * np.pi * 180 * t) * exp_env(n, 0.005)
    return body + click


def snare():
    n = int(RATE * 0.18)
    noise = np.random.randn(n).astype(np.float32) * exp_env(n, 0.04)
    tone = 0.25 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * exp_env(n, 0.12)
    return noise + tone


def hat():
    n = int(RATE * 0.05)
    return np.random.randn(n).astype(np.float32) * exp_env(n, 0.008)


def clave():
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    return 0.5 * np.sin(2 * np.pi * 1000 * t) * exp_env(n, 0.01)


def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)


# ─────────── pre‑render loops ───────────
def build_loop(base_only=False):
    buf = np.zeros(int(RATE * BAR_SEC), np.float32)
    q = 60 / BPM

    def put(s, sec):
        i = int(sec * RATE)
        buf[i : i + s.size] += s[: max(0, buf.size - i)]

    put(kick(), 0)
    put(kick(), 1.5 * q)
    put(kick(), 2.5 * q)
    put(snare(), 1 * q)
    put(snare(), 3 * q)
    if not base_only:
        for off in np.arange(0, 4 * q, q / 2):
            put(hat(), off)
        put(clave(), 0)
        put(clave(), 2 * q)
    return buf / np.max(np.abs(buf))


LOOP_BASE = build_loop(True)
LOOP_HC = build_loop(False) - LOOP_BASE
BLEN = LOOP_BASE.size


# ─────────── pad modes ───────────
def tri_pad(ns, ang):
    beauty = max(0, 1 - ang / 40)
    root = midi2hz(60)
    notes = [root]
    if beauty > 0.25:
        notes += [root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
    if beauty > 0.60:
        notes += [root * 2 ** (9 / 12), root * 2 ** (14 / 12)]
    if beauty > 0.90:
        notes += [root * 2]
    vib = 0.004 * beauty
    t = np.arange(ns) / RATE
    buf = np.zeros(ns, np.float32)
    for f in notes:
        buf += sum(
            ((-1) ** k) * np.sin(2 * np.pi * f * (2 * k + 1) * t) / ((2 * k + 1) ** 2)
            for k in range(4)
        )
    buf /= len(notes)
    buf *= 1 + np.sin(2 * np.pi * 4 * t) * vib
    return buf * VOL_PAD


def saw_pad(ns, ang):
    bright = max(0, 1 - ang / 40)
    root = midi2hz(48)  # lower
    t = np.arange(ns) / RATE
    wave = sum(np.sin(2 * np.pi * root * (k + 1) * t) / (k + 1) for k in range(20))
    lp_cut = 300 + bright * 2500
    rc = 1 / (2 * np.pi * lp_cut)
    alpha = SLICE_SEC / (rc + SLICE_SEC)
    saw_pad.prev = alpha * wave + (1 - alpha) * getattr(saw_pad, "prev", 0)
    return saw_pad.prev.astype(np.float32) * VOL_PAD


saw_pad.prev = 0


def fm_bell(ns, ang):
    bright = max(0, 1 - ang / 40)
    carrier = midi2hz(72)  # C5
    mod = carrier * 1.414
    t = np.arange(ns) / RATE
    buf = np.sin(2 * np.pi * carrier * t + 5 * bright * np.sin(2 * np.pi * mod * t))
    buf *= exp_env(ns, int(0.3 * RATE))
    return buf * VOL_PAD * 1.2


PAD_MODES = {1: tri_pad, 2: saw_pad, 3: fm_bell}
current_mode = 1


# ─────────── audio thread ───────────
def audio_thread():
    global current_mode
    l2_mm, ang = state_q.get()
    base_idx = hc_idx = 0
    while True:
        # grab latest state
        try:
            while True:
                l2_mm, ang = state_q.get_nowait()
        except queue.Empty:
            pass
        # grab mode switch
        try:
            while True:
                current_mode = mode_q.get_nowait()
        except queue.Empty:
            pass

        ns = int(RATE * SLICE_SEC)
        seg = (
            lambda loop, idx: loop[idx : idx + ns]
            if idx + ns < BLEN
            else np.concatenate((loop[idx:], loop[: (idx + ns) % BLEN]))
        )
        beat = seg(LOOP_BASE, base_idx) * VOL_BEAT_BASE
        base_idx = (base_idx + ns) % BLEN

        closeness = max(0, 1 - l2_mm / 10)
        beat += seg(LOOP_HC, hc_idx) * VOL_HC * (closeness**2)
        beat *= 1 + closeness * VOL_BEAT_GAIN / VOL_BEAT_BASE
        hc_idx = (hc_idx + ns) % BLEN

        pad = PAD_MODES[current_mode](ns, ang)
        mix = beat + pad
        peak = np.max(np.abs(mix))
        mix /= peak if peak > 1 else 1
        sa.play_buffer((mix * 32767).astype(np.int16), 1, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


threading.Thread(target=audio_thread, daemon=True).start()


# ─────────── non‑blocking key listener ───────────
def key_listener():
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch in "123":
                    mode_q.put(int(ch))
                    print(f"\nAngle mode → {ch}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)


threading.Thread(target=key_listener, daemon=True).start()


# ─────────── robot loop ───────────
def main():
    best = float("inf")
    with ReachyMini() as r:
        r.disable_motors()
        target = np.eye(4)
        try:
            while True:
                j, _ = r._get_current_joint_positions()
                pose = r.head_kinematics.fk(j)
                l2, ang, magic = distance_between_poses(target, pose)
                l2_mm = l2 * 1000
                ang_deg = np.degrees(ang)
                best = min(best, magic)
                state_q.put((l2_mm, ang_deg))

                print(
                    f"\rl2 {l2_mm:6.2f} mm | ang {ang_deg:5.2f}° | magic {magic:6.3f} | best {best:6.3f}",
                    end="",
                )
                if l2_mm < 5 and ang_deg < 5:
                    print("\nTODO success – new target.")
                    target = r.head_kinematics.random_pose()
                    best = float("inf")
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
