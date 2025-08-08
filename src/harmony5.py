#!/usr/bin/env python3
"""
Reachy‑Mini sonic game v4 – delta‑cues version (mono)

Dependencies:
    pip install simpleaudio numpy reachy_mini
"""

import time, threading, queue, collections
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ------------------ audio constants ------------------
RATE = 44100
VOL_MAIN = 0.33
VOL_CUE = 0.45
ENV_DECAY = 0.08
MAX_SUBDIV = 4  # rhythmic density @ angle 0
CUE_WINDOW = 0.3  # seconds cue buffers last
DELTA_THRESH = 0.01  # 1 % relative change before cue fires

state_q = queue.Queue()  # (dist_mm, ang_deg)
cue_q = queue.Queue()  # 'up' | 'down' | None


# ------------------ helpers ------------------
def midi2hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def distance_period(mm):  # 0‑200 → 0.1‑0.6 s
    return float(np.interp(mm, [0, 200], [0.10, 0.60]))


def distance_root(mm):  # 0‑200 → C5‑A3
    return float(np.interp(mm, [0, 200], [72, 57]))


def angle_subdiv(deg):  # 0‑30 → 4‑1
    return int(np.round(np.interp(deg, [0, 30], [MAX_SUBDIV, 1])))


def angle_tension(deg):  # 0‑30 → 0‑1
    return float(np.clip(deg / 30, 0, 1))


def angle_swing(deg):  # 0‑30 → 0.5‑0
    return float(np.interp(deg, [0, 30], [0.5, 0]))


def env(length, amp=1.0):
    n = int(RATE * length)
    t = np.linspace(0, length, n, False)
    return (np.exp(-4 * t / length) * amp).astype(np.float32)


def synth_hit(freqs, length, amp=1.0):
    e = env(length, amp)
    t = np.arange(e.size) / RATE
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return sig * e


def choose_chord(root_hz, tension):
    f = [
        root_hz,
        root_hz * 2 ** (4 / 12),  # M3
        root_hz * 2 ** (7 / 12),
    ]  # P5
    if tension < 0.2:
        f += [root_hz * 2 ** (9 / 12), root_hz * 2 ** (14 / 12)]  # 6 & 9 shimmer
    elif tension > 0.6:
        f += [root_hz * 2 ** (1 / 12), root_hz * 2 ** (6 / 12)]  # crunch
    return f


# cue buffers (pre‑rendered)
CUE_UP = synth_hit([midi2hz(84), midi2hz(88)], 0.08, VOL_CUE)  # up M3
CUE_DOWN = synth_hit([midi2hz(84), midi2hz(83)], 0.08, VOL_CUE)  # down m2
CUE_UP = (CUE_UP / np.max(np.abs(CUE_UP)) * 32767).astype(np.int16)
CUE_DOWN = (CUE_DOWN / np.max(np.abs(CUE_DOWN)) * 32767).astype(np.int16)


# ------------------ audio thread ------------------
def beat_buffer(dist_mm, ang_deg):
    period = distance_period(dist_mm)
    root_hz = midi2hz(distance_root(dist_mm))
    subdivisions = angle_subdiv(ang_deg)
    tension = angle_tension(ang_deg)
    swing = angle_swing(ang_deg)

    hit_len = min(ENV_DECAY, period / subdivisions * 0.8)
    buf = np.zeros(int(RATE * period), np.float32)
    chord = choose_chord(root_hz, tension)

    step = period / subdivisions
    for i in range(subdivisions):
        onset = i * step + (step * swing if i % 2 and subdivisions > 1 else 0)
        idx = int(onset * RATE)
        hit = synth_hit(chord, hit_len, 1.0 if i == 0 else 0.7)
        end = min(buf.size, idx + hit.size)
        buf[idx:end] += hit[: end - idx]

    # normalise & convert
    if (peak := np.max(np.abs(buf))) > 0:
        buf = buf / peak * VOL_MAIN
    return (buf * 32767).astype(np.int16), period


def audio_loop():
    dist, ang = 200.0, 30.0
    next_tick = time.time()
    while True:
        # latest continuous state
        try:
            while True:
                dist, ang = state_q.get_nowait()
        except queue.Empty:
            pass
        beat, period = beat_buffer(dist, ang)

        # play continuous beat
        sa.play_buffer(beat, 1, 2, RATE)

        # check for cue
        try:
            cue = cue_q.get_nowait()
            if cue == "up":
                sa.play_buffer(CUE_UP, 1, 2, RATE)
            elif cue == "down":
                sa.play_buffer(CUE_DOWN, 1, 2, RATE)
        except queue.Empty:
            pass

        next_tick += period
        time.sleep(max(0, next_tick - time.time()))


threading.Thread(target=audio_loop, daemon=True).start()


# ------------------ main robot loop ------------------
def main():
    prev_score = None
    # moving average to smooth out tiny jiggles
    avg = collections.deque([], maxlen=5)

    with ReachyMini() as reachy:
        reachy.disable_motors()
        target_pose = np.eye(4)

        while True:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)
            l2, ang, score = distance_between_poses(target_pose, pose)
            l2_mm = l2 * 1000
            ang_deg = np.degrees(ang)

            # send state to audio
            state_q.put((l2_mm, ang_deg))

            # delta cue logic
            avg.append(score)
            mean = sum(avg) / len(avg)
            if prev_score is not None:
                delta = (mean - prev_score) / prev_score
                if delta < -DELTA_THRESH:
                    cue_q.put("up")
                elif delta > DELTA_THRESH:
                    cue_q.put("down")
            prev_score = mean

            # text HUD
            print(
                f"\rl2 {l2_mm:6.1f} mm | ang {ang_deg:5.2f}° | score {score:7.4f}",
                end="",
            )

            # TODO success (keep placeholder)
            if l2_mm < 5 and ang_deg < 2:
                print("\nTODO success reached. New target.")
                target_pose = reachy.head_kinematics.random_pose()
                prev_score = None
                avg.clear()

            time.sleep(0.02)


if __name__ == "__main__":
    main()
