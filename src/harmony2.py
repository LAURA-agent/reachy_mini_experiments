#!/usr/bin/env python3
"""
Reachy‑Mini sonic distance game v2
‑ single file, zero audio config
‑ angle -> stereo pan
‑ distance -> ping rate and pitch
"""

import time, threading, queue
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ---------- sound helper section ----------
RATE = 44100  # samples per second
VOL = 0.3  # master volume 0‑1
DECAY = 5.0  # envelope decay constant

_beep_q = queue.Queue()


def midi_to_hz(m):
    """MIDI note number to frequency."""
    return 440.0 * 2 ** ((m - 69) / 12)


def make_triad(freq_root, dur=0.15):
    """Return stereo int16 buffer holding a decaying major triad."""
    f2 = freq_root * 2 ** (4 / 12)  # major 3rd
    f3 = freq_root * 2 ** (7 / 12)  # perfect 5th
    t = np.arange(int(RATE * dur)) / RATE
    env = np.exp(-DECAY * t)
    mono = (np.sin(2 * np.pi * f * t) for f in (freq_root, f2, f3))
    mono = sum(mono) / 3.0 * env * VOL
    return (mono * 32767).astype(np.int16)


def audio_thread():
    while True:
        buf, pan = _beep_q.get()
        left = (buf * (1.0 - pan)).astype(np.int16)
        right = (buf * pan).astype(np.int16)
        stereo = np.column_stack((left, right))
        sa.play_buffer(stereo, 2, 2, RATE)  # 2 bytes per sample


def start_audio():
    threading.Thread(target=audio_thread, daemon=True).start()


def schedule_ping(distance_mm, angle_deg):
    """Put a ping buffer and pan value on the queue."""
    # distance 0‑200 mm -> MIDI 72‑60 (C5‑C4)
    midi_root = np.interp(distance_mm, [0, 200], [72, 60])
    freq = midi_to_hz(midi_root)
    buf = make_triad(freq)
    # angle 0‑30° -> pan 0.5‑0.0 (centre‑left)
    pan = np.interp(angle_deg, [0, 30], [0.5, 0.0])
    _beep_q.put((buf, pan))


# ---------- mapping helpers ----------
def period_from_distance(d_mm):
    """distance 0‑200 mm -> period 0.05‑0.6 s (inverse linear)."""
    return np.interp(d_mm, [0, 200], [0.05, 0.6])


# ---------- main game loop ----------
def main():
    start_audio()
    with ReachyMini() as reachy:
        reachy.disable_motors()
        target_pose = np.eye(4)
        best = float("inf")
        last_ping = time.time()

        while True:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)

            l2, ang, score = distance_between_poses(target_pose, pose)
            l2_mm = l2 * 1000
            ang_deg = np.degrees(ang)
            best = min(best, score)

            now = time.time()
            if now - last_ping >= period_from_distance(l2_mm):
                schedule_ping(l2_mm, ang_deg)
                last_ping = now

            print(
                f"\rl2: {l2_mm:6.1f} mm | ang: {ang_deg:5.2f}° | best: {best:6.2f}",
                end="",
            )

            if l2_mm < 5 and ang_deg < 2:
                print("\nTODO: success sound here")
                best = float("inf")
                last_ping = now  # avoid burst of old period

            time.sleep(0.02)


if __name__ == "__main__":
    main()
