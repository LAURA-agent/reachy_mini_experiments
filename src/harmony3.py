#!/usr/bin/env python3
"""
Reachy-Mini Musical Feedback v3
Mono output, musical and rhythmic:
- Distance controls tempo & pitch
- Angle controls harmonic richness (1-3 notes)
"""

import time, threading, queue
import numpy as np
import simpleaudio as sa
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ---------- Audio Engine ----------
RATE = 44100
VOL = 0.25
BLOCK_DUR = 0.25  # chord/arpeggio slice
DECAY = 5.0

_beep_q = queue.Queue()


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def make_arpeggio(freqs, dur=BLOCK_DUR):
    """Combine frequencies in a quick arp-like pattern."""
    t = np.arange(int(RATE * dur)) / RATE
    env = np.exp(-DECAY * t)
    # 3 segments, one note at a time
    seg_len = len(t) // len(freqs)
    wave = np.zeros_like(t)
    for i, f in enumerate(freqs):
        seg = slice(i * seg_len, (i + 1) * seg_len)
        wave[seg] = np.sin(2 * np.pi * f * t[seg])
    wave *= env * VOL
    return (wave * 32767).astype(np.int16)


def audio_thread():
    while True:
        buf = _beep_q.get()
        sa.play_buffer(buf, 1, 2, RATE)  # mono


def start_audio():
    threading.Thread(target=audio_thread, daemon=True).start()


def schedule_arpeggio(distance_mm, angle_deg):
    # distance -> root note (C3=48 .. C5=72)
    midi_root = np.interp(distance_mm, [0, 200], [72, 48])
    root = midi_to_hz(midi_root)
    third = root * 2 ** (4 / 12)
    fifth = root * 2 ** (7 / 12)

    # angle -> how many notes (1..3)
    n_notes = int(np.interp(angle_deg, [0, 30], [3, 1]))
    freqs = [root]
    if n_notes >= 2:
        freqs.append(third)
    if n_notes >= 3:
        freqs.append(fifth)

    buf = make_arpeggio(freqs)
    _beep_q.put(buf)


def period_from_distance(d_mm):
    return np.interp(d_mm, [0, 200], [0.25, 1.0])  # fast when close


# ---------- Main Loop ----------
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
                schedule_arpeggio(l2_mm, ang_deg)
                last_ping = now

            print(
                f"\rl2: {l2_mm:6.1f} mm | ang: {ang_deg:5.2f}° | best: {best:6.2f}",
                end="",
            )

            if l2_mm < 5 and ang_deg < 2:
                print("\nTODO: success sound")
                target_pose = reachy.head_kinematics.random_pose()
                best = float("inf")
                last_ping = now

            time.sleep(0.02)


if __name__ == "__main__":
    main()
