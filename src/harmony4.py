#!/usr/bin/env python3
"""
Reachy Mini sonic distance game v3 (mono friendly)

Mappings
--------
distance (mm):
  200 -> slow (0.6 s/beat), low pitch (approx A3)
    |
    v
    0 -> fast (0.10 s/beat), high pitch (approx C5)

angle (deg):
  30 -> sparse 1 hit/beat, straight, dissonant cluster
    |
    v
    0 -> 4 subdivisions/beat w swing, consonant add6/9 shimmer, grace sparkles

Success condition prints TODO (no wav).
"""

import time
import threading
import queue
import numpy as np
import simpleaudio as sa

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ------------------- audio config -------------------
RATE = 44100  # Hz
VOL = 0.35  # master volume 0-1
MAX_SUBDIV = 4  # max rhythmic density when well aligned
ENV_DECAY = 0.08  # seconds note tail (base, scaled by beat len)
SPARK_ODDS = 0.25  # chance of sparkle hit when well aligned

_state_q = queue.Queue()  # (distance_mm, angle_deg) from main loop


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _angle_to_subdiv(angle_deg):
    # small angle -> more notes
    return int(np.round(np.interp(angle_deg, [0, 30], [MAX_SUBDIV, 1])))


def _angle_to_tension(angle_deg):
    # 0 -> consonant, 1 -> crunchy
    return float(np.clip(angle_deg / 30.0, 0.0, 1.0))


def _angle_to_swing(angle_deg):
    # more swing when well aligned
    return float(np.interp(angle_deg, [0, 30], [0.5, 0.0]))  # 0-50 percent swing


def _distance_to_period(distance_mm):
    return float(np.interp(distance_mm, [0, 200], [0.10, 0.60]))  # seconds/beat


def _distance_to_root_midi(distance_mm):
    # closer -> higher
    return float(np.interp(distance_mm, [0, 200], [72, 57]))  # C5 down to A3


def _make_env(length_s, amp=1.0):
    # simple fast attack exp decay
    n = max(1, int(RATE * length_s))
    t = np.linspace(0, length_s, n, endpoint=False)
    env = np.exp(-4 * (t / length_s)) * amp
    return env.astype(np.float32)


def _synth_hit(freqs, length_s, amp=1.0):
    env = _make_env(length_s, amp)
    t = np.arange(env.size) / RATE
    sig = np.zeros_like(env)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)
    sig /= len(freqs)
    sig *= env
    return sig


def _choose_chord(root_hz, tension):
    """
    Return list of freqs.
    tension=0 -> consonant (major +6/9 shimmer)
    tension=1 -> crunchy cluster (add 2, b9-ish, tritone)
    """
    # always include root
    freqs = [root_hz]
    if tension <= 0.01:
        # lovely chord: 3, 5, 6, 9 partial shimmer
        freqs += [
            root_hz * 2 ** (4 / 12),  # M3
            root_hz * 2 ** (7 / 12),  # P5
            root_hz * 2 ** (9 / 12),  # M6
            root_hz * 2 ** (14 / 12),  # 9
        ]
    else:
        # gradually mix in dissonance as tension rises
        # base triad
        freqs += [
            root_hz * 2 ** (4 / 12),
            root_hz * 2 ** (7 / 12),
        ]
        # dissonant additions weight by tension
        if tension > 0.2:
            freqs.append(root_hz * 2 ** (1 / 12))  # m2 cluster
        if tension > 0.5:
            freqs.append(root_hz * 2 ** (6 / 12))  # Tritone (#11)
        if tension > 0.8:
            freqs.append(root_hz * 2 ** (10 / 12))  # b7 grind
    return freqs


def _maybe_sparkle(root_hz, tension):
    """
    Optional short high ping when well aligned.
    More likely when tension < 0.25.
    """
    if tension >= 0.25:
        return None
    if np.random.rand() > SPARK_ODDS:
        return None
    # sparkle frequencies: 2 octaves up + fifth
    f = root_hz * 4
    return _synth_hit([f, f * 2 ** (7 / 12)], 0.04, amp=0.4)


def _make_beat(distance_mm, angle_deg):
    """
    Build mono float32 buffer for one beat according to current state.
    """
    period = _distance_to_period(distance_mm)
    root_m = _distance_to_root_midi(distance_mm)
    root_hz = midi_to_hz(root_m)
    subdiv = _angle_to_subdiv(angle_deg)
    tension = _angle_to_tension(angle_deg)
    swing = _angle_to_swing(angle_deg)

    # time grid for event onsets
    # swing pushes even subdivisions later
    base_step = period / subdiv
    onsets = []
    for i in range(subdiv):
        o = i * base_step
        if i % 2 == 1:  # swing the offbeats
            o += base_step * swing
            if o > period:
                o = period  # clamp
        onsets.append(o)

    # synth each event
    buf_n = int(RATE * period)
    buf = np.zeros(buf_n, dtype=np.float32)
    chord_freqs = _choose_chord(root_hz, tension)
    hit_len = min(ENV_DECAY, base_step * 0.8)  # scale to grid

    for i, o in enumerate(onsets):
        start = int(o * RATE)
        hit = _synth_hit(chord_freqs, hit_len, amp=1.0 if i == 0 else 0.7)
        end = min(buf_n, start + hit.size)
        buf[start:end] += hit[: end - start]

    # sparkle
    spark = _maybe_sparkle(root_hz, tension)
    if spark is not None:
        s_start = int((period * 0.75) * RATE)  # near end of beat
        s_end = min(buf_n, s_start + spark.size)
        buf[s_start:s_end] += spark[: s_end - s_start]

    # normalize soft clip
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = (buf / peak) * VOL

    # convert to int16 mono
    return (buf * 32767).astype(np.int16), period


def _audio_thread():
    # consume latest state; if multiple arrive, take the newest
    dist = 200.0
    ang = 30.0
    next_update = time.time()
    while True:
        # drain queue
        try:
            while True:
                dist, ang = _state_q.get_nowait()
        except queue.Empty:
            pass

        buf, period = _make_beat(dist, ang)
        sa.play_buffer(buf, 1, 2, RATE)  # mono
        next_update += period
        sleep_time = max(0.0, next_update - time.time())
        time.sleep(sleep_time)


def start_audio_thread():
    threading.Thread(target=_audio_thread, daemon=True).start()


def update_audio_state(distance_mm, angle_deg):
    # push latest numbers; audio thread keeps newest
    _state_q.put((float(distance_mm), float(angle_deg)))


# ------------------- main reachy loop -------------------
def main():
    start_audio_thread()
    with ReachyMini() as reachy:
        reachy.disable_motors()
        target_pose = np.eye(4)
        best = float("inf")

        while True:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)
            l2, ang, score = distance_between_poses(target_pose, pose)

            l2_mm = float(l2 * 1000.0)
            ang_deg = float(np.degrees(ang))
            best = min(best, score)

            # update audio engine
            update_audio_state(l2_mm, ang_deg)

            print(
                f"\rl2: {l2_mm:6.1f} mm | ang: {ang_deg:5.2f}° | best: {best:6.2f}",
                end="",
            )

            # success zone check (no wav yet)
            if l2_mm < 5 and ang_deg < 2:
                print("\nTODO success: reached pose. Picking new target.")
                target_pose = reachy.head_kinematics.random_pose()
                best = float("inf")

            time.sleep(0.02)


if __name__ == "__main__":
    main()
