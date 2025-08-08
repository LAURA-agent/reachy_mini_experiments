#!/usr/bin/env python3
"""Reachy Mini head‑pose game with rhythmic audio feedback.

* **Speed‑run (default)** – clear 3 targets as fast as possible.
* **--precision**        – 30 s countdown, aim for lowest magic score.
* **--cheats**           – print live pose & current target for debugging.
* Difficulty thresholds  – easy 25 | medium 12 | hard 6.

Audio mapping (unchanged from v11)
----------------------------------
Distance → kick / snare / hats volume
Angle    → rim‑woodblock volume
Both follow the `(1 − err/40)¹·⁷` curve so the groove stays balanced.
"""

from __future__ import annotations

import argparse
import queue
import random
import sys
import threading
import time
from typing import NoReturn

import numpy as np
import simpleaudio as sa
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses
from scipy.spatial.transform import Rotation as R

# ═════════════════════════════ AUDIO CONSTANTS ═════════════════════════════
RATE: int = 44_100
SLICE_SEC: float = 0.05
BPM: int = 96
BAR_SEC: float = 60 / BPM * 4  # length of 1 four‑beat bar

BASE_DIST_VOL = 0.15  # baseline kick/snare volume when far
MAX_DIST_GAIN = 0.55  # extra loudness when l2→0
MAX_ANG_GAIN = 0.45  # rim layer loudness when angle→0

state_q: queue.Queue[tuple[float, float]] = queue.Queue()  # (l2_mm, ang_deg)


# ═════════════════════════════ DRUM SYNTH HELPERS ══════════════════════════
def _exp_env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / RATE / tau, dtype=np.float32)


def _kick() -> np.ndarray:
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * _exp_env(n, 0.15)
    click = 0.14 * np.sin(2 * np.pi * 180 * t) * _exp_env(n, 0.004)
    return body + click


def _snare() -> np.ndarray:
    n = int(RATE * 0.18)
    noise = np.random.randn(n).astype(np.float32) * _exp_env(n, 0.05)
    tone = 0.3 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * _exp_env(n, 0.12)
    return noise + tone


def _hat() -> np.ndarray:
    n = int(RATE * 0.05)
    return np.random.randn(n).astype(np.float32) * _exp_env(n, 0.008)


def _rim() -> np.ndarray:
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    body = 0.6 * np.sin(2 * np.pi * 1_200 * t) * _exp_env(n, 0.01)
    noise = 0.3 * np.random.randn(n).astype(np.float32) * _exp_env(n, 0.01)
    return body + noise


# ═══════════════════ PRE‑RENDER ONE BAR OF EACH LOOP ═══════════════════════
def _build_dist_loop() -> np.ndarray:
    """Kick/snare/hats groove – always audible."""
    buf = np.zeros(int(RATE * BAR_SEC), np.float32)
    q = 60 / BPM

    def put(s: np.ndarray, pos: float) -> None:  # write with wrap‑guard
        i = int(pos * RATE)
        buf[i : i + s.size] += s[: max(0, buf.size - i)]

    put(_kick(), 0)
    put(_kick(), 1.5 * q)
    put(_kick(), 2.5 * q)
    put(_snare(), 1 * q)
    put(_snare(), 3 * q)
    for off in np.arange(0, 4 * q, q / 2):
        put(_hat(), off)
    return buf / np.max(np.abs(buf))


def _build_ang_loop() -> np.ndarray:
    """High‑pitched rim / shaker layer – volume follows angle."""
    buf = np.zeros(int(RATE * BAR_SEC), np.float32)
    q = 60 / BPM

    def put(s: np.ndarray, pos: float) -> None:
        i = int(pos * RATE)
        buf[i : i + s.size] += s[: max(0, buf.size - i)]

    for off in [0.75 * q, 2.25 * q]:
        put(_rim(), off)  # complement hats
    shaker = np.random.randn(int(RATE * 0.04)).astype(np.float32) * _exp_env(
        int(RATE * 0.04), 0.005
    )
    for off in np.arange(0, 4 * q, q / 4):
        put(shaker, off)
    return buf / np.max(np.abs(buf))


DIST_LOOP = _build_dist_loop()
ANG_LOOP = _build_ang_loop()
DLEN, ALEN = DIST_LOOP.size, ANG_LOOP.size


# ═════════════════════════════ SUCCESS CHIME ═══════════════════════════════
def _success_chime() -> None:
    """Play a short major‑triad bell."""
    dur = 0.3
    n = int(RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    carrier = np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1_320 * t)
    carrier *= np.exp(-6 * t)
    sa.play_buffer((carrier * 32_767).astype(np.int16), 1, 2, RATE)


# ════════════════════════════ POSE UTILITIES ═══════════════════════════════
def random_target_pose() -> np.ndarray:
    """Random pose within ±10 ° orientation & specified mm bounds."""
    x = random.uniform(-8, 5)
    y = random.uniform(-10, 10)
    z = random.uniform(-15, 5)
    roll = random.uniform(-10, 10)
    pitch = random.uniform(-10, 10)
    yaw = random.uniform(-10, 10)

    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    pose[:3, 3] = np.array([x, y, z]) / 1_000  # mm → m
    return pose


def _pose_to_xyz_rpy(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = pose[:3, 3] * 1_000
    rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    return xyz, rpy


def print_pose(pose: np.ndarray, label: str = "pose") -> None:
    """Print pose as x y z (mm) | roll pitch yaw (°)."""
    xyz, rpy = _pose_to_xyz_rpy(pose)
    x, y, z = xyz
    roll, pitch, yaw = rpy
    print(
        f"{label}: "
        f"x={x:6.1f} mm y={y:6.1f} mm z={z:6.1f} mm | "
        f"roll={roll:6.1f}° pitch={pitch:6.1f}° yaw={yaw:6.1f}°"
    )


# ═════════════════════════════ AUDIO THREAD ═══════════════════════════════
def _audio_thread() -> None:
    """Stream 50 ms slices mixed from the two pre‑rendered loops."""
    # unblock immediately with neutral values
    l2_mm, ang_deg = 40.0, 40.0
    d_idx = a_idx = 0
    while True:
        try:  # drain latest pose state
            while True:
                l2_mm, ang_deg = state_q.get_nowait()
        except queue.Empty:
            pass

        dist_gain = (max(0.0, 1 - l2_mm / 40)) ** 0.7
        ang_gain = (max(0.0, 1 - ang_deg / 40)) ** 0.7

        ns = int(RATE * SLICE_SEC)

        def seg(loop: np.ndarray, idx: int) -> np.ndarray:
            return (
                loop[idx : idx + ns]
                if idx + ns < loop.size
                else np.concatenate((loop[idx:], loop[: (idx + ns) % loop.size]))
            )

        dist_slice = seg(DIST_LOOP, d_idx) * (BASE_DIST_VOL + dist_gain * MAX_DIST_GAIN)
        ang_slice = seg(ANG_LOOP, a_idx) * (ang_gain * MAX_ANG_GAIN)

        d_idx = (d_idx + ns) % DLEN
        a_idx = (a_idx + ns) % ALEN

        mix = dist_slice + ang_slice
        peak = np.max(np.abs(mix))
        if peak > 1:
            mix /= peak
        sa.play_buffer((mix * 32_767).astype(np.int16), 1, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


# ═════════════════════════════ GAME MODES ═════════════════════════════════
def speed_run(threshold: float, cheats: bool) -> None:
    """Three‑target speed‑run; score = total time."""
    targets = [np.eye(4)] + [random_target_pose() for _ in range(2)]
    current = 0
    best_magic = float("inf")
    start = time.monotonic()

    with ReachyMini() as reachy:
        reachy.disable_motors()

        while current < 3:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)

            if cheats:
                print_pose(pose, "current")
                print_pose(targets[current], "target ")

            t_dist, a_dist, magic = distance_between_poses(targets[current], pose)
            best_magic = min(best_magic, magic)

            state_q.put((t_dist * 1_000, np.degrees(a_dist)))

            print(
                f"\rl2={t_dist * 1000:6.1f} mm "
                f"angle={np.degrees(a_dist):5.2f}° "
                f"magic={magic:6.2f} "
                f"best={best_magic:6.2f}",
                end="",
                flush=True,
            )

            if magic < threshold:
                _success_chime()
                current += 1
                best_magic = float("inf")
                if current < 3:
                    print("\nNext target!")

            time.sleep(0.02)

    print(f"\n🎉  Finished in {time.monotonic() - start:.2f} s!")


def precision_mode() -> None:
    """30‑second countdown, aim for the lowest magic score."""
    deadline = time.monotonic() + 30.0
    best_magic = float("inf")

    with ReachyMini() as reachy:
        reachy.disable_motors()
        print("Precision mode – 30 s to hit the lowest magic score!")
        while (remain := deadline - time.monotonic()) > 0:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)
            t_dist, a_dist, magic = distance_between_poses(np.eye(4), pose)
            best_magic = min(best_magic, magic)

            state_q.put((t_dist * 1_000, np.degrees(a_dist)))

            print(
                f"\rtime left={remain:4.1f} s "
                f"magic={magic:6.2f} best={best_magic:6.2f}",
                end="",
                flush=True,
            )
            time.sleep(0.02)

    print(f"\n⌛️  Best magic score = {best_magic:.2f}")


# ═══════════════════════════════════ CLI ══════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini head‑pose game")
    parser.add_argument(
        "-d",
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="distance+angle threshold (default: medium = 12)",
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="30 s precision‑challenge mode",
    )
    parser.add_argument(
        "--cheats",
        action="store_true",
        help="print live pose and current target",
    )
    return parser.parse_args()


# ═════════════════════════════ MAIN ENTRY ════════════════════════════════
def main() -> NoReturn:
    """Entry point – start audio thread, then chosen game mode."""
    # start continuous audio
    state_q.put((40.0, 40.0))  # seed with neutral error so audio begins
    threading.Thread(target=_audio_thread, daemon=True).start()

    args = _parse_args()
    if args.precision:  # precision mode overrides difficulty
        precision_mode()
        sys.exit()

    thresholds = {"easy": 25.0, "medium": 12.0, "hard": 6.0}
    speed_run(thresholds[args.difficulty], cheats=args.cheats)
    sys.exit()


if __name__ == "__main__":
    main()
