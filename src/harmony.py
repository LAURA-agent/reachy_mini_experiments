import numpy as np, time
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

import simpleaudio as sa
import numpy as np, queue, threading, time

RATE = 44100  # Sa/s
BLOCK = 0.05  # seconds per audio slice
CH = 2  # stereo
VOL = 0.25  # master volume

_cmd = queue.Queue()


def _sine_block(freq_l, freq_r, vib=0.0):
    """Return a 50 ms stereo numpy.int16 buffer."""
    t = np.arange(int(RATE * BLOCK)) / RATE
    l = np.sin(2 * np.pi * freq_l * (1 + vib * np.sin(2 * np.pi * 6 * t)) * t)
    r = np.sin(2 * np.pi * freq_r * (1 + vib * np.sin(2 * np.pi * 6 * t)) * t)
    stereo = np.column_stack((l, r)) * VOL
    return (stereo * 32767).astype(np.int16)


def audio_thread():
    last = (440, 440, 0)  # default A4
    while True:
        try:
            last = _cmd.get(timeout=BLOCK * 0.9)
        except queue.Empty:
            pass
        buf = _sine_block(*last)
        sa.play_buffer(buf, CH, 2, RATE)  # 2 bytes = 16‑bit


def start_audio():
    threading.Thread(target=audio_thread, daemon=True).start()


def update_tone(freq, pan, vib):
    """pan: 0‑1 (0 = left, 0.5 = center, 1 = right)."""
    l = freq * (1.0 - pan)
    r = freq * pan
    _cmd.put((l, r, vib))


start_audio()


def map_distance_to_pitch(d_mm):  # 0‑200 mm → 220‑880 Hz
    return np.interp(d_mm, [0, 200], [880, 220])


def map_angle_to_pan(a_deg):  # 0‑30° → centre‑to‑left
    return np.interp(a_deg, [0, 30], [0.5, 0.0])


def map_angle_to_vib(a_deg):  # 0‑30° → 0‑6 %
    return np.interp(a_deg, [0, 30], [0.0, 0.06])


with ReachyMini() as reachy_mini:
    reachy_mini.disable_motors()
    target_pose = np.eye(4)  # origin
    best_score = float("inf")

    while True:
        joints, _ = reachy_mini._get_current_joint_positions()
        pose = reachy_mini.head_kinematics.fk(joints)

        l2, ang, magic = distance_between_poses(target_pose, pose)
        l2_mm = l2 * 1000
        ang_deg = np.degrees(ang)
        best_score = min(best_score, magic)

        # --- send live tone update ---
        pitch = map_distance_to_pitch(l2_mm)
        pan = map_angle_to_pan(ang_deg)
        vib = map_angle_to_vib(ang_deg)
        update_tone(pitch, pan, vib)

        print(
            f"\rl2: {l2_mm:6.1f} mm | ang: {ang_deg:5.2f}° | best: {best_score:6.2f}",
            end="",
        )
        if l2_mm < 5 and ang_deg < 2:  # sweet spot?
            sa.WaveObject.from_wave_file("win.wav").play()  # success jingle
            target_pose = reachy_mini.head_kinematics.random_pose()
            best_score = float("inf")
        time.sleep(0.05)
