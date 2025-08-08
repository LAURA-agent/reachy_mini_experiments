# list_gm.py
from scamp import Session

sess = Session()  # boots FluidSynth
impl = sess._playback  # SoundfontPlaybackImplementation

get_name = getattr(impl, "get_program_name", lambda p: f"GM {p}")

print(f"Presets in {sess.default_soundfont}:\n")
for prog in range(128):
    print(f"{prog:3}  {get_name(prog)}")
