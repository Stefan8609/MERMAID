import numpy as np
from pathlib import Path
import obspy

"""
Save all MERMAID locations and times to a NumPy .npz file for later use.
Includes files not in the tomocat1.txt file, which only associated event records

Times saved in same style as JOEL as defined in tomocat1_description
"""

from datetime import datetime
import pandas as pd


def datetime_to_tomocat_time(dt):
    """
    Convert a first-sample datetime into TOMOCAT-style string:

        datetime(2026, 3, 30, 8, 54, 50, 418058)
        -> 2026-03-30T08:54:500.418058
    """
    sec = f"{dt.second:02d}"
    b_offset = dt.microsecond / 1_000_000.0

    b_string = f"{b_offset:.6f}".rstrip("0").rstrip(".")
    if b_string == "0":
        b_string = "0.0"

    return f"{dt:%Y-%m-%dT%H:%M}:{sec}{b_string}"

def save_loc_time(input_dir, output_dir):
    """
    Read all SAC files in input_dir, extract stla, stlo, and starttime, convert to TOMOCAT-style time,
    """
    sac_files = sorted(input_dir.glob("*.sac"))
    print(f"Found {len(sac_files)} SAC files")

    stla = []
    stlo = []
    seismogram_time = []
    filename=[]

    for sac_file in sac_files:

        print(f"Processing {sac_file.name}")
        st = obspy.read(str(sac_file))
        tr = st[0]

        filename.append(sac_file.name)
        stla.append(tr.stats.sac.stla)
        stlo.append(tr.stats.sac.stlo)
        
        time = tr.stats.starttime.datetime
        time_tomocat = datetime_to_tomocat_time(time)
        seismogram_time.append(time_tomocat)

    stla = np.array(stla)
    stlo = np.array(stlo)
    seismogram_time = np.array(seismogram_time)

    output_file = output_dir / "mermaid_loc_time.npz"
    np.savez(output_file, filename=filename, stla=stla, stlo=stlo, seismogram_time=seismogram_time)


if __name__ == "__main__":
    input_dir = Path("./Mermaid_Data_Joel")
    output_dir = Path("./saved_data")

    save_loc_time(input_dir, output_dir)