"""Download the DC motor data used in the paper.

The files belong to their authors and are distributed in the sysidentpy-data
repository (MIT licence). They are saved next to this script as u_cc.csv
(input) and y_cc.csv (output), the names expected by the MATLAB and Python code.
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/main/datasets/generator/"
HERE = Path(__file__).resolve().parent

for remote, local in [("x_cc.csv", "u_cc.csv"), ("y_cc.csv", "y_cc.csv")]:
    target = HERE / local
    urlretrieve(BASE + remote, target)
    print(f"{remote} -> {target}")
