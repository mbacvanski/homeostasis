"""H48d: transfer test — do existing follower champions follow the LIVE pacemaker?"""
from __future__ import annotations
import json
from pathlib import Path
from h48c_live_chain import cosim

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    cands = {
        "h38c-ellipse-follower": json.loads((LAB / "h38c_interp.json").read_text())[-1],
        "h34-orbit-follower": json.loads((LAB / "h34_joint.json").read_text())[-1],
    }
    for name, log in cands.items():
        g, s = log["champion"], log["champ_seed"]
        r = cosim(g, s, n=7200)
        print(f"{name:24s} near4 {r['near4']:.2f}  dist {r['dist']:.2f}")

if __name__ == "__main__":
    main()
