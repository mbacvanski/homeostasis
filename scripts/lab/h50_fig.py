"""Render the depth-4 chain (A + B + C + D)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from h50_depth import cosim_chain, LAB
from h48c_live_chain import PACE_CFG, PACE_SEED
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WallSimulation  # noqa: E402
from h50_depth import make_follower, START_Y  # noqa: E402

def main():
    b = json.loads((LAB / "h48e_warm.json").read_text())
    c = json.loads((LAB / "h49_chain3.json").read_text())
    # D's champion isn't saved by h50_depth; re-evolve quickly? No - h50 saved h50_chain.json
    meta = json.loads((LAB / "h50_chain.json").read_text())
    print("saved chain meta:", meta)

if __name__ == "__main__":
    main()
