"""H94: derive the ridge law from measured duty rates + one timescale."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    rows = json.load(open(LAB / "cluster1_results.json"))
    cells = defaultdict(list)
    for r in rows:
        if set(r["res"].keys()) != {"leak", "weight_lr"}:
            continue
        cells[(round(r["res"]["leak"], 4), round(r["res"]["weight_lr"], 4))].append(r)
    leaks = sorted({k[0] for k in cells})
    wlrs = sorted({k[1] for k in cells})
    crest = {}
    for lk in leaks:
        scores = {w: np.mean([x["score_late"] for x in cells[(lk, w)]])
                  for w in wlrs if (lk, w) in cells}
        wstar = max(scores, key=scores.get)
        f = np.mean([x["prop_spiked"] for x in cells[(lk, wstar)]])
        crest[lk] = (wstar, f, scores[wstar])
    # fit single P_eff in log space
    lw = np.array([np.log(crest[lk][0]) for lk in leaks])
    lf = np.array([np.log(max(crest[lk][1], 1e-6)) for lk in leaks])
    logP = np.mean(np.log(2 * np.pi) - lf - lw)
    P_eff = float(np.exp(logP))
    print(f"fitted P_eff = {P_eff:.0f} steps")
    ok = 0
    for lk in leaks:
        wstar, f, sc = crest[lk]
        pred = 2 * np.pi / (f * P_eff)
        ratio = pred / wstar
        good = 1 / 1.5 <= ratio <= 1.5
        ok += good
        print(f"leak {lk:<5} crest wlr {wstar:<6} f {f:.3f}  pred {pred:.3f}"
              f"  ratio {ratio:.2f}  {'OK' if good else 'MISS'}")
    print(f"within 1.5x: {ok}/{len(leaks)}")
    ll = np.array([np.log(lk) for lk in leaks])
    c_f = np.corrcoef(lw, -lf)[0, 1]
    c_l = np.corrcoef(lw, ll)[0, 1]
    print(f"shape: corr(log w*, -log f) = {c_f:+.3f} vs corr(log w*, log leak) = {c_l:+.3f}")
    (LAB / "h94_derive.json").write_text(json.dumps(dict(
        P_eff=P_eff, crest={str(k): v for k, v in crest.items()},
        ok=int(ok), corr_f=float(c_f), corr_leak=float(c_l))))

if __name__ == "__main__":
    main()
