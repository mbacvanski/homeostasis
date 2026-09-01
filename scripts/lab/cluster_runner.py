"""Cluster chunk runner: read a JSONL task file, run each task, write JSONL.

Usage: python cluster_runner.py TASKS.jsonl OUT.jsonl
Tasks are dicts for common.run_closed_loop / run_open_loop with a "mode" key
("closed"/"open") and arbitrary passthrough keys starting with "_".
Single-process (BLAS pinned by common); parallelism comes from the Slurm array.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import run_closed_loop, run_open_loop
from pong_eval import eval_pong


def main():
    tasks_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_tmp = out_path.with_suffix(".part")
    with out_tmp.open("w") as out:
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            mode = task.get("mode")
            if mode == "pong":
                r = eval_pong(task)
            elif mode == "open":
                r = run_open_loop(task)
            else:
                r = run_closed_loop(task)
            r.pop("snaps", None)
            r.pop("f_t", None)
            for k, v in task.items():
                if k.startswith("_"):
                    r[k] = v
            out.write(json.dumps(r) + "\n")
            out.flush()
    out_tmp.rename(out_path)
    print(f"done {tasks_path.name}: {out_path}")


if __name__ == "__main__":
    main()
