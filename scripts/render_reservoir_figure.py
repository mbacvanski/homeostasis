"""Static slide figure: the reservoir's actual connectivity, no text.

Left: the sensor array (amber), drawn as an arc, with every real
input-adjacency edge into the reservoir. Middle: the recurrent reservoir
(real adjacency, paper config, seed 0 - the same network as the tracking
videos). Right: the two effectors with their real readout connections.
No labels or callouts; annotation happens on the slide.

Usage: python scripts/render_reservoir_figure.py
Writes scripts/out/reservoir_figure.png (white) and _transparent.png.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from homeostasis import VariableTrackingSimulation

INK = "#1c2733"
EDGE = "#d9dee6"
AMBER = "#c9820a"
NODE = "#5b8fd6"
NODE_FILL = "#dbe7f6"
GREEN = "#0c9c62"
BLUE = "#2266cc"

sim = VariableTrackingSimulation(seed=0)
net = sim.network
adj = net.adjacency
in_adj = net.input_adjacency          # (n_sensors, n_nodes)
out_adj = net.output_adjacency        # (n_nodes, 2)
n = net.config.n_nodes
offs = sim.env.config.sensor_offsets  # degrees, ordered

# ---- layout ----------------------------------------------------------------
rng = np.random.default_rng(7)

# reservoir: sunflower disc with jitter, radius 1.0, centered at origin
k = np.arange(n) + 0.5
r = np.sqrt(k / n) * 1.0
th = k * np.pi * (3.0 - np.sqrt(5.0))
pos = np.column_stack([r * np.cos(th), r * np.sin(th)])
pos += rng.normal(0, 0.035, pos.shape)

# sensors: an arc on the left, ordered by tuning angle, bulging away
order = np.argsort(offs)
t = np.linspace(-1.05, 1.05, len(offs))
s_pos = np.column_stack([-2.35 + 0.30 * (t ** 2), t * 1.30])[np.argsort(order)]

# effectors: right side
e_pos = np.array([[2.35, 0.52], [2.35, -0.52]])

fig = plt.figure(figsize=(12.8, 7.2), dpi=225)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-2.85, 2.85)
ax.set_ylim(-1.60, 1.60)
ax.set_aspect("equal")
ax.axis("off")

# ---- edges (bottom to top) -------------------------------------------------
HIGHLIGHT = bool(int(__import__("os").environ.get("HIGHLIGHT", "0")))
dim = 0.45 if HIGHLIGHT else 1.0

si, ni = np.nonzero(in_adj)
ax.add_collection(LineCollection(
    np.stack([s_pos[si], pos[ni]], axis=1),
    colors=AMBER, linewidths=0.45, alpha=0.09 * dim, zorder=1))

es, ed = np.nonzero(adj)
ax.add_collection(LineCollection(
    np.stack([pos[es], pos[ed]], axis=1),
    colors=INK, linewidths=0.40, alpha=0.07 * dim, zorder=2))

for e in (0, 1):
    src = np.flatnonzero(out_adj[:, e])
    ax.add_collection(LineCollection(
        np.stack([pos[src], np.repeat(e_pos[None, e], len(src), 0)], axis=1),
        colors=(GREEN if e == 0 else BLUE), linewidths=0.7,
        alpha=0.30 * dim, zorder=3))

if HIGHLIGHT:
    # one sensor's fan, one recurrent node's out-edges, one readout edge -
    # a single input -> reservoir -> output path to talk over, no text
    s_pick = int(np.argmin(np.abs(offs)))          # the straight-ahead sensor
    targets = np.flatnonzero(in_adj[s_pick])
    ax.add_collection(LineCollection(
        np.stack([np.repeat(s_pos[None, s_pick], len(targets), 0),
                  pos[targets]], axis=1),
        colors=AMBER, linewidths=1.6, alpha=0.85, zorder=7))
    readout_nodes = np.flatnonzero(out_adj[:, 0])
    mid = targets[np.argmax(adj[targets][:, readout_nodes].sum(1))]
    hops = np.flatnonzero(adj[mid])
    ax.add_collection(LineCollection(
        np.stack([np.repeat(pos[None, mid], len(hops), 0), pos[hops]],
                 axis=1),
        colors=INK, linewidths=1.3, alpha=0.55, zorder=7))
    ends = [h for h in hops if out_adj[h, 0]] or [readout_nodes[0]]
    ax.add_collection(LineCollection(
        [[pos[ends[0]], e_pos[0]]],
        colors=GREEN, linewidths=2.0, alpha=0.9, zorder=7))
    ax.scatter(*s_pos[s_pick], s=64, c=AMBER, edgecolors="white",
               linewidths=0.8, zorder=8)
    ax.scatter(pos[targets][:, 0], pos[targets][:, 1], s=115, c="#f3d9ae",
               edgecolors=AMBER, linewidths=1.3, zorder=8)
    ax.scatter(*pos[mid], s=130, c="#f3d9ae", edgecolors=INK,
               linewidths=1.5, zorder=9)
    ax.scatter(*pos[ends[0]], s=115, c="#cdeadd", edgecolors=GREEN,
               linewidths=1.4, zorder=8)

# ---- nodes -----------------------------------------------------------------
ax.scatter(s_pos[:, 0], s_pos[:, 1], s=34, c=AMBER, edgecolors="white",
           linewidths=0.6, zorder=5)
ax.scatter(pos[:, 0], pos[:, 1], s=105, c=NODE_FILL, edgecolors=NODE,
           linewidths=1.1, zorder=5)
for e, col in ((0, GREEN), (1, BLUE)):
    ax.add_patch(Circle(e_pos[e], 0.155, facecolor="white", edgecolor=col,
                        lw=2.4, zorder=6))

out = pathlib.Path("scripts/out")
out.mkdir(parents=True, exist_ok=True)
stem = "reservoir_figure_highlight" if HIGHLIGHT else "reservoir_figure"
fig.savefig(out / f"{stem}.png", dpi=225, facecolor="white")
fig.savefig(out / f"{stem}_transparent.png", dpi=225, transparent=True)
print(f"in-edges {len(si)}, recurrent {len(es)}, "
      f"readout {int(out_adj.sum())}; saved 2 files", flush=True)
