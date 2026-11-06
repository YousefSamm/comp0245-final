"""Top-down view of one obstacle test scene: the paths of the expert and three BaseNets, the boxes, and the
driving corridor of the corridor-feature network at the moment it first sees a box in its path.

    python results/figures/plot_obstacle_trajectories.py            (from robotnik_arm_task/; scene 29312)
    python results/figures/plot_obstacle_trajectories.py 29206      (another scene)

Reads data/raw_obs/ep_<scene>.npz (expert) and data/test_obs_obs_{nolidar,multi,corr}/ep_<scene>.npz
(closed-loop runs saved by evaluate_obs.py), writes obstacle_trajectories.png next to this script.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(os.path.dirname(FIG_DIR))
sys.path.insert(0, os.path.join(TASK_DIR, "yousef"))
from features import base_goal

SCENE = int(sys.argv[1]) if len(sys.argv) > 1 else 29312
CORR_HALF_W, BUMPER, F_SAFE = 0.505, 0.36, 1.0            # = expert_avoid.HALF_W, FRONT, F_SAFE
HL, HW = 0.36, 0.305                                      # chassis half-length, half-width
LIDAR_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 37)     # front scan, = world.LIDAR_ANGLES
plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})


def rect(cx, cy, yaw, x0, x1, y0, y1):
    """Corners of the rectangle [x0, x1] x [y0, y1] given in the frame of a body at (cx, cy, yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.column_stack([cx + c * pts[:, 0] - s * pts[:, 1], cy + s * pts[:, 0] + c * pts[:, 1]])


exp = np.load(os.path.join(TASK_DIR, "data", "raw_obs", f"ep_{SCENE:05d}.npz"))
runs = [("Expert", exp, "k", "--"),
        ("No LiDAR", None, "#d62728", "-"),
        ("2 front features", None, "#ff7f0e", "-"),
        ("Corridor features", None, "#2ca02c", "-")]
for i, tag in [(1, "nolidar"), (2, "multi"), (3, "corr")]:
    name, _, col, ls = runs[i]
    runs[i] = (name, np.load(os.path.join(TASK_DIR, "data", f"test_obs_obs_{tag}", f"ep_{SCENE:05d}.npz")), col, ls)

fig, ax = plt.subplots(figsize=(3.4, 3.3))
for x, y, h, yaw in exp["obstacles"]:
    ax.add_patch(Polygon(rect(x, y, yaw, -h, h, -h, h), closed=True, fc="0.55", ec="0.3", lw=0.6, zorder=2))

AFTER_HIT = 2.0                                           # s of path drawn after the first contact
for name, d, col, ls in runs:
    end = len(d["t"])
    if int(d["n_collision_steps"]) > 0:                   # a collision can make the simulation unstable:
        end = int(np.searchsorted(d["t"], d["t"][np.argmax(d["collision"] > 0)] + AFTER_HIT))   # stop soon after it
    p = d["base_pos"][:end:20]
    ax.plot(p[:, 0], p[:, 1], color=col, ls=ls, lw=1.1, zorder=3,
            label=f"{name}" + (" (collided, x)" if int(d["n_collision_steps"]) > 0 else ""))
    hit = d["collision"] > 0
    if hit.any():
        k = int(np.argmax(hit))
        ax.plot(*d["base_pos"][k], marker="x", color=col, ms=6, mew=1.5, zorder=5)

# corridor of the corridor-feature network the first time a box is in it (within F_SAFE) while driving
d = runs[3][1]
r = d["lidar"]
px, py = r * np.cos(LIDAR_ANGLES), r * np.sin(LIDAR_ANGLES)
in_corr = (r < 2.99) & (px > 0) & (np.abs(py) < CORR_HALF_W) & (px - BUMPER < F_SAFE)
moving = np.hypot(d["base_vel"][:, 0], d["base_vel"][:, 1]) > 0.08     # driving, not turning in place
cand = in_corr.any(axis=1) & moving
k = int(np.argmax(cand)) if cand.any() else len(r) // 3
bx, by, yaw = d["base_pos"][k, 0], d["base_pos"][k, 1], d["base_bearing"][k]
ax.add_patch(Polygon(rect(bx, by, yaw, BUMPER, BUMPER + F_SAFE, -CORR_HALF_W, CORR_HALF_W), closed=True,
                     fc="#2ca02c", alpha=0.15, ec="#2ca02c", lw=0.6, ls=":", zorder=1, label="corridor (1 m ahead)"))
ax.add_patch(Polygon(rect(bx, by, yaw, -HL, HL, -HW, HW), closed=True, fc="none", ec="#2ca02c", lw=0.9, zorder=4))

st, tgt = exp["base_init"], exp["target"]
goal, _ = base_goal(tgt)
ax.plot(st[0], st[1], "k^", ms=5, zorder=6, label="start")
ax.plot(*np.asarray(goal)[:2], "ks", ms=4, mfc="white", zorder=6, label="parking spot")
ax.plot(tgt[0], tgt[1], "*", color="#d62728", ms=9, zorder=6, label="target")
pts = np.vstack([exp["base_pos"], [st[:2]], [tgt[:2]], exp["obstacles"][:, :2]])
lo, hi = pts.min(0) - 0.7, pts.max(0) + 0.7
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
          handlelength=1.6, columnspacing=0.9)
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
out = os.path.join(FIG_DIR, "obstacle_trajectories.png")
fig.savefig(out, dpi=300)
print("saved", out, f"(scene {SCENE}, corridor drawn at t = {d['t'][k]:.1f} s)")
