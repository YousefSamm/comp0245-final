"""Does the expert's fixed approach direction explain the collisions?

The base goal is always on the line from the world origin to the target (features.base_goal), so a robot
starting on the far side of the target has to drive past it and round to the origin side. For the 95 solved
obstacle test scenes this script compares far-side and near-side starts: extra distance, collisions and
episode time for the expert and the networks, with Fisher's exact test on the collision counts.

    python yousef/obstacles/analyze_approach_obs.py      (from robotnik_arm_task/; reads data/raw_obs and results/)
"""
import os, sys, glob
from math import comb
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from features import base_goal
TASK_DIR = os.path.dirname(os.path.dirname(HERE))


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda k: comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
    p0 = p(a)
    return sum(p(k) for k in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(k) <= p0 + 1e-12)


rows = []
for f in sorted(glob.glob(os.path.join(TASK_DIR, "data", "raw_obs", "ep_29*.npz"))):
    d = np.load(f)
    if not bool(d["success"]):
        continue
    tgt = d["target"]; g, _ = base_goal(tgt); g = np.asarray(g)[:2]
    st, t2 = d["base_init"][:2], tgt[:2]
    standoff = np.linalg.norm(t2 - g)
    cosang = np.dot(st - t2, g - t2) / np.linalg.norm(st - t2) / np.linalg.norm(g - t2)
    angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))      # at the target, between start and parking spot
    pos, drive = d["base_pos"], d["t"] < float(d["nav_time"])
    rows.append({"ep": int(os.path.basename(f)[3:8]), "angle": angle, "far": angle > 90,
                 "extra_m": np.linalg.norm(g - st) - max(np.linalg.norm(st - t2) - standoff, 0.0),
                 "passed_target": bool(drive.any() and np.linalg.norm(pos[drive] - t2, axis=1).min() < standoff - 0.05),
                 "expert_col": int(d["n_collision_steps"]) > 0, "expert_dur": float(d["t"][-1])})
G = pd.DataFrame(rows).set_index("ep")
models = [m for m in ["nolidar", "multi", "corr"] if os.path.exists(os.path.join(TASK_DIR, "results", f"obs_obs_{m}.csv"))]
for m in models:
    x = pd.read_csv(os.path.join(TASK_DIR, "results", f"obs_obs_{m}.csv")).set_index("ep")
    G[f"{m}_col"], G[f"{m}_dur"] = x.collided.astype(bool), x.duration

far, near = G[G.far], G[~G.far]
print(f"{len(G)} scenes | far-side starts: {len(far)} ({100 * G.far.mean():.0f} %) | expert passed the target: {int(G.passed_target.sum())}")
print(f"median extra driving vs the nearest parking spot: far {far.extra_m.median():.2f} m, near {near.extra_m.median():.2f} m\n")
print(f"{'':14s}{'collided far':>14s}{'collided near':>15s}{'Fisher p':>10s}{'median s far':>14s}{'near':>7s}")
for m in ["expert"] + models:
    c, t = f"{m}_col", f"{m}_dur"
    a, b, cc, dd = int(far[c].sum()), int((~far[c]).sum()), int(near[c].sum()), int((~near[c]).sum())
    print(f"{m:14s}{100 * far[c].mean():13.1f}%{100 * near[c].mean():14.1f}%{fisher(a, b, cc, dd):10.2f}"
          f"{far[t].median():14.1f}{near[t].median():7.1f}")
