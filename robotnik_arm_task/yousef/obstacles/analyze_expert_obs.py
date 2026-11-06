"""Summary of the obstacle-aware EXPERT from its saved demonstrations in data/raw_obs/ (no simulation).

    python yousef/obstacles/analyze_expert_obs.py            # all scenes
    python yousef/obstacles/analyze_expert_obs.py --test     # only the 100 test scenes (IDs >= 29000)

Prints success, collisions, timing, how much it detours and how close it gets to boxes, per scene set and
per number of boxes, and saves one row per episode to results/obs_expert.csv (for plots).
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")          # pandas groupby / empty-slice notices
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from features import base_goal

TASK_DIR = os.path.dirname(os.path.dirname(HERE))
# chassis-edge start of each of the 72 rays (= world.LIDAR360_R0), so clearance = range - R0
ANG = -np.pi + np.deg2rad(5.0) * np.arange(72)
R0 = np.minimum(0.36 / np.maximum(np.abs(np.cos(ANG)), 1e-6), 0.305 / np.maximum(np.abs(np.sin(ANG)), 1e-6)) + 0.03

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()


def scene_set(ep):
    return "test" if ep >= 29000 else ("multi-box" if ep >= 24000 else "single-box")


rows = []
files = sorted(glob.glob(os.path.join(TASK_DIR, "data", "raw_obs", "ep_*.npz")))
for f in files:
    ep = int(os.path.basename(f)[3:8])
    if args.test and ep < 29000:
        continue
    d = np.load(f)
    pos, t = d["base_pos"], d["t"]
    goal, _ = base_goal(d["target"])
    goal = np.asarray(goal).reshape(-1)[:2]
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(goal - pos[0]))
    nav = float(d["nav_time"])
    drive = t < nav if nav > 0 else np.ones(len(t), bool)
    clear = (d["lidar360"][::10] - R0).min() if len(d["obstacles"]) else np.nan
    rows.append({"ep": ep, "set": scene_set(ep), "n_boxes": len(d["obstacles"]), "success": bool(d["success"]),
                 "collided": int(d["n_collision_steps"]) > 0, "collision_steps": int(d["n_collision_steps"]),
                 "duration": float(t[-1]), "nav_time": nav, "final_error_cm": 100 * float(d["final_error"]),
                 "path_m": path, "straight_m": straight, "detour": path / max(straight, 1e-6),
                 "avoiding_frac": float(d["avoiding"][drive].mean()) if drive.any() else 0.0,
                 "min_clearance_m": float(clear)})

df = pd.DataFrame(rows)
out = os.path.join(TASK_DIR, "results", "obs_expert.csv")
df.to_csv(out, index=False)


def summary(g):
    s = g[g.success]
    return pd.Series({"episodes": len(g), "success %": 100 * g.success.mean(), "collided %": 100 * g.collided.mean(),
                      "median duration s": s.duration.median(), "median nav time s": s.nav_time.median(),
                      "median detour": s.detour.median(), "time avoiding %": 100 * s.avoiding_frac.mean(),
                      "median closest box m": s.min_clearance_m.median()})


pd.set_option("display.width", 160, "display.precision", 2)
print("\n=== Expert per scene set ===")
print(df.groupby("set").apply(summary).T)
print("\n=== Expert per number of boxes (all sets) ===")
print(df.groupby("n_boxes").apply(summary).T)
t = df[(df.set == "test") & df.success]
print(f"\nTest scenes it solved: {len(t)}; collided in {int(t.collided.sum())}: {t[t.collided].ep.tolist()}")
print(f"saved {out}")
