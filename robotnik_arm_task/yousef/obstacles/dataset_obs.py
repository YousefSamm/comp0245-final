"""BaseNet_obs dataset: the 10 BaseNet inputs (features.base_features) + the 2 LiDAR inputs
(world.lidar_features: angle of the closest front hit, proximity) = 12 inputs; output (v, w).

Episodes: data/raw_obs/ep_*.npz, successful AND collision-free only. Split by episode:
IDs >= 29000 = obstacle test set (never trained on); the rest -> train / val (15 %).
Every 10th row (1 kHz -> 100 Hz), labels = the expert's clean command (wheel_exp).
Normalisation stats from the training split only.

Usage (from robotnik_arm_task/):  python yousef/obstacles/dataset_obs.py [--lidar front2|sect8]
  front2 (default): 2 features from the front 180 deg -> data/processed_obs_step10.npz      (12 inputs)
  sect8           : 8 sector proximities from the full 360 deg -> data/processed_obs_step10_sect8.npz (18 inputs)
  corr            : 6 corridor features from the front scan (what the expert steers from) -> ..._corr.npz (16 inputs)
  --dagger        : also add the data/dagger_obs_r*/ runs to the training split (-> ..._dagger.npz)
  --dagger_near   : the same, but only DAgger rows with a box within 1.5 m, every 40th step instead of every
                    10th (about 5 % of the training rows instead of 21 %) (-> ..._daggernear.npz)
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # yousef/  (features.py)
sys.path.insert(0, HERE)                                   # yousef/obstacles/  (world.py)
from features import base_features, wheels_to_vw
from world import lidar_features, lidar_sector_features, lidar_corridor_features

LIDAR_DIMS = {"front2": 2, "sect8": 8, "corr": 6}

TASK_DIR = os.path.dirname(os.path.dirname(HERE))
OBS_DIR = os.path.join(TASK_DIR, "data", "raw_obs")
TEST_FROM = 29000


def split_obs(val_frac=0.15, seed=0):
    train_pool, test, skipped = [], [], 0
    for f in sorted(glob.glob(os.path.join(OBS_DIR, "ep_*.npz"))):
        d = np.load(f)
        if not bool(d["success"]) or int(d["n_collision_steps"]) > 0:
            skipped += 1
            continue
        (test if int(os.path.basename(f)[3:8]) >= TEST_FROM else train_pool).append(f)
    pool = list(np.random.default_rng(seed).permutation(train_pool))
    n_val = int(val_frac * len(pool))
    return pool[n_val:], pool[:n_val], test, skipped


def lidar_inputs(d, lidar="front2"):
    """The LiDAR part of BaseNet_obs's input. Same function at run time (policy_obs.py)."""
    if lidar == "front2":
        return lidar_features(d["lidar"])
    if lidar == "corr":
        return lidar_corridor_features(d["lidar"])
    return lidar_sector_features(d["lidar360"])


def obs_features(d, lidar="front2"):
    """(N, 10 + LIDAR_DIMS[lidar]): BaseNet's 10 inputs + the LiDAR inputs."""
    xb, _ = base_features(d)
    return np.hstack([xb, lidar_inputs(d, lidar)]).astype(np.float32)


def build_split(files, step=10, lidar="front2"):
    X, Y, ph, ep = [], [], [], []
    for f in files:
        d = np.load(f)
        s = slice(None, None, step)
        X.append(obs_features(d, lidar)[s]); Y.append(wheels_to_vw(d["wheel_exp"])[s])
        ph.append(d["phase"][s]); ep.append(np.full(len(X[-1]), int(os.path.basename(f)[3:8])))
    if not X:
        return {k: np.zeros((0, n), np.float32) for k, n in [("Xb", 10 + LIDAR_DIMS[lidar]), ("Yb", 2)]} | {"phase": np.zeros(0), "ep_id": np.zeros(0)}
    return {"Xb": np.concatenate(X), "Yb": np.concatenate(Y), "phase": np.concatenate(ph), "ep_id": np.concatenate(ep)}


def build_dataset(step=10, out_path=None, lidar="front2", dagger=None):
    """dagger: None, "all" or "near" (only DAgger rows with a box within 1.5 m)."""
    suffix = ("" if lidar == "front2" else f"_{lidar}") + {None: "", "all": "_dagger", "near": "_daggernear"}[dagger]
    out_path = out_path or os.path.join(TASK_DIR, "data", f"processed_obs_step{step}{suffix}.npz")
    tr, va, te, skipped = split_obs()
    print(f"obstacle episodes: train {len(tr)}, val {len(va)}, test {len(te)}  (skipped {skipped}: failed or collided)")
    splits = {n: build_split(f, step, lidar) for n, f in [("train", tr), ("val", va), ("test", te)]}
    arrays = {f"{k}_{n}": v for n, sp in splits.items() for k, v in sp.items()}
    for k in ["Xb", "Yb"]:
        x = splits["train"][k]
        arrays[f"{k}_mu"], arrays[f"{k}_sd"] = x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32)
    if dagger:   # as in the main pipeline: DAgger rows go to the training split only, stats unchanged
        from dagger_obs import dagger_rows
        Xd, Yd = dagger_rows(step * (4 if dagger == "near" else 1), lidar, near_only=(dagger == "near"))
        arrays["Xb_train"] = np.concatenate([arrays["Xb_train"], Xd])
        arrays["Yb_train"] = np.concatenate([arrays["Yb_train"], Yd])
    np.savez(out_path, **arrays)
    print("saved", out_path, "| train rows", arrays["Xb_train"].shape, "| val rows", arrays["Xb_val"].shape)
    return arrays


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar", choices=list(LIDAR_DIMS), default="front2")
    ap.add_argument("--dagger", action="store_true")
    ap.add_argument("--dagger_near", action="store_true")
    a = ap.parse_args()
    build_dataset(lidar=a.lidar, dagger="near" if a.dagger_near else ("all" if a.dagger else None))
