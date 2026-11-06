"""DAgger for BaseNet_obs: BaseNet_obs drives among the boxes in TRAINING scenes, the obstacle-aware expert
runs alongside every step (rollout_obs.py), and its wheel commands become extra labels for BaseNet_obs.

Collect (from robotnik_arm_task/, 4 terminals or the run_dagger_obs.sh script):
    ROBOENV_HEADLESS=1 python yousef/obstacles/dagger_obs.py --round 1 --shard 0 --nshards 4
    -> data/dagger_obs_r1/ep_<id>.npz  (same ID as the training scene it replays)

Use:  python yousef/obstacles/dataset_obs.py --dagger   (adds every data/dagger_obs_r*/ run to the training split)

Which labels are kept (dagger_rows):
  * every 10th step (100 Hz), as in the expert data
  * only while the base is more than MIN_GOAL_DIST from its goal and the expert is not in its "done" phase:
    once BaseNet has parked (up to 0.3 m off), the expert's phase machine still says "drive to the exact
    goal", which is not what BaseNet should learn while the arm reaches (same issue as in the main DAgger)
  * runs are cut at the first arm joint speed above QD_CUT (the simulation going unstable after a collision),
    the same threshold as the main pipeline's DAgger
Only training-split scenes are replayed, so validation and the obstacle test set stay untouched.
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
from dataset_obs import split_obs, obs_features
from features import wheels_to_vw
from world import lidar_features

TASK_DIR = os.path.dirname(os.path.dirname(HERE))
QD_CUT = 15.0          # rad/s
MIN_GOAL_DIST = 0.35   # m (BaseNet counts as parked within 0.30 m)


def dagger_rows(step=10, lidar="front2", near_only=False):
    """(X, Y) from all data/dagger_obs_r*/ runs: BaseNet_obs inputs and the shadow expert's (v, w).
    near_only: keep only rows with a box within 1.5 m of the chassis (front proximity > 0), so free driving and
    parking are learned from the expert's own demonstrations only."""
    assert lidar in ("front2", "corr"), "DAgger runs log the front scan only"
    X, Y, n_ep = [], [], 0
    for f in sorted(glob.glob(os.path.join(TASK_DIR, "data", "dagger_obs_r*", "ep_*.npz"))):
        d = dict(np.load(f))
        if len(d["t"]) == 0:
            continue
        fast = np.where(np.abs(d["qd_arm"]).max(axis=1) > QD_CUT)[0]
        end = int(fast[0]) if len(fast) else len(d["t"])
        x = obs_features(d, lidar)[:end]
        y = wheels_to_vw(d["wheel_exp"])[:end]
        keep = (x[:, 2] > MIN_GOAL_DIST) & (d["phase"][:end] != 3) & (np.arange(end) % step == 0)
        if near_only:
            keep &= lidar_features(d["lidar"][:end])[:, 1] > 0
        X.append(x[keep]); Y.append(y[keep]); n_ep += 1
    if not X:
        return np.zeros((0, 10 + {"front2": 2, "corr": 6}[lidar]), np.float32), np.zeros((0, 2), np.float32)
    X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)
    print(f"DAgger (obstacles): {n_ep} runs -> {len(X)} rows")
    return X, Y


if __name__ == "__main__":
    import argparse
    import torch
    torch.set_num_threads(1)
    from rollout_obs import run_policy_episode
    from policy_obs import BCPolicyObs

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)               # -> data/dagger_obs_r<round>/
    parser.add_argument("--n", type=int, default=160)                 # training scenes per round
    parser.add_argument("--base", default="base_net_obs_multi")       # the BaseNet_obs that drives
    parser.add_argument("--arm", default="arm_net")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()

    train_f, _, _, _ = split_obs()                                    # training split only (same split as the dataset)
    rng = np.random.default_rng(100 + args.round)
    pick = sorted(rng.choice(len(train_f), size=min(args.n, len(train_f)), replace=False))
    files = [train_f[i] for i in pick][args.shard::args.nshards]
    out_subdir = f"dagger_obs_r{args.round}"
    pol = BCPolicyObs(os.path.join(TASK_DIR, "checkpoints"), base_file=f"{args.base}.pt", arm_file=f"{args.arm}.pt")
    for f in files:
        ep = int(os.path.basename(f)[3:8])
        if os.path.exists(os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")):
            continue
        e = np.load(f)
        print(f"=== DAGGER OBS r{args.round}: scene {ep} ({len(e['obstacles'])} boxes) ===")
        try:
            run_policy_episode(e["target"], ep, tuple(e["base_init"]), pol, net_parts=("base", "arm"),
                               out_subdir=out_subdir, obstacles=e["obstacles"])
        except Exception as ex:
            print(f"!!! scene {ep} failed: {ex}")
