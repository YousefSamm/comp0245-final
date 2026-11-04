"""DAgger (Dataset Aggregation): everything in one place.

1) COLLECT (command line): the current networks drive on TRAINING-split configurations (never val/test);
   the shadow expert's labels are saved with every run in data/dagger_r<round>/.
       ROBOENV_HEADLESS=1 ROBOENV_MAX_STEPS=20000 python yousef/dagger.py --round 1 --shard 0 --nshards 4
   (run shards 0..3 in 4 terminals; interrupted runs resume)

2) AGGREGATE (in the training notebook):
       from dagger import add_dagger
       D = add_dagger(D)        # appends all data/dagger_r*/ rows to the ARM training set

Labels are used for ArmNet only: when BaseNet drives, the expert's base phase machine (align/drive/orient)
does not follow the robot's real path, so its wheel labels are unreliable; its arm labels are not.
"""
import os
import glob
import numpy as np

from features import arm_features

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# 2) Aggregate: load DAgger runs as extra ArmNet training rows (no simulator needed)
# ==============================================================================
def build_dagger(step=10, max_qd=15.0, max_base_speed=1.0):
    """Arm features + clean expert torque labels from all data/dagger_r*/ runs.
    Each run is cut at the first sign of the simulation blowing up (huge joint or base speeds).
    max_qd = 15 rad/s: a short (~10 ms) wrist spike of ~8 rad/s at the hold -> reach switch is normal
    and must NOT cut the run (a 5 rad/s limit discarded most of the near-target data)."""
    files = sorted(glob.glob(os.path.join(TASK_DIR, "data", "dagger_r*", "ep_*.npz")))
    assert len(files) > 0, "No DAgger runs found in data/dagger_r*/"
    Xa, Ya, n_rows = [], [], 0
    for f in files:
        d = np.load(f)
        xa, _ = arm_features(d)
        ya = d["torque_exp"]                                   # label = shadow expert, not what was applied
        bad = (np.abs(d["qd_arm"]).max(axis=1) > max_qd) | \
              (np.linalg.norm(d["base_vel"][:, :2], axis=1) > max_base_speed)
        n = int(np.argmax(bad)) if bad.any() else len(xa)      # keep rows before the first bad one
        s = slice(0, n, step)
        Xa.append(xa[s]); Ya.append(ya[s]); n_rows += len(xa[s])
    print(f"DAgger: {len(files)} runs, {n_rows} arm rows")
    return np.concatenate(Xa).astype(np.float32), np.concatenate(Ya).astype(np.float32)


def add_dagger(D, step=10):
    """Returns a copy of the dataset dict D with the DAgger rows appended to the ARM training set.
    Normalisation stats and val/test are left unchanged."""
    Xa, Ya = build_dagger(step)
    D = dict(D)
    D["Xa_train"] = np.concatenate([D["Xa_train"], Xa])
    D["Ya_train"] = np.concatenate([D["Ya_train"], Ya])
    print("Xa_train now", D["Xa_train"].shape)
    return D


# ==============================================================================
# 1) Collect: let the current networks drive, record the shadow expert's labels
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import torch
    torch.set_num_threads(1)
    from rollout import run_policy_episode, PARTS          # imported here: the simulator is only needed to collect
    from policy import BCPolicy
    from dataset import split_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)        # -> data/dagger_r<round>/
    parser.add_argument("--n", type=int, default=160)          # training configs per round
    parser.add_argument("--parts", choices=PARTS.keys(), default="both")
    parser.add_argument("--shard", type=int, default=0)        # this terminal's share of the n configs
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()

    pol = BCPolicy(os.path.join(TASK_DIR, "checkpoints"))
    train_f, _, _ = split_episodes()                           # clean TRAIN split only: no val/test configs
    rng = np.random.default_rng(1000 + args.round)             # same selection in every terminal
    chosen = sorted(rng.choice(train_f, size=min(args.n, len(train_f)), replace=False))
    chosen = chosen[args.shard::args.nshards]
    out_subdir = f"dagger_r{args.round}"
    for k, f in enumerate(chosen):
        ep = int(os.path.basename(f)[3:8])
        if os.path.exists(os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")):
            continue                                           # resume support
        print(f"=== DAgger round {args.round}: config {k+1}/{len(chosen)} (ep {ep}) ===")
        e = np.load(f)
        try:
            run_policy_episode(e["target"], ep, tuple(e["base_init"]), pol,
                               net_parts=PARTS[args.parts], out_subdir=out_subdir)
        except Exception as ex:
            print(f"!!! ep {ep} failed: {ex}")
