"""Obstacle test-set evaluation: the networks drive the obstacle test episodes (expert demos with IDs >= 29000,
collected with collect_obs.py) among the same boxes.

Usage (from robotnik_arm_task/), 4 terminals:
    ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_1box --base base_net_obs_1box --shard 0 --nshards 4
    ... shards 1-3, then:
    python yousef/obstacles/evaluate_obs.py --tag obs_1box --summary_only
No-LiDAR baseline (the main BaseNet in the same scenes):  --tag obs_nolidar --base base_net
Runs -> data/test_obs_<tag>/, table -> results/obs_<tag>.csv. Finished episodes are skipped (resumable).
"""
import os, sys, csv, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(1, os.path.dirname(HERE))
TASK_DIR = os.path.dirname(os.path.dirname(HERE))
TEST_FROM = 29000

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--base", default="base_net_obs")     # BaseNet checkpoint (without .pt)
parser.add_argument("--arm", default="arm_net")           # ArmNet checkpoint (without .pt)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--summary_only", action="store_true")
args = parser.parse_args()

test_f = [f for f in sorted(glob.glob(os.path.join(TASK_DIR, "data", "raw_obs", "ep_*.npz")))
          if int(os.path.basename(f)[3:8]) >= TEST_FROM and bool(np.load(f)["success"])]
out_subdir = f"test_obs_{args.tag}"
print(f"{len(test_f)} obstacle test episodes")

if not args.summary_only:
    import torch
    torch.set_num_threads(1)
    from rollout_obs import run_policy_episode
    from policy_obs import BCPolicyObs
    pol = BCPolicyObs(os.path.join(TASK_DIR, "checkpoints"), base_file=f"{args.base}.pt", arm_file=f"{args.arm}.pt")
    for f in test_f[args.shard::args.nshards]:
        ep = int(os.path.basename(f)[3:8])
        if os.path.exists(os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")):
            continue
        e = np.load(f)
        print(f"=== OBSTACLE TEST {args.tag}: ep {ep} ({len(e['obstacles'])} boxes) ===")
        try:
            run_policy_episode(e["target"], ep, tuple(e["base_init"]), pol, net_parts=("base", "arm"),
                               out_subdir=out_subdir, obstacles=e["obstacles"])
        except Exception as ex:
            print(f"!!! ep {ep} failed: {ex}")

# ---- summary (also after running) ----
from metrics import run_metrics
rows = []
for f in test_f:
    ep = int(os.path.basename(f)[3:8])
    p = os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")
    if not os.path.exists(p):
        continue
    r = np.load(p)
    m = run_metrics(r, expert=np.load(f)); m["ep"] = ep
    m["n_boxes"] = int(len(r["obstacles"])); m["collision_steps"] = int(r["n_collision_steps"])
    m["collided"] = m["collision_steps"] > 0
    rows.append(m)
if rows:
    os.makedirs(os.path.join(TASK_DIR, "results"), exist_ok=True)
    out = os.path.join(TASK_DIR, "results", f"obs_{args.tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    R = lambda k: np.array([r[k] for r in rows])
    print(f"\n======== OBSTACLE TEST: {args.tag}  ({len(rows)}/{len(test_f)} episodes) ========")
    print(f"collided          : {100 * R('collided').mean():5.1f} %  of episodes")
    for c in (5, 2, 1):
        print(f"success @ {c} cm    : {100 * R(f'success_{c}cm').mean():5.1f} %   (and no collision: {100 * (R(f'success_{c}cm') & ~R('collided')).mean():5.1f} %)")
    print(f"base goal error   : median {100 * np.median(R('base_goal_err')):.1f} cm")
    print(f"blew up           : {int(R('blew_up').sum())}")
    print("saved", out)
