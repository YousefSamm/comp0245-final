"""Test-set evaluation: run the trained networks on ALL successful test episodes (IDs >= 9000).

Usage (from robotnik_arm_task/), 4 terminals:
    ROBOENV_HEADLESS=1 ROBOENV_MAX_STEPS=25000 python yousef/evaluate.py --parts both --tag dagger2_full --shard 0 --nshards 4
    ... shards 1..3, then the totals:
    python yousef/evaluate.py --parts both --tag dagger2_full --summary_only

--tag names the model version being tested (whatever checkpoints/base_net.pt + arm_net.pt are now).
Runs are saved to data/test_<parts>_<tag>/, the per-episode table to results/test_<parts>_<tag>.csv.
Already-finished episodes are skipped, so an interrupted evaluation resumes.
"""
import os
import sys
import csv
import argparse
import numpy as np

from dataset import split_episodes

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS_CHOICES = ["base", "arm", "both"]

parser = argparse.ArgumentParser()
parser.add_argument("--parts", choices=PARTS_CHOICES, default="both")
parser.add_argument("--tag", required=True)                # name of the model version
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--summary_only", action="store_true")  # just summarise already-saved runs
args = parser.parse_args()

_, _, test_f = split_episodes()                            # the 98 successful clean test episodes
out_subdir = f"test_{args.parts}_{args.tag}"

if not args.summary_only:
    import torch
    torch.set_num_threads(1)
    from rollout import run_policy_episode, PARTS          # simulator only needed when running
    from policy import BCPolicy
    pol = BCPolicy(os.path.join(TASK_DIR, "checkpoints"))
    for f in test_f[args.shard::args.nshards]:
        ep = int(os.path.basename(f)[3:8])
        if os.path.exists(os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")):
            continue
        print(f"=== TEST {args.parts}/{args.tag}: ep {ep} ===")
        e = np.load(f)
        try:
            run_policy_episode(e["target"], ep, tuple(e["base_init"]), pol,
                               net_parts=PARTS[args.parts], out_subdir=out_subdir)
        except Exception as ex:
            print(f"!!! ep {ep} failed: {ex}")

# ---- summary over all test runs saved so far ----------------------------------------
from metrics import run_metrics, THRESHOLDS
rows = []
for f in test_f:
    ep = int(os.path.basename(f)[3:8])
    p = os.path.join(TASK_DIR, "data", out_subdir, f"ep_{ep:05d}.npz")
    if os.path.exists(p):
        m = run_metrics(np.load(p), np.load(f)); m["ep"] = ep; rows.append(m)
if not rows:
    print("no test runs found"); sys.exit(0)

os.makedirs(os.path.join(TASK_DIR, "results"), exist_ok=True)
csv_path = os.path.join(TASK_DIR, "results", f"test_{args.parts}_{args.tag}.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

n = len(rows)
print(f"\n================ TEST SET: {args.parts} / {args.tag}  ({n}/{len(test_f)} episodes) ================")
for th in THRESHOLDS:
    c = int(th * 100)
    sr = np.mean([r[f"success_{c}cm"] for r in rows]) * 100
    tt = np.nanmedian([r[f"time_to_{c}cm"] for r in rows])
    print(f"success @ {c} cm : {sr:5.1f} %   median time to reach: {tt:5.1f} s")
print(f"best EE error    : median {np.median([r['best_err'] for r in rows])*100:.1f} cm")
print(f"final EE error   : median {np.median([r['final_err'] for r in rows])*100:.1f} cm")
print(f"base goal error  : median {np.median([r['base_goal_err'] for r in rows])*100:.1f} cm")
print(f"blew up          : {sum(r['blew_up'] for r in rows)} episodes")
print(f"expert reference : 100 % @ 1 cm on these episodes, median duration "
      f"{np.median([r['expert_duration'] for r in rows]):.1f} s")
print("saved", csv_path)
