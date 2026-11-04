"""Run ONE episode with the trained networks driving and print a summary. Best with the GUI (no ROBOENV_HEADLESS).

Usage (from robotnik_arm_task/):
    python yousef/run_demo.py --parts both --ep 9000     # both networks (full learned policy)
    python yousef/run_demo.py --parts base               # BaseNet drives, expert moves the arm
    python yousef/run_demo.py --parts arm                # expert drives the base, ArmNet moves the arm
Prefix with ROBOENV_MAX_STEPS=25000 so a failing run still ends (and saves) after 25 s.
--ep: any TEST episode ID (9000-9099); its target and start pose are reused, so the expert's run is the reference.
"""
import os
import argparse
import numpy as np
import torch
torch.set_num_threads(1)                       # faster for tiny one-row predictions

from rollout import run_policy_episode, PARTS, TASK_DIR
from metrics import run_metrics
from policy import BCPolicy
from features import base_goal, wrap

parser = argparse.ArgumentParser()
parser.add_argument("--parts", choices=PARTS.keys(), default="both")
parser.add_argument("--ep", type=int, default=9000)
args = parser.parse_args()

pol = BCPolicy(os.path.join(TASK_DIR, "checkpoints"))
e = np.load(os.path.join(TASK_DIR, "data", "raw", f"ep_{args.ep:05d}.npz"))
out_subdir = f"eval_{args.parts}"
run_policy_episode(e["target"], args.ep, tuple(e["base_init"]), pol,
                   net_parts=PARTS[args.parts], out_subdir=out_subdir)

# ---- summary --------------------------------------------------------------------
r = np.load(os.path.join(TASK_DIR, "data", out_subdir, f"ep_{args.ep:05d}.npz"))
m = run_metrics(r)
print("\n================ RUN SUMMARY ================")
print(f"networks driving : {args.parts}")
print(f"stop message     : {r['stop_message']}")
print(f"best EE error    : {m['best_err']*100:.1f} cm   success @ 1/2/5 cm: "
      f"{m['success_1cm']}/{m['success_2cm']}/{m['success_5cm']}   final error: {m['final_err']*100:.1f} cm")
print(f"expert episode   : final error {float(e['final_error'])*100:.1f} cm, duration {e['t'][-1]:.1f} s "
      f"(this run: {r['t'][-1]:.1f} s)")
print(f"base goal error  : {m['base_goal_err']:.3f} m, bearing error {m['base_bear_err']:.3f} rad "
      f"(expert accepts up to ~0.15 m / 0.05 rad)")
dw = np.abs(r["wheel_cmds"] - r["wheel_exp"]).max(axis=1)      # rad/s
dt = np.abs(r["torque_arm"] - r["torque_exp"]).max(axis=1)     # N·m
for name, diff, thr in [("wheels", dw, 0.5), ("arm torque", dt, 2.0)]:
    big = np.where(diff > thr)[0]
    if len(big):
        i = big[0]
        print(f"{name:10s}: first differs from the expert by > {thr} at t = {r['t'][i]:.2f} s "
              f"(phase {int(r['phase'][i])}); max difference {diff.max():.2f}")
    else:
        print(f"{name:10s}: always within {thr} of the expert (max {diff.max():.2f})")
