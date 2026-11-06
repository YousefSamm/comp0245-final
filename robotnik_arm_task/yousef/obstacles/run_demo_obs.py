"""Watch ONE obstacle test scene in the PyBullet GUI.

    python yousef/obstacles/run_demo_obs.py --ep 29003                          # corridor-feature BaseNet (best)
    python yousef/obstacles/run_demo_obs.py --ep 29003 --base base_net_obs_multi # 2 front features
    python yousef/obstacles/run_demo_obs.py --ep 29003 --base base_net           # no LiDAR (original BaseNet)
    python yousef/obstacles/run_demo_obs.py --ep 29003 --expert                  # the obstacle-aware expert drives

--ep is an obstacle test scene (IDs 29000-29324, see data/raw_obs/). The run is saved to data/demo_obs/,
which no dataset or evaluation reads, so nothing here changes the data. Press q in the GUI window to stop.
"""
import os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
TASK_DIR = os.path.dirname(os.path.dirname(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--ep", type=int, default=29000)
parser.add_argument("--base", default="base_net_obs_corr")
parser.add_argument("--arm", default="arm_net")
parser.add_argument("--expert", action="store_true", help="the expert drives (base and arm); the networks only run alongside")
args = parser.parse_args()

scene = os.path.join(TASK_DIR, "data", "raw_obs", f"ep_{args.ep:05d}.npz")
if not os.path.exists(scene):
    sys.exit(f"no scene {scene}")
e = np.load(scene)
print(f"scene {args.ep}: {len(e['obstacles'])} boxes | target {np.round(e['target'], 2)} | expert success {bool(e['success'])}")

import torch
from rollout_obs import run_policy_episode
from policy_obs import BCPolicyObs
pol = BCPolicyObs(os.path.join(TASK_DIR, "checkpoints"), base_file=f"{args.base}.pt", arm_file=f"{args.arm}.pt")
parts = () if args.expert else ("base", "arm")
run_policy_episode(e["target"], args.ep, tuple(e["base_init"]), pol, net_parts=parts,
                   out_subdir="demo_obs", obstacles=e["obstacles"])

r = np.load(os.path.join(TASK_DIR, "data", "demo_obs", f"ep_{args.ep:05d}.npz"))
err = np.linalg.norm(r["ee_pos"] - r["target"], axis=1)
print(f"\n{'expert' if args.expert else args.base}: collided {int(r['n_collision_steps']) > 0} "
      f"({int(r['n_collision_steps'])} contact steps) | closest {100 * err.min():.1f} cm | final {100 * err[-1]:.1f} cm "
      f"| duration {r['t'][-1]:.1f} s | {r['stop_message']}")
