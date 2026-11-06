# Extension: obstacle avoidance

Static boxes are placed between the robot and its parking spot, and BaseNet has to drive around them
using a LiDAR. **ArmNet is not changed**: it is the final `dagger4_gravres` model from the main pipeline.
Nothing here modifies the template or the main pipeline; everything is in this folder.

Run all commands from `robotnik_arm_task/`.

## How it works

**The world (`world.py`).** Each scene has 0, 2, 3 or 4 boxes (probabilities 0.1 / 0.3 / 0.3 / 0.3),
sides 0.20–0.30 m with random yaw, placed between 10 % and 90 % of the line from the start to the base
goal and up to 1 m to either side, so some block the path and some do not. They are kept at least 0.6 m
from the target, 0.8 m from the base goal and 0.75 m from the start (the reach and the turns in place are
never blocked), and 1.0 m apart (the robot is 0.61 m wide). The start is 2.0–3.2 m from the base goal.

**The LiDAR.** A planar scan at 0.3 m height, 72 rays every 5° (3 m range), each starting at the edge of
the chassis so the robot never sees itself. It is updated every 10 simulation steps (100 Hz).

**The expert (`expert_avoid.py`).** The template's `base_controller` plus a reactive layer that is only
active while driving, using the full 360° scan:
- a box in the driving corridor (chassis half-width + 0.2 m) within 1.0 m of the bumper: slow down and
  steer away at up to 0.3 rad/s, keeping the chosen side until the box is passed;
- within 0.35 m: stop and turn in place until 0.6 m ahead is free (hysteresis);
- never turn towards a box right beside the chassis; once the way to the goal is clear, turn in place to
  face it again; turn in place only if the 360° scan shows room for the chassis corners;
- `v·|ω| ≤ 0.03` while avoiding, because the template's wheels have a lateral friction of 0.01 and the
  base slides sideways if it turns fast while moving; the base may park up to 0.25 m from its goal
  instead of circling it.

**BaseNet_obs.** The same MLP as BaseNet with 12 inputs: the original 10 plus two LiDAR features from the
**front 180°** only (`world.lidar_features`): the angle of the ray with the smallest clearance from the
chassis, and a proximity `clip((1.5 − clearance)/1.5, 0, 1)` (0 when nothing is within 1.5 m).
Trained only on successful, collision-free episodes. `policy_obs.py` detects 10 vs 12 inputs, so the
original BaseNet can be run in the same scenes as a no-LiDAR baseline.

## Reproducing

```bash
# 1. expert demonstrations -> data/raw_obs/
#    single-box scenes (an earlier version of the sampler, before CLEAR_EACH/LATERAL/PATH_FRAC were changed)
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 21 --start_id 20000 > log_o0.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 22 --start_id 21000 > log_o1.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 23 --start_id 22000 > log_o2.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 24 --start_id 23000 > log_o3.txt
#    multi-box scenes (current world.py)
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 31 --start_id 24000 > log_m0.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 32 --start_id 25000 > log_m1.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 33 --start_id 26000 > log_m2.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 100 --seed 34 --start_id 27000 > log_m3.txt
#    test scenes (IDs >= 29000, never trained on)
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 25 --seed 91 --start_id 29000 > log_t0.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 25 --seed 92 --start_id 29100 > log_t1.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 25 --seed 93 --start_id 29200 > log_t2.txt
ROBOENV_HEADLESS=1 python yousef/obstacles/collect_obs.py --n 25 --seed 94 --start_id 29300 > log_t3.txt

# 2. dataset (base features + 2 LiDAR features) -> data/processed_obs_step10.npz, then train
python yousef/obstacles/dataset_obs.py
python yousef/obstacles/train_obs.py --version multi      # -> checkpoints/base_net_obs_multi.pt
#    (base_net_obs_1box.pt was trained the same way when only IDs 20000-23999 existed)

# 3. closed-loop test, 4 shards each, then the summary -> results/obs_<tag>.csv
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_multi   --base base_net_obs_multi --shard 0 --nshards 4
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_1box    --base base_net_obs_1box  --shard 0 --nshards 4
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_nolidar --base base_net           --shard 0 --nshards 4
#    ... shards 1-3 for each, then e.g.
python yousef/obstacles/evaluate_obs.py --tag obs_multi --summary_only
```

To watch the expert with the LiDAR rays drawn (GUI, red = hit, green = free), use IDs that aren't part of
the data set, e.g. `python yousef/obstacles/collect_obs.py --n 3 --seed 0 --start_id 19000 --rays`.

## Results

95 test scenes the expert solved (mostly 2–3 boxes), ArmNet the same in every row.

| BaseNet | collided | @2 cm | @2 cm, no collision | median time |
|---|---|---|---|---|
| no LiDAR (original BaseNet) | 29.5 % | 74.7 % | 68.4 % | 17.6 s |
| LiDAR, single-box training | 22.1 % | 43.2 % | 36.8 % | 48.0 s |
| LiDAR, single- + multi-box training | 18.9 % | 60.0 % | 54.7 % | 24.4 s |
| expert (360° LiDAR) | 6.3 % | 100 % | 93.7 % | 17.4 s |

Paired McNemar tests for single + multi-box training: against single-box training, success at 2 cm
p = 0.011 (+26/−10), collisions p = 0.63; against no LiDAR, collisions p = 0.08, success at 2 cm p = 0.029
(the baseline is still better at reaching).

The expert succeeded in 395 of the 400 single-box demonstrations (1 collision) and 373 of the 400
multi-box ones (31 collisions).

**Why the gap remains.** The expert's avoidance depends on internal state (the side it has committed to,
when a stop-and-turn ends) that two features cannot show, the same partial observability as `q_des` in the
main task. And most collisions are with boxes beside or behind the robot while it turns, outside the
front 180° that BaseNet sees.

## Files

| File | What it does |
|---|---|
| `world.py` | box sampling and spawning, LiDAR scans (360° and front 180°), the 2 LiDAR features, collision check |
| `expert_avoid.py` | the obstacle-aware expert (template `base_controller` + reactive LiDAR steering) |
| `collect_obs.py` | expert demonstrations with obstacles (a copy of `../collect_data.py` with boxes and LiDAR) |
| `dataset_obs.py` | 12-input BaseNet dataset from successful, collision-free episodes |
| `train_obs.py` | trains BaseNet_obs (`--version` names the checkpoint) |
| `rollout_obs.py`, `policy_obs.py` | closed-loop run with obstacles; loads a 10- or 12-input BaseNet |
| `evaluate_obs.py` | the obstacle test set → `results/obs_<tag>.csv` (adds `n_boxes`, `collision_steps`, `collided`) |
