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

**BaseNet_obs.** The same MLP as BaseNet: the original 10 inputs plus LiDAR features, trained only on
successful, collision-free episodes. Three LiDAR feature sets were tried (`dataset_obs.py --lidar ...`):

| `--lidar` | Inputs | What they are |
|---|---|---|
| `front2` (default) | 12 | angle of the ray with the smallest clearance from the chassis (front 180°) and a proximity `clip((1.5 − clearance)/1.5, 0, 1)`, 0 when nothing is within 1.5 m (`world.lidar_features`) |
| `sect8` | 18 | proximity of the closest box in 8 sectors of 45° around the robot, from the 360° scan (`world.lidar_sector_features`) |
| **`corr`** | 16 | **what the expert steers from** (`world.lidar_corridor_features`, same computation as `expert_avoid.corridor`): proximity of the nearest hit inside the driving corridor and the side it is on, closest proximity left and right, plus the two `front2` features |

`policy_obs.py` picks the features from the network's input size (10 = no LiDAR, 12, 16, 18), so the
original BaseNet can also be run in the same scenes as a no-LiDAR baseline.

**DAgger (`dagger_obs.py`).** BaseNet_obs drives 160 training scenes while the obstacle-aware expert runs
alongside; its wheel commands become extra labels. Labels within 0.35 m of the base goal or in the expert's
"done" phase are dropped (there the expert's phase machine no longer follows the robot, as in the main
DAgger), and runs are cut at the first arm joint speed above 15 rad/s.

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
#    the best model, corridor features (no DAgger):
python yousef/obstacles/dataset_obs.py --lidar corr
python yousef/obstacles/train_obs.py --version corr --lidar corr       # -> checkpoints/base_net_obs_corr.pt

# 3. closed-loop test, 4 shards each, then the summary -> results/obs_<tag>.csv
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_multi   --base base_net_obs_multi --shard 0 --nshards 4
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_1box    --base base_net_obs_1box  --shard 0 --nshards 4
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_nolidar --base base_net           --shard 0 --nshards 4
ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_corr    --base base_net_obs_corr  --shard 0 --nshards 4
#    ... shards 1-3 for each, then e.g.
python yousef/obstacles/evaluate_obs.py --tag obs_multi --summary_only
```

The other experiments are one script each (dataset, training and test in one go, all from data that
already exists except the DAgger runs):

```bash
bash yousef/obstacles/run_sect8.sh        # 8 sectors from the 360 deg scan        -> results/obs_obs_sect8.csv
bash yousef/obstacles/run_dagger_obs.sh   # DAgger round (collects 160 runs first) -> results/obs_obs_dagger1.csv
bash yousef/obstacles/run_fixes.sh        # DAgger rows near boxes only + corridor features
                                          #   -> results/obs_obs_daggernear.csv, results/obs_obs_corr.csv
```

**Watching a run in the GUI** (one test scene; saved to `data/demo_obs/`, which nothing else reads):

```bash
python yousef/obstacles/run_demo_obs.py --ep 29206                            # corridor features (best)
python yousef/obstacles/run_demo_obs.py --ep 29206 --base base_net_obs_multi   # 2 front features
python yousef/obstacles/run_demo_obs.py --ep 29206 --base base_net             # no LiDAR
python yousef/obstacles/run_demo_obs.py --ep 29206 --expert                    # the expert drives
```

Scenes 29206, 29305 and 29312 (3 boxes each) are ones where the no-LiDAR and 2-feature networks collide and
the corridor network does not.

## Results

95 test scenes the expert solved (mostly 2–3 boxes), ArmNet the same in every row. Rows marked * were
trained on the single- and multi-box demonstrations.

| BaseNet | collided | @2 cm | @2 cm, no collision | @5 cm | median time |
|---|---|---|---|---|---|
| no LiDAR (original BaseNet) | 29.5 % | 74.7 % | 68.4 % | 76.8 % | 17.6 s |
| 2 features, single-box training | 22.1 % | 43.2 % | 36.8 % | 53.7 % | 48.0 s |
| 2 features, + multi-box training* | 18.9 % | 60.0 % | 54.7 % | 74.7 % | 24.4 s |
| 8 sectors (360°)* | 35.8 % | 37.9 % | 33.7 % | 42.1 % | 48.0 s |
| 2 features + DAgger, all labels* | 15.8 % | 13.7 % | 13.7 % | 17.9 % | 48.0 s |
| 2 features + DAgger, near boxes only* | 14.7 % | 37.9 % | 35.8 % | 48.4 % | 35.1 s |
| **corridor features*** | **4.2 %** | **74.7 %** | **72.6 %** | **86.3 %** | **19.5 s** |
| expert (360° LiDAR) | 6.3 % | 100 % | 93.7 % | | 17.4 s |

Paired McNemar tests for the corridor features: against 2 features, collisions +3/−17 (p = 0.003) and
success at 2 cm +25/−11 (p = 0.029); against no LiDAR, collisions +0/−24 (p < 0.001), success at 2 cm
p = 1.0, at 5 cm p = 0.049; against the expert, collisions p = 0.75.

The expert succeeded in 395 of the 400 single-box demonstrations (1 collision) and 373 of the 400
multi-box ones (31 collisions).

**What the experiments show.**
- Two features of the closest hit cannot separate a box in the path from one beside the robot, and the
  expert's decisions depend on exactly that. Giving BaseNet the expert's own corridor quantities closes the
  gap to the expert's collision rate without DAgger or more data.
- Seeing more is not the same as seeing the right thing: 8 sectors from the full scan were active in 90 %
  of samples (boxes beside or already passed), and collisions got worse.
- DAgger lowered collisions, but the shadow expert's labels near the base goal are unreliable (its phase
  machine does not follow the robot), and without them the base often failed its final turn to face the
  target.
- Success at 2 cm with the corridor features equals the no-LiDAR baseline, so it is now limited by the reach
  from a less precise parking pose, not by the boxes. The expert's sticky side and stop-and-turn state are
  still not observed; a recurrent policy is the next step.

## Files

| File | What it does |
|---|---|
| `world.py` | box sampling and spawning, LiDAR scans (360° and front 180°), the 2 LiDAR features, collision check |
| `expert_avoid.py` | the obstacle-aware expert (template `base_controller` + reactive LiDAR steering) |
| `collect_obs.py` | expert demonstrations with obstacles (a copy of `../collect_data.py` with boxes and LiDAR) |
| `dataset_obs.py` | BaseNet_obs dataset from successful, collision-free episodes (`--lidar front2/sect8/corr`) |
| `train_obs.py` | trains BaseNet_obs (`--version` names the checkpoint, `--lidar` / `--dagger` / `--dagger_near` pick the dataset) |
| `dagger_obs.py` | DAgger runs for BaseNet_obs and the label filter used by `dataset_obs.py --dagger` |
| `run_sect8.sh`, `run_dagger_obs.sh`, `run_fixes.sh` | one-command experiments (see Reproducing) |
| `rollout_obs.py`, `policy_obs.py` | closed-loop run with obstacles; loads a 10- or 12-input BaseNet |
| `run_demo_obs.py` | one test scene in the GUI with any BaseNet, or with the expert driving |
| `evaluate_obs.py` | the obstacle test set → `results/obs_<tag>.csv` (adds `n_boxes`, `collision_steps`, `collided`) |
