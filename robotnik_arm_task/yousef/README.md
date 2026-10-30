# Expert data collection (Yousef)

Scripts to generate the expert demonstrations (training + test data) for the
behaviour-cloning task. Running the commands below with the **same seeds** reproduces
**exactly the same episodes** (same targets, same start poses), so everyone in the group
trains and tests on identical data.

The dataset itself is **not in the repo** (≈1.9 GB, ignored by `.gitignore`), so generate it
locally with the steps below.

---

## 1. Requirements

- Your usual **RoboEnv** environment (the one used for the labs: `pybullet`, `pinocchio`,
  `simulation_and_control`).
- The template folders `configs/` and `models/` inside `robotnik_arm_task/` (see the template README).
- **≥ 3 GB free disk space.** Check with `df -h ~`. A full disk corrupts episode files.

Folder layout:
```
robotnik_arm_task/
├── robotnik_arm_task.py      # template (unchanged)
├── robotnik_arm_control.py   # template expert controller (unchanged)
├── configs/  models/         # template assets
├── data/raw/                 # ← episodes are written here (git-ignored)
└── yousef/
    ├── collect_data.py       # runs the expert and records episodes
    ├── inspect_data.py       # checks the collected data
    └── README.md
```

---

## 2. Collect the data

Run **from the `robotnik_arm_task/` folder**. Open **5 terminals** (the commands are
independent and can run in parallel), activate RoboEnv in each, and run one command per terminal:

```bash
# Training pool: 4 × 300 = 1,200 episodes
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 1 --start_id 0    > log_s1.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 2 --start_id 1000 > log_s2.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 3 --start_id 2000 > log_s3.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 4 --start_id 3000 > log_s4.txt

# Test set: 100 episodes, NEVER used for training
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 100 --seed 900 --start_id 9000 > log_test.txt
```

- `ROBOENV_HEADLESS=1` runs without the GUI (much faster: ≈ 5 s per episode).
- `> log_xx.txt` stores the printed output in a log file instead of the terminal.
- Total time: ≈ 25–30 min with the 5 commands in parallel.

**Interrupted?** Just run the same command again. Episodes that already exist in
`data/raw/` are skipped, and the missing ones are generated with exactly the same
configuration they would have had.
⚠ If a run stopped because the **disk was full**, the last file of that run may be broken
(0 bytes). Delete it before re-running, otherwise it is skipped as "already collected".

---

## 3. Dataset definition

| Set | Seed | File IDs | Episodes |
|---|---|---|---|
| Training pool | 1 | `ep_00000` – `ep_00299` | 300 |
| Training pool | 2 | `ep_01000` – `ep_01299` | 300 |
| Training pool | 3 | `ep_02000` – `ep_02299` | 300 |
| Training pool | 4 | `ep_03000` – `ep_03299` | 300 |
| **Test set** | **900** | **`ep_09000` – `ep_09099`** | **100** |

**Rule: file IDs ≥ 9000 are the test set. Never train or tune hyperparameters on them.**
The train/validation split is done *by episode* inside the training pool (IDs < 9000).

### How each episode is generated (`collect_data.py`, `__main__` block)

- **Target** (world frame): horizontal distance `r ~ U(0.8, 2.0)` m from the world origin,
  direction `θ ~ U(−π, π)` (all around the robot), height `z ~ U(0.1, 1.25)` m.
- **Start pose of the base**: `x, y ~ U(−1, 1)` m, heading `yaw ~ U(−π, π)` rad
  (applied with `resetBasePositionAndOrientation`, keeping the spawn height).
- Samples where the target is closer than **0.6 m** (horizontally) to the start position are
  rejected and re-drawn (rejected samples do not use up an episode ID).
- **Expert**: the template controller with its default parameters
  (`REACH_TOLERANCE = 0.01 m`, `BASE_NAV_TIMEOUT = 25 s`, max 40,000 steps, fingers fixed at 0.1 rad).
- Simulation step: 1 ms. **Every step is recorded** (float32); subsampling is done later
  when building the training set.

---

## 4. Check your data

```bash
ls data/raw | wc -l                 # → 1300
python yousef/inspect_data.py | tail -1
```

Expected result: **1300 files, ≈ 1283/1300 successful** (≈ 98.7 %; the expert itself fails on a few
hard targets). Failed episodes are excluded from training.

**Quick check that you have the same data as me:** the first episode of seed 1 must be
```python
import numpy as np
d = np.load("data/raw/ep_00000.npz")
print(d["target"].round(2), d["base_init"].round(2))
# → [-1.35  0.43  0.27] [ 0.9  -0.38 -0.48]
```
Configs (targets/starts) are identical by construction. The recorded trajectories should
also match as long as everyone uses the same RoboEnv/PyBullet versions.

---

## 5. What is inside each episode file (`ep_XXXXX.npz`)

Load with `d = np.load(path)`; access with `d["key"]`. `N` = number of steps (1 kHz).

**Per-step arrays**

| Key | Shape | Meaning |
|---|---|---|
| `t` | (N,) | time since episode start (s) |
| `base_pos` | (N, 2) | base x, y in the world frame (m) |
| `base_bearing` | (N,) | base heading / yaw (rad) |
| `base_vel` | (N, 3) | base vx, vy (world frame, m/s) and yaw rate (rad/s) |
| `q_arm` | (N, 7) | arm joint angles = `q_mes[4:11]` (rad) |
| `qd_arm` | (N, 7) | arm joint velocities = `qd_mes[4:11]` (rad/s) |
| `qd_wheels` | (N, 4) | measured wheel speeds = `qd_mes[0:4]` (rad/s) |
| `ee_pos` | (N, 3) | end-effector position (forward kinematics, world frame) |
| `wheel_cmds` | (N, 4) | **expert label**: wheel velocity commands [FR, FL, BL, BR] (rad/s) |
| `torque_arm` | (N, 7) | **expert label**: arm joint torques (N·m, clipped to ±30) |
| `phase` | (N,) | expert's base phase: 0 align, 1 drive, 2 orient, 3 done (arm reaching) |
| `q_des_arm` | (N, 7) | expert's internal joint reference (analysis only, not available at run time) |
| `tau_g` | (N, 7) | gravity-compensation torque g(q) of the arm (N·m) |

**Per-episode values**

| Key | Meaning |
|---|---|
| `target` | target position (x, y, z), world frame |
| `base_init` | start pose of the base (x, y, yaw) |
| `success` | `True` if the target was reached within 1 cm |
| `final_error` | final end-effector distance to the target (m) |
| `stop_message` | why the episode ended |
| `nav_time` | time when the base finished navigating (s); ≥ 25 means it timed out |

Joint order in `q_mes` / `qd_mes` (17 values): 0–3 wheels (FR, FL, BL, BR), 4–10 arm joints 1–7,
11–16 fingers. Arm joints 1, 3, 5, 7 are continuous (no limits), so use sin/cos when used as features.
