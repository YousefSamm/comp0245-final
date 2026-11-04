# Behaviour cloning pipeline (Yousef)

This folder has everything needed to go from the expert controller in the template to a pair of neural
networks that drive the robot on their own: data collection, features, training, DAgger and evaluation.

The idea is that the report can be written from what's here. Every number in the results tables can be
regenerated with the commands below, and the CSV files in `results/` contain the raw per-episode numbers
for plots.

If something doesn't work or doesn't make sense, ask me before changing anything, because later steps
depend on earlier ones.

---

## 1. What we built, in short

The template has an expert that drives the base to a spot near the target and then reaches with the arm.
We replace it with two MLPs:

- **BaseNet**: 10 inputs (where the parking spot is from the robot's point of view, heading errors, base
  speed) → forward speed and turning rate (v, ω), converted to the 4 wheel commands.
- **ArmNet**: 25 inputs (joint angles and velocities, hand-to-target error in the robot frame, and a few
  signals that tell it whether the base has finished) → 7 joint torques.

Both are trained to copy the expert (behaviour cloning), then tested in closed loop, meaning the networks
are actually in control of the simulated robot.

---

## 2. Setup

- Use the usual RoboEnv environment from the labs, plus PyTorch
  (`uv pip install torch --index-url https://download.pytorch.org/whl/cu128`, or the CPU version).
- Run everything from the `robotnik_arm_task/` folder.
- `configs/` and `models/` from the template must be inside `robotnik_arm_task/`.
- Keep at least ~5 GB of free disk space. All the data together is about 2.5 GB, and a full disk
  leaves broken episode files behind.

Nothing in `data/` or `checkpoints/` is on GitHub (too big). Section 3 regenerates the data. For the
DAgger data you also need my checkpoints, see 3.3.

---

## 3. Getting exactly the same data

Every collection command uses a fixed seed, so the targets and start poses come out identical on
every machine. The recorded trajectories should also match as long as we use the same RoboEnv/PyBullet
versions.

Run each block in separate terminals (they're independent), with `ROBOENV_HEADLESS=1` so there's no GUI.
If a run gets interrupted, just start the same command again: finished episodes are skipped.

### 3.1 Clean expert demonstrations → `data/raw/` (≈ 30 min)

```bash
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 1   --start_id 0    > log_s1.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 2   --start_id 1000 > log_s2.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 3   --start_id 2000 > log_s3.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 300 --seed 4   --start_id 3000 > log_s4.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --n 100 --seed 900 --start_id 9000 > log_test.txt
```

- IDs 0–3299: training pool (split 85/15 into train/val by episode, seed 0).
- **IDs 9000–9099: test set. Never train or tune on these.**
- Targets: 0.8–2.0 m from the world origin in any direction, height 0.1–1.25 m.
  Start pose: x, y in [−1, 1] m, any heading. Targets closer than 0.6 m to the start are re-drawn.
- Result: 1300 episodes, 1283 successful (98.7 %). The 17 failures are excluded from training.

Quick check: `ep_00000.npz` should have target `[-1.35, 0.43, 0.27]` and start `[0.9, -0.38, -0.48]`.

### 3.2 Noise-injected expert demonstrations → `data/raw_noise/` (≈ 10 min)

```bash
ROBOENV_HEADLESS=1 python yousef/collect_data.py --noise --n 100 --seed 11 --start_id 5000 > log_n1.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --noise --n 100 --seed 12 --start_id 6000 > log_n2.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --noise --n 100 --seed 13 --start_id 7000 > log_n3.txt
ROBOENV_HEADLESS=1 python yousef/collect_data.py --noise --n 100 --seed 14 --start_id 8000 > log_n4.txt
```

The expert still drives, but random pushes are added to the commands sent to the robot. Each push is
held for 0.2 s: 2.5 N·m per arm joint, 0.015 m/s and 0.06 rad/s on the base. There are no arm pushes in
the last 5 cm before the target and no base pushes once the base has parked. What we store as the label
is the expert's **clean** command (`torque_exp`, `wheel_exp`), so the data shows the expert correcting
itself. Section 5 explains why this was needed.

### 3.3 DAgger runs → `data/dagger_r1/`, `data/dagger_r2/` (≈ 10 min per round)

```bash
ROBOENV_HEADLESS=1 ROBOENV_MAX_STEPS=20000 python yousef/dagger.py --round 1 --shard 0 --nshards 4 > log_d0.txt
#   ... same with --shard 1, 2, 3 in three more terminals
```

Here the **networks** drive and the expert only gives labels. That means the data depends on which
networks were loaded (`checkpoints/base_net.pt` and `arm_net.pt`) when it was collected, so it can't be
regenerated from seeds alone. To get my DAgger data you need my checkpoints. I'll share a zip of
`checkpoints/`, or you can use my `data/dagger_r*/` folders directly.

| Round | Networks that drove | Then retrained |
|---|---|---|
| 1 | BaseNet + ArmNet trained on clean + noise data (`arm_net_noise.pt`) | ArmNet → `arm_net_dagger1.pt` |
| 2 | BaseNet + `arm_net_dagger1.pt` | ArmNet → `dagger2_full` (the current `arm_net.pt`) |

Each round uses 160 configurations from the **training split only**, so val and test stay untouched.

---

## 4. Building the datasets and training

1. **`01_features_dataset.ipynb`**: sanity checks on the features, then builds the processed datasets
   (every 10th step, i.e. 100 Hz, with normalisation stats from the training split only):
   - `build_dataset(step=10)` → `data/processed_step10.npz` (clean only)
   - `build_dataset(step=10, use_noise=True)` → `data/processed_step10_noise.npz` (clean + noise)
2. **`02_training.ipynb`**: set the options in the first cell, then Restart & Run All.
   - `DATASET` = `"clean"` or `"noise"`
   - `USE_DAGGER`: adds all `data/dagger_r*/` runs to ArmNet's training data (not BaseNet's, see 5.4)
   - `TRAIN_BASE` / `TRAIN_ARM`: which network to retrain
   - `VERSION`: name for the checkpoints. They're saved as `<net>_<VERSION>.pt` and also as the
     "active" `<net>.pt` that everything else loads.

Both networks are 3 hidden layers of 256 units with ReLU, trained with Adam (lr 1e-3, batch 1024) on
MSE of normalised outputs, with early stopping on the validation loss (patience 5, best epoch kept).

---

## 5. How the work progressed and why

This is the part the report should tell. Each step came from a specific failure we observed.

### 5.1 Offline it looked perfect
Validation error on the expert's own trajectories was tiny (BaseNet ≈ 0.0015, ArmNet ≈ 0.0066 normalised
MSE, so roughly 99 % of the variance explained). A one-step check on a test episode gave almost the same
commands as the expert in every phase.

### 5.2 In closed loop the robot flew
With the networks in control, the arm went wild after a couple of seconds and the physics blew up.
Testing each network separately (`run_demo.py --parts base` / `--parts arm`) showed:
- **BaseNet was fine.** It parked about 7 cm from its goal. It only looked like a failure because the
  expert's internal phase logic, which we used to decide "base done", got stuck. We now decide that from
  the measured pose: within 0.15 m, within 0.1 rad, and stopped.
- **ArmNet couldn't hold the arm still**, already 0.1 s into the episode.

### 5.3 Why the arm couldn't hold: no stiffness was learned
The expert holds the arm with a PD law, τ = 100·(q_init − q) − 30·q̇ + gravity. Nudging a joint by 0.01 rad
should change its torque by about −1 N·m. We checked what the network had learned by nudging its inputs
(the stiffness test at the end of `02_training.ipynb`):

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| learned dτ/dq (expert −100) | −69 | −88 | **+0.3** | −19 | −27 | **+0.5** | −49 |
| learned dτ/dq̇ (expert −30) | −30 | −31 | −16 | −29 | −25 | **+0.3** | −21 |

Joints 3 and 6 had no restoring torque at all. The reason is the data: the expert held the arm so
precisely that the joints almost never moved away from their holding pose, so there was nothing to learn
the correction from. This is the classic behaviour-cloning problem: small errors take the robot to
states the training data never covered, and the errors grow from there.

**Fix: noise injection (3.2).** Pushing the robot off its path during collection, while recording the
expert's corrections, gave the network examples of how to recover. After this the arm held still during
navigation.

### 5.4 Then the arm wouldn't start reaching
The expert doesn't jump into the reach. It moves a hidden reference pose (`q_des`) towards the target a
little every step, and the torque pulls the arm after it. At the first moment of the reach that
reference still equals the current pose, so the expert's torque is still basically a holding torque.
The network can't see `q_des`, so it kept outputting a holding torque, the arm never moved, and the
situation never changed.

**Fix: DAgger (3.3).** Let the networks drive, and at every step record what the expert would have done
in that exact state. In the stuck state the expert's answer is "push towards the target", which was
exactly what the network had never seen. The new pairs are added to the training data and ArmNet is
retrained. After round 1 the arm started reaching (about 1.2 cm from the target on episode 9030). Round 2
added labels for states close to the target and improved the precision.

DAgger labels are only used for ArmNet. While BaseNet drives, the expert's base logic doesn't follow the
robot's actual path, so its wheel labels aren't reliable.

### 5.5 A bug worth mentioning
DAgger runs were cut at the first sign of the simulation blowing up (very high joint speeds). The first
threshold, 5 rad/s, was too strict: a harmless 10 ms wrist spike (about 8 rad/s) at the moment the reach
starts triggered it, so most of the reaching data was thrown away. Raising it to 15 rad/s and retraining
on the full data is the difference between `dagger2_cut` and `dagger2_full` below. That makes it a
natural ablation showing that labels near the target are what improved precision.

---

## 6. Results so far

98 test episodes (IDs 9000–9099 that the expert solved), both networks driving. "Success at X cm"
means the hand got within X cm of the target at some point. The expert's own controller uses 1 cm, but
that's too strict to summarise a learned policy, so we report three thresholds.

| Version (ArmNet) | @5 cm | @2 cm | @1 cm | median best error | median final error | blew up |
|---|---|---|---|---|---|---|
| `clean` | to do | | | | | |
| `noise` | to do | | | | | |
| `dagger1` | to do | | | | | |
| `dagger2_cut` | 95.9 % | 66.3 % | 12.2 % | 1.3 cm | 2.2 cm | 0 |
| `dagger2_full` | 94.9 % | 83.7 % | 10.2 % | 1.0 cm | 1.3 cm | 0 |
| expert | 100 % | 100 % | 100 % | < 1 cm | < 1 cm | 0 |

Median time to get within 5 cm is about 8.6 s, against about 9.9 s for a full expert episode.
Most episodes end just above 1 cm, which suggests a small constant offset rather than random misses.

To fill in the missing rows, copy a backed-up arm checkpoint over the active one and evaluate it under
its own tag, e.g.:

```bash
cp checkpoints/arm_net_noise.pt checkpoints/arm_net.pt
ROBOENV_HEADLESS=1 ROBOENV_MAX_STEPS=25000 python yousef/evaluate.py --parts both --tag noise --shard 0 --nshards 4
#   ... shards 1–3, then:
python yousef/evaluate.py --parts both --tag noise --summary_only
cp checkpoints/arm_net_dagger2_full.pt checkpoints/arm_net.pt      # put the best one back afterwards
```

---

## 7. Using the results for plots

Each evaluation writes `results/test_<parts>_<tag>.csv` with one row per test episode:

| Column | Meaning |
|---|---|
| `ep` | test episode ID |
| `best_err`, `final_err` | closest and final hand-to-target distance (m) |
| `success_1cm`, `success_2cm`, `success_5cm` | got within that distance at some point |
| `time_to_1cm`, `time_to_2cm`, `time_to_5cm` | when it first got there (s), empty if never |
| `duration`, `expert_duration` | length of this run and of the expert's run on the same episode (s) |
| `base_goal_err`, `base_bear_err` | how far from its parking spot (m) and heading (rad) the base ended |
| `blew_up` | the simulation became unstable |

A starting point:

```python
import pandas as pd, matplotlib.pyplot as plt
tags = ["dagger2_cut", "dagger2_full"]           # add clean, noise, dagger1 once they exist
df = {t: pd.read_csv(f"results/test_both_{t}.csv") for t in tags}

# success rate per version and threshold
rates = pd.DataFrame({t: [d[f"success_{c}cm"].mean() * 100 for c in (5, 2, 1)] for t, d in df.items()},
                     index=["5 cm", "2 cm", "1 cm"])
rates.plot.bar(ylabel="success (%)"); plt.tight_layout(); plt.show()

# distribution of the closest distance reached
for t, d in df.items():
    plt.hist(d["best_err"] * 100, bins=30, alpha=0.6, label=t)
plt.xlabel("best hand-to-target error (cm)"); plt.ylabel("episodes"); plt.legend(); plt.show()
```

Individual runs (full trajectories, both the network's and the expert's commands) are saved in
`data/test_both_<tag>/ep_XXXXX.npz` if you want time plots of one episode.

---

## 8. Files

| File | What it does |
|---|---|
| `collect_data.py` | expert demonstrations, clean or `--noise` |
| `features.py` | builds the 10 BaseNet and 25 ArmNet inputs from the raw signals (same code for training and running) |
| `dataset.py` | episode split, features, subsampling, normalisation, saving |
| `models.py`, `train.py` | the MLP and the training loop |
| `policy.py` | loads the active checkpoints, turns one step of sensor data into commands |
| `rollout.py` | simulation loop where the networks drive and the expert runs alongside for comparison |
| `run_demo.py` | one run with the GUI and a short summary |
| `dagger.py` | DAgger: collecting runs, and `add_dagger` for training |
| `evaluate.py` | runs the whole test set and writes `results/*.csv` |
| `metrics.py` | the metrics in section 7 |
| `01_features_dataset.ipynb`, `02_training.ipynb` | checks, dataset building, training |
| `notes/` | day-by-day logs with more detail on every step and every bug |

Everything outside `yousef/` is either the template (unchanged) or Mathilda's work.

### Data and checkpoints (local only)

| Path | Content |
|---|---|
| `data/raw/` | 1300 clean expert episodes |
| `data/raw_noise/` | 400 noise-injected expert episodes (IDs 5000–8099) |
| `data/dagger_r1/`, `data/dagger_r2/` | DAgger runs |
| `data/processed_step10*.npz` | built datasets |
| `data/test_both_<tag>/` | saved test runs per model version |
| `checkpoints/` | `base_net.pt` and `arm_net.pt` are the active ones; `*_clean`, `*_noise`, `*_dagger1`, `*_dagger2_cut` are earlier versions |

---

## 9. Things to keep in mind for the report

- Offline accuracy said almost nothing about closed-loop success. That gap is the main finding.
- Each fix targets one observed failure, with evidence: the stiffness test for noise injection, and the
  stuck reach for DAgger.
- Limitations: a remaining ~1 cm offset; a short wrist jerk when the reach starts; DAgger labels from an
  expert that isn't in control are an approximation (its hidden `q_des` evolves differently); only 2
  DAgger rounds; simulation only.
- The test set was never used for training or tuning. DAgger only ever used training configurations.
