# COMP0245 – Session log: Mon 2 Nov 2026 (13:40) → Tue 3 Nov 2026 (01:35)

**Phase:** from collected data → features → training dataset → first trained network (BaseNet).
**Deadline:** Fri 6 Nov 2026 (report + code). Lab demo window: 30 Oct – 6 Nov (required, not marked).

---

## 0. Starting point (end of previous session)

- `robotnik_arm_task/data/raw/` contains **1,300 expert episodes** (`.npz`, one file per episode, 1 kHz, 19 keys each).
  - Training pool: IDs `0–299` (seed 1), `1000–1299` (seed 2), `2000–2299` (seed 3), `3000–3299` (seed 4)
  - Test set: IDs `9000–9099` (seed 900). **Rule: IDs ≥ 9000 are never used for training.**
- **1,283 / 1,300 successful (98.7 %)** → the expert's upper bound.
- Code lives in `robotnik_arm_task/yousef/` (`collect_data.py`, `inspect_data.py`, `README.md`) and is pushed to GitHub; data is git-ignored.

---

## 1. Design decisions taken today

### 1.1 Two networks: BaseNet + ArmNet
- Proposed by Yousef; agreed. Justified by the structure of the expert itself: `base_controller` → wheels, `arm_controller` → torques, almost independent.
- Benefits: smaller focused inputs; no loss balancing between rad/s and N·m; easier diagnosis (base vs arm failure); BaseNet can predict **(v, ω)** instead of 4 wheel speeds.
- **Plan for the report:** also train **one joint MLP** (all inputs → all 11 outputs) as a baseline, so "two networks vs one" is an experiment with evidence.

### 1.2 What the 4 phases mean (`phase` field)
Defined in `collect_data.py`: `phase_id = 3 if base_state.done else {"align":0, "drive":1, "orient":2}[base_state.phase]`

| Phase | Name | Robot | Base | Arm |
|---|---|---|---|---|
| 0 | align | turns on the spot to face the base goal | rotating | holds start pose |
| 1 | drive | drives to the base goal | moving | holds |
| 2 | orient | at goal, turns on the spot to face the target | rotating | holds |
| 3 | done | base stopped, arm reaches | wheel cmds = 0 | reaching |

- **Phases 0, 1, 2 = navigation; only phase 3 = manipulation.** (Phase 2 is still the base turning.)
- Phase 3 also starts if the 25 s navigation timeout expires.
- Arm torques in phases 0–2 are **not zero**: the expert applies PD + gravity compensation to hold the start configuration.

### 1.3 `phase` is NOT an input
- It is the expert's internal state; the model will not have it at run time.
- Used only for: analysis (errors per phase, data balance), and possibly as a label for an auxiliary output later.

### 1.4 Split by columns (features), not by rows (timesteps)
- **Every row (all phases) goes to both networks; each network gets only its own columns.**
- Why not drop rows by phase:
  - BaseNet without phase-3 rows never learns to **stop** at the goal.
  - ArmNet without phase-0–2 rows never learns to **hold** while the base moves (it would reach too early).
  - At run time both networks run every step; no phase variable tells them whose turn it is.

### 1.5 How the arm knows when to start reaching → "let it switch by itself" (Option A)
Options discussed:
- **A (chosen):** ArmNet gets 3 observable "base done" signals (distance to base goal, goal-bearing error sin/cos, base speed) and learns the switching.
- **B:** explicit switch flag as one input (training: `phase == 3`; run time: a latched rule or BaseNet output ≈ 0). Simpler but hand-coded; train/run-time timing mismatch risk.
- **Gated / mixture-of-experts:** a third classifier network predicts "base done". Possible "novel idea" extra.
- Note: the 3 extra features are computationally negligible; the real difference is learned vs hand-coded switching.
- Possible report experiment: A vs B (closed-loop success, arm error).

### 1.6 Notebook vs `.py`
- Hybrid: **functions in `.py` modules** (used by both training and the run-time policy, so features are computed identically), **notebook for calling / inspecting / training / plotting**.
- Notebook uses `%load_ext autoreload` + `%autoreload 2` so edits to `.py` files are picked up.
- Before committing a notebook: **Restart & Run All**.

Planned structure:
```
yousef/
├── features.py     ✅ feature engineering (shared by training + policy)
├── dataset.py      ✅ split, subsample, stack, normalise, save
├── models.py       ✅ MLP class
├── train.py        ✅ DataLoaders + training loop with early stopping
├── features.ipynb  ✅ notebook used today
├── policy.py       ⏳ loads trained nets → callable(obs) → (wheel_cmds, torque_arm)
└── evaluate.py     ⏳ closed-loop runs on the test set (headless) → metrics
```

### 1.7 Why failed episodes are excluded from training
- BC copies what is in the data; failed episodes contain timeouts (arm reaching from the wrong place) and stalls against unreachable targets, often with long "stuck" tails.
- Counter-argument: same expert controller; actions aren't wrong per state.
- Only 17/1,300 (1.3 %) → small effect either way. **Decision: exclude from training, keep for analysis, state the choice in the report.**

---

## 2. Inspecting `.npz` files (notebook snippets)

- List keys/shapes: `for k in d.files: print(k, d[k].shape, d[k].dtype)`
- Time plot of wheel commands, arm torques and EE error with the phases shaded (`fill_between` with `where=(phase==p)`).
- Top-down base path: `plt.plot(d["base_pos"][:,0], d["base_pos"][:,1])` + start + target.
- Phase balance across episodes (concatenate `phase`, percentage per value).
- Lesson: when `glob` finds nothing it returns an empty list silently → always `print(len(files))`.

---

## 3. `yousef/features.py`

### 3.1 Constants (copied from the template, with source comments)
- Copied rather than imported, because importing the template files also loads PyBullet/pinocchio.
```python
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]      # robotnik_arm_task.py
STANDOFF_POINTS   = [0.80, 0.65, 0.65, 0.35]
WHEEL_RADIUS = 0.11                             # BaseControllerParams
WHEEL_BASE_WIDTH = 0.66
CONT_JOINTS = [0, 2, 4, 6]   # q_arm columns of continuous joints 1,3,5,7
LIM_JOINTS  = [1, 3, 5]      # limited joints 2,4,6
```

### 3.2 Functions (all vectorised over N rows)
| Function | What it does |
|---|---|
| `wrap(a)` | angle → [−π, π] via `arctan2(sin a, cos a)` |
| `base_goal(target)` | expert's standoff formula → `goal_xy`, `goal_bearing` (bearing from the **world origin** to the target; Option 1 from earlier) |
| `world_to_base(vec_xy, yaw)` | rotate world vectors into the robot frame: fwd = cosψ·x + sinψ·y, left = −sinψ·x + cosψ·y |
| `base_features(d)` | BaseNet input matrix (N, 10) + names |
| `arm_features(d)` | ArmNet input matrix (N, 25) + names |
| `wheels_to_vw(w)` | `[rw, lw, lw, rw]` → (v, ω): v = r(rw+lw)/2, ω = r(rw−lw)/W |
| `vw_to_wheels(vw)` | inverse, for the policy at run time |

### 3.3 BaseNet inputs (10)
| # | Name | Meaning |
|---|---|---|
| 0–1 | `goal_fwd`, `goal_left` | vector robot → base goal, robot frame |
| 2 | `goal_dist` | distance to base goal |
| 3–4 | `sin_head`, `cos_head` | heading error towards the goal ("am I facing the goal?") |
| 5–6 | `sin_bear`, `cos_bear` | error to the final goal bearing ("am I facing the final direction?") |
| 7–8 | `v_fwd`, `v_left` | base velocity, robot frame |
| 9 | `yaw_rate` | base angular velocity |

Why sin/cos: −3.13 and +3.13 rad are nearly the same heading but far apart as raw numbers.

### 3.4 ArmNet inputs (25)
| Count | Features |
|---|---|
| 4 + 4 | sin and cos of joints 1, 3, 5, 7 (continuous) |
| 3 | raw joints 2, 4, 6 (limited) |
| 7 | joint velocities `qd1–qd7` |
| 3 | EE → target error: `ee_err_fwd`, `ee_err_left` (robot frame), `ee_err_z` |
| 1 + 2 + 1 | "base done" signals: `goal_dist`, `sin_bear`, `cos_bear`, `base_speed` (norm of `v_fwd`, `v_left`) |

### 3.5 Labels
- BaseNet: `wheels_to_vw(wheel_cmds)` → (v, ω), shape (N, 2)
- ArmNet: `torque_arm`, shape (N, 7)
- Never used as inputs: `phase`, `q_des_arm` (expert internals), `t`.

### 3.6 Bugs fixed while writing it
| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'base_pose'` | typo | key is `base_pos` |
| `ValueError: truth value of an array…` | `d[Xb[:, 7:9]]` (array used as dict key) | `Xb[:, 7:9]` |
| `IndentationError` risk | `X = …` indented 1 space, rest 4 | same 4-space indent |
| `ValueError: … 2 dimension(s) … 3 dimension(s)` | `np.linalg.norm([Xb[:, 7:9]], …)` (extra list brackets) | remove `[ ]` |

Debug tip: print the shape of every array before `np.column_stack`.
Note: `np.load(path, 'r')` passes `mmap_mode='r'` (no effect for `.npz`); `np.load(path)` is enough.

### 3.7 Checkpoints (all passed)
| # | Test | Result |
|---|---|---|
| 1 | `world_to_base([[0,1]], [π/2])` | `[[1.0, 6.1e-17]]` → (1, 0), 6e-17 = float rounding |
| 2 | max goal distance in phase 3 (ep_00000) | **0.0706 m** (< 0.08 expert `pos_tol`) |
| 3 | wheel round trip `vw_to_wheels(wheels_to_vw(w)) − w` | **2.4e-7** (float32 precision) |
| 4 | shapes / NaNs | `(14061, 10) (14061, 25) 14061 False False` |

### 3.8 Finding: "base done" is not a simple distance threshold
- Over the first 100 episodes the worst goal distance during phase 3 was **0.161 m** (> 0.08).
- 32 of the 100 episodes exceeded 0.08 m; **all had `nav_time` < 25 s → no timeouts.**
- Two causes, both from the expert's own logic:
  1. **Overshoot rule** in `base_controller` (drive phase): `passed_goal = dist_to_goal < 0.15 and cos(err) < 0` → it may finish driving up to 0.15 m from the goal (cases 0.10–0.16 m: ep 15, 25, 31, 33, 54, 56, …).
  2. **Drift**: turning on the spot in the orient phase (skid-steer slip) + reaction forces of the reaching arm add a few mm–cm (cases 0.080–0.095 m).
- **Report point:** in ≈ 1/3 of episodes the arm starts reaching with the base 8–16 cm from its nominal goal → justifies several "done" signals (distance + bearing + speed) instead of a single distance rule.

### 3.9 Moved from notebook to file
Functions moved into `yousef/features.py`; the notebook imports them:
```python
%load_ext autoreload
%autoreload 2
import sys; sys.path.insert(0, "/home/yousef/ucl_ws/comp0245-final/robotnik_arm_task/yousef")
from features import *
```
Checkpoint 4 re-run with the imported version: same output.

---

## 4. `yousef/dataset.py`

### 4.1 Two kinds of split (clarified)
| Split | Divides | Where |
|---|---|---|
| By episode (train/val/test) | which **episodes** go where | `split_episodes()` |
| By network (base/arm) | which **columns** each net gets | `base_features()` / `arm_features()` inside `build_split()` |

Both networks use the same episodes and timesteps, different columns. Same split for both keeps the test set consistent when they run together in the simulator.

### 4.2 `split_episodes(val_frac=0.15, seed=0)`
- Paths: `TASK_DIR = dirname(dirname(abspath(__file__)))` → `robotnik_arm_task/`; `RAW_DIR = TASK_DIR/data/raw`.
- Keeps only `success == True`; IDs ≥ 9000 → test; others → pool.
- Pool shuffled **by episode** with `np.random.default_rng(0)`, then 15 % val / 85 % train.
- **Bug:** `TASK_DIR` had only one `dirname` → looked in `yousef/data/raw` → 0 files → `0 0 0`. Fixed with two `dirname`s. Recommended: `assert len(files) > 0, …` after the glob.
- **Result: train 1008 / val 177 / test 98 = 1283** ✅; no overlap (`set() set()`). Test = 98 because 2 test episodes failed.

### 4.3 `build_split(file_list, step=10)`
For each episode: `base_features`, `arm_features`, `wheels_to_vw(wheel_cmds)`, `torque_arm`; keep every 10th row (`slice(None, None, 10)` → **1 kHz → 100 Hz**); also stores `phase` and `ep_id` per row (analysis only); concatenates all episodes.

### 4.4 `norm_stats(X)`
Per-column mean and std **from the training split only**, std + `1e-6` (avoid division by zero), float32.

### 4.5 `build_dataset(step=10)`
Builds train/val/test, computes stats, saves everything once to
`robotnik_arm_task/data/processed_step10.npz` (git-ignored).

### 4.6 Bugs fixed
| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'torque_arms'` | typo | `torque_arm` |
| `v.shaep` in notebook | typo | `v.shape` |
| `ValueError: need at least one array to concatenate` | extra `"t": []` key never appended | removed it (or append `d["t"][s]`) |
| (silent) only 1 episode per split | `return` indented inside the `for` loop | moved back to the `for` level |
| (risk) division by zero | no epsilon in `norm_stats` | `+ 1e-6` |

### 4.7 Key names (renamed by Claude at Yousef's request)
- Data: `Xb_train, Yb_train, Xa_train, Ya_train, phase_train, ep_id_train` (+ `_val`, `_test`)
- Stats: `Xb_mu, Xb_sd, Yb_mu, Yb_sd, Xa_mu, Xa_sd, Ya_mu, Ya_sd`

### 4.8 Final dataset (no NaNs)
| Split | Rows | Xb | Yb | Xa | Ya |
|---|---|---|---|---|---|
| train | 1,026,259 | (·, 10) | (·, 2) | (·, 25) | (·, 7) |
| val | 175,714 | (·, 10) | (·, 2) | (·, 25) | (·, 7) |
| test | 99,966 | (·, 10) | (·, 2) | (·, 25) | (·, 7) |

Reload any time (seconds):
```python
D = dict(np.load("/home/yousef/ucl_ws/comp0245-final/robotnik_arm_task/data/processed_step10.npz"))
```

---

## 5. Environment: installing PyTorch with CUDA

- The notebook kernel uses `~/ucl_ws/LAB_SESSION_COMP_0245_2026_PUBLIC/.venv` (Python 3.10), which had **no torch**.
- Reusing another install is fragile (venv isolation, Python/NumPy version mismatch); torch is needed in this venv anyway because the policy runs the network inside PyBullet.
- The venv was made with **uv** → `No module named pip`. Install with:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```
  (fallback: `python -m ensurepip --upgrade` then `python -m pip install …`)
- CUDA notes: PyTorch wheels bundle their own CUDA runtime; `nvcc` (12.8 here) doesn't matter, the **driver** must support ≥ 12.8 (`nvidia-smi`, top-right).
- GPU speed-up for these small MLPs is limited by DataLoader overhead; bigger batches (4096–8192, lr ≈ 2e-3) or putting the whole dataset on the GPU help.
- After installing: restart the kernel → all variables lost → reload `D` (`NameError: name 'D' is not defined`). Keep the load cell near the top of the notebook.
- Device: `device = "cuda" if torch.cuda.is_available() else "cpu"`, and pass `device=device` to `train_model`.

---

## 6. `yousef/models.py`

```python
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_layers=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))      # no activation (regression)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```
- BaseNet = `MLP(10, 2)`, ArmNet = `MLP(25, 7)`.
- **Bug:** `NotImplementedError: … missing the required "forward" function` — `forward` was indented inside `__init__`, and the output layer + `self.net` were inside the `for` loop. Fixed indentation.
- Verified: `10 → 256 → 256 → 256 → 2` (3 × Linear+ReLU, final Linear), `m(torch.zeros(5,10)).shape == [5, 2]`.

---

## 7. `yousef/train.py`

- `make_loaders(D, net="b"|"a", batch_size=1024)`: normalises X and Y with the training stats, wraps them in `TensorDataset`; train loader `shuffle=True`, val loader `shuffle=False`, batch × 4.
- `train_model(model, train_dl, val_dl, epochs=30, lr=1e-3, patience=5, device)`:
  - Adam, MSE on normalised outputs
  - `model.train()` → `zero_grad` → forward → `backward` → `step`; `model.eval()` + `torch.no_grad()` for validation
  - **Early stopping**: keeps the best validation state (`state_dict` cloned), stops after 5 epochs without improvement, restores the best weights.

---

## 8. Result: BaseNet trained

| Epoch | Train | Val |
|---|---|---|
| 1 | 0.0176 | 0.0059 |
| 2 | 0.0040 | 0.0037 |
| 3 | 0.0028 | 0.0029 |
| 5 | 0.0021 | 0.0024 |
| 7 | 0.0018 | 0.0021 |
| 11 | 0.0014 | 0.0020 |
| 15 | 0.0012 | 0.0018 |
| 16 | 0.0012 | 0.0017 |
| **20** | 0.0011 | **0.0016 (best)** |
| 22 | 0.0010 | 0.0027 |
| 25 | 0.0010 | 0.0021 → early stop |

Interpretation:
- Losses are on normalised outputs (1.0 = no better than predicting the mean) → **val 0.0016 ≈ R² 0.998**.
- Small train/val gap (0.0010 vs 0.0016) → mild overfitting.
- Val loss fluctuates (fixed LR) → optional LR scheduler (e.g. ReduceLROnPlateau).
- Best epoch (20) restored by early stopping.
- **Caution:** this is open-loop accuracy on the expert's own trajectories. Closed-loop performance (the network driving the robot) must be tested separately.

---

## 9. Next steps (in order)

1. **Save BaseNet** with its normalisation stats (not yet confirmed done):
   ```python
   CKPT = "/home/yousef/ucl_ws/comp0245-final/robotnik_arm_task/checkpoints"
   os.makedirs(CKPT, exist_ok=True)
   torch.save({"state_dict": base_net.state_dict(), "in_dim": 10, "out_dim": 2,
               "X_mu": D["Xb_mu"], "X_sd": D["Xb_sd"], "Y_mu": D["Yb_mu"], "Y_sd": D["Yb_sd"]},
              f"{CKPT}/base_net.pt")
   ```
2. **Train ArmNet**: `make_loaders(D, "a")`, `MLP(25, 7)`; save as `arm_net.pt` with `Xa`/`Ya` stats. Expect a higher loss than BaseNet.
3. **Errors in physical units, per phase** (`eval_physical`): RMSE in m/s, rad/s and N·m, overall and per phase (expect the largest errors in orient and at phase switches).
4. **`policy.py`**: load both nets + stats; at each step compute features from the live state → normalise → predict → un-normalise → `vw_to_wheels`, clip torques to ±30 N·m.
5. **`evaluate.py`**: run the policy in PyBullet on the test configurations (IDs 9000+), headless; success rate, final error, time — vs the expert's 98.7 %. **This is the lab demo.**
6. Experiments (if time): two nets vs one joint MLP; Option A vs B; feature ablations; history/DAgger; robustness (noise, delay); efficiency (inference time, parameters).

### Remaining timeline
| Day | Goal |
|---|---|
| Tue 3 Nov | ArmNet, `policy.py`, `evaluate.py`, closed loop working (lab demo) |
| Wed 4 Nov | experiments |
| Thu 5 Nov | robustness + efficiency, report |
| Fri 6 Nov | final pass, README, submit |

---

## 10. For the README / AI-use declaration
- Written by Claude at Yousef's request (earlier sessions): the resume + error-handling fixes in `collect_data.py`, the path changes when moving files into `yousef/`, the `yousef/README.md`.
- Today: Claude renamed the saved array keys in `dataset.py` (`Xb_train`, `Xb_sd`, …). All other code today was written by Yousef from Claude's hints/skeletons.
- Check COMP0245's policy on AI assistance and acknowledge it as required.
