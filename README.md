# COMP0245 Final Assessment, Group 12: Behaviour Cloning of a Mobile Manipulator

Mathilda Weston, Yousef Sammour, Samadrita Basu

We replace the template's hand-designed controller (a Robotnik skid-steer base that drives to a target,
then a Kinova JACO2 arm that reaches it) with two MLPs trained by behaviour cloning:

- **BaseNet**: 10 inputs → forward speed and turn rate (v, ω) → the 4 wheel velocity commands
- **ArmNet**: 25 inputs → the 7 arm joint torques

Clean demonstrations alone fail in closed loop. Noise injection (DART), four rounds of DAgger with a
shadow expert, and a gravity residual (ArmNet predicts τ − g(q)) bring the hand within 2 cm of unseen
targets in **89.9 % of 494 test episodes**. As an extension, a BaseNet given the driving-corridor
quantities the expert steers from, measured with a LiDAR, avoids static boxes as well as the expert
(4.2 % collisions against 29.5 % without LiDAR).

## Where things are

| Path | What it is |
|---|---|
| `robotnik_arm_task/robotnik_arm_task.py`, `robotnik_arm_control.py` | the template and its expert controller (not modified) |
| `robotnik_arm_task/yousef/` | the full pipeline: data collection, features, training, DAgger, evaluation. **[README](robotnik_arm_task/yousef/README.md)** |
| `robotnik_arm_task/yousef/obstacles/` | the obstacle-avoidance extension. **[README](robotnik_arm_task/yousef/obstacles/README.md)** |
| `robotnik_arm_task/results/` | one CSV per evaluated model, one row per test episode (every number in the report comes from these) |
| `robotnik_arm_task/results/figures/` | the result figures in the report and the script that makes them |
| `robotnik_arm_task/data_proc_*.py`, `robotnik_arm_task/notebooks/` | Mathilda's data processing and training notebooks |

`data/` and `checkpoints/` are not on GitHub (several GB). All expert data is regenerated from fixed
seeds with the commands in the READMEs above.

## Quick start

Use the RoboEnv environment from the labs plus PyTorch, copy `models/` and `configs/` from RoboEnv into
`robotnik_arm_task/`, and run everything from `robotnik_arm_task/`.

```bash
python robotnik_arm_task.py                                                    # the original expert
python yousef/evaluate.py --parts both --tag dagger4_gravres --summary_only    # re-print a result from its saved runs
python results/figures/plot_report_figures.py                                  # remake the report figures from results/*.csv
```

Running the trained networks needs the checkpoints (`checkpoints/base_net.pt`, `arm_net.pt`), which we
can share on request.

## Main results

Closed loop, both networks driving, on test episodes the expert solved (IDs ≥ 9000, never used for
training or tuning). Success at X cm means the hand got within X cm of the target.

| ArmNet version | Test episodes | @1 cm | @2 cm | @5 cm |
|---|---|---|---|---|
| clean demonstrations only | 98 | 0.0 % | 0.0 % | 2.0 % |
| + noise injection | 98 | 0.0 % | 0.0 % | 6.1 % |
| + DAgger round 1 | 98 | 13.3 % | 84.7 % | 96.9 % |
| DAgger round 3 | 494 | 14.4 % | 87.7 % | 97.6 % |
| DAgger round 4, no residual | 494 | 14.4 % | 85.4 % | 95.1 % |
| **DAgger round 4 + gravity residual (final)** | 494 | 14.6 % | **89.9 %** | 96.4 % |

All rounds, ablations and the McNemar tests are in `yousef/README.md` and the report.

## Branches

Everything is on `main`. The obstacle extension is the `yousef/obstacles/` folder.
