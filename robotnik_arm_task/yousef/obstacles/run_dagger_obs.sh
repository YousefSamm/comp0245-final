#!/usr/bin/env bash
# One DAgger round for BaseNet_obs, then retrain and test:
#   base_net_obs_multi drives 160 training scenes (4 in parallel) -> data/dagger_obs_r1/
#   dataset with the DAgger rows -> train base_net_obs_dagger1.pt -> closed-loop test -> results/obs_obs_dagger1.csv
# Run from robotnik_arm_task/ in the RoboEnv environment:   bash yousef/obstacles/run_dagger_obs.sh
set -e
for s in 0 1 2 3; do
  ROBOENV_HEADLESS=1 python yousef/obstacles/dagger_obs.py --round 1 --base base_net_obs_multi \
      --shard $s --nshards 4 > log_dgo$s.txt 2>&1 &
done
wait
python yousef/obstacles/dataset_obs.py --dagger                             > log_dgo_dataset.txt 2>&1
python yousef/obstacles/train_obs.py --version dagger1 --dagger            > log_dgo_train.txt 2>&1
for s in 0 1 2 3; do
  ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_dagger1 --base base_net_obs_dagger1 \
      --shard $s --nshards 4 > log_edg$s.txt 2>&1 &
done
wait
python yousef/obstacles/evaluate_obs.py --tag obs_dagger1 --summary_only   > log_dgo_summary.txt 2>&1
echo "done: results/obs_obs_dagger1.csv, summary in log_dgo_summary.txt"
