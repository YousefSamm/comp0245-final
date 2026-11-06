#!/usr/bin/env bash
# Two offline fixes, trained and tested side by side (no new data collection):
#   A  daggernear: front-2 features, expert data + only the DAgger rows with a box within 1.5 m
#   B  corr      : 6 corridor features (what the expert steers from), expert data only
# Run from robotnik_arm_task/ in the RoboEnv environment:   bash yousef/obstacles/run_fixes.sh
set -e
python yousef/obstacles/dataset_obs.py --dagger_near                       > log_fix_dataset_A.txt 2>&1
python yousef/obstacles/dataset_obs.py --lidar corr                        > log_fix_dataset_B.txt 2>&1
python yousef/obstacles/train_obs.py --version daggernear --dagger_near    > log_fix_train_A.txt 2>&1
python yousef/obstacles/train_obs.py --version corr --lidar corr           > log_fix_train_B.txt 2>&1
for s in 0 1 2 3; do
  ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_daggernear --base base_net_obs_daggernear \
      --shard $s --nshards 4 > log_efa$s.txt 2>&1 &
  ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_corr --base base_net_obs_corr \
      --shard $s --nshards 4 > log_efb$s.txt 2>&1 &
done
wait
python yousef/obstacles/evaluate_obs.py --tag obs_daggernear --summary_only > log_fix_summary_A.txt 2>&1
python yousef/obstacles/evaluate_obs.py --tag obs_corr --summary_only       > log_fix_summary_B.txt 2>&1
echo "done: results/obs_obs_daggernear.csv, results/obs_obs_corr.csv"
