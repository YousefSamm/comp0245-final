#!/usr/bin/env bash
# BaseNet_obs with 8 sector proximities from the full 360 deg scan (18 inputs) instead of the
# 2 front-180 deg features: dataset -> training -> closed-loop test on the 95 obstacle scenes.
# Uses the data already in data/raw_obs/ (lidar360 was saved in every episode); no new collection.
# Run from robotnik_arm_task/ in the RoboEnv environment:   bash yousef/obstacles/run_sect8.sh
set -e
python yousef/obstacles/dataset_obs.py --lidar sect8                        > log_sect8_dataset.txt 2>&1
python yousef/obstacles/train_obs.py --version sect8 --lidar sect8         > log_sect8_train.txt 2>&1
for s in 0 1 2 3; do
  ROBOENV_HEADLESS=1 python yousef/obstacles/evaluate_obs.py --tag obs_sect8 --base base_net_obs_sect8 \
      --shard $s --nshards 4 > log_es$s.txt 2>&1 &
done
wait
python yousef/obstacles/evaluate_obs.py --tag obs_sect8 --summary_only     > log_sect8_summary.txt 2>&1
echo "done: results/obs_obs_sect8.csv, summary in log_sect8_summary.txt"
