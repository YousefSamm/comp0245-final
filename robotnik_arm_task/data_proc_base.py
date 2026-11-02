import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

BASE_FEATS = [
    "base_pos",
    "base_bearing",
    "base_vel",
    "target",
    "wheel_cmds"
]

STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]

BASE_COLS = [
    "ep_id", #0
    "t", #1
    "base_pos_x", #2
    "base_pos_y", #3
    "base_bearing", #4
    "base_vel_x", #5
    "base_vel_y", #6    
    "base_target_x", #7
    "base_target_y", #8
    "wheel_cmd_1", #9
    "wheel_cmd_2", #10
    "wheel_cmd_3", #11
    "wheel_cmd_4" #12
]



def combine_data():

    all_ids = ["00", "01", "02", "03", "09"]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(cur_dir, "data", "raw")
    

    blocks = []
    n_skipped_failed = 0

    for fname in sorted(os.listdir(raw_data_path)):
        # must be a finished episode file, not a leftover temp file
        if not fname.endswith(".npz") or fname.endswith(".tmp.npz"):
            continue
        # ep_00xxx, ep_01xxx, ep_02xxx, ep_03xxx -> training pool only
     
        if not any(fname.startswith(f"ep_{tid}") for tid in all_ids):
            continue

        path = os.path.join(raw_data_path, fname)

        data = np.load(path)
        if not bool(data["success"]):              # exclude failed expert episodes
            n_skipped_failed += 1
            continue

        #Add ep id
        ep_id = int(fname[3:8])          # "ep_00012.npz" -> 12
        n_obs = data['t'].shape[0]

        ep_inputs = np.column_stack([
            np.full(n_obs, ep_id),       # column 0: episode id
            data["t"],                   # column 1: time
        ])

        for feat in BASE_FEATS:
            d = data[feat]
            if len(d.shape) == 1:
                d = d.reshape(-1, 1)

            if feat == "base_vel":
                d = d[:, :2]
                

            if feat == "target": #Standoff logic
                base_init_xy = data["base_init"][:2]
                ee_target = data["target"]
                ee_target_xy = ee_target[:2]
                standoff = float(np.interp(ee_target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
                
                dist_total = np.linalg.norm(ee_target_xy - base_init_xy)
                dist_goal = (dist_total - standoff) / dist_total

                base_target_xy = base_init_xy + dist_goal * (ee_target_xy - base_init_xy)
                
                d = np.ones((n_obs, 2), dtype=np.float32) * base_target_xy

            assert ep_inputs.shape[0] == d.shape[0]
            ep_inputs = np.concatenate((ep_inputs, d), axis=1)
        

        blocks.append(ep_inputs)
    
    arr = np.vstack(blocks)


    print(f"Built array {arr.shape} from {len(blocks)} episodes "
        f"({n_skipped_failed} failed excluded)")
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    np.savez(os.path.join(cur_dir, "data", "proc", f"base_combined_{stamp}"), arr=arr, cols=np.array(BASE_COLS))
    
    return arr, stamp



def proc_data(arr, stamp=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ep_id = arr[:, 0]

    # Calculate base error
    base_pos_error = arr[:, 7:9] - arr[:, 2:4] # target - base_pos

    # Take sin and cos of base bearing
    base_bearing = arr[:, 4]
    base_bearing_sin = np.sin(base_bearing)
    base_bearing_cos = np.cos(base_bearing)

    # Normalise data
    nomalized_pos_error_x = (base_pos_error[:, 0] - base_pos_error.mean(axis=0)[0]) / np.sqrt(base_pos_error.var(axis=0)[0])
    nomalized_pos_error_y = (base_pos_error[:, 1] - base_pos_error.mean(axis=0)[1]) / np.sqrt(base_pos_error.var(axis=0)[1])

    normalized_vel_x = (arr[:, 5] - arr[:, 5].mean()) / np.sqrt(arr[:, 5].var())
    normalized_vel_y = (arr[:, 6] - arr[:, 6].mean()) / np.sqrt(arr[:, 6].var())

    # construct array
    proc_arr = np.column_stack([
        ep_id,
        nomalized_pos_error_x,
        nomalized_pos_error_y,
        base_bearing_sin,
        base_bearing_cos,
        normalized_vel_x,
        normalized_vel_y,
        arr[:, 9:]
    ])

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Filter based on train vs test

    train_arr = proc_arr[proc_arr[:, 0] < 9000]
    test_arr = proc_arr[proc_arr[:, 0] >= 9000]

    np.savez(os.path.join(cur_dir, "data", "proc", f"base_train_proc_{stamp}"), arr=train_arr, cols=np.array(["ep_id", "pos_error_x", "pos_error_y", "bearing_sin", "bearing_cos", "vel_x", "vel_y", "wheel_cmd_1", "wheel_cmd_2", "wheel_cmd_3", "wheel_cmd_4"]))
    np.savez(os.path.join(cur_dir, "data", "proc", f"base_test_proc_{stamp}"), arr=test_arr, cols=np.array(["ep_id", "pos_error_x", "pos_error_y", "bearing_sin", "bearing_cos", "vel_x", "vel_y", "wheel_cmd_1", "wheel_cmd_2", "wheel_cmd_3", "wheel_cmd_4"]))
    
    return f"Saved processed train/test arrays with stamp {stamp} to data/proc"


def main():
    arr, stamp = combine_data()
    proc_data(arr, stamp=stamp)



if __name__ == "__main__":

    main()