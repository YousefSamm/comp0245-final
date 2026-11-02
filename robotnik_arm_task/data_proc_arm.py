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

ARM_FEATS = [
    "ee_pos",
    "q_arm",
    "qd_arm",
    "target",
    "torque_arm"
]


ARM_COLS = [
    "ep_id", #0
    "t", #1
    "ee_pos_x", #2
    "ee_pos_y", #3
    "ee_pos_z", #4
    "j1_pos", #5
    "j2_pos", #6
    "j3_pos", #7
    "j4_pos", #8
    "j5_pos", #9
    "j6_pos", #10
    "j7_pos", #11
    "j1_vel", #12
    "j2_vel", #13
    "j3_vel", #14
    "j4_vel", #15
    "j5_vel", #16
    "j6_vel", #17
    "j7_vel", #18       
    "target_x", #19
    "target_y", #20
    "target_z", #21
    "torque_1", #22
    "torque_2", #23
    "torque_3", #24
    "torque_4", #25
    "torque_5", #26
    "torque_6", #27
    "torque_7" #28
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

        for feat in ARM_FEATS:
            d = data[feat]
            
            if feat == "target":
                d = np.tile(d, (n_obs, 1))

            if len(d.shape) == 1:
                d = d.reshape(-1, 1)
                
            assert ep_inputs.shape[0] == d.shape[0]
            ep_inputs = np.concatenate((ep_inputs, d), axis=1)
        

        blocks.append(ep_inputs)
    
    arr = np.vstack(blocks)


    print(f"Built array {arr.shape} from {len(blocks)} episodes "
        f"({n_skipped_failed} failed excluded)")
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    np.savez(os.path.join(cur_dir, "data", "proc", f"arm_combined_{stamp}"), arr=arr, cols=np.array(ARM_COLS))

    
    return arr, stamp



def proc_data(arr, stamp=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ep_id = arr[:, 0]

    # Calculate ee error
    ee_pos_error =  arr[:, 19:22] - arr[:, 2:5] # target - ee_pos

    # Take sin and cos of all joint angles and velocities
    j_sin = np.sin(arr[:, 5:12])
    j_cos = np.cos(arr[:, 5:12])

    # Normalise data
    normalized_pos_error_x = (ee_pos_error[:, 0] - ee_pos_error.mean(axis=0)[0]) / np.sqrt(ee_pos_error.var(axis=0)[0])
    normalized_pos_error_y = (ee_pos_error[:, 1] - ee_pos_error.mean(axis=0)[1]) / np.sqrt(ee_pos_error.var(axis=0)[1])
    normalized_pos_error_z = (ee_pos_error[:, 2] - ee_pos_error.mean(axis=0)[2]) / np.sqrt(ee_pos_error.var(axis=0)[2])

    # Normalise joint velocities
    normalized_j_vel = (arr[:, 12:19] - arr[:, 12:19].mean(axis=0)) / np.sqrt(arr[:, 12:19].var(axis=0))


    # construct array
    proc_arr = np.column_stack([
        ep_id,
        normalized_pos_error_x,
        normalized_pos_error_y,
        normalized_pos_error_z,
        j_sin,
        j_cos,
        normalized_j_vel,
        arr[:, 22:] # torques
    ])

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Filter based on train vs test

    train_arr = proc_arr[proc_arr[:, 0] < 9000]
    test_arr = proc_arr[proc_arr[:, 0] >= 9000]

    np.savez(os.path.join(cur_dir, "data", "proc", f"arm_train_proc_{stamp}"), arr=train_arr, cols=np.array(["ep_id", "pos_error_x", "pos_error_y", "pos_error_z", "j1_sin", "j2_sin", "j3_sin", "j4_sin", "j5_sin", "j6_sin", "j7_sin", "j1_cos", "j2_cos", "j3_cos", "j4_cos", "j5_cos", "j6_cos", "j7_cos", "j1_vel", "j2_vel", "j3_vel", "j4_vel", "j5_vel", "j6_vel", "j7_vel", "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", "tau_7"]))
    np.savez(os.path.join(cur_dir, "data", "proc", f"arm_test_proc_{stamp}"), arr=test_arr, cols=np.array(["ep_id", "pos_error_x", "pos_error_y", "pos_error_z", "j1_sin", "j2_sin", "j3_sin", "j4_sin", "j5_sin", "j6_sin", "j7_sin", "j1_cos", "j2_cos", "j3_cos", "j4_cos", "j5_cos", "j6_cos", "j7_cos", "j1_vel", "j2_vel", "j3_vel", "j4_vel", "j5_vel", "j6_vel", "j7_vel", "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", "tau_7"]))
    
    return f"Saved processed train/test arrays with stamp {stamp} to data/proc"


def main():
    arr, stamp = combine_data()
    proc_data(arr, stamp=stamp)



if __name__ == "__main__":

    main()