import numpy as np, glob, os
from features import base_features, arm_features, wheels_to_vw



TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(TASK_DIR, "data", "raw")

def split_episodes(val_frac = 0.15, seed = 0):
    """
    Returns three list of file paths: train, val, test.
    Only successful episodes. ID >= 9000 = test set.
    """
    files = sorted (glob.glob(os.path.join(RAW_DIR, "ep_*.npz")))
    pool, test = [], []

    for f in files:
        ep_id = int(os.path.basename(f)[3:8])
        if not bool(np.load(f)["success"]):
            continue
        if ep_id >= 9000:
            test.append(f)
        else:
            pool.append(f)

    rng = np.random.default_rng(seed)
    pool = list(rng.permutation(pool))
    n_val = int(val_frac * len(pool))
    val ,train = pool[:n_val], pool[n_val:]
    return train, val, test

def build_split(file_list, step = 10):
    """
    Returns a dict of stacked arrays for one split (train/val/test).
    step = 10 keeps every 10th row: 1Khz -> 100 Hz."""
    out = {"Xb": [], "Yb": [], "Xa": [], "Ya": [], "phase":[], "ep_id": []}
    for f in file_list:
        d = np.load(f)
        xb, _ = base_features(d)
        xa, _ = arm_features(d)
        yb = wheels_to_vw(d["wheel_cmds"])
        ya = d["torque_arm"]

        s = slice(None, None, step)
        out["Xb"].append(xb[s])
        out["Yb"].append(yb[s])
        out["Xa"].append(xa[s])
        out["Ya"].append(ya[s])
        out["phase"].append(d["phase"][s])
        ep = int(os.path.basename(f)[3:8])
        out["ep_id"].append(np.full(len(xb[s]), ep))

    return {k: np.concatenate(v) for k, v in out.items()}

def norm_stats(X):
    """
    Returns mean and std for each column of X.
    """
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return mu.astype(np.float32), std.astype(np.float32)

def build_dataset(step = 10, out_path = None):
    out_path = out_path or os.path.join(TASK_DIR, "data", f"processed_step{step}.npz")
    train_f, val_f, test_f = split_episodes()
    splits = {"train": build_split(train_f, step),
             "val": build_split(val_f, step), 
             "test": build_split(test_f, step)}
    arrays = {}

    for name, sp in splits.items():
        for k, v in sp.items():
            arrays[f"{k}_{name}"] = v          # e.g. "Xb_train", "Ya_val"
    for k in ["Xb", "Yb", "Xa", "Ya"]:
        arrays[f"{k}_mu"], arrays[f"{k}_sd"] = norm_stats(splits["train"][k])   # TRAIN only

    np.savez(out_path, **arrays)
    print ("saved", out_path)
    return arrays