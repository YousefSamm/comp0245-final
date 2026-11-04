import os
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ARM_FEATS = [
    "ee_pos",
    "q_arm",
    "qd_arm",
    "target",
    "torque_arm",
    "phase",
]

# Column layout of the combined array returned by combine_data()
# (phase column is dropped after filtering)
ARM_COLS = [
    "ep_id",                                                    # 0
    "t",                                                        # 1
    "ee_pos_x", "ee_pos_y", "ee_pos_z",                         # 2-4
    "j1_pos", "j2_pos", "j3_pos", "j4_pos",
    "j5_pos", "j6_pos", "j7_pos",                               # 5-11
    "j1_vel", "j2_vel", "j3_vel", "j4_vel",
    "j5_vel", "j6_vel", "j7_vel",                               # 12-18
    "target_x", "target_y", "target_z",                         # 19-21
    "torque_1", "torque_2", "torque_3", "torque_4",
    "torque_5", "torque_6", "torque_7",                         # 22-28
]

PROC_COLS = (
    ["ep_id", "pos_error_x", "pos_error_y", "pos_error_z"]
    + [f"j{i}_sin" for i in range(1, 8)]
    + [f"j{i}_cos" for i in range(1, 8)]
    + [f"j{i}_vel" for i in range(1, 8)]
    + [f"tau_{i}" for i in range(1, 8)]
)  # 32 columns

# Episode-ID prefixes to load. ep_00xxx..ep_03xxx -> training pool,
# ep_09xxx -> test pool (IDs >= TEST_EP_THRESHOLD).
EP_PREFIXES = ["00", "01", "02", "03", "09"]
TEST_EP_THRESHOLD = 9000

# Phases to keep: 2.0 = orient, 3.0 = stop
KEEP_PHASES = (2.0, 3.0)

# Normalisation options
CENTER_POS_ERROR = False     # False -> scale only, so zero error stays zero input
NORMALIZE_TORQUES = True     # normalise targets too (stats saved for un-normalising)
EPS = 1e-8                   # std floor to avoid divide-by-zero


def combine_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(cur_dir, "data", "raw")

    blocks = []
    n_skipped_failed = 0

    for fname in sorted(os.listdir(raw_data_path)):
        # must be a finished episode file, not a leftover temp file
        if not fname.endswith(".npz") or fname.endswith(".tmp.npz"):
            continue
        if not any(fname.startswith(f"ep_{p}") for p in EP_PREFIXES):
            continue

        path = os.path.join(raw_data_path, fname)

        with np.load(path) as data:
            if not bool(data["success"]):  # exclude failed expert episodes
                n_skipped_failed += 1
                continue

            ep_id = int(fname[3:8])  # "ep_00012.npz" -> 12
            n_obs = data["t"].shape[0]

            ep_inputs = np.column_stack([
                np.full(n_obs, ep_id),  # column 0: episode id
                data["t"],              # column 1: time
            ])

            for feat in ARM_FEATS:
                d = data[feat]

                if feat == "target":
                    d = np.tile(d, (n_obs, 1))

                if d.ndim == 1:
                    d = d.reshape(-1, 1)

                assert ep_inputs.shape[0] == d.shape[0], (
                    f"{fname}: feature '{feat}' has {d.shape[0]} rows, "
                    f"expected {ep_inputs.shape[0]}"
                )
                ep_inputs = np.concatenate((ep_inputs, d), axis=1)

        blocks.append(ep_inputs)

    if not blocks:
        raise RuntimeError(
            f"No usable episodes found in {raw_data_path} "
            f"(prefixes {EP_PREFIXES}, {n_skipped_failed} failed episodes skipped)."
        )

    arr = np.vstack(blocks)

    # Keep only orient and stop phases, then drop the phase column (last)
    phase = arr[:, -1]
    arr = arr[np.isin(phase, KEEP_PHASES)][:, :-1]

    assert arr.shape[1] == len(ARM_COLS), (
        f"Expected {len(ARM_COLS)} columns, got {arr.shape[1]}"
    )

    print(f"Built array {arr.shape} from {len(blocks)} episodes "
          f"({n_skipped_failed} failed excluded)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return arr, stamp



def _safe_std(x):
    std = x.std(axis=0)
    return np.where(std < EPS, 1.0, std)


def fit_stats(train_arr):
    """Compute normalisation stats from TRAINING rows only (raw combined array)."""
    pos_err = train_arr[:, 19:22] - train_arr[:, 2:5]  # target - ee_pos
    vel = train_arr[:, 12:19]
    tau = train_arr[:, 22:29]

    stats = {
        "pos_err_mean": pos_err.mean(axis=0) if CENTER_POS_ERROR else np.zeros(3),
        "pos_err_std": _safe_std(pos_err),
        "vel_mean": vel.mean(axis=0),
        "vel_std": _safe_std(vel),
    }
    if NORMALIZE_TORQUES:
        stats["tau_mean"] = tau.mean(axis=0)
        stats["tau_std"] = _safe_std(tau)
    else:
        stats["tau_mean"] = np.zeros(7)
        stats["tau_std"] = np.ones(7)
    return stats


def build_features(arr, stats, isinsim=False):
    """Turn a raw combined array into the processed 32-column array."""

    if isinsim:
        #Add two columns to the front of the array for ep_id and t proxy
        arr = np.column_stack([np.zeros(arr.shape[0]), np.zeros(arr.shape[0]), arr])


    ep_id = arr[:, 0]

    pos_err = arr[:, 19:22] - arr[:, 2:5]  # target - ee_pos
    pos_err = (pos_err - stats["pos_err_mean"]) / stats["pos_err_std"]

    # sin / cos of joint angles only (not velocities)
    j_sin = np.sin(arr[:, 5:12])
    j_cos = np.cos(arr[:, 5:12])

    vel = (arr[:, 12:19] - stats["vel_mean"]) / stats["vel_std"]
    if isinsim:
        tau = np.zeros((arr.shape[0], 7))

    else:
        tau = (arr[:, 22:29] - stats["tau_mean"]) / stats["tau_std"]

    return np.column_stack([ep_id, pos_err, j_sin, j_cos, vel, tau])


def proc_data(arr, stamp=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "data", "proc")
    os.makedirs(out_dir, exist_ok=True)

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Split first, so stats come from training data only
    is_test = arr[:, 0] >= TEST_EP_THRESHOLD
    raw_train, raw_test = arr[~is_test], arr[is_test]

    if len(raw_train) == 0:
        raise RuntimeError("No training rows (ep_id < %d)." % TEST_EP_THRESHOLD)
    if len(raw_test) == 0:
        print("Warning: no test rows (ep_id >= %d)." % TEST_EP_THRESHOLD)

    stats = fit_stats(raw_train)
    train_arr = build_features(raw_train, stats)
    test_arr = build_features(raw_test, stats)

    cols = np.array(PROC_COLS)
    assert train_arr.shape[1] == len(cols)

    np.savez(os.path.join(out_dir, f"arm_train_proc_{stamp}"), arr=train_arr, cols=cols)
    np.savez(os.path.join(out_dir, f"arm_test_proc_{stamp}"), arr=test_arr, cols=cols)
    np.savez(os.path.join(out_dir, f"arm_norm_stats_{stamp}"), **stats)

    print(f"train {train_arr.shape}, test {test_arr.shape}")
    return f"Saved processed train/test arrays and norm stats with stamp {stamp} to data/proc"


def main():
    arr, stamp = combine_data()
    print(proc_data(arr, stamp=stamp))


if __name__ == "__main__":
    main()