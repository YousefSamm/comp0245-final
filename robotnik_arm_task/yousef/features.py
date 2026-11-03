import numpy as np

# --- Constants copied from the template (keep in sync!) -----------------
# robotnik_arm_task.py
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# robotnik_arm_control.py → BaseControllerParams
WHEEL_RADIUS = 0.11
WHEEL_BASE_WIDTH = 0.66
# arm joints 1,3,5,7 are continuous (no limits); 2,4,6 are limited
CONT_JOINTS = [0, 2, 4, 6]     # column indices in q_arm
LIM_JOINTS  = [1, 3, 5]

def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def base_goal(target):
    """
    target: (3,) world-frame target, Returns goal_xy (2,), goal_bearing (float)
    """
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2]
    goal_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    goal_xy = target_xy - standoff * np.array([np.cos(goal_bearing), np.sin(goal_bearing)])

    return goal_xy, goal_bearing

def world_to_base(vec_xy, yaw):
    """
    vec_xy: (N,2) world-frame vectors, yaw: (N,) robot heading
    Returns (N, 2) [forward, left] components in the robot frame
    
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = vec_xy[:, 0], vec_xy[:, 1]
    fwd = c * x + s * y
    left = -s * x + c * y
    return np.column_stack([fwd, left])

def base_features(d):
    """
    d: one loaded episode (np.load result). Returns X (N, 10) float 32 and feature names
    
    """

    pos, yaw, vel = d["base_pos"], d["base_bearing"], d["base_vel"]
    goal_xy, goal_b = base_goal(d["target"])

    err_w = goal_xy - pos
    err_r = world_to_base(err_w, yaw)
    dist = np.linalg.norm(err_r, axis=1)
    head_err = wrap(np.arctan2(err_w[:, 1], err_w[:, 0]) - yaw)
    bear_err = wrap(goal_b - yaw)
    vel_r = world_to_base(vel[:, :2], yaw)
    yaw_rate = vel[:, 2]

    X = np.column_stack([
        err_r, dist, 
        np.sin(head_err), np.cos(head_err),
        np.sin(bear_err), np.cos(bear_err),
        vel_r, yaw_rate
    ]).astype(np.float32)
    names = ["goal_fwd", "goal_left", "goal_dist", "sin_head", "cos_head",
             "sin_bear", "cos_bear", "v_fwd", "v_left", "yaw_rate"]

    return X, names

def arm_features(d):
    """
    Returns X (N, 25) float 32 and feature names.
    """

    q, q_d = d["q_arm"], d["qd_arm"]
    yaw = d["base_bearing"]

    # joints: sin/cos for continuous, raw value for limited
    q_sin = np.sin(q[:, CONT_JOINTS])
    q_cos = np.cos(q[:, CONT_JOINTS])
    q_lim = q[:, LIM_JOINTS]

    # EE -> target error in the robot frame (xy rotated, z unchanged)

    e = d["target"] - d["ee_pos"]
    e_xy_r = world_to_base(e[:, :2], yaw)
    e_z = e[:, 2:3]

    Xb, _ = base_features(d)
    dist = Xb[:, 2:3]  # goal distance from base features
    bear_sc = Xb[:, 5:7]  # sin/cos of bearing error from base features
    speed = np.linalg.norm(Xb[:, 7:9], axis=1, keepdims=True)

    X = np.column_stack([q_sin, q_cos, q_lim, q_d, e_xy_r, e_z, dist, bear_sc, speed]).astype(np.float32)
    names = ([f"sin_q{j+1}" for j in CONT_JOINTS] +
             [f"cos_q{j+1}" for j in CONT_JOINTS] +
             [f"q{j+1}" for j in LIM_JOINTS] +
             [f"qd{j+1}" for j in range(7)] +
             ["ee_err_fwd", "ee_err_left", "ee_err_z", "goal_dist", "sin_bear", "cos_bear", "base_speed"])
    return X, names

def wheels_to_vw(w):
    """
    w: (N,4) wheels cmds [rw, lw, lw, rw]. Returns (N,2) [v, w].
    """
    v = WHEEL_RADIUS * (w[:, 0] + w[:, 1]) / 2.0
    w = WHEEL_RADIUS * (w[:, 0] - w[:, 1]) / WHEEL_BASE_WIDTH
    return np.column_stack([v, w])

def vw_to_wheels(vw):
    """
    Inverse: (N,2) [v, w] -> (N,4) [rw, lw, lw, rw].
    """
    v, w = vw[:, 0], vw[:, 1]
    rw = (v + w * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
    lw = (v - w * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
    return np.column_stack([rw, lw, lw, rw]).astype(np.float32)