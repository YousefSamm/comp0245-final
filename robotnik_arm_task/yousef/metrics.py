"""Evaluation metrics of one saved run (no simulator needed, so it also works in notebooks)."""
import numpy as np

THRESHOLDS = (0.01, 0.02, 0.05)      # m: success if the EE got this close to the target at some point


def run_metrics(r, expert=None):
    """r: a saved run (np.load). Returns a dict of evaluation metrics."""
    from features import base_goal, wrap
    err = np.linalg.norm(r["ee_pos"] - r["target"], axis=1)
    m = {"ep": int(r.get("ep_id", -1)) if hasattr(r, "get") else -1,
         "best_err": float(err.min()), "final_err": float(err[-1]), "duration": float(r["t"][-1])}
    for th in THRESHOLDS:
        hit = np.where(err < th)[0]
        m[f"success_{int(th*100)}cm"] = bool(len(hit))
        m[f"time_to_{int(th*100)}cm"] = float(r["t"][hit[0]]) if len(hit) else np.nan
    goal_xy, goal_b = base_goal(r["target"])
    m["base_goal_err"] = float(np.linalg.norm(r["base_pos"][-1] - goal_xy))
    m["base_bear_err"] = float(abs(wrap(goal_b - r["base_bearing"][-1])))
    # did the simulation blow up? (huge joint or base speeds)
    m["blew_up"] = bool((np.abs(r["qd_arm"]).max() > 15.0) or (np.linalg.norm(r["base_vel"][:, :2], axis=1).max() > 1.0))
    if expert is not None:
        m["expert_duration"] = float(expert["t"][-1])
    return m
