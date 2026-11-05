"""Obstacle-aware expert for the base: the template base_controller plus reactive LiDAR steering.

The template (robotnik_arm_control.py) is NOT modified. We call its base_controller and, only while it is
driving, adjust its (v, w) from the closest LiDAR hit in the front 180 deg: its angle and its clearance
from the chassis. Those are exactly the two LiDAR inputs BaseNet_obs gets, so the behaviour is observable
and can be cloned. With nothing within C_SAFE the command is exactly the template's.

No sliding: skid-steer wheels slip sideways when the robot turns fast while moving. The template keeps
|w| <= 0.3 rad/s while driving and only turns faster in place; so do we: either drive with |w| <= W_DRIVE,
or stop and turn in place.

Privileged safety check (expert only): before turning in place, the expert checks with the true box layout
that its rear corners (not seen by the front LiDAR) have room to swing.
"""
import numpy as np
from robotnik_arm_control import base_controller
from world import LIDAR_ANGLES, LIDAR_R0, LIDAR_RMAX, FRONT_IDX

HALF_W = 0.305 + 0.20   # m: half-width of the driving corridor (chassis half-width + 20 cm safety)
FRONT = 0.36            # m: centre -> front bumper
F_SAFE = 1.00           # m from the bumper to a box IN the corridor: start slowing and steering away
F_STOP = 0.35           # m: stop and turn in place
F_GO = 0.60             # m: ... until the corridor is clear this far (hysteresis)
SIDE_CLOSE = 0.20       # m: a box beside the corridor this close -> don't turn towards it
REALIGN = 0.25          # rad (~15 deg): once the way to the goal is clear, turn in place to face it if this far off
REALIGN_NEAR = 0.6      # rad: near the goal, only realign if this far off (and not within 0.35 m of it)
PARK_TOL = 0.25         # m: within this of the base goal but facing away -> park here (template orient)
NEAR_GOAL = 0.6         # m: closer than this to the base goal, the template finishes on its own (no realign,
                        # no minimum speed): the direction to a goal 10 cm away swings wildly and made it wobble
W_DRIVE = 0.3           # rad/s: max turning rate while moving (the template's limit while driving)
W_TURN = 1.0            # rad/s: turning in place (the template's own in-place rate)
V_MAX = 0.25            # m/s: the template's max_lin_vel
ROBOT_R = 0.47          # m: chassis half-diagonal (what sweeps when turning in place)
VW_MAX = 0.03           # (m/s)*(rad/s): max speed x turn rate while avoiding; above this the skid-steer
                        # base slides sideways (0.25 m/s at 0.3 rad/s slid ~0.2 m/s in the GUI test)
V_AVOID = 0.12          # m/s: min speed while going round boxes (the template slows to ~0 when facing away from its goal)
WHEEL_RADIUS, WHEEL_BASE_WIDTH = 0.11, 0.66


def wheels_to_vw(w):
    return WHEEL_RADIUS * (w[0] + w[1]) / 2.0, WHEEL_RADIUS * (w[0] - w[1]) / WHEEL_BASE_WIDTH


def vw_to_wheels(v, w):
    rw = (v + w * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
    lw = (v - w * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
    return np.array([rw, lw, lw, rw])


def _box_dist(p, o):
    """Distance from point p (x, y) to box o = [x, y, half_side, yaw] (0 inside)."""
    x, y, h, yaw = o
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = p[0] - x, p[1] - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return float(np.hypot(max(abs(lx) - h, 0.0), max(abs(ly) - h, 0.0)))


def can_turn(ranges360):
    """From the expert's 360 deg scan: the chassis corners can sweep a full circle without touching a box."""
    return bool(np.all(ranges360 > ROBOT_R + 0.15))      # 15 cm safety round the corners


def goal_line_clear(ranges, err, dist):
    """True if no LiDAR hit lies in the corridor the chassis would drive along to the goal."""
    if abs(err) > np.pi / 2:                     # goal behind: nothing to see that way (the box was passed)
        return True
    hit = ranges < LIDAR_RMAX - 0.01
    px, py = ranges * np.cos(LIDAR_ANGLES), ranges * np.sin(LIDAR_ANGLES)
    gx, gy = np.cos(err), np.sin(err)
    along, across = px * gx + py * gy, np.abs(-px * gy + py * gx)
    return not np.any(hit & (along > 0) & (along < dist + 0.3) & (across < 0.31 + 0.15))


def corridor(ranges):
    """From one scan: free distance from the bumper to the nearest hit inside the driving corridor, the side
    (+1 left, -1 right) that hit is on, and the clearance + angle of the closest hit overall."""
    hit = ranges < LIDAR_RMAX - 0.01
    px, py = ranges * np.cos(LIDAR_ANGLES), ranges * np.sin(LIDAR_ANGLES)
    inside = hit & (px > 0) & (np.abs(py) < HALF_W)
    if inside.any():
        k = np.argmin(np.where(inside, px, np.inf))
        free, side_of_hit = float(px[k] - FRONT), float(np.sign(py[k]) or 1.0)
    else:
        free, side_of_hit = np.inf, 0.0
    clear = ranges - LIDAR_R0
    i = int(np.argmin(clear))
    return free, side_of_hit, float(clear[i]), float(LIDAR_ANGLES[i])


def base_controller_avoid(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params,
                          ranges360):
    """Template base_controller + LiDAR steering. The expert sees 360 deg (ranges360, 72 rays); steering uses
    the front 180 deg part, the 360 deg view is used to check there is room to turn in place.
    Returns (wheel_cmds, avoiding)."""
    ranges = ranges360[FRONT_IDX]
    pos = np.asarray(base_pos[:2], float)
    to_goal = np.asarray(goal_pos, float) - pos
    err_goal = np.arctan2(to_goal[1], to_goal[0]) - base_bearing
    err_goal = float(np.arctan2(np.sin(err_goal), np.cos(err_goal)))
    free, side_of_hit, c, a = corridor(ranges)
    turning = getattr(state, "avoid_turn", 0.0)          # +1/-1 while a stop-and-turn is in progress

    # Face the goal again (template align = turn in place) once nothing blocks the corridor, the way to the
    # goal is clear and there is room to turn. The template's drive mode alone (0.3 rad/s) would circle it.
    dist = float(np.linalg.norm(to_goal))
    far = dist > NEAR_GOAL
    lim = REALIGN if far else REALIGN_NEAR
    # Near the goal and not facing it (it slid sideways on the way in): accept the position and let the
    # template turn to the final bearing (orient), instead of circling the goal trying to hit 8 cm.
    if not state.done and state.phase == "drive" and dist < PARK_TOL and abs(err_goal) > 0.5:
        state.phase = "orient"
    if dist > 0.35 and not state.done and state.phase == "drive" and not turning and free >= F_SAFE \
            and abs(err_goal) > lim and goal_line_clear(ranges, err_goal, dist) \
            and can_turn(ranges360):
        state.phase = "align"

    wheels = base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params)
    if state.done or state.phase != "drive":
        state.avoid_turn = 0.0
        return wheels, False

    # Stop and turn in place: starts when a box in the corridor is closer than F_STOP, lasts until F_GO is free
    if turning and free >= F_GO:
        turning = 0.0
    if not turning and free < F_STOP and can_turn(ranges360):
        turning = getattr(state, "avoid_side", 0.0) or -side_of_hit   # keep the side already chosen
    state.avoid_turn = turning
    if turning:
        return vw_to_wheels(0.0, turning * W_TURN), True

    v, w = wheels_to_vw(wheels)
    avoiding = False
    if free < F_SAFE:                                    # box in the corridor ahead: slow down, steer away
        s = float(np.clip((F_SAFE - free) / (F_SAFE - F_STOP), 0.0, 1.0))
        # pick the side once and keep it until the corridor is clear (no left-right chatter between two boxes)
        side = getattr(state, "avoid_side", 0.0) or -side_of_hit
        state.avoid_side = side
        v = V_MAX * (1.0 - 0.6 * s) if free >= F_STOP else 0.0
        w = side * W_DRIVE * (0.5 + 0.5 * s)
        avoiding = True
    else:
        state.avoid_side = 0.0
        if c < SIDE_CLOSE and np.sign(w) == np.sign(a):  # box right beside: don't turn towards it
            w = 0.0
            avoiding = True
    if far and abs(err_goal) > REALIGN and free >= F_SAFE:   # facing away from the goal, can't turn back yet:
        v = max(v, V_AVOID)                              # keep moving until there is room / the way is clear
        avoiding = True
    # Never turn towards a box that is right beside the chassis (front corners swing into it).
    if c < 2 * SIDE_CLOSE and np.sign(w) == np.sign(a):
        w = 0.0 if v > 0 else -np.sign(a) * W_DRIVE      # stopped: rotate away from the closest box instead
        avoiding = True
    w = float(np.clip(w, -W_DRIVE, W_DRIVE))
    if abs(w) > 1e-3:                                    # turning while moving (also the template's own corrections):
        v = min(v, VW_MAX / abs(w))                      # slow down so it arcs instead of sliding
    return vw_to_wheels(v, w), avoiding
