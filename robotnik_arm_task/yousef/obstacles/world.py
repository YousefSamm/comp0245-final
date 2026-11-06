"""Obstacle world: sampling and spawning obstacles, the 2-D LiDAR scan, and the 2 LiDAR features.

Shared by data collection (collect_obs.py), the dataset (dataset_obs.py) and closed-loop runs
(rollout_obs.py), so training and run time compute the features with the same code.
Nothing here imports the simulator wrapper; the PyBullet client is passed in.
"""
import numpy as np
import pybullet as p

# ------------------------------------------------------------------------------
# Obstacles: static boxes (square footprint, random rotation) standing on the floor
# ------------------------------------------------------------------------------
OBS_HALF = (0.10, 0.15)        # m, half the side length (sides 0.20-0.30 m), sampled uniformly
OBS_HEIGHT = 1.0               # m, taller than the LiDAR plane
N_OBS = [0, 2, 3, 4]           # number of boxes in an episode ...
N_OBS_P = [0.1, 0.3, 0.3, 0.3] # ... and its probability (10 % empty scenes: drive normally when nothing is near)
PATH_FRAC = (0.10, 0.90)       # boxes sit between 10 % and 90 % of the start -> base-goal line ...
LATERAL = 1.0                  # m ... up to this far to either side of it (some block the way, some do not)
START_DIST = (2.0, 3.2)        # m, start -> base-goal distance (collect_obs.py samples the start this far out)
MIN_PATH = 1.6                 # m, shorter start -> base-goal lines get no boxes
# Keep-out radii ("donut" holes), measured from the edge of the box's bounding circle (half-side * sqrt 2):
CLEAR_TARGET = 0.6             # m around the target (x, y): the arm's reaching space stays free
CLEAR_GOAL = 0.80              # m around the base goal: the base can park and turn in place (chassis corner ~0.47 m)
CLEAR_START = 0.75             # m around the start pose: it turns in place there (chassis corner ~0.47 m)
CLEAR_EACH = 1.0              # m between two boxes' bounding circles: robot 0.61 m + 0.2 m safety each side

# ------------------------------------------------------------------------------
# LiDAR: a horizontal fan of rays over the front 180 deg
# ------------------------------------------------------------------------------
LIDAR_N = 37                   # rays, every 5 deg
LIDAR_FOV = np.pi              # front 180 deg (the base never drives backwards)
LIDAR_ANGLES = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, LIDAR_N)   # robot frame, 0 = straight ahead, + = left
# Each ray starts just outside the chassis rectangle (half-length 0.36 m, half-width 0.305 m), so it never hits
# the robot itself but still sees obstacles right next to it.
CHASSIS_HL, CHASSIS_HW = 0.36, 0.305
LIDAR_R0 = np.minimum(CHASSIS_HL / np.maximum(np.abs(np.cos(LIDAR_ANGLES)), 1e-6),
                      CHASSIS_HW / np.maximum(np.abs(np.sin(LIDAR_ANGLES)), 1e-6)) + 0.03   # (LIDAR_N,) m
LIDAR_RMAX = 3.0               # m
LIDAR_Z = 0.3                  # m above the floor
# Expert's 360 deg scan: 72 rays every 5 deg; the network's front 180 deg rays are a subset of it.
LIDAR360_ANGLES = -np.pi + np.deg2rad(5.0) * np.arange(72)                 # -180 ... +175 deg, robot frame
FRONT_IDX = np.arange(18, 55)                                               # -90 ... +90 deg = LIDAR_ANGLES
LIDAR360_R0 = np.minimum(CHASSIS_HL / np.maximum(np.abs(np.cos(LIDAR360_ANGLES)), 1e-6),
                         CHASSIS_HW / np.maximum(np.abs(np.sin(LIDAR360_ANGLES)), 1e-6)) + 0.03
assert np.allclose(LIDAR360_ANGLES[FRONT_IDX], LIDAR_ANGLES)
FEAT_RANGE = 1.5              # m of clearance from the chassis: obstacles further than this give proximity 0


def sample_obstacles(rng, target, base_init, base_goal):
    """Returns a (k, 4) array of boxes [x, y, half_side, yaw], k from N_OBS. Pure numpy, reproducible.
    Fewer than k are returned if the keep-out zones leave no room for more."""
    start = np.asarray(base_init[:2], dtype=float)
    goal = np.asarray(base_goal, dtype=float)
    tgt = np.asarray(target[:2], dtype=float)
    path = goal - start
    length = np.linalg.norm(path)
    k = int(rng.choice(N_OBS, p=N_OBS_P))
    if k == 0 or length < MIN_PATH:
        return np.zeros((0, 4))
    u_dir = path / length
    n_dir = np.array([-u_dir[1], u_dir[0]])
    obs = []
    for _ in range(50 * k):                      # rejection sampling
        if len(obs) == k:
            break
        h = rng.uniform(*OBS_HALF)
        r = h * np.sqrt(2.0)                     # bounding circle of the square footprint
        c = start + rng.uniform(*PATH_FRAC) * path + rng.uniform(-LATERAL, LATERAL) * n_dir
        if np.linalg.norm(c - tgt) < CLEAR_TARGET + r:
            continue
        if np.linalg.norm(c - goal) < CLEAR_GOAL + r:
            continue
        if np.linalg.norm(c - start) < CLEAR_START + r:
            continue
        if any(np.linalg.norm(c - o[:2]) < CLEAR_EACH + r + o[2] * np.sqrt(2.0) for o in obs):
            continue
        obs.append([c[0], c[1], h, rng.uniform(0, np.pi / 2)])
    return np.array(obs).reshape(-1, 4)


def spawn_obstacles(client, obstacles):
    """Creates one static box per row [x, y, half_side, yaw] of `obstacles`. Returns their PyBullet body ids."""
    ids = []
    for x, y, h, yaw in obstacles:
        half = [float(h), float(h), OBS_HEIGHT / 2]
        col = client.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = client.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.55, 0.6, 1.0])
        ids.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                          basePosition=[float(x), float(y), OBS_HEIGHT / 2],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, float(yaw)])))
    return ids


def lidar_scan(client, base_pos, base_bearing, obstacle_ids, line_ids=None):
    """Ranges (LIDAR_N,) in m from the base centre along LIDAR_ANGLES; LIDAR_RMAX where nothing is hit.
    Only obstacle hits count (floor, target marker and the robot itself are ignored).
    GUI only: pass line_ids (a list, empty at first) to draw the rays; the same 37 lines are moved each
    call instead of new ones being created, which would slow the GUI down to a crawl."""
    dirs = base_bearing + LIDAR_ANGLES
    c, s = np.cos(dirs), np.sin(dirs)
    x0, y0 = base_pos[0], base_pos[1]
    froms = [[x0 + r0 * ci, y0 + r0 * si, LIDAR_Z] for r0, ci, si in zip(LIDAR_R0, c, s)]
    tos = [[x0 + LIDAR_RMAX * ci, y0 + LIDAR_RMAX * si, LIDAR_Z] for ci, si in zip(c, s)]
    ranges = np.full(LIDAR_N, LIDAR_RMAX)
    if obstacle_ids:
        ids = set(obstacle_ids)
        for i, hit in enumerate(client.rayTestBatch(froms, tos)):
            if hit[0] in ids:
                ranges[i] = LIDAR_R0[i] + hit[2] * (LIDAR_RMAX - LIDAR_R0[i])
    if line_ids is not None:                     # red = hit, green = free
        for old in line_ids:                     # remove the previous fan, then draw the new one
            client.removeUserDebugItem(old)
        line_ids.clear()
        for i, (f, t, rng_i) in enumerate(zip(froms, tos, ranges)):
            frac = (rng_i - LIDAR_R0[i]) / (LIDAR_RMAX - LIDAR_R0[i])
            end = [f[k] + frac * (t[k] - f[k]) for k in range(3)]
            col = [1, 0, 0] if rng_i < LIDAR_RMAX else [0, 0.7, 0]
            line_ids.append(client.addUserDebugLine(f, end, col, 1))
    return ranges


def lidar_features(ranges):
    """The 2 BaseNet LiDAR inputs from scans of shape (LIDAR_N,) or (T, LIDAR_N), using the clearance from
    the chassis (range minus the ray's start distance), so a box beside the robot counts as close:
    angle of the ray with the smallest clearance (rad, robot frame, + = left; 0 when nothing is close) and
    proximity in [0, 1] (0 = clearance >= FEAT_RANGE, 1 = touching the chassis)."""
    clear = np.atleast_2d(ranges) - LIDAR_R0
    i = np.argmin(clear, axis=1)
    d = clear[np.arange(len(clear)), i]
    prox = np.clip((FEAT_RANGE - d) / FEAT_RANGE, 0.0, 1.0)
    ang = np.where(prox > 0, LIDAR_ANGLES[i], 0.0)
    return np.column_stack([ang, prox]).astype(np.float32)


def robot_hits_obstacle(client, robot_id, obstacle_ids):
    """True if any part of the robot touches any obstacle this step."""
    return any(len(client.getContactPoints(bodyA=robot_id, bodyB=o)) > 0 for o in obstacle_ids)


def lidar_scan_360(client, base_pos, base_bearing, obstacle_ids):
    """The expert's 360 deg scan: ranges (72,) in m from the base centre along LIDAR360_ANGLES.
    ranges[FRONT_IDX] is exactly the network's front 180 deg scan (same rays as lidar_scan)."""
    dirs = base_bearing + LIDAR360_ANGLES
    c, s = np.cos(dirs), np.sin(dirs)
    x0, y0 = base_pos[0], base_pos[1]
    froms = [[x0 + r0 * ci, y0 + r0 * si, LIDAR_Z] for r0, ci, si in zip(LIDAR360_R0, c, s)]
    tos = [[x0 + LIDAR_RMAX * ci, y0 + LIDAR_RMAX * si, LIDAR_Z] for ci, si in zip(c, s)]
    ranges = np.full(len(LIDAR360_ANGLES), LIDAR_RMAX)
    if obstacle_ids:
        ids = set(obstacle_ids)
        for i, hit in enumerate(client.rayTestBatch(froms, tos)):
            if hit[0] in ids:
                ranges[i] = LIDAR360_R0[i] + hit[2] * (LIDAR_RMAX - LIDAR360_R0[i])
    return ranges
