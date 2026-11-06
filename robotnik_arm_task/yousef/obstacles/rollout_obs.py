"""OBSTACLE version of yousef/rollout.py (left unchanged): the trained networks drive among the boxes.

Differences from rollout.py:
  * the episode's boxes are spawned (obstacles=[...] from the expert's test episode)
  * LiDAR at 100 Hz: the expert's 360 deg scan, the network's front 180 deg scan (obs["lidar"])
  * the shadow expert is the obstacle-aware one (expert_avoid.base_controller_avoid)
  * robot-box contact logged per step; base nav timeout 45 s; "base done" accepts up to 0.3 m from the goal
    (the obstacle expert parks up to 0.25 m off after drifting round boxes)

Simulation loop in which the TRAINED NETWORKS drive the robot (library module, no command line).

    run_policy_episode(...)  one closed-loop episode; the expert runs alongside every step ("shadow expert")
                             and its commands are logged as wheel_exp / torque_exp
    run_metrics(r)           evaluation metrics of a saved run (success @ 1/2/5 cm, errors, ...)
    PARTS                    which parts the networks control: "base", "arm" or "both"

Used by: run_demo.py (single GUI run), dagger.py (DAgger data collection), evaluate.py (test-set evaluation).
"""
import numpy as np
import time
import os
import sys

# This script lives in robotnik_arm_task/yousef/. The template files (configs/, models/,
# robotnik_arm_control.py) and the dataset (data/raw/) live one level up, in robotnik_arm_task/.
HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)                      # yousef/obstacles/
sys.path.insert(1, os.path.dirname(HERE))     # yousef/
sys.path.append(TASK_DIR)

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "60000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper, wrap_angle
from metrics import run_metrics, THRESHOLDS     # re-exported for convenience
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)
from world import spawn_obstacles, lidar_scan_360, robot_hits_obstacle, FRONT_IDX
from expert_avoid import base_controller_avoid
LIDAR_EVERY = 10              # steps: LiDAR + contact check at 100 Hz (as in collect_obs.py)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 45.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01
# ==============================================================================


def run_policy_episode(target, ep_id, base_init, policy, net_parts=("base", "arm"), out_subdir="eval_obs",
                       obstacles=np.zeros((0, 4))):
    """Run ONE episode where the networks drive.
    policy(obs) -> (wheel_cmds, torque_arm); net_parts says which parts it controls:
    ("base",) or ("arm",) to test one network at a time (the expert does the other part), ("base","arm") for both."""
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = TASK_DIR
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    # Randomised initial base pose
    if base_init is not None:
        x0, y0, yaw0 = base_init
        z0 = sim.GetBasePosition()[2]                    # keep the spawn height
        client.resetBasePositionAndOrientation(
            rid, [x0, y0, z0], p.getQuaternionFromEuler([0, 0, yaw0])
        )
        sim.ReceiveObservation()                         # refresh the measurements
    # Visualize the target reach reference in the PyBullet 3D environment as a red dot
    target_pos_list = target.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )
    obstacle_ids = spawn_obstacles(client, obstacles)

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
    sim.ReceiveObservation() 
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    print("==================================================================")
    print(f"Target Cartesian EE Position : {target}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target - init_ee_pos):.4f}")
    print("==================================================================")

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0

    # Mobile base goal location facing the target
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    # Stop conditions (step cap, target reached, unreachable-target stall), see StopCriteria
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)

    # Command interface
    cmd = MotorCommands()

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    log = {
        "t": [], "base_pos": [], "base_bearing": [], 
        "q_arm": [], "qd_arm": [], "qd_wheels": [], 
        "ee_pos": [], 
        "wheel_cmds": [], "torque_arm": [],
        "base_vel": [], "phase": [], "q_des_arm": [],
        "tau_g": [],
        "wheel_exp": [], "torque_exp": [],    # what the expert WOULD send (= the applied cmds when policy is None)
        "lidar": [], "avoiding": [], "collision": []
    }
    stop_message = "quit"
    nav_done_time = -1.0
    net_base_done = False      # latched: the BaseNet-driven base has arrived at its goal pose
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        lin_vel, ang_vel = client.getBaseVelocity(rid)
        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        step_i = int(round(current_time / time_step))
        if step_i % LIDAR_EVERY == 0:
            ranges360 = lidar_scan_360(client, base_pos, base_bearing, obstacle_ids)
            ranges = ranges360[FRONT_IDX]                                    # what the network gets
            touching = robot_hits_obstacle(client, rid, obstacle_ids)
        wheel_cmds, avoiding = base_controller_avoid(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params,
            ranges360
        )
        base_navigation_done = base_state.done
        if base_navigation_done and nav_done_time < 0:
            nav_done_time = current_time          # when the base finished (≥ 25 s means it timed out)

        # When BaseNet drives, the expert's phase machine (align -> drive -> orient -> done) does not
        # follow the robot's real path and can get stuck in "drive" even though the base has arrived.
        # So "base done" is decided from the MEASURED pose instead, with tolerances matching what the
        # expert itself accepts in the data (stops up to ~0.15 m from the goal, bearing within ~0.05 rad).
        if "base" in net_parts:
            d_goal = np.linalg.norm(base_goal_pos - base_pos[:2])
            b_err = abs(wrap_angle(base_goal_bearing - base_bearing))
            speed = np.hypot(lin_vel[0], lin_vel[1])
            if d_goal < 0.30 and b_err < 0.1 and speed < 0.02 and abs(ang_vel[2]) < 0.05:
                net_base_done = True
            reach_active = base_navigation_done or net_base_done
        else:
            reach_active = base_navigation_done   # expert drives the base: use its own phase machine

        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=reach_active,
        )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # Keep the expert's commands (for logging / comparison / DAgger labels)
        wheel_exp, tau_exp = wheel_cmds.copy(), torque_arm.copy()

        # The networks drive; the expert above still runs every step so its commands can be logged
        tau_g_now = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)[6:13]   # 7 arm joints
        obs = {"base_pos": base_pos, "base_bearing": base_bearing,
               "base_vel": np.array([lin_vel[0], lin_vel[1], ang_vel[2]]),
               "q_arm": q_mes[4:11], "qd_arm": qd_mes[4:11],
               "ee_pos": ee_pos, "target": target, "tau_g": tau_g_now, "lidar": ranges}
        w_net, tau_net = policy(obs)
        if "base" in net_parts:
            wheel_cmds = w_net
        if "arm" in net_parts:
            torque_arm = tau_net
        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Stop criteria: maximum steps, target reached within fine tolerance, or unreachable target
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, reach_active)
        if stop_message:
            print(stop_message)
            break

        # Data log section
        log["t"].append(current_time)
        log["base_pos"].append(base_pos[:2])
        log["base_bearing"].append(base_bearing)
        log["q_arm"].append(q_mes[4:11])
        log["qd_arm"].append(qd_mes[4:11])
        log["qd_wheels"].append(qd_mes[0:4])
        log["ee_pos"].append(ee_pos)
        log["wheel_cmds"].append(wheel_cmds)
        log["torque_arm"].append(torque_arm)
        log["wheel_exp"].append(wheel_exp)
        log["torque_exp"].append(tau_exp)
        log["base_vel"].append([lin_vel[0], lin_vel[1], ang_vel[2]])      # vx, vy, yaw rate (world frame)
        phase_id = 3 if base_state.done else {"align": 0, "drive": 1, "orient": 2}[base_state.phase]
        log["phase"].append(phase_id)                                       # expert's hidden phase
        log["q_des_arm"].append(q_des_arm)                                  # expert's hidden reference
        log["tau_g"].append(tau_g_now)
        log["lidar"].append(ranges)
        log["avoiding"].append(avoiding)
        log["collision"].append(touching)
        
        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} | phase {base_state.phase} | nav_done {base_navigation_done}")
    out_dir = os.path.join(cur_dir, "data", out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"ep_{ep_id:05d}.npz"),
        **{k: np.array(v, dtype=np.float32) for k, v in log.items()},
        target=target,
        base_init=np.array(base_init if base_init is not None else [0.0, 0.0, 0.0]),
        success=bool(stop_message) and stop_message.startswith("Target reached"),
        final_error=np.linalg.norm(target - ee_pos),
        stop_message=str(stop_message),
        nav_time=float(nav_done_time),
        obstacles=np.asarray(obstacles, dtype=np.float32),
        n_collision_steps=int(np.sum(log["collision"])),
    )
    print(f"Boxes: {len(obstacles)} | collision steps: {int(np.sum(log['collision']))}")

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {target}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(target - ee_pos):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")
    client.disconnect()
    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return


# Which parts the networks control in each mode
PARTS = {"base": ("base",), "arm": ("arm",), "both": ("base", "arm")}
