import numpy as np
import time
import os
import sys

# This script lives in robotnik_arm_task/yousef/. The template files (configs/, models/,
# robotnik_arm_control.py) and the dataset (data/raw/) live one level up, in robotnik_arm_task/.
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)


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
BASE_NAV_TIMEOUT = 25.0
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


# Noise injection (DART-style): random pushes are added to the commands that are APPLIED to the robot,
# while the expert's clean commands are recorded as labels (wheel_exp / torque_exp). The robot is thus
# pushed slightly off the expert's path and the data shows how the expert corrects it.
# Each push is held for NOISE_HOLD seconds: noise that changes every 1 ms averages out before the arm moves.
NOISE_HOLD = 0.2          # s
NOISE_TORQUE_STD = 2.5    # N·m per arm joint (≈ 0.025 rad deviation with the expert's Kp = 100)
NOISE_V_STD = 0.015       # m/s, base forward speed
NOISE_W_STD = 0.06        # rad/s, base turning rate
NOISE_ARM_OFF_DIST = 0.05 # m: arm pushes stop once the EE is this close, so the expert can finish (1 cm tolerance)
WHEEL_RADIUS, WHEEL_BASE_WIDTH = 0.11, 0.66     # BaseControllerParams


def run_episode(target, ep_id, base_init=None, noise=False):    # base_init = (x, y, yaw)
    """Run ONE expert demonstration and save it to data/raw/ (or data/raw_noise/ when noise=True)."""
    noise_rng = np.random.default_rng(100000 + ep_id)          # reproducible noise per episode
    tau_push, vw_push, next_push_t = np.zeros(7), np.zeros(2), 0.0
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
        "wheel_exp": [], "torque_exp": []    # the expert's CLEAN commands (= applied ones when noise=False)
    }
    stop_message = "quit"
    nav_done_time = -1.0
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
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        base_navigation_done = base_state.done
        if base_navigation_done and nav_done_time < 0:
            nav_done_time = current_time          # when the base finished (≥ 25 s means it timed out)


        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # Clean expert commands = training labels
        wheel_exp, tau_exp = wheel_cmds.copy(), torque_arm.copy()

        # Noise injection: perturb the APPLIED commands only
        if noise:
            if current_time >= next_push_t:                       # draw a new push every NOISE_HOLD s
                tau_push = noise_rng.normal(0.0, NOISE_TORQUE_STD, 7)
                vw_push = noise_rng.normal(0.0, [NOISE_V_STD, NOISE_W_STD])
                next_push_t = current_time + NOISE_HOLD
            if ee_dist_error > NOISE_ARM_OFF_DIST:               # no pushes in the final approach
                torque_arm = np.clip(torque_arm + tau_push, -30.0, 30.0)
            if not base_navigation_done:                          # base pushes only while navigating
                dlw = (vw_push[0] - vw_push[1] * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
                drw = (vw_push[0] + vw_push[1] * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
                wheel_cmds = wheel_cmds + np.array([drw, dlw, dlw, drw])

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
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
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
        log["tau_g"].append(pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)[6:13])
        
        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} | phase {base_state.phase} | nav_done {base_navigation_done}")
    out_dir = os.path.join(cur_dir, "data", "raw_noise" if noise else "raw")
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
        noise=bool(noise),
    )

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

   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--noise", action="store_true", help="noise-injected demos -> data/raw_noise/")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ep_id = args.start_id
    end_id = args.start_id + args.n
    while ep_id < end_id:
        # random target
        r = rng.uniform(0.8, 2.0)
        ang = rng.uniform(-np.pi, np.pi)
        z = rng.uniform(0.1, 1.25)
        target = np.array([r*np.cos(ang), r*np.sin(ang), z])

        # random start pose (x, y, yaw)
        base_init = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi))

        # skip targets that are too close to the start position
        if np.linalg.norm(target[:2] - np.array(base_init[:2])) < 0.6:
            continue

        # Resume support: skip episodes that were already collected. The random target/start are
        # drawn BEFORE this check, so the random sequence (and every episode's config) is unchanged.
        path = os.path.join(TASK_DIR, "data", "raw_noise" if args.noise else "raw", f"ep_{ep_id:05d}.npz")
        if os.path.exists(path):
            ep_id += 1
            continue

        print(f"=== Episode {ep_id}: target {np.round(target, 3)}, start {np.round(base_init, 3)} ===")
        try:
            run_episode(target, ep_id, base_init, noise=args.noise)
        except Exception as e:
            # One failed episode must not kill the whole batch
            print(f"!!! Episode {ep_id} failed: {e}")
        ep_id += 1