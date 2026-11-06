"""Policy for the obstacle scenes: BaseNet_obs (12 inputs: 10 BaseNet + 2 LiDAR) + the usual ArmNet.
Also runs a plain 10-input BaseNet (e.g. base_net.pt) without LiDAR, as the no-LiDAR baseline."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from policy import BCPolicy
from features import base_features, arm_features, vw_to_wheels
from dataset_obs import lidar_inputs


class BCPolicyObs(BCPolicy):
    def __init__(self, ckpt_dir, base_file="base_net_obs.pt", arm_file="arm_net.pt"):
        super().__init__(ckpt_dir, base_file=base_file, arm_file=arm_file)
        n_in = self.base.net.net[0].in_features                  # 10: no LiDAR, 12: front2, 16: corr, 18: sect8
        self.lidar_mode = {10: None, 12: "front2", 16: "corr", 18: "sect8"}[n_in]
        self.uses_lidar = self.lidar_mode is not None
        print(f"BCPolicyObs: BaseNet LiDAR inputs: {self.lidar_mode}")

    def __call__(self, obs):
        d = {"base_pos": obs["base_pos"][None, :2], "base_bearing": np.array([obs["base_bearing"]]),
             "base_vel": obs["base_vel"][None, :], "q_arm": obs["q_arm"][None, :],
             "qd_arm": obs["qd_arm"][None, :], "ee_pos": obs["ee_pos"][None, :], "target": obs["target"]}
        Xb, _ = base_features(d)
        if self.uses_lidar:
            Xb = np.hstack([Xb, lidar_inputs(obs, self.lidar_mode)]).astype(np.float32)
        Xa, _ = arm_features(d)
        wheel_cmds = vw_to_wheels(self.base(Xb))[0]
        torque_arm = self.arm(Xa)[0]
        if self.arm.gravity_residual:
            torque_arm = torque_arm + obs["tau_g"]
        return wheel_cmds, np.clip(torque_arm, -30.0, 30.0)
