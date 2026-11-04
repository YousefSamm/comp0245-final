import os, numpy as np, torch
from features import base_features, arm_features, vw_to_wheels
from models import MLP

class LoadedNet:
    """One trained net + its normalisation stats. Call with raw features, get raw outputs."""
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = MLP(ck["in_dim"], ck["out_dim"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.X_mu, self.X_sd = ck["X_mu"], ck["X_sd"]
        self.Y_mu, self.Y_sd = ck["Y_mu"], ck["Y_sd"]

    def __call__(self, X):
        x = (X - self.X_mu)/self.X_sd
        with torch.no_grad():
            y = self.net(torch.from_numpy(x.astype(np.float32))).numpy()

        return y * self.Y_sd + self.Y_mu

class BCPolicy:
    def __init__(self, ckpt_dir):
        self.base = LoadedNet(os.path.join(ckpt_dir, "base_net.pt"))
        self.arm = LoadedNet(os.path.join(ckpt_dir, "arm_net.pt"))

    def __call__(self, obs):
        """obs: dict with the live measurements of ONE time step.
        Returns wheel_cmds (4,), torque_arm(7,)."""

        d = {"base_pos":    obs["base_pos"][None, :2],
            "base_bearing": np.array([obs["base_bearing"]]),
            "base_vel":     obs["base_vel"][None, :],
            "q_arm":        obs["q_arm"][None, :],
            "qd_arm":       obs["qd_arm"][None, :],
            "ee_pos":       obs["ee_pos"][None, :],
            "target":       obs["target"]
            }
        Xb, _ = base_features(d)
        Xa, _ = arm_features(d)
        wheel_cmds = vw_to_wheels(self.base(Xb))[0]
        torque_arm = np.clip(self.arm(Xa)[0], -30.0, +30.0)
        return wheel_cmds, torque_arm