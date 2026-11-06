"""Train BaseNet_obs (12 inputs -> v, w) on data/processed_obs_step10.npz.
Saves checkpoints/base_net_obs_<VERSION>.pt and checkpoints/base_net_obs.pt; the main base_net.pt is NOT touched.

Usage (from robotnik_arm_task/):  python yousef/obstacles/train_obs.py --version v1
"""
import os, sys, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from models import MLP
from train import make_loaders, train_model

TASK_DIR = os.path.dirname(os.path.dirname(HERE))
parser = argparse.ArgumentParser()
parser.add_argument("--version", default="v1")
args = parser.parse_args()

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
D = dict(np.load(os.path.join(TASK_DIR, "data", "processed_obs_step10.npz")))
print("train", D["Xb_train"].shape, "val", D["Xb_val"].shape, "|", device)

train_dl, val_dl = make_loaders(D, "b")
net, hist = train_model(MLP(12, 2), train_dl, val_dl, device=device)

ck = {"state_dict": net.state_dict(), "in_dim": 12, "out_dim": 2,
      "X_mu": D["Xb_mu"], "X_sd": D["Xb_sd"], "Y_mu": D["Yb_mu"], "Y_sd": D["Yb_sd"],
      "version": f"obs_{args.version}", "lidar_features": True}
ckdir = os.path.join(TASK_DIR, "checkpoints")
torch.save(ck, os.path.join(ckdir, f"base_net_obs_{args.version}.pt"))
torch.save(ck, os.path.join(ckdir, "base_net_obs.pt"))
print(f"saved base_net_obs_{args.version}.pt and base_net_obs.pt")

# offline check: RMSE of v (m/s) and w (rad/s), overall and while the expert is avoiding vs not
net.eval()
x = (D["Xb_val"] - D["Xb_mu"]) / D["Xb_sd"]
with torch.no_grad():
    p = net(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy() * D["Yb_sd"] + D["Yb_mu"]
err = p - D["Yb_val"]
near = D["Xb_val"][:, 11] > 0                             # proximity > 0: a box within 1.5 m
print("val RMSE [v, w]:", np.sqrt((err ** 2).mean(0)).round(4))
print("  box nearby  (%4.1f%% rows):" % (100 * near.mean()), np.sqrt((err[near] ** 2).mean(0)).round(4))
print("  no box near (%4.1f%% rows):" % (100 * (~near).mean()), np.sqrt((err[~near] ** 2).mean(0)).round(4))
