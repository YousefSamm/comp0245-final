import numpy as np, glob, os
# Dataset lives in robotnik_arm_task/data/raw (one level up from this script)
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(TASK_DIR, "data", "raw", "*.npz")))
ok = 0
for f in files:
    d = np.load(f)
    ok += bool(d["success"])
    print(f"{f[-12:]}  success={bool(d['success'])}  err={float(d['final_error']):.4f}  T={d['t'][-1]:.1f}s  nav={float(d['nav_time']):.1f}s")
print(f"\n{ok}/{len(files)} succeeded")
