"""Where does the final model fail? Success within 2 cm on the 494 test episodes against target height and
horizontal distance, with Fisher's exact test for low targets (z < 0.3 m) against the rest.

    python yousef/analyze_failures.py                       (from robotnik_arm_task/)
    python yousef/analyze_failures.py --tag dagger3         (another evaluated model)

Reads results/test_both_<tag>.csv and the targets in data/raw/ep_<id>.npz.
"""
import os, sys, argparse
from math import comb
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="dagger4_gravres")
args = parser.parse_args()


def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda k: comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
    p0 = p(a)
    return sum(p(k) for k in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(k) <= p0 + 1e-12)


x = pd.read_csv(os.path.join(TASK_DIR, "results", f"test_both_{args.tag}.csv")).set_index("ep")
tg = {ep: np.load(os.path.join(TASK_DIR, "data", "raw", f"ep_{ep:05d}.npz"))["target"] for ep in x.index}
x["z"] = [tg[e][2] for e in x.index]
x["r"] = [np.linalg.norm(tg[e][:2]) for e in x.index]
x["fail"] = ~x.success_2cm.astype(bool)

pd.set_option("display.precision", 1)
print(f"{args.tag}: {len(x)} episodes, {int(x.fail.sum())} miss 2 cm "
      f"({int(((x.best_err >= .02) & (x.best_err < .05)).sum())} within 2-5 cm, {int((x.best_err >= .05).sum())} beyond 5 cm, "
      f"{int(x.blew_up.sum())} blew up)\n")
for col, bins, name in [("z", [0.1, 0.3, 0.5, 0.8, 1.1, 1.25], "target height (m)"),
                        ("r", [0.8, 1.1, 1.4, 1.7, 2.0], "target distance from origin (m)")]:
    g = x.groupby(pd.cut(x[col], bins, include_lowest=True), observed=True)
    print(name); print(pd.DataFrame({"episodes": g.size(), "miss 2 cm (%)": 100 * g.fail.mean()}), "\n")
low = x.z < 0.3
a, b, c, d = int(x.fail[low].sum()), int((~x.fail[low]).sum()), int(x.fail[~low].sum()), int((~x.fail[~low]).sum())
print(f"z < 0.3 m: {a}/{a + b} miss ({100 * a / (a + b):.1f} %) vs {c}/{c + d} ({100 * c / (c + d):.1f} %) above, "
      f"Fisher p = {fisher(a, b, c, d):.2g}")
