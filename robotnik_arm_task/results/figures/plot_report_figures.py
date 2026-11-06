"""Figures for the report, made from the per-episode result tables in results/.

    python results/figures/plot_report_figures.py      (from robotnik_arm_task/)

final_error_cdf_494.png : share of the 494 test episodes within each error threshold for the three final
                          ArmNet versions. Solid = closest approach to the target (what success uses),
                          dashed = error at the end of the episode.
obstacle_results.png    : collisions and success within 2 cm in the 95 obstacle scenes for each BaseNet
                          (all trained on single- and multi-box demonstrations),
                          with the expert's collision rate as a dotted line.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.dirname(FIG_DIR)
EXPERT_COLLIDED = 6.3   # % of the 95 obstacle test scenes in which the expert touched a box

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})


def final_error_cdf():
    versions = [("dagger3", "DAgger Round 3", "#1f77b4"),
                ("dagger4", "DAgger Round 4, no residual", "#ff7f0e"),
                ("dagger4_gravres", "DAgger Round 4 (Gravity Res.)", "#2ca02c")]
    th = np.logspace(np.log10(0.3), 1, 400)   # cm
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for tag, label, c in versions:
        x = pd.read_csv(os.path.join(RES_DIR, f"test_both_{tag}.csv"))
        best, final = x.best_err.values * 100, x.final_err.values * 100
        ax.plot(th, [100 * (best <= t).mean() for t in th], color=c, lw=1.3, label=label)
        ax.plot(th, [100 * (final <= t).mean() for t in th], color=c, lw=0.9, ls="--")
    for t in (1, 2, 5):
        ax.axvline(t, color="0.8", lw=0.6, zorder=0)
    ax.plot([], [], color="k", lw=1.3, label="closest approach (success)")
    ax.plot([], [], color="k", lw=0.9, ls="--", label="final error")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xlim(0.3, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Error threshold (cm)")
    ax.set_ylabel("Episodes within threshold (%)")
    ax.legend(frameon=True, framealpha=1, edgecolor="none", fontsize=6.3, loc="lower right",
              handlelength=1.8, bbox_to_anchor=(1.0, 0.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "final_error_cdf_494.png"), dpi=300)
    plt.close(fig)


def obstacle_results():
    rows = [("nolidar", "No\nLiDAR"), ("multi", "2 front\nfeatures"), ("sect8", "8 sectors\n(360\u00b0)"),
            ("daggernear", "2 features\n+ DAgger"), ("corr", "Corridor\nfeatures")]
    collided, s2, s2_clean = [], [], []
    for tag, _ in rows:
        x = pd.read_csv(os.path.join(RES_DIR, f"obs_obs_{tag}.csv"))
        c, s = x.collided.astype(bool), x.success_2cm.astype(bool)
        collided.append(100 * c.mean())
        s2.append(100 * s.mean())
        s2_clean.append(100 * (s & ~c).mean())
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    i, w = np.arange(len(rows)), 0.27
    bars = [(collided, "Collided", "#d62728"), (s2, "Success @2 cm", "#1f77b4"),
            (s2_clean, "@2 cm, no collision", "#2ca02c")]
    for k, (vals, label, c) in enumerate(bars):
        b = ax.bar(i + (k - 1) * w, vals, w, label=label, color=c)
        for r, val in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=5)
    ax.axhline(EXPERT_COLLIDED, color="#d62728", lw=0.7, ls=":", label=f"Expert collided ({EXPERT_COLLIDED} %)")
    ax.set_xticks(i)
    ax.set_xticklabels([r[1] for r in rows], fontsize=6.5)
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obstacle_results.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    final_error_cdf()
    obstacle_results()
    print("saved final_error_cdf_494.png and obstacle_results.png in", FIG_DIR)
