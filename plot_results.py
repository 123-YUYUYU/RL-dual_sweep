"""Render training and evaluation figures from recorded public-demo CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes = read_csv(args.run / "evaluation/episodes.csv")
    trace = read_csv(args.run / "evaluation/trace.csv")
    history = read_csv(args.run / "training.csv")
    policies = list(dict.fromkeys(row["policy"] for row in episodes))
    colors = ["#64748b", "#06b6d4", "#a78bfa", "#f59e0b", "#2563eb"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, metric, title in zip(axes[0], ("completion_rate", "energy_j"),
                                 ("Completion rate (all arrivals)", "Total energy (J)")):
        groups = [[float(row[metric]) for row in episodes if row["policy"] == policy] for policy in policies]
        ax.bar(policies, [np.mean(group) for group in groups],
               yerr=[np.std(group) for group in groups], color=colors, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.15)
    for policy, color in zip(policies, colors):
        rows = [row for row in trace if row["policy"] == policy]
        slots = sorted({int(row["slot"]) for row in rows})
        groups = [[float(row["queue"]) for row in rows if int(row["slot"]) == slot] for slot in slots]
        axes[1, 0].plot(slots, [np.mean(group) for group in groups], label=policy, color=color)
    axes[1, 0].set_title("Queued and running tasks (test episode mean)")
    axes[1, 0].set_xlabel("Slot")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([int(row["episode"]) + 1 for row in history],
                    [float(row["reward"]) for row in history], color="#2563eb")
    axes[1, 1].set_title("Training episode reward")
    axes[1, 1].set_xlabel("Episode")
    fig.suptitle("Device-edge scheduling | synthetic simulation", fontsize=15, weight="bold")
    path = args.output or args.run / "figures/dashboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}; error bars are population std across test traces, not training seeds.")


if __name__ == "__main__":
    main()
