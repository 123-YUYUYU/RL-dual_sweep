"""Evaluate saved models and baselines on the same held-out traces."""
import argparse
from pathlib import Path

from scheduling.cli import load_run, positive
from scheduling.experiment import evaluate, split_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--episodes", type=positive, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, agent, guide, metadata = load_run(args.run)
    rng = split_seeds(metadata["seed"])[3]
    seeds = rng.choice(2**31, size=args.episodes, replace=False)
    evaluate(config, agent, guide, seeds, args.output or args.run / "evaluation")


if __name__ == "__main__":
    main()
