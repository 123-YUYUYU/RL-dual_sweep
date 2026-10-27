"""Common CLI validation and artifact loading."""
import argparse
import json
from pathlib import Path

from .config import load_config
from .outer import DQN, SupervisedGuide


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return number


def training_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=nonnegative, default=7)
    parser.add_argument("--episodes", type=positive, default=20)
    parser.add_argument("--label-samples", type=positive, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--no-supervision", action="store_true", help="Train RL without supervised exploration suggestions")
    return parser


def load_run(path):
    path = Path(path)
    config = load_config(path / "config.json")
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return config, DQN.load(path / "policy.npz"), SupervisedGuide.load(path / "guide.npz"), metadata
