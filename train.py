"""Train a public reference DQN with optional supervised exploration."""
from scheduling.cli import training_parser
from scheduling.config import load_config
from scheduling.experiment import train


def main():
    args = training_parser(__doc__).parse_args()
    train(load_config(args.config), args.output, args.seed, args.episodes,
          args.label_samples, not args.no_supervision)
    print(f"Saved inference checkpoints and training records to {args.output}")


if __name__ == "__main__":
    main()
