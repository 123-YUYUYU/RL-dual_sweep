"""Run supervised labeling, RL training and paired evaluation end to end."""
from scheduling.cli import positive, training_parser
from scheduling.config import load_config
from scheduling.experiment import evaluate, split_seeds, train


def main():
    parser = training_parser(__doc__)
    parser.add_argument("--eval-episodes", type=positive, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    agent, guide = train(config, args.output, args.seed, args.episodes,
                         args.label_samples, not args.no_supervision)
    rng = split_seeds(args.seed)[3]
    seeds = rng.choice(2**31, size=args.eval_episodes, replace=False)
    evaluate(config, agent, guide, seeds, args.output / "evaluation")
    print(f"Artifacts saved to {args.output}")


if __name__ == "__main__":
    main()
