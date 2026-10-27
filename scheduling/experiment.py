"""Training and paired-trace evaluation utilities shared by CLI entry points."""
import csv
import json
import platform
from pathlib import Path

import numpy as np

from .environment import EdgeSchedulingEnv
from .outer import DQN, SupervisedGuide


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_seeds(seed):
    # Each stream owns a separate seed subtree: labels, RL, initialization, test.
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]


def collect_labels(config, samples, rng, output=None):
    env = EdgeSchedulingEnv(config)
    observation, _ = env.reset(seed=int(rng.integers(2**31)))
    observations, rewards = [], []
    for sample in range(samples):
        action_rewards = []
        for action in range(env.n_actions):
            trial = env.fork()
            _, reward, _, _, _ = trial.step(action)
            # One-step supervision is a public surrogate, not an optimal oracle.
            action_rewards.append(reward)
        observations.append(observation.copy())
        rewards.append(action_rewards)
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(env.n_actions)))
        if terminated or truncated:
            observation, _ = env.reset(seed=int(rng.integers(2**31)))
    observations, rewards = np.stack(observations), np.array(rewards)
    if output is not None:
        np.savez(output, observations=observations, action_rewards=rewards)
    guide = SupervisedGuide()
    guide.fit(observations, rewards)
    return guide


def train(config, output, seed=7, episodes=20, label_samples=64, assisted=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    label_rng, train_rng, init_rng, _ = split_seeds(seed)
    print(f"Generating {label_samples} supervised states...", flush=True)
    guide = collect_labels(config, label_samples, label_rng, output / "supervision.npz")
    guide.save(output / "guide.npz")
    env = EdgeSchedulingEnv(config)
    agent = DQN(len(env.observation()), env.n_actions, init_rng)
    history = []
    episode_seeds = train_rng.choice(2**31, size=episodes, replace=False)
    for episode in range(episodes):
        episode_seed = int(episode_seeds[episode])
        observation, _ = env.reset(seed=episode_seed)
        epsilon = max(0.1, 1.0 - episode / max(1, episodes - 1))
        losses = []
        while not env.done:
            action = agent.act(observation, train_rng, epsilon, guide if assisted else None)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.remember(observation, action, reward, next_observation, terminated or truncated)
            loss = agent.train_batch(train_rng)
            if loss is not None:
                losses.append(loss)
            observation = next_observation
        history.append(dict(episode=episode, seed=episode_seed, epsilon=epsilon,
                            td_loss=float(np.mean(losses)) if losses else None, **env.summary()))
        if episode == 0 or (episode + 1) % 5 == 0 or episode + 1 == episodes:
            print(f"Episode {episode+1}/{episodes}: completion={env.summary()['completion_rate']:.2f}, "
                  f"reward={env.total['reward']:.2f}", flush=True)
    agent.save(output / "policy.npz")
    write_csv(output / "training.csv", history)
    metadata = dict(seed=seed, episodes=episodes, label_samples=label_samples, assisted=assisted,
                    episode_seeds=[int(value) for value in episode_seeds],
                    observation_size=len(env.observation()), actions=env.n_actions,
                    checkpoint_type="inference_only", numpy_version=np.__version__,
                    python_version=platform.python_version(), schema_version=3)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return agent, guide


def evaluate(config, agent, guide, seeds, output):
    output = Path(output)
    probe = EdgeSchedulingEnv(config)
    if agent.parameters[0].shape[0] != len(probe.observation()) or agent.actions != probe.n_actions:
        raise ValueError("Policy observation/action dimensions do not match this environment")
    if guide.weights.shape != (len(probe.observation()) + 1, probe.n_actions):
        raise ValueError("Guide dimensions do not match this environment")
    episodes, trace_rows, events = [], [], []
    names = ("local_first", "balanced", "edge_first", "supervised", "rl")
    for seed in seeds:
        for name in names:
            env = EdgeSchedulingEnv(config)
            observation, _ = env.reset(seed=int(seed))
            steps = []
            while not env.done:
                if name == "local_first":
                    # A compatible edge remains available for advanced requests.
                    action = env.encode(env.edges, 0, 0)
                elif name == "balanced":
                    action = env.encode(env.edges, 1, 0)
                elif name == "edge_first":
                    action = env.encode(env.edges, 2, 0)
                elif name == "supervised":
                    action = guide.suggest(observation)
                else:
                    action = int(np.argmax(agent.values(observation)))
                observation, _, _, _, info = env.step(action)
                row = dict(policy=name, seed=int(seed), action=action, **info)
                steps.append(row)
                trace_rows.append(row)
            episodes.append(dict(policy=name, seed=int(seed), **env.summary(),
                                 mean_queue=float(np.mean([row["queue"] for row in steps])),
                                 mean_active_edges=float(np.mean([row["active_edges"] for row in steps])),
                                 inner_converged_fraction=float(np.mean([row["inner_converged"] for row in steps])),
                                 max_compute_violation=max(row["compute_violation"] for row in steps),
                                 max_power_violation=max(row["power_violation"] for row in steps)))
            # Homogeneous event schema, including rejected and unfinished tasks.
            all_events = env.events + [dict(task_id=task.task_id, status="pending", time_s=config.horizon * config.slot_seconds) for task in env.tasks]
            for event in all_events:
                events.append(dict(policy=name, seed=int(seed), **{
                    key: event.get(key) for key in ("task_id", "status", "node", "model_rank", "required_rank", "time_s", "latency_s")
                }))
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "trace.csv", trace_rows)
    if events:
        write_csv(output / "tasks.csv", events)
    summary = []
    metrics = ("completion_rate", "deadline_miss_rate", "overflow_rate", "energy_j", "mean_latency_s",
               "mean_queue", "mean_active_edges", "inner_converged_fraction")
    for name in names:
        selected = [row for row in episodes if row["policy"] == name]
        aggregate = dict(policy=name, episodes=len(selected))
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            aggregate[metric] = float(np.mean(values)) if values else None
            aggregate[metric + "_std"] = float(np.std(values)) if values else None
        summary.append(aggregate)
        print(f"{name:14s} completion={aggregate['completion_rate']:.3f} "
              f"miss={aggregate['deadline_miss_rate']:.3f} energy={aggregate['energy_j']:.2f} J")
    write_csv(output / "summary.csv", summary)
    (output / "evaluation.json").write_text(json.dumps(dict(seeds=[int(seed) for seed in seeds],
        aggregation="unweighted episode means; std uses ddof=0; latency excludes episodes with no completions"), indent=2) + "\n", encoding="utf-8")
    return summary
