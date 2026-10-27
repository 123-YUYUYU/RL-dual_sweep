from pathlib import Path
import tempfile
import unittest

import numpy as np

from scheduling.config import Config, load_config
from scheduling.experiment import evaluate, train
from scheduling.outer import DQN, SupervisedGuide


class LearningTests(unittest.TestCase):
    def test_reward_updates_network_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(9)
        agent = DQN(3, 2, rng, hidden=8)
        observation = np.array([0.2, 0.5, 0.1])
        before = agent.values(observation)[1]
        for _ in range(32):
            agent.remember(observation, 1, 2.0, observation, True)
        for _ in range(40):
            self.assertTrue(np.isfinite(agent.train_batch(rng)))
        self.assertGreater(agent.values(observation)[1], before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            agent.save(path)
            np.testing.assert_array_equal(agent.values(observation), DQN.load(path).values(observation))

    def test_terminal_transitions_do_not_bootstrap(self):
        rng = np.random.default_rng(0)
        agent = DQN(2, 2, rng, hidden=4)
        for parameter in agent.parameters:
            parameter.fill(0)
        agent.target[-1].fill(100)
        agent.remember(np.zeros(2), 0, 0.0, np.zeros(2), True)
        self.assertEqual(agent.train_batch(rng, batch_size=1), 0.0)

    def test_guide_suggestion_and_serialization(self):
        guide = SupervisedGuide()
        x = np.array([[0, 0], [1, 1], [0.5, 0.5]])
        guide.fit(x, np.tile([-1.0, 2.0], (3, 1)))
        self.assertEqual(guide.suggest(x[0]), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.npz"
            guide.save(path)
            self.assertEqual(SupervisedGuide.load(path).suggest(x[0]), 1)

    def test_supervised_exploration_is_actually_consulted(self):
        class Guide:
            calls = 0
            def suggest(self, observation):
                self.calls += 1
                return 1
        rng = np.random.default_rng(1)
        agent, guide = DQN(2, 2, rng), Guide()
        for _ in range(30):
            agent.act(np.zeros(2), rng, epsilon=1.0, guide=guide)
        self.assertGreater(guide.calls, 0)
        guide.calls = 0
        agent.act(np.zeros(2), rng, epsilon=0.0, guide=guide)
        self.assertEqual(guide.calls, 0)

    def test_training_and_paired_evaluation_artifacts(self):
        config = Config(horizon=8, inner_iterations=5)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            agent, guide = train(config, directory, episodes=5, label_samples=4)
            self.assertGreater(agent.updates, 0)
            summary = evaluate(config, agent, guide, [41, 42], directory / "eval")
            self.assertEqual(len(summary), 5)
            self.assertTrue((directory / "eval/trace.csv").exists())
            self.assertTrue((directory / "eval/tasks.csv").exists())
            self.assertEqual(load_config(directory / "config.json").horizon, 8)
            restored = DQN.load(directory / "policy.npz")
            np.testing.assert_array_equal(restored.values(np.zeros(agent.parameters[0].shape[0])),
                                          agent.values(np.zeros(agent.parameters[0].shape[0])))


class ConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self):
        for values in (dict(horizon=0), dict(horizon=1.5), dict(arrival_rate=-1),
                       dict(slot_seconds=float("nan")), dict(device_ghz=[]),
                       dict(device_power_w=[1]), dict(deadline_slots=(2.5, 3)),
                       dict(advanced_task_fraction=2), dict(edge_model_rank=[0]),
                       dict(input_mbit=(2, 1)), dict(energy_weight=-1),
                       dict(model_parameters_b=(0.6,)), dict(model_parameters_b=(0.6, -1.7))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values).validate()

    def test_unknown_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"arrival_typo": 4}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
