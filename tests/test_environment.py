import unittest

import numpy as np

from scheduling.config import Config
from scheduling.environment import EdgeSchedulingEnv
from scheduling.inner import allocate
from scheduling.scenario import Task


class EnvironmentTests(unittest.TestCase):
    def test_trace_is_reproducible_and_action_independent(self):
        first, second = EdgeSchedulingEnv(), EdgeSchedulingEnv()
        first.reset(13)
        second.reset(13)
        for _ in range(8):
            first.step(first.encode(0, 0))
            second.step(second.encode(second.edges, 2))
        np.testing.assert_array_equal(first.trace.channel, second.trace.channel)
        np.testing.assert_array_equal(first.trace.background, second.trace.background)
        self.assertEqual([[task.work_gcycles for task in batch] for batch in first.trace.arrivals],
                         [[task.work_gcycles for task in batch] for batch in second.trace.arrivals])
        self.assertNotEqual(first.observation().tolist(), second.observation().tolist())

    def test_task_accounting_and_capacity_over_entire_episode(self):
        env = EdgeSchedulingEnv(Config(horizon=20, arrival_rate=5, queue_limit=8, inner_iterations=10))
        env.reset(18)
        rng = np.random.default_rng(2)
        while not env.done:
            observation, reward, _, _, info = env.step(int(rng.integers(env.n_actions)))
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertTrue(np.isfinite(reward))
            self.assertLessEqual(info["compute_violation"], 1e-9)
            self.assertLessEqual(info["power_violation"], 1e-9)
            self.assertLessEqual(info["inner_objective_after"], info["inner_objective_before"] + 1e-8)
            self.assertLessEqual(len(env.tasks), env.config.queue_limit)
        total = env.summary()
        self.assertEqual(total["arrived"], sum(total[key] for key in ("completed", "missed", "overflow", "pending")))
        self.assertEqual(len(env.events) + len(env.tasks), total["arrived"])
        self.assertGreater(total["overflow"], 0)

    def test_upload_and_compute_are_sequential_and_stateful(self):
        env = EdgeSchedulingEnv(Config(arrival_rate=0, horizon=6, slot_seconds=0.5))
        env.tasks = [Task(0, 0, 0, 6, 0.01, 0.1, 1)]
        env.total["arrived"] = 1
        env.step(env.encode(env.edges, 2))
        self.assertEqual(len(env.tasks), 1)
        task = env.tasks[0]
        self.assertEqual(task.upload_left, 0)
        self.assertAlmostEqual(task.compute_left, 0.1 * env.config.model_parameters_b[1] / env.config.model_parameters_b[0])
        node = task.node
        # Once assigned, a task cannot migrate or have its model changed.
        _, _, _, _, info = env.step(env.encode(0, 0))
        self.assertEqual(info["completed"], 1)
        self.assertGreaterEqual(info["active_edges"], 1)
        self.assertEqual(env.events[-1]["node"], node)
        self.assertEqual(env.events[-1]["model_rank"], 1)

    def test_deadlines_drop_unserviceable_tasks(self):
        env = EdgeSchedulingEnv(Config(arrival_rate=0, horizon=3))
        env.tasks = [Task(0, 0, 0, 1, 1.0, 2.0, 1)]
        env.total["arrived"] = 1
        _, reward, _, _, info = env.step(env.encode(0, 0))
        self.assertEqual(info["missed"], 1)
        self.assertFalse(env.tasks)
        self.assertLess(reward, 0)

    def test_model_compatibility_and_device_ownership(self):
        env = EdgeSchedulingEnv(Config(arrival_rate=0))
        env.tasks = [Task(0, 2, 0, 10, 1.0, 1.0, 0), Task(1, 1, 0, 10, 1.0, 1.0, 1)]
        env._dispatch(env.decode(env.encode(env.edges, 0)))
        self.assertEqual(env.tasks[0].node, 2)
        advanced = env.tasks[1]
        self.assertGreaterEqual(advanced.node, env.devices)
        self.assertGreaterEqual(env.config.edge_model_rank[advanced.node - env.devices], advanced.required_rank)

    def test_fork_never_mutates_live_state(self):
        env = EdgeSchedulingEnv()
        observation, _ = env.reset(8)
        original_tasks = repr(env.tasks)
        original_totals = env.total.copy()
        clone = env.fork()
        clone.step(clone.encode(clone.edges, 2))
        np.testing.assert_array_equal(env.observation(), observation)
        self.assertEqual(repr(env.tasks), original_tasks)
        self.assertEqual(env.total, original_totals)
        self.assertEqual(env.time, 0)

    def test_empty_episode_and_terminal_guard(self):
        env = EdgeSchedulingEnv(Config(horizon=1, arrival_rate=0))
        _, reward, terminated, truncated, info = env.step(env.encode(0))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0)
        self.assertTrue(info["inner_converged"])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_invalid_actions_do_not_advance_time(self):
        env = EdgeSchedulingEnv()
        for action in (-1, env.n_actions, 1.5, True):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.time, 0)
        with self.assertRaises(ValueError):
            env.encode(1.5)

    def test_cpu_progress_and_energy_use_consistent_units(self):
        env = EdgeSchedulingEnv(Config(arrival_rate=0, horizon=3, device_ghz=(1.0,),
                                       device_power_w=(0.4,), slot_seconds=0.5))
        env.tasks = [Task(0, 0, 0, 3, 0.0, 2.0, 0)]
        env.total["arrived"] = 1
        _, _, _, _, info = env.step(env.encode(0, 0))
        self.assertAlmostEqual(env.tasks[0].compute_left, 1.5)
        self.assertAlmostEqual(info["energy_j"], 0.06 * 0.5)

    def test_shared_transmit_budget_is_respected(self):
        config = Config()
        env = EdgeSchedulingEnv(config)
        env.tasks = [Task(i, 0, 0, 10, 1.0, 2.0, 1) for i in range(10)]
        env._dispatch(env.decode(env.encode(env.edges, 2)))
        allocation = allocate(env.tasks, config, env.capacities(), env.trace.channel[0])
        self.assertLessEqual(allocation.power.sum(), config.device_power_w[0] + 1e-9)
        self.assertTrue(np.all(allocation.rate > 0))


if __name__ == "__main__":
    unittest.main()
