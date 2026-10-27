"""Public DQN and supervised guide, implemented with NumPy only."""
from pathlib import Path

import numpy as np


class SupervisedGuide:
    """Multi-output ridge regression predicts one-step action rewards."""

    def fit(self, observations, rewards):
        self.mean = observations.mean(axis=0)
        self.scale = np.maximum(observations.std(axis=0), 0.05)
        features = self._features(observations)
        penalty = 0.1 * np.eye(features.shape[1])
        penalty[0, 0] = 1e-6
        self.weights = np.linalg.solve(features.T @ features + penalty, features.T @ rewards)

    def _features(self, observations):
        z = (np.asarray(observations) - self.mean) / self.scale
        if z.ndim == 1:
            return np.concatenate(([1.0], z))
        return np.column_stack((np.ones(len(z)), z))

    def suggest(self, observation):
        return int(np.argmax(self._features(observation) @ self.weights))

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            obj.mean, obj.scale, obj.weights = (data[key].copy() for key in ("mean", "scale", "weights"))
        return obj


class DQN:
    """Two-layer Q network with replay, target network and Adam updates.

    This general-purpose reference agent deliberately does not reproduce the
    research policy, feature engineering, reward calibration or training stack.
    """

    def __init__(self, observation_size, actions, rng, hidden=48):
        self.actions = actions
        self.parameters = [
            rng.normal(0, np.sqrt(2 / observation_size), (observation_size, hidden)),
            np.zeros(hidden), rng.normal(0, 0.05, (hidden, actions)), np.zeros(actions),
        ]
        self.target = [value.copy() for value in self.parameters]
        self.first_moment = [np.zeros_like(value) for value in self.parameters]
        self.second_moment = [np.zeros_like(value) for value in self.parameters]
        self.updates = 0
        self.replay = []
        self.replay_position = 0

    def values(self, observations, target=False):
        w1, b1, w2, b2 = self.target if target else self.parameters
        return np.maximum(np.asarray(observations) @ w1 + b1, 0) @ w2 + b2

    def act(self, observation, rng, epsilon=0.0, guide=None):
        if rng.random() < epsilon:
            if guide is not None and rng.random() < 0.5:
                return guide.suggest(observation)
            return int(rng.integers(self.actions))
        return int(np.argmax(self.values(observation)))

    def remember(self, observation, action, reward, next_observation, terminal):
        item = (observation.copy(), action, reward, next_observation.copy(), terminal)
        if len(self.replay) < 5000:
            self.replay.append(item)
        else:
            self.replay[self.replay_position] = item
        self.replay_position = (self.replay_position + 1) % 5000

    def train_batch(self, rng, batch_size=32):
        if len(self.replay) < batch_size:
            return None
        indices = rng.choice(len(self.replay), batch_size, replace=False)
        batch = [self.replay[i] for i in indices]
        x = np.stack([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_x = np.stack([item[3] for item in batch])
        terminal = np.array([item[4] for item in batch], dtype=float)
        # Double-DQN selection/evaluation; terminal transitions never bootstrap.
        next_actions = np.argmax(self.values(next_x), axis=1)
        target = rewards + 0.95 * (1 - terminal) * self.values(next_x, target=True)[np.arange(batch_size), next_actions]
        w1, b1, w2, b2 = self.parameters
        pre_activation = x @ w1 + b1
        hidden = np.maximum(pre_activation, 0)
        prediction = hidden @ w2 + b2
        error = prediction[np.arange(batch_size), actions] - target
        loss = np.mean(np.where(np.abs(error) <= 1, 0.5 * error**2, np.abs(error) - 0.5))
        grad_output = np.zeros_like(prediction)
        grad_output[np.arange(batch_size), actions] = np.clip(error, -1, 1) / batch_size
        grad_hidden = (grad_output @ w2.T) * (pre_activation > 0)
        gradients = [x.T @ grad_hidden, grad_hidden.sum(axis=0), hidden.T @ grad_output, grad_output.sum(axis=0)]
        norm = np.sqrt(sum(np.sum(grad**2) for grad in gradients))
        self.updates += 1
        for i, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            gradient = gradient / max(1.0, norm / 5)
            self.first_moment[i] = 0.9 * self.first_moment[i] + 0.1 * gradient
            self.second_moment[i] = 0.999 * self.second_moment[i] + 0.001 * gradient**2
            moment = self.first_moment[i] / (1 - 0.9**self.updates)
            variance = self.second_moment[i] / (1 - 0.999**self.updates)
            parameter -= 0.001 * moment / (np.sqrt(variance) + 1e-8)
        if self.updates % 50 == 0:
            self.target = [value.copy() for value in self.parameters]
        return float(loss)

    def save(self, path):
        # Inference checkpoint only; optimizer/replay are intentionally not saved.
        np.savez(Path(path), **{f"p{i}": value for i, value in enumerate(self.parameters)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            parameters = [data[f"p{i}"].copy() for i in range(4)]
        obj = cls(parameters[0].shape[0], parameters[3].size, np.random.default_rng(0), parameters[0].shape[1])
        obj.parameters = parameters
        obj.target = [value.copy() for value in parameters]
        return obj
