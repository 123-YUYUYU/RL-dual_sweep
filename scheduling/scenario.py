"""Pre-generated exogenous traces keep comparisons independent of actions."""
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Task:
    task_id: int
    device: int
    arrived: int
    deadline: int
    input_mbit: float
    work_gcycles: float
    required_rank: int
    node: int = -1
    model_rank: int = -1
    upload_left: float = 0.0
    compute_left: float = 0.0


@dataclass
class Trace:
    arrivals: list
    channel: np.ndarray
    background: np.ndarray


def generate_trace(config: Config, seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    devices, edges = len(config.device_ghz), len(config.edge_ghz)
    arrivals = []
    task_id = 0
    channel = np.empty((config.horizon, devices, edges))
    background = np.empty((config.horizon, edges))
    previous_channel = rng.uniform(3.0, 12.0, (devices, edges))
    previous_load = rng.uniform(0.05, 0.35, edges)
    for slot in range(config.horizon):
        # Bursty load and correlated channels/resource availability.
        rate = config.arrival_rate * (1.0 + 0.5 * np.sin(2 * np.pi * slot / 25))
        count = int(rng.poisson(rate))
        batch = []
        for _ in range(count):
            batch.append(Task(
                task_id, int(rng.integers(devices)), slot,
                slot + int(rng.integers(config.deadline_slots[0], config.deadline_slots[1] + 1)),
                float(rng.uniform(*config.input_mbit)), float(rng.uniform(*config.work_gcycles)),
                int(rng.random() < config.advanced_task_fraction),
            ))
            task_id += 1
        arrivals.append(batch)
        previous_channel = np.clip(0.85 * previous_channel + 0.15 * rng.uniform(2, 14, (devices, edges)), 1, 15)
        previous_load = np.clip(0.8 * previous_load + 0.2 * rng.uniform(0, 0.6, edges), 0, 0.8)
        channel[slot], background[slot] = previous_channel, previous_load
    return Trace(arrivals, channel, background)
