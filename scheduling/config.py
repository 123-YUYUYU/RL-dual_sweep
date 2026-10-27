"""Validated public simulation parameters, unrelated to research settings."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np


@dataclass
class Config:
    horizon: int = 60
    slot_seconds: float = 0.5
    arrival_rate: float = 2.5
    queue_limit: int = 64
    device_ghz: tuple = (1.0, 1.3, 1.6, 1.2)
    edge_ghz: tuple = (4.0, 6.0, 8.0)
    device_power_w: tuple = (0.4, 0.5, 0.6, 0.4)
    edge_model_rank: tuple = (0, 1, 1)
    # Nominal Qwen3 sizes in billions; a workload scaling input, not profiling.
    model_parameters_b: tuple = (0.6, 1.7)
    bandwidth_mhz: float = 2.0
    input_mbit: tuple = (0.3, 1.2)
    work_gcycles: tuple = (0.5, 2.5)
    deadline_slots: tuple = (4, 10)
    advanced_task_fraction: float = 0.25
    inner_iterations: int = 30
    inner_tolerance: float = 1e-4
    energy_weight: float = 0.1

    def validate(self):
        for name in ("horizon", "queue_limit", "inner_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("slot_seconds", "bandwidth_mhz", "inner_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("arrival_rate", "energy_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("device_ghz", "edge_ghz", "device_power_w"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"{name} must be a nonempty positive vector")
        if len(self.device_power_w) != len(self.device_ghz):
            raise ValueError("Each device needs a power budget")
        model_sizes = np.asarray(self.model_parameters_b, dtype=float)
        if (model_sizes.shape != (2,) or not np.all(np.isfinite(model_sizes))
                or not 0 < model_sizes[0] <= model_sizes[1]):
            raise ValueError("model_parameters_b must contain two increasing positive sizes")
        if len(self.edge_model_rank) != len(self.edge_ghz) or any(
            rank not in (0, 1) for rank in self.edge_model_rank
        ):
            raise ValueError("Each edge needs a model rank of 0 or 1")
        for name in ("input_mbit", "work_gcycles", "deadline_slots"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 < values[0] <= values[1]:
                raise ValueError(f"{name} must be an increasing positive pair")
        if any(int(x) != x for x in self.deadline_slots):
            raise ValueError("deadline_slots must contain integers")
        if not np.isfinite(self.advanced_task_fraction) or not 0 <= self.advanced_task_fraction <= 1:
            raise ValueError("advanced_task_fraction must be in [0, 1]")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def load_config(path=None):
    values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Config(**values).validate()
