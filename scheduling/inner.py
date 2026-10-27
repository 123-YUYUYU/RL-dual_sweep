"""Public projected-gradient resource allocator; no specialized research solver."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    frequency: np.ndarray
    power: np.ndarray
    rate: np.ndarray
    iterations: int
    converged: bool
    objective_before: float
    objective_after: float
    compute_violation: float
    power_violation: float


def project(values, budget, floor=1e-4):
    """Euclidean projection onto a capacity simplex with positive shares."""
    if len(values) == 0:
        return values.copy()
    floor = min(floor, budget / (2 * len(values)))
    remaining = budget - floor * len(values)
    shifted = values - floor
    if np.maximum(shifted, 0).sum() <= remaining:
        return np.maximum(shifted, 0) + floor
    ordered = np.sort(shifted)[::-1]
    offsets = (np.cumsum(ordered) - remaining) / np.arange(1, len(values) + 1)
    index = np.flatnonzero(ordered > offsets)[-1]
    return np.maximum(shifted - offsets[index], 0) + floor


def _descent(initial, groups, budgets, evaluate, max_iter, tolerance):
    x = initial.copy()
    if not len(x):
        return x, 0, True, 0.0, 0.0
    start, _ = evaluate(x)
    converged = False
    for iteration in range(1, max_iter + 1):
        value, gradient = evaluate(x)
        # Stopping uses a unit-step projected gradient residual, independent
        # of the backtracking step, so a tiny trial step is not convergence.
        projected = x.copy()
        for indices, budget in zip(groups, budgets):
            projected[indices] = project(x[indices] - gradient[indices], budget)
        if np.max(np.abs(projected - x)) <= tolerance:
            converged = True
            break
        step = 1.0
        for _ in range(35):
            trial = x.copy()
            for indices, budget in zip(groups, budgets):
                trial[indices] = project(x[indices] - step * gradient[indices], budget)
            trial_value, _ = evaluate(trial)
            if trial_value <= value + 1e-4 * float(gradient @ (trial - x)):
                x = trial
                break
            step *= 0.5
        else:
            break
    return x, iteration, converged, start, evaluate(x)[0]


def allocate(tasks, config, capacity, channel):
    """Allocate CPU GHz and transmit W under per-node/per-device budgets.

    Upload and computation are separate stages. Radio bandwidth is shared
    equally between current uploads; power and CPU shares are optimized.
    """
    n = len(tasks)
    frequency, power, rate = (np.zeros(n) for _ in range(3))
    devices = len(config.device_ghz)
    compute = np.array([i for i, task in enumerate(tasks) if task.node >= 0 and task.upload_left <= 1e-9], dtype=int)
    uploads = np.array([i for i, task in enumerate(tasks) if task.node >= devices and task.upload_left > 1e-9], dtype=int)
    nodes = np.array([tasks[i].node for i in compute], dtype=int)
    owners = np.array([tasks[i].device for i in uploads], dtype=int)
    compute_groups = [np.flatnonzero(nodes == node) for node in range(len(capacity))]
    radio_groups = [np.flatnonzero(owners == device) for device in range(devices)]
    initial_cpu = np.zeros(len(compute))
    initial_power = np.zeros(len(uploads))
    for group, budget in zip(compute_groups, capacity):
        if len(group):
            initial_cpu[group] = budget / len(group)
    for group, budget in zip(radio_groups, config.device_power_w):
        if len(group):
            initial_power[group] = budget / len(group)
    work = np.array([tasks[i].compute_left for i in compute])
    bits = np.array([tasks[i].upload_left for i in uploads])
    gains = np.array([channel[tasks[i].device, tasks[i].node - devices] for i in uploads])
    # Orthogonal per-device channels; no inter-device interference in this model.
    bandwidth = np.array([config.bandwidth_mhz / np.sum(owners == owner) for owner in owners])

    def cpu_objective(x):
        return float(np.sum(work / x + config.energy_weight * x**2)), -work / x**2 + 2 * config.energy_weight * x

    def radio_objective(p):
        rates = bandwidth * np.log2(1 + gains * p)
        derivative = bandwidth * gains / ((1 + gains * p) * np.log(2))
        return (float(np.sum(bits / rates + config.energy_weight * p**2)),
                -bits * derivative / rates**2 + 2 * config.energy_weight * p)

    cpu, ci, cc, cb, ca = _descent(initial_cpu, compute_groups, capacity, cpu_objective,
                                  config.inner_iterations, config.inner_tolerance)
    radio, pi, pc, pb, pa = _descent(initial_power, radio_groups, config.device_power_w,
                                    radio_objective, config.inner_iterations, config.inner_tolerance)
    frequency[compute], power[uploads] = cpu, radio
    rate[uploads] = bandwidth * np.log2(1 + gains * radio)
    cpu_violation = max([0.0] + [float(cpu[group].sum() - budget) for group, budget in zip(compute_groups, capacity)])
    power_violation = max([0.0] + [float(radio[group].sum() - budget) for group, budget in zip(radio_groups, config.device_power_w)])
    return Allocation(frequency, power, rate, max(ci, pi), cc and pc, cb + pb, ca + pa,
                      cpu_violation, power_violation)
