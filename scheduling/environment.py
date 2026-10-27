"""Finite-horizon, stateful device-edge inference scheduling simulator."""
import copy
from dataclasses import dataclass

import numpy as np

from .config import Config
from .inner import allocate
from .scenario import generate_trace


@dataclass(frozen=True)
class Decision:
    edge_budget: int
    preference: int  # local, balanced, edge
    model_rank: int


class EdgeSchedulingEnv:
    """reset/step interface with observation, reward and episode statistics.

    Node IDs [0, D) are devices; [D, D+E) are edges. Tasks retain their
    destination after admission and pass through upload then compute stages.
    """

    def __init__(self, config=None):
        self.config = (config or Config()).validate()
        self.devices = len(self.config.device_ghz)
        self.edges = len(self.config.edge_ghz)
        self.n_actions = (self.edges + 1) * 3 * 2
        self.reset(seed=0)

    def decode(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError(f"action must be an integer in [0, {self.n_actions})")
        return Decision(int(action) // 6, (int(action) // 2) % 3, int(action) % 2)

    def encode(self, edge_budget, preference=1, model_rank=0):
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
               for value in (edge_budget, preference, model_rank)):
            raise ValueError("Decision components must be integers")
        if not 0 <= edge_budget <= self.edges or preference not in (0, 1, 2) or model_rank not in (0, 1):
            raise ValueError("Invalid decision components")
        return int(edge_budget * 6 + preference * 2 + model_rank)

    def reset(self, seed=0):
        self.trace = generate_trace(self.config, seed)
        self.time = 0
        self.tasks = []
        self.events = []
        self.latencies = []
        self.total = dict(arrived=0, completed=0, missed=0, overflow=0, energy_j=0.0, reward=0.0)
        self._overflow_pending = 0
        self.done = False
        self._spawn()
        return self.observation(), {"seed": seed}

    def fork(self):
        """Copy endogenous state while sharing the read-only exogenous trace."""
        clone = copy.copy(self)
        clone.tasks = copy.deepcopy(self.tasks)
        clone.total = self.total.copy()
        clone.events = self.events.copy()
        clone.latencies = self.latencies.copy()
        return clone

    def _spawn(self):
        batch = self.trace.arrivals[self.time]
        self.total["arrived"] += len(batch)
        for template in batch:
            if len(self.tasks) >= self.config.queue_limit:
                self.total["overflow"] += 1
                self._overflow_pending += 1
                self.events.append(dict(task_id=template.task_id, status="overflow", time_s=self.time * self.config.slot_seconds))
            else:
                self.tasks.append(copy.copy(template))

    def capacities(self):
        slot = min(self.time, self.config.horizon - 1)
        return np.concatenate((self.config.device_ghz, np.asarray(self.config.edge_ghz) * (1 - self.trace.background[slot])))

    def observation(self):
        capacity = self.capacities()
        slot = min(self.time, self.config.horizon - 1)
        queued = np.zeros(self.devices)
        node_work = np.zeros(self.devices + self.edges)
        for task in self.tasks:
            if task.node < 0:
                queued[task.device] += 1
            else:
                node_work[task.node] += task.compute_left
        slack = np.mean([task.deadline - self.time for task in self.tasks]) if self.tasks else 0
        features = [
            self.time / self.config.horizon,
            len(self.tasks) / self.config.queue_limit,
            slack / max(self.config.deadline_slots),
            sum(task.upload_left for task in self.tasks) / 20,
            sum(task.compute_left if task.node >= 0 else task.work_gcycles for task in self.tasks) / 30,
            self.trace.channel[slot].mean() / 15,
            sum(task.required_rank for task in self.tasks) / max(1, len(self.tasks)),
        ]
        return np.clip(np.concatenate((features, node_work / capacity / 5,
                                       self.trace.background[slot], queued / self.config.queue_limit)), 0, 5).astype(float)

    def _dispatch(self, decision):
        capacity = self.capacities()
        channel = self.trace.channel[self.time]
        occupied = {task.node - self.devices for task in self.tasks if task.node >= self.devices}
        active = set(occupied)
        ranking = np.argsort(-capacity[self.devices:] * channel.mean(axis=0))
        for edge in ranking:
            if len(active) >= decision.edge_budget:
                break
            active.add(int(edge))
        backlog = np.zeros(len(capacity))
        for task in self.tasks:
            if task.node >= 0:
                backlog[task.node] += task.compute_left
        for task in sorted(self.tasks, key=lambda item: (item.deadline, item.task_id)):
            if task.node >= 0:
                continue
            rank = max(task.required_rank, decision.model_rank)
            model_scale = self.config.model_parameters_b[rank] / self.config.model_parameters_b[0]
            work = task.work_gcycles * model_scale
            candidates = []
            if rank == 0:  # devices host only the lightweight model
                local_cost = (backlog[task.device] + work) / capacity[task.device]
                local_bias = (0.0, 0.2, 2.0)[decision.preference]
                candidates.append((local_cost + local_bias, task.device))
            for edge in sorted(active):
                if self.config.edge_model_rank[edge] < rank:
                    continue
                node = self.devices + edge
                upload = task.input_mbit / (self.config.bandwidth_mhz * np.log2(1 + channel[task.device, edge] * self.config.device_power_w[task.device]))
                edge_bias = (5.0, 0.0, -0.5)[decision.preference]
                score = upload + (backlog[node] + work) / capacity[node] + edge_bias
                candidates.append((score, node))
            if not candidates:
                continue  # remain queued until a compatible resource is activated
            _, node = min(candidates)
            task.node, task.model_rank = node, rank
            task.upload_left = task.input_mbit if node >= self.devices else 0.0
            task.compute_left = work
            backlog[node] += work
        return active

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; call reset before step")
        decision = self.decode(action)
        active = self._dispatch(decision)
        allocation = allocate(self.tasks, self.config, self.capacities(), self.trace.channel[self.time])
        dt = self.config.slot_seconds
        energy = 0.3 * len(active) * dt  # synthetic edge idle power, W
        completed, missed, offloaded = 0, 0, 0
        survivors = []
        for i, task in enumerate(self.tasks):
            finish_offset = dt
            if task.node >= 0:
                if task.upload_left > 1e-9:
                    used = min(dt, task.upload_left / max(allocation.rate[i], 1e-12))
                    task.upload_left = max(0.0, task.upload_left - allocation.rate[i] * dt)
                    energy += allocation.power[i] * used
                else:
                    frequency = allocation.frequency[i]
                    used = min(dt, task.compute_left / max(frequency, 1e-12))
                    task.compute_left = max(0.0, task.compute_left - frequency * dt)
                    coefficient = 0.06 if task.node < self.devices else 0.025
                    energy += coefficient * frequency**3 * used
                    finish_offset = used
            if task.node >= 0 and task.upload_left <= 1e-9 and task.compute_left <= 1e-9:
                completed += 1
                offloaded += int(task.node >= self.devices)
                latency = (self.time - task.arrived) * dt + finish_offset
                self.latencies.append(latency)
                self.events.append(dict(task_id=task.task_id, status="completed", node=task.node,
                                        model_rank=task.model_rank, required_rank=task.required_rank,
                                        time_s=self.time * dt + finish_offset, latency_s=latency))
            elif self.time + 1 >= task.deadline:
                missed += 1
                self.events.append(dict(task_id=task.task_id, status="deadline_miss", time_s=(self.time + 1) * dt))
            else:
                survivors.append(task)
        self.tasks = survivors
        self.total["completed"] += completed
        self.total["missed"] += missed
        self.total["energy_j"] += energy
        overflow = self._overflow_pending
        self._overflow_pending = 0
        reward = completed - 2 * (missed + overflow) - self.config.energy_weight * energy - 0.03 * len(self.tasks)
        slot = self.time
        self.time += 1
        self.done = self.time >= self.config.horizon
        if self.done:
            reward -= 0.5 * len(self.tasks)
        else:
            self._spawn()
        self.total["reward"] += reward
        info = dict(
            slot=slot, arrivals=len(self.trace.arrivals[slot]), completed=completed,
            missed=missed, overflow=overflow, queue=len(self.tasks), energy_j=energy,
            active_edges=len(active), requested_edges=decision.edge_budget,
            edge_completions=offloaded, reward=reward,
            inner_iterations=allocation.iterations, inner_converged=int(allocation.converged),
            inner_objective_before=allocation.objective_before, inner_objective_after=allocation.objective_after,
            compute_violation=allocation.compute_violation, power_violation=allocation.power_violation,
            cpu_utilization=float(allocation.frequency.sum() / np.sum(np.concatenate((
                self.config.device_ghz, np.asarray(self.config.edge_ghz) * (1 - self.trace.background[slot]))))),
            transmit_power_utilization=float(allocation.power.sum() / np.sum(self.config.device_power_w)),
            waiting=sum(task.node < 0 for task in self.tasks),
            uploading=sum(task.node >= 0 and task.upload_left > 1e-9 for task in self.tasks),
            computing=sum(task.node >= 0 and task.upload_left <= 1e-9 for task in self.tasks),
            mean_channel=float(self.trace.channel[slot].mean()),
            background_load=float(self.trace.background[slot].mean()),
        )
        # This is a finite-horizon task, so there is no terminal bootstrap.
        return self.observation(), float(reward), self.done, False, info

    def summary(self):
        arrived = max(1, self.total["arrived"])
        return dict(self.total, pending=len(self.tasks),
                    completion_rate=self.total["completed"] / arrived,
                    deadline_miss_rate=self.total["missed"] / arrived,
                    overflow_rate=self.total["overflow"] / arrived,
                    mean_latency_s=float(np.mean(self.latencies)) if self.latencies else None,
                    p95_latency_s=float(np.percentile(self.latencies, 95)) if self.latencies else None)
