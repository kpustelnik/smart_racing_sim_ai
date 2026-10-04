"""
The Roblox-backed environment shared by every trainer.
"""

import time
import uuid
from collections import deque
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from stable_baselines3.common.callbacks import BaseCallback

from . import latency
from .base import TrainingBridge


class RobloxFreezeCallback(BaseCallback):
    """
    Freezes Roblox physics while SB3 is busy updating the policy between rollouts.
    """

    LOG_INTERVAL_SECONDS = 30.0

    def __init__(self, bridge: TrainingBridge, verbose: int = 0):
        super().__init__(verbose)
        self.bridge = bridge
        self._last_log = 0.0
        self._rollouts = 0
        print("[Callback] Initializing RobloxFreezeCallback.")

    def _log_occasionally(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_log < self.LOG_INTERVAL_SECONDS:
            return
        self._last_log = now
        print(f"{message} (rollouts so far: {self._rollouts})")

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        self._log_occasionally("Rollout end (Freeze)")
        self.bridge.freeze(True)

    def _on_rollout_start(self) -> None:
        self._log_occasionally("Rollout start (Unfreeze)")
        self.bridge.freeze(False)

    def _on_step(self) -> bool:
        return True


class PettingZooWSEnv(ParallelEnv):
    """PettingZoo Parallel Environment that communicates via the WebSocket bridge."""

    metadata = {"render_modes": ["human"], "name": "petting_zoo_ws_env"}

    def __init__(
        self,
        bridge: TrainingBridge,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        num_venvs: int = 4,
        stack_size: int = 1,
        autodetect_state_dim: bool = True,
    ):
        self.venvs_id = [str(uuid.uuid4()) for _ in range(num_venvs)]
        self.data_bridge = bridge
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.possible_agents: list[str] = []
        self.agents_per_venv: dict[str, list[str]] = {}
        for venv_id in self.venvs_id:
            agents_venv = [f"car_{i}_env_{venv_id}" for i in range(num_agents)]
            self.possible_agents.extend(agents_venv)
            self.agents_per_venv[venv_id] = agents_venv
        self.agents = self.possible_agents[:]

        self.stack_size = stack_size
        self._step_latency = latency.get_tracker("step", report_every=200)
        self._warned_widths: set[int] = set()
        self._stacks: dict[str, deque] = {}
        self.render_mode = None

        # Spawn agents on init
        for venv_id in self.venvs_id:
            self.data_bridge.spawn_agents(venv_id, self.agents_per_venv[venv_id])

        # The observation width is decided in Lua by the car's ray Config
        if autodetect_state_dim:
            detected = self._detect_state_dim()
            if detected is not None and detected != self.state_dim:
                print(
                    f"[env] Roblox is sending {detected} values per observation; using that instead of the configured BASE_STATE_DIM of {self.state_dim}. "
                )
                self.state_dim = detected

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

    DETECT_ATTEMPTS = 3

    def _detect_state_dim(self) -> Optional[int]:
        """Width of the first observation Roblox sends, or None if none arrives."""
        for agent in self.possible_agents[: self.DETECT_ATTEMPTS]:
            try:
                data = self.data_bridge.get_observation(agent)
            except Exception:
                return None
            if not data:
                continue
            raw = data.get("obs")
            if raw is None:
                continue
            width = int(np.asarray(raw, dtype=np.float32).reshape(-1).shape[0])
            if width > 0:
                return width
        return None

    def observation_space(self, agent: str) -> gym.Space:
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim * self.stack_size,), dtype=np.float32
        )

    def action_space(self, agent: str) -> gym.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def _report_width_mismatch(self, received: int) -> None:
        if received in self._warned_widths:
            return
        self._warned_widths.add(received)
        print(
            f"[env] OBSERVATION WIDTH MISMATCH: Roblox sent {received} values per frame, "
            f"but this trainer is configured for {self.state_dim}."
        )

    def _as_observation(self, raw: Any) -> np.ndarray:
        if raw is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        array = np.asarray(raw, dtype=np.float32).reshape(-1)
        if array.shape[0] != self.state_dim:
            self._report_width_mismatch(int(array.shape[0]))
            padded = np.zeros(self.state_dim, dtype=np.float32)
            usable = min(array.shape[0], self.state_dim)
            padded[:usable] = array[:usable]
            array = padded

        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _stacked(self, agent: str, observation: np.ndarray, restart: bool = False) -> np.ndarray:
        """
        Append observation to the agent's history and return the flattened stack.
        """
        history = self._stacks.get(agent)
        if history is None or restart:
            history = deque([observation] * self.stack_size, maxlen=self.stack_size)
            self._stacks[agent] = history
        else:
            history.append(observation)
        return np.concatenate(list(history)).astype(np.float32)

    @staticmethod
    def _as_scalar(raw: Any, default: float = 0.0) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    def observe_all_data(self) -> Dict[str, Any]:
        data_map = {}
        missing = 0
        for agent in self.agents:
            data = self.data_bridge.get_observation(agent)
            if data:
                data_map[agent] = data
            else:
                missing += 1
        if missing:
            self._step_latency.record_timeout(missing)
        return data_map

    def close(self):
        for venv_id in self.venvs_id:
            self.data_bridge.close_environment(venv_id)
        self.data_bridge.close_all()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.data_bridge.clear_observations()
        self.agents = self.possible_agents[:]

        for venv_id in self.venvs_id:
            self.data_bridge.reset_agents(venv_id, self.agents_per_venv[venv_id])

        data_map = self.observe_all_data()

        obs = {}
        infos = {}
        self._stacks.clear()
        for agent in self.agents:
            frame = self._as_observation(data_map.get(agent, {}).get("obs"))
            obs[agent] = self._stacked(agent, frame, restart=True)
            infos[agent] = {}

        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        serializable_actions = {agent: acts.tolist() for agent, acts in actions.items()}

        with latency.Stopwatch(self._step_latency):
            for venv_id in self.venvs_id:
                venv_actions = {
                    a: serializable_actions[a]
                    for a in self.agents_per_venv[venv_id]
                    if a in serializable_actions
                }
                self.data_bridge.send_actions(venv_id, venv_actions)

            data_map = self.observe_all_data()

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            agent_data = data_map.get(agent) or {}
            rewards[agent] = self._as_scalar(agent_data.get("reward"), 0.0)
            terminations[agent] = bool(agent_data.get("terminated", False))
            truncations[agent] = bool(agent_data.get("truncated", False))
            ended = terminations[agent] or truncations[agent]

            frame = self._as_observation(agent_data.get("obs"))
            infos[agent] = {}

            if ended:
                final = agent_data.get("last_observation")
                if final is not None:
                    infos[agent]["terminal_observation"] = self._stacked(
                        agent, self._as_observation(final)
                    )
                observations[agent] = self._stacked(agent, frame, restart=True)
            else:
                observations[agent] = self._stacked(agent, frame)
            if truncations[agent] and not terminations[agent]:
                infos[agent]["TimeLimit.truncated"] = True

        return observations, rewards, terminations, truncations, infos
