"""
Base classes and interfaces for model trainers.

This module provides:
- BridgeClosed: raised when the Roblox connection behind a trainer goes away
- TrainingBridge: A facade over DataBridge with documented, limited API
- ModelTrainer: Abstract base class for all model trainers
"""

import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BridgeClosed(RuntimeError):
    """
    Raised once the WebSocket behind a trainer has gone away.
    """


class TrainingBridge:
    """
    Facade for DataBridge that exposes only the methods needed by model trainers.

    This provides a clean, documented interface for:
    - Sending commands to Roblox (spawn, reset, action, close)
    - Receiving observations from agents
    - Managing observation queues
    """

    def __init__(self, data_bridge: Any):
        """
        Initialize the training bridge facade.

        Args:
            data_bridge: The underlying DataBridge instance
        """
        self._bridge = data_bridge

    # --- Command Methods (send to Roblox) ---

    def freeze(self, status: bool) -> None:
        """
        Request Roblox to freeze or unfreeze agents' physics.

        Args:
            status: Whether the environment should be frozen or unfrozen.
        """
        self._bridge.send_command("FREEZE", "", {"status": status})

    def spawn_agents(self, env_id: str, agents: list[str]) -> None:
        """
        Request Roblox to spawn agents for a virtual environment.

        Args:
            env_id: Unique identifier for the virtual environment
            agents: List of agent IDs to spawn (e.g., ["car_0_env_xxx", "car_1_env_xxx"])
        """
        self._bridge.send_command("SPAWN_AGENTS", env_id, {"agents": agents})

    def reset_agents(self, env_id: str, agents: list[str]) -> None:
        """
        Request Roblox to reset/respawn agents.

        Args:
            env_id: Unique identifier for the virtual environment
            agents: List of agent IDs to reset
        """
        self._bridge.send_command("RESET_AGENTS", env_id, {"agents": agents})

    def remove_agents(self, env_id: str, agents: list[str]) -> None:
        """
        Request Roblox to remove agents

        Args:
            env_id: Unique identifier for the virtual environment
            agents: List of agent IDs to remove
        """
        self._bridge.send_command("REMOVE_AGENTS", env_id, {"agents": agents})

    def send_actions(self, env_id: str, actions: Dict[str, list[float]]) -> None:
        """
        Send actions to agents in Roblox.

        Args:
            env_id: Unique identifier for the virtual environment
            actions: Dict mapping agent_id -> [throttle, steering, nitro]
        """
        self._bridge.send_command("ACTION", env_id, actions)

    def close_environment(self, env_id: str) -> None:
        """
        Request Roblox to close/cleanup a virtual environment.

        Args:
            env_id: Unique identifier for the virtual environment to close
        """
        self._bridge.send_command("CLOSE", env_id)

    def close_all(self) -> None:
        """
        Request Roblox to close/cleanup all virtual environments.
        """
        self._bridge.send_command("CLOSE_ALL", '')

    def update_collisions(self, status: bool) -> None:
        """
        Request Roblox to update collisions between agents.
        """
        self._bridge.send_command("UPDATE_COLLISIONS", '', {"enable_collisions": status})

    # --- Observation Methods (receive from Roblox) ---

    def get_observation(self, agent: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest observation data for an agent.

        Blocks until data is available (with 10s timeout).

        Args:
            agent: Agent ID to get observation for

        Returns:
            Dict with keys: 'obs', 'reward', 'terminated', 'truncated'
            Or None if timeout/error

        Raises:
            BridgeClosed: if the Roblox connection went away while waiting
        """
        data = self._bridge.get_latest_obs(agent)
        if self._bridge.is_closed():
            raise BridgeClosed("Roblox disconnected")
        return data

    def clear_observations(self) -> None:
        """
        Clear all queued observations.

        Call this before a reset to ensure fresh data.
        """
        self._bridge.clear_data()


class ModelTrainer(ABC):
    """
    Abstract base class for all model trainers.

    Subclasses must implement:
    - NAME: short identifier
    - ALGORITHM: the Stable-Baselines3 algorithm class
    - HYPERPARAMETERS: Class-level dict with model-specific hyperparameters
    - create_model(): Factory method to create the RL model

    train() and use()

    Example usage:
        trainer = PPOTrainer(bridge, "my_model")
        trainer.train()
    """

    # Override in subclasses with model-specific hyperparameters
    NAME: str = ""
    ALGORITHM: Any = None
    HYPERPARAMETERS: Dict[str, Any] = {}
    
    LOAD_PARAMETERS: Dict[str, Any] = {}

    # Shared constants (can be overridden).
    RAYCASTS = 16
    NITRO_FUEL_STATE = 1
    VELOCITY_STATE = 1
    TRACK_REVERSE_STATE = 1
    PREV_ACTIONS = 3  # one previous action vector
    BASE_STATE_DIM = (
        RAYCASTS + NITRO_FUEL_STATE + VELOCITY_STATE + TRACK_REVERSE_STATE + PREV_ACTIONS
    )
    STACK_SIZE = 4 # How many past observations are concatenated into each input
    ACTION_DIM = 3 # Throttle, Steering, Nitro
    NUM_AGENTS = 5 # How many agents per environment
    NUM_VENVS = 5 # How many parralel environments

    MODELS_DIR = "saved_models"
    LOGS_DIR = "sb3_logs"
    CHECKPOINT_FREQ = 10_000

    def __init__(self, bridge: TrainingBridge, model_id: str):
        """
        Initialize the trainer.

        Args:
            bridge: TrainingBridge facade for Roblox communication
            model_id: Unique identifier for this model instance
        """
        if not model_id or os.path.basename(model_id) != model_id or model_id in {".", ".."}:
            raise ValueError(f"Unusable model id: {model_id!r}")

        self.bridge = bridge
        self.model_id = model_id

    @abstractmethod
    def create_model(self, env: Any, tensorboard_log: Optional[str] = None) -> Any:
        """
        Create and return the RL model.

        Args:
            env: The vectorized environment
            tensorboard_log: Directory to write TensorBoard events to

        Returns:
            Stable-Baselines3 model instance
        """

    # --- Paths ---

    @property
    def model_path(self) -> str:
        return os.path.join(self.MODELS_DIR, f"{self.model_id}.zip")

    @property
    def stats_path(self) -> str:
        return os.path.join(self.MODELS_DIR, f"{self.model_id}_vecnormalize.pkl")

    @property
    def tensorboard_path(self) -> str:
        return os.path.join(self.LOGS_DIR, f"{self.model_id}_{self.NAME}")

    # --- Shared setup ---

    def build_env(self, training: bool) -> Any:
        """
        Build the wrapped vector environment, restoring saved normalization stats if there are any.
        """
        import supersuit as ss
        from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
        from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

        from .env import PettingZooWSEnv

        env = PettingZooWSEnv(
            self.bridge,
            num_agents=self.NUM_AGENTS,
            state_dim=self.BASE_STATE_DIM,
            action_dim=self.ACTION_DIM,
            num_venvs=self.NUM_VENVS,
            stack_size=self.STACK_SIZE,
        )
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = SB3VecEnvWrapper(env)
        env = VecMonitor(env)

        if os.path.exists(self.stats_path):
            try:
                print(f"[{self.model_id}] Loading normalization stats...")
                normalized = VecNormalize.load(self.stats_path, env)
                normalized.training = training
                normalized.norm_reward = training
                return normalized
            except Exception:
                traceback.print_exc()
                print(f"[{self.model_id}] Load failed. Creating new VecNormalize.")

        return VecNormalize(
            env,
            norm_obs=True,
            norm_reward=training,
            clip_obs=10.0,
            training=training,
        )

    def load_model(self, env: Any) -> Any:
        return self.ALGORITHM.load(
            self.model_path,
            env=env,
            device=self.HYPERPARAMETERS["device"],
            custom_objects=self.LOAD_PARAMETERS or None,
        )

    def load_or_create_model(self, env: Any) -> Any:
        """Load the saved model if there is one, otherwise create a fresh one."""
        if os.path.exists(self.model_path):
            try:
                print(f"[{self.model_id}] Loading existing {self.NAME.upper()} model...")
                model = self.load_model(env)
                model.tensorboard_log = self.tensorboard_path
                return model
            except Exception:
                traceback.print_exc()
                print(f"[{self.model_id}] Load failed. Creating new {self.NAME.upper()} model.")
        else:
            print(f"[{self.model_id}] Creating new {self.NAME.upper()} model.")

        return self.create_model(env, tensorboard_log=self.tensorboard_path)

    def save(self, model: Any, env: Any) -> None:
        model.save(self.model_path)
        env.save(self.stats_path)
        print(f"[{self.model_id}] Saved {self.NAME.upper()} model and normalization stats.")

    def train(self) -> None:
        """Run the training loop until Roblox disconnects."""
        from stable_baselines3.common.callbacks import CheckpointCallback

        from .env import RobloxFreezeCallback

        print(f"[{self.model_id}] {self.NAME.upper()} Training thread started.")

        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

        env = self.build_env(training=True)
        model = self.load_or_create_model(env)

        print(f"[{self.model_id}] TensorBoard logs: {self.tensorboard_path}")
        print(f"[{self.model_id}] Run 'tensorboard --logdir {self.LOGS_DIR}' to view training progress.")

        callbacks = [
            CheckpointCallback(
                save_freq=self.CHECKPOINT_FREQ,
                save_path=self.MODELS_DIR,
                save_vecnormalize=True,
                name_prefix=self.model_id,
            ),
            RobloxFreezeCallback(self.bridge),
        ]

        try:
            while True:
                model.learn(
                    total_timesteps=self.HYPERPARAMETERS["total_timesteps"],
                    callback=callbacks,
                    reset_num_timesteps=False,
                )
                self.save(model, env)
        except BridgeClosed:
            print(f"[{self.model_id}] Roblox disconnected; stopping training.")
            try:
                self.save(model, env)
            except Exception:
                traceback.print_exc()
        except Exception:
            print(f"[{self.model_id}] Training Error:")
            traceback.print_exc()
        finally:
            env.close()

    def use(self) -> None:
        """Run the inference loop (no training)."""
        print(f"[{self.model_id}] {self.NAME.upper()} Inference thread started.")

        if not os.path.exists(self.model_path):
            print(f"[{self.model_id}] ERROR: Model not found at {self.model_path}")
            print(f"[{self.model_id}] Please train a model first using --mode train")
            return

        env = self.build_env(training=False)

        print(f"[{self.model_id}] Loading {self.NAME.upper()} model from {self.model_path}...")
        model = self.load_model(env)

        print(f"[{self.model_id}] Starting inference loop...")
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, _rewards, _dones, _infos = env.step(action)
        except BridgeClosed:
            print(f"[{self.model_id}] Roblox disconnected; stopping inference.")
        except Exception:
            print(f"[{self.model_id}] Inference Error:")
            traceback.print_exc()
        finally:
            env.close()

    @classmethod
    def get_description(cls) -> str:
        """Return a human-readable description of this trainer."""
        return cls.__doc__ or cls.__name__
