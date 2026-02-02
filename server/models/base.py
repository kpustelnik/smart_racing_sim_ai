"""
Base classes and interfaces for model trainers.

This module provides:
- TrainingBridge: A facade over DataBridge with documented, limited API
- ModelTrainer: Abstract base class for all model trainers
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class TrainingBridge:
    """
    Facade for DataBridge that exposes only the methods needed by model trainers.
    
    This provides a clean, documented interface for:
    - Sending commands to Roblox (spawn, reset, action, close)
    - Receiving observations from agents
    - Managing observation queues
    
    Attributes:
        num_agents: Number of agents per environment
    """
    
    def __init__(self, data_bridge: Any):
        """
        Initialize the training bridge facade.
        
        Args:
            data_bridge: The underlying DataBridge instance
        """
        self._bridge = data_bridge
        self.num_agents = data_bridge.num_agents
    
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
        self._bridge.send_command("RESET", env_id, {"agents": agents})
    
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
        """
        return self._bridge.get_latest_obs(agent)
    
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
    - HYPERPARAMETERS: Class-level dict with model-specific hyperparameters
    - create_model(): Factory method to create the RL model
    - train(): Main training loop
    
    Example usage:
        trainer = PPOTrainer(bridge, "my_model")
        trainer.train()
    """
    
    # Override in subclasses with model-specific hyperparameters
    HYPERPARAMETERS: Dict[str, Any] = {}
    
    # Shared constants (can be overridden)
    RAYCASTS = 10
    NITRO_FUEL_STATE = 1
    VELOCITY_STATE = 1
    PREV_ACTIONS = 0
    BASE_STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS + VELOCITY_STATE  # 12
    STACK_SIZE = 4
    ACTION_DIM = 3
    NUM_AGENTS = 5
    NUM_VENVS = 4
    
    MODELS_DIR = "saved_models"
    LOGS_DIR = "sb3_logs"
    
    def __init__(self, bridge: TrainingBridge, model_id: str):
        """
        Initialize the trainer.
        
        Args:
            bridge: TrainingBridge facade for Roblox communication
            model_id: Unique identifier for this model instance
        """
        self.bridge = bridge
        self.model_id = model_id
    
    @abstractmethod
    def create_model(self, env: Any) -> Any:
        """
        Create and return the RL model.
        
        Args:
            env: The vectorized environment
            
        Returns:
            Stable-Baselines3 model instance
        """
        pass
    
    @abstractmethod
    def train(self) -> None:
        """
        Run the training loop.
        
        This should:
        1. Create the environment
        2. Create/load the model
        3. Run model.learn()
        4. Save the model
        """
        pass
    
    @abstractmethod
    def use(self) -> None:
        """
        Run inference loop (no training).
        
        This should:
        1. Create the environment
        2. Load the existing model
        3. Run inference loop using model.predict()
        """
        pass
    
    @classmethod
    def get_description(cls) -> str:
        """Return a human-readable description of this trainer."""
        return cls.__doc__ or cls.__name__
