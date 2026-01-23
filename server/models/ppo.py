"""
PPO (Proximal Policy Optimization) Trainer

A policy gradient method that uses clipped surrogate objective for stable training.
Good for continuous action spaces and environments with high variance.

Hyperparameters:
- learning_rate: 0.0003
- batch_size: 64 (smaller for on-policy updates)
- n_steps: 1024 (steps to collect before each update)
- ent_coef: 0.01 (entropy coefficient for exploration)
- net_arch: [256, 256]
"""

import os
import numpy as np
from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from pettingzoo import ParallelEnv
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
import uuid

from .base import ModelTrainer, TrainingBridge


class RobloxFreezeCallback(BaseCallback):
    """Callback to log rollout start/end events."""
    
    def __init__(self, bridge: TrainingBridge, verbose=0):
        super().__init__(verbose)
        self.bridge = bridge
        print("[Callback] Initializing RobloxFreezeCallback.")

    def _on_rollout_end(self) -> None:
        print("Rollout end (Freeze)")
        self.bridge.freeze(True)

    def _on_rollout_start(self) -> None:
        print("Rollout start (Unfreeze)")
        self.bridge.freeze(False)
        
    def _on_step(self) -> bool:
        return True


class PettingZooWSEnv(ParallelEnv):
    """PettingZoo Parallel Environment that communicates via WebSocket bridge."""
    
    metadata = {"render_modes": ["human"], "name": "petting_zoo_ws_env"}
    
    def __init__(self, bridge: TrainingBridge, num_agents: int, state_dim: int, action_dim: int, num_venvs: int = 4):
        self.venvs_id = [str(uuid.uuid4()) for _ in range(num_venvs)]
        self.data_bridge = bridge
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.possible_agents = []
        self.agents_per_venv = {}
        for venv_id in self.venvs_id:
            agents_venv = [f"car_{i}_env_{venv_id}" for i in range(num_agents)]
            self.possible_agents.extend(agents_venv)
            self.agents_per_venv[venv_id] = agents_venv
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.render_mode = None
        
        # Spawn agents on init
        for venv_id in self.venvs_id:
            self.data_bridge.spawn_agents(venv_id, self.agents_per_venv[venv_id])

    def observation_space(self, agent: str) -> gym.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
    
    def action_space(self, agent: str) -> gym.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
    
    def observe_all_data(self) -> Dict[str, Any]:
        data_map = {}
        for agent in self.agents:
            data = self.data_bridge.get_observation(agent)
            if data:
                data_map[agent] = data
        return data_map
    
    def close(self):
        for venv_id in self.venvs_id:
            self.data_bridge.close_environment(venv_id)

    def reset(self, seed=None, options=None):
        self.data_bridge.clear_observations()
        self.agents = self.possible_agents[:]
        
        for venv_id in self.venvs_id:
            self.data_bridge.reset_agents(venv_id, self.agents_per_venv[venv_id])

        data_map = self.observe_all_data()
        
        obs = {}
        infos = {}
        for agent in self.agents:
            agent_obs = data_map.get(agent, {}).get('obs', np.zeros(self.state_dim, dtype=np.float32))
            obs[agent] = np.array(agent_obs, dtype=np.float32)
            infos[agent] = {}
            
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        serializable_actions = {agent: acts.tolist() for agent, acts in actions.items()}
        for venv_id in self.venvs_id:
            venv_actions = {a: serializable_actions[a] for a in self.agents_per_venv[venv_id] if a in serializable_actions}
            self.data_bridge.send_actions(venv_id, venv_actions)

        data_map = self.observe_all_data()
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            agent_data = data_map.get(agent)
            
            if agent_data:
                observations[agent] = np.array(agent_data.get('obs', np.zeros(self.state_dim)), dtype=np.float32)
                rewards[agent] = float(agent_data.get('reward', 0.0))
                terminations[agent] = bool(agent_data.get('terminated', False))
                truncations[agent] = bool(agent_data.get('truncated', False))
            else:
                observations[agent] = np.zeros(self.state_dim, dtype=np.float32)
                rewards[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False

        return observations, rewards, terminations, truncations, infos


class PPOTrainer(ModelTrainer):
    """
    PPO Trainer - Proximal Policy Optimization
    
    Best for:
    - Continuous action spaces
    - Environments requiring exploration
    - Stable, reliable training
    """
    
    HYPERPARAMETERS = {
        "total_timesteps": 200_000,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_steps": 1024,
        "ent_coef": 0.01,
        "net_arch": [256, 256],
        "device": "cpu",
    }
    
    def create_model(self, env: Any, tensorboard_log: Optional[str] = None) -> PPO:
        """Create a new PPO model with configured hyperparameters."""
        hp = self.HYPERPARAMETERS
        policy_kwargs = dict(net_arch=hp["net_arch"])
        
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            ent_coef=hp["ent_coef"],
            policy_kwargs=policy_kwargs,
            device=hp["device"],
            tensorboard_log=tensorboard_log,
        )
    
    def train(self) -> None:
        """Run PPO training loop."""
        print(f"[{self.model_id}] PPO Training thread started.")
        
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        
        # Create environment
        env = PettingZooWSEnv(
            self.bridge,
            num_agents=self.NUM_AGENTS,
            state_dim=self.BASE_STATE_DIM,
            action_dim=self.ACTION_DIM,
            num_venvs=self.NUM_VENVS,
        )
        env = ss.frame_stack_v1(env, stack_size=self.STACK_SIZE)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = SB3VecEnvWrapper(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        model_path = os.path.join(self.MODELS_DIR, f"{self.model_id}.zip")
        stats_path = os.path.join(self.MODELS_DIR, f"{self.model_id}_vecnormalize.pkl")
        
        # TensorBoard log path
        tb_log_path = os.path.join(self.LOGS_DIR, f"{self.model_id}_ppo")
        
        # Load or create model
        if os.path.exists(model_path):
            try:
                print(f"[{self.model_id}] Loading existing PPO model...")
                model = PPO.load(model_path, env=env, device=self.HYPERPARAMETERS["device"])
                model.tensorboard_log = tb_log_path
                if os.path.exists(stats_path):
                    print(f"[{self.model_id}] Loading normalization stats...")
                    env = VecNormalize.load(stats_path, env)
            except Exception as e:
                print(f"[{self.model_id}] Load failed: {e}. Creating new PPO model.")
                model = self.create_model(env, tensorboard_log=tb_log_path)
        else:
            print(f"[{self.model_id}] Creating new PPO model.")
            model = self.create_model(env, tensorboard_log=tb_log_path)
        
        print(f"[{self.model_id}] TensorBoard logs: {tb_log_path}")
        print(f"[{self.model_id}] Run 'tensorboard --logdir {self.LOGS_DIR}' to view training progress.")

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.MODELS_DIR,
            save_vecnormalize=True,
            name_prefix=self.model_id,
        )
        freeze_callback = RobloxFreezeCallback(self.bridge)

        try:
            while True:
                model.learn(
                    total_timesteps=self.HYPERPARAMETERS["total_timesteps"],
                    callback=[checkpoint_callback, freeze_callback],
                    reset_num_timesteps=False,
                )
                
                model.save(os.path.join(self.MODELS_DIR, self.model_id))
                env.save(stats_path)
                print(f"[{self.model_id}] Saved PPO model and normalization stats.")
            
        except Exception as e:
            print(f"[{self.model_id}] Training Error: {e}")
        finally:
            env.close()
