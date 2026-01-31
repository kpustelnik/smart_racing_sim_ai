"""
SAC (Soft Actor-Critic) Trainer

An off-policy algorithm that maximizes both expected return and entropy.
Excellent sample efficiency and stable training for continuous control.

Hyperparameters:
- learning_rate: 0.0003
- batch_size: 512 (larger for off-policy replay buffer sampling)
- ent_coef: "auto" (automatic entropy tuning)
- buffer_size: 1_000_000 (replay buffer size)
"""

import os
import numpy as np
from typing import Any, Dict

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from pettingzoo import ParallelEnv
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
import uuid

from .base import ModelTrainer, TrainingBridge


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

            infos[agent] = {}

        return observations, rewards, terminations, truncations, infos


class SACTrainer(ModelTrainer):
    """
    SAC Trainer - Soft Actor-Critic
    
    Best for:
    - Sample efficiency (off-policy)
    - Continuous action spaces
    - Automatic entropy tuning
    """
    
    HYPERPARAMETERS = {
        "total_timesteps": 200_000,
        "learning_rate": 0.0003,
        "batch_size": 512,
        "ent_coef": "auto",
        "buffer_size": 1_000_000,
        "net_arch": [256, 256],
        "device": "cpu",
    }
    
    def create_model(self, env: Any, tensorboard_log: str = None) -> SAC:
        """Create a new SAC model with configured hyperparameters."""
        hp = self.HYPERPARAMETERS
        policy_kwargs = dict(net_arch=hp["net_arch"])
        
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp["learning_rate"],
            batch_size=hp["batch_size"],
            ent_coef=hp["ent_coef"],
            buffer_size=hp["buffer_size"],
            policy_kwargs=policy_kwargs,
            device=hp["device"],
            tensorboard_log=tensorboard_log,
        )
    
    def train(self) -> None:
        """Run SAC training loop."""
        print(f"[{self.model_id}] SAC Training thread started.")
        
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
        tb_log_path = os.path.join(self.LOGS_DIR, f"{self.model_id}_sac")
        
        # Load or create model
        if os.path.exists(model_path):
            try:
                print(f"[{self.model_id}] Loading existing SAC model...")
                model = SAC.load(model_path, env=env, device=self.HYPERPARAMETERS["device"])
                model.tensorboard_log = tb_log_path
                if os.path.exists(stats_path):
                    print(f"[{self.model_id}] Loading normalization stats...")
                    env = VecNormalize.load(stats_path, env)
            except Exception as e:
                print(f"[{self.model_id}] Load failed: {e}. Creating new SAC model.")
                model = self.create_model(env, tensorboard_log=tb_log_path)
        else:
            print(f"[{self.model_id}] Creating new SAC model.")
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

        try:
            model.learn(
                total_timesteps=self.HYPERPARAMETERS["total_timesteps"],
                callback=checkpoint_callback,
                reset_num_timesteps=False,
            )
            
            model.save(os.path.join(self.MODELS_DIR, self.model_id))
            env.save(stats_path)
            print(f"[{self.model_id}] Saved SAC model and normalization stats.")
            
        except Exception as e:
            print(f"[{self.model_id}] Training Error: {e}")
        finally:
            env.close()

    def use(self) -> None:
        """Run SAC inference loop (no training)."""
        print(f"[{self.model_id}] SAC Inference thread started.")
        
        model_path = os.path.join(self.MODELS_DIR, f"{self.model_id}.zip")
        stats_path = os.path.join(self.MODELS_DIR, f"{self.model_id}_vecnormalize.pkl")
        
        if not os.path.exists(model_path):
            print(f"[{self.model_id}] ERROR: Model not found at {model_path}")
            print(f"[{self.model_id}] Please train a model first using --mode train")
            return
        
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
        
        # Load VecNormalize stats if available
        if os.path.exists(stats_path):
            print(f"[{self.model_id}] Loading normalization stats...")
            env = VecNormalize.load(stats_path, env)
            env.training = False  # Disable updating normalization stats
            env.norm_reward = False  # Don't need reward normalization for inference
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
        
        # Load model
        print(f"[{self.model_id}] Loading SAC model from {model_path}...")
        model = SAC.load(model_path, env=env, device=self.HYPERPARAMETERS["device"])
        
        print(f"[{self.model_id}] Starting inference loop...")
        
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                
        except Exception as e:
            print(f"[{self.model_id}] Inference Error: {e}")
        finally:
            env.close()
