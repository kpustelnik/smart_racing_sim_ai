import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import threading
import queue
import os
import shutil
from typing import List, Dict, Optional, Any

# RL Libraries
import gymnasium as gym
from gymnasium import spaces
# CHANGED: Import PPO instead of SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# PettingZoo & SuperSuit
from pettingzoo import ParallelEnv
import supersuit as ss

# --- CONSTANTS ---
RAYCASTS = 16
NITRO_FUEL_STATE = 1
VELOCITY_STATE = 1 
PREV_ACTIONS = 0 

BASE_STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS + VELOCITY_STATE 
STACK_SIZE = 4
ACTION_DIM = 3
NUM_AGENTS = 5

# --- HYPERPARAMS (PPO Specific) ---
TOTAL_TIMESTEPS = 500_000 
LEARNING_RATE = 0.0003
BATCH_SIZE = 64        # Smaller batch size for PPO updates
N_STEPS = 1024         # Steps to collect per agent before updating
MODELS_DIR = "saved_models"
LOGS_DIR = "sb3_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI()

# --- COMPATIBILITY ADAPTER ---
class GymnasiumToSB3Adapter(VecEnv):
    def __init__(self, venv):
        self.venv = venv
        self.render_mode = getattr(venv, "render_mode", None)
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space
        )

    def reset(self):
        obs, _ = self.venv.reset()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, term, trunc, info = self.venv.step_wait()
        dones = term | trunc
        
        if isinstance(info, dict):
            new_infos = []
            keys = info.keys()
            for i in range(self.num_envs):
                sub_info = {k: info[k][i] for k in keys}
                if dones[i]:
                    if "terminal_observation" not in sub_info and "final_observation" in sub_info:
                        sub_info["terminal_observation"] = sub_info["final_observation"]
                new_infos.append(sub_info)
            info = new_infos
            
        return obs, reward, dones, info

    def close(self):
        self.venv.close()
    
    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *args, **kwargs):
        return self.venv.env_method(method_name, *args, **kwargs)
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# --- BRIDGE MECHANISM ---
class DataBridge:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.action_queue = queue.Queue()
        self.obs_queue = queue.Queue()

    def clear(self):
        with self.action_queue.mutex:
            self.action_queue.queue.clear()
        with self.obs_queue.mutex:
            self.obs_queue.queue.clear()

    def put_actions(self, actions):
        with self.action_queue.mutex:
            self.action_queue.queue.clear()
        self.action_queue.put(actions)

    def put_observations(self, obs_data):
        self.obs_queue.put(obs_data)

    def get_latest_observation(self):
        data = None
        try:
            while True:
                data = self.obs_queue.get_nowait()
        except queue.Empty:
            pass
        if data is None:
            data = self.obs_queue.get()
        return data

bridges: Dict[str, DataBridge] = {}

# --- PETTINGZOO PARALLEL ENVIRONMENT ---
class RobloxRacingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "roblox_racing_v1"}

    def __init__(self, bridge: DataBridge):
        self.render_mode = None 
        self.bridge = bridge
        self.possible_agents = [f"car_{i}" for i in range(bridge.num_agents)]
        self.agents = self.possible_agents[:]
        
        self.observation_space = lambda agent: spaces.Box(
            low=-np.inf, high=np.inf, shape=(BASE_STATE_DIM,), dtype=np.float32
        )
        self.action_space = lambda agent: spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        data = self.bridge.get_latest_observation()
        raw_obs = data['obs']
        
        observations = {}
        infos = {}
        for idx, agent in enumerate(self.agents):
            observations[agent] = raw_obs[idx]
            infos[agent] = {}
        return observations, infos

    def step(self, actions):
        ordered_actions = np.zeros((self.bridge.num_agents, ACTION_DIM), dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                ordered_actions[i] = actions[agent]
        
        self.bridge.put_actions(ordered_actions)
        data = self.bridge.get_latest_observation()
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for idx, agent in enumerate(self.possible_agents):
            observations[agent] = data['obs'][idx]
            rewards[agent] = float(data['rewards'][idx])
            terminations[agent] = bool(data['terms'][idx])
            truncations[agent] = bool(data['truncs'][idx])
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos

# --- TRAINING THREAD (PPO VERSION) ---
def train_model(model_id: str):
    print(f"[{model_id}] Training thread started (PPO).")
    bridge = bridges[model_id]
    
    # 1. Base Env
    env = RobloxRacingEnv(bridge)
    
    # 2. Wrappers
    env = ss.frame_stack_v1(env, stack_size=STACK_SIZE)
    gym_vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = GymnasiumToSB3Adapter(gym_vec_env)

    model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    
    # PPO CONFIGURATION
    # ent_coef=0.01: Forces exploration (prevents early convergence to crashing)
    # n_steps=1024: Collects 1024 steps per car before updating weights
    policy_kwargs = dict(net_arch=[256, 256])
    
    if os.path.exists(model_path):
        try:
            print(f"[{model_id}] Loading existing model...")
            model = PPO.load(model_path, env=env, device="cpu")
            #model.ent_coef = 0.01 # Force exploration on load
        except Exception as e:
            print(f"[{model_id}] Load failed: {e}. Creating new PPO.")
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, 
                        n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=0.01, 
                        policy_kwargs=policy_kwargs, device="cpu")
    else:
        print(f"[{model_id}] Creating new PPO model.")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, 
                    n_steps=N_STEPS, batch_size=BATCH_SIZE, ent_coef=0.01, 
                    policy_kwargs=policy_kwargs, device="cpu")

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODELS_DIR, name_prefix=model_id)
    
    while True:
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)
            model.save(os.path.join(MODELS_DIR, model_id))
            print(f"[{model_id}] Saved model.")
        except Exception as e:
            print(f"[{model_id}] Training Error: {e}")
            break

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    await websocket.accept()
    if model_id not in bridges: bridges[model_id] = DataBridge(num_agents=NUM_AGENTS)
    bridge = bridges[model_id]
    
    bridge.clear()
    
    thread_name = f"train_{model_id}"
    if not any(t.name == thread_name and t.is_alive() for t in threading.enumerate()):
        threading.Thread(target=train_model, args=(model_id,), name=thread_name, daemon=True).start()

    batch_obs = np.zeros((NUM_AGENTS, BASE_STATE_DIM), dtype=np.float32)
    batch_rewards = np.zeros((NUM_AGENTS,), dtype=np.float32)
    batch_terms = np.zeros((NUM_AGENTS,), dtype=bool)
    batch_truncs = np.zeros((NUM_AGENTS,), dtype=bool)
    no_op_action = np.zeros((NUM_AGENTS, ACTION_DIM), dtype=np.float32).tolist()

    try:
        while True:
            raw_data = await websocket.receive_json()
            data_list = raw_data if isinstance(raw_data, list) else [raw_data]
            
            batch_obs.fill(0); batch_rewards.fill(0); batch_terms.fill(False); batch_truncs.fill(False)
            
            for i in range(NUM_AGENTS):
                item = data_list[i] if i < len(data_list) else None
                if item:
                    try:
                        batch_obs[i] = np.array(item.get('current_observation', []), dtype=np.float32)
                        batch_rewards[i] = float(item.get('reward', 0.0))
                        batch_terms[i] = bool(item.get('terminated', False))
                        batch_truncs[i] = bool(item.get('truncated', False))
                    except: batch_terms[i] = True
                else:
                    batch_terms[i] = True

            bridge.put_observations({'obs': batch_obs, 'rewards': batch_rewards, 'terms': batch_terms, 'truncs': batch_truncs})
            
            if not bridge.action_queue.empty():
                actions = bridge.action_queue.get()
                await websocket.send_json({"actions": actions.tolist()})
            else:
                await websocket.send_json({"actions": no_op_action})

    except WebSocketDisconnect:
        print(f"[{model_id}] Client disconnected.")
    except Exception as e:
        print(f"[{model_id}] Critical Connection Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)