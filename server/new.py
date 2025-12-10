import asyncio
import functools


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
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# PettingZoo & SuperSuit
from pettingzoo import ParallelEnv
import supersuit as ss

# --- CONSTANTS ---
RAYCASTS = 10
NITRO_FUEL_STATE = 1
VELOCITY_STATE = 1 
PREV_ACTIONS = 0 

BASE_STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS + VELOCITY_STATE # 12
STACK_SIZE = 4
ACTION_DIM = 3
NUM_AGENTS = 5

# --- HYPERPARAMS ---
TOTAL_TIMESTEPS = 200_000 
LEARNING_RATE = 0.0003
BATCH_SIZE = 512
MODELS_DIR = "saved_models"
LOGS_DIR = "sb3_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# --- BRIDGE MECHANISM ---
class DataBridge:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.obs_queues: dict[str, queue.Queue] = {}

    def get_queue(self, agent: str) -> queue.Queue:
        if agent not in self.obs_queues:
            self.obs_queues[agent] = queue.Queue()
        return self.obs_queues[agent]

    def put_data(self, agent: str, obs_data):
        queue = self.get_queue(agent)
        queue.put(obs_data)

    def get_latest_data(self, agent: str):
        c_queue = self.get_queue(agent)
        latest_data = None
        try:
            while True:
                latest_data = c_queue.get_nowait()
        except queue.Empty:
            pass
        # Wait for the data if its missing
        if latest_data is None:
            latest_data = c_queue.get()
        return latest_data
    
    def clear_data(self):
        for agent in self.obs_queues:
            c_queue = self.obs_queues[agent]
            with c_queue.mutex:
                c_queue.queue.clear()

# --- PETTINGZOO PARALLEL ENVIRONMENT ---
class PettingZooWSEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "petting_zoo_ws_env"}
    def __init__(self, bridge: DataBridge, websocket: WebSocket, num_agents: int):
        self.data_bridge = bridge
        self.websocket = websocket
        
        self.possible_agents = [f"car_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

        self.render_mode = None
        asyncio.run(self.websocket.send_json({ "command": "INIT", "agents": self.agents })) # TODO Create universal interface

    def observation_space(self, agent: str) -> gym.Space:
        return spaces.Box(
            low=-1, high=1, shape=(BASE_STATE_DIM,), dtype=np.float32
        )
    
    def action_space(self, agent: str) -> gym.Space:
        return spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
    
    def observe(self, agent: str) -> np.ndarray:
        return self.data_bridge.get_latest_data(agent)
    
    def observe_all(self) -> Dict[str, np.ndarray]:
        observations = {}
        for agent in self.possible_agents:
            observations[agent] = self.observe(agent)
        return observations
    
    def close(self):
        asyncio.run(self.websocket.send_json({ "command": "CLOSE" })) # TODO Create universal interface

    def reset(self, seed=None, options=None):
        for agent in self.possible_agents:
            self.data_bridge.clear_data()
        asyncio.run(self.websocket.send_json({ "command": "RESET" })) # TODO Create universal interface

        data = self.observe_all()
        obs = {}
        infos = {}
        for agent, agent_data in data.items():
            obs[agent] = agent_data['obs']
            infos[agent] = {}
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        # Send the action
        asyncio.run(self.websocket.send_json({
            "command": "ACTION",
            "data": { agent: actions[agent].tolist() for agent in actions }
        }))

        # Await the observation
        data = self.observe_all()
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        for agent, agent_data in data.items():
            observations[agent] = agent_data['obs']
            rewards[agent] = float(agent_data['rewards'])
            terminations[agent] = bool(agent_data['terms'])
            truncations[agent] = bool(agent_data['truncs'])
            infos[agent] = {}

        return observations, rewards, terminations, truncations, infos

# --- TRAINING THREAD ---
def train_model(model_id: str, websocket: WebSocket, bridge: DataBridge):
    
    print(f"[{model_id}] Training thread started.")
    '''
    bridge = bridges[model_id]
    
    # 1. Base Env
    env = RobloxRacingEnv(bridge)
    
    # 2. SuperSuit Wrappers
    env = ss.frame_stack_v1(env, stack_size=STACK_SIZE)
    gym_vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 3. MANUAL ADAPTER
    env = GymnasiumToSB3Adapter(gym_vec_env)
    
    # 4. Normalize
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    
    if os.path.exists(model_path):
        try:
            print(f"[{model_id}] Loading existing model...")
            model = SAC.load(model_path, env=env)
        except Exception as e:
            print(f"[{model_id}] Load failed: {e}. Creating new.")
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, ent_coef="auto")
    else:
        print(f"[{model_id}] Creating new SAC model.")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, ent_coef="auto")

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODELS_DIR, name_prefix=model_id)
    
    while True:
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)
            model.save(os.path.join(MODELS_DIR, model_id))
            print(f"[{model_id}] Saved model.")
        except Exception as e:
            print(f"[{model_id}] Training Error: {e}")
            break
    '''

app = FastAPI()

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    await websocket.accept() # Accept the websocket connection

    bridge = DataBridge(num_agents=NUM_AGENTS)
    
    thread_name = f"train_{model_id}"
    thread = threading.Thread(target=train_model, args=(model_id, websocket, bridge), name=thread_name, daemon=True)
    thread.start()

    try:
        while True:
            raw_data = await websocket.receive_json()

            

            '''
            data_list = raw_data if isinstance(raw_data, list) else [raw_data]
            
            # Reset buffers
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
                await websocket.send_json({"actions": []})
            '''
    except (WebSocketDisconnect, Exception) as e:
        print(f"[{model_id}] Connection closed: {e}")
        thread.join()
        print("Thread has been killed")
    
    '''
    if model_id not in bridges: bridges[model_id] = DataBridge(num_agents=NUM_AGENTS)
    bridge = bridges[model_id]
    

    batch_obs = np.zeros((NUM_AGENTS, BASE_STATE_DIM), dtype=np.float32)
    batch_rewards = np.zeros((NUM_AGENTS,), dtype=np.float32)
    batch_terms = np.zeros((NUM_AGENTS,), dtype=bool)
    batch_truncs = np.zeros((NUM_AGENTS,), dtype=bool)
    '''

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)