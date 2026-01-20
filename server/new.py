import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import threading
import queue
import os
from typing import List, Dict, Optional, Any

# RL Libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# PettingZoo & SuperSuit
from pettingzoo import ParallelEnv
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

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
        # Incoming data - observation/reward
        self.obs_queues: dict[str, queue.Queue] = {}
        # Outgoing commands - actions/spawns
        self.command_queue: queue.Queue = queue.Queue()

    def get_agent_queue(self, agent: str) -> queue.Queue:
        if agent not in self.obs_queues:
            self.obs_queues[agent] = queue.Queue()
        return self.obs_queues[agent]

    # Called by WebSocket Reader
    def put_incoming_data(self, agent: str, obs_data):
        q = self.get_agent_queue(agent)
        q.put(obs_data)

    # Called by environment - thread
    def get_latest_obs(self, agent: str):
        c_queue = self.get_agent_queue(agent)
        try:
            # Blocking get with timeout to prevent deadlock if something crashes
            return c_queue.get(timeout=10) 
        except queue.Empty:
            return None

    # Called by environment - thread
    def send_command(self, command: str, data: dict = None):
        payload = {"command": command, "data": data if data else {}}
        self.command_queue.put(payload)

    # Called by WebSocket Sender
    def get_outgoing_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_data(self):
        for agent in self.obs_queues:
            c_queue = self.obs_queues[agent]
            with c_queue.mutex:
                c_queue.queue.clear()


# --- PETTINGZOO PARALLEL ENVIRONMENT ---
class PettingZooWSEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "petting_zoo_ws_env"}
    
    def __init__(self, bridge: DataBridge, num_agents: int):
        self.data_bridge = bridge
        self.possible_agents = [f"car_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

        self.render_mode = None
        
        # Backend decides to spawn agents on INIT
        self.data_bridge.send_command("SPAWN_AGENTS", {"agents": self.agents})

    def observation_space(self, agent: str) -> gym.Space:
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(BASE_STATE_DIM,), dtype=np.float32
        )
    
    def action_space(self, agent: str) -> gym.Space:
        return spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
    
    def observe(self, agent: str) -> np.ndarray:
        data = self.data_bridge.get_latest_obs(agent)
        if data is None:
            # Fallback or handling for dead agents if needed
            return np.zeros(BASE_STATE_DIM, dtype=np.float32)
        return data
    
    # Helper to get full packets (obs + rewards + done)
    def observe_all_data(self) -> Dict[str, Any]:
        data_map = {}
        for agent in self.agents:
            data = self.data_bridge.get_latest_obs(agent)
            if data:
                data_map[agent] = data
        return data_map
    
    def close(self):
        self.data_bridge.send_command("CLOSE")

    # Reset environment and respawn/reset agents
    def reset(self, seed=None, options=None):
        self.data_bridge.clear_data()
        self.agents = self.possible_agents[:]
        
        self.data_bridge.send_command("RESET", {"agents": self.agents})

        # Wait for initial observation after reset
        obs = {}
        infos = {}
        
        # We need to wait for the first packet from Roblox after reset
        data_map = self.observe_all_data()
        
        for agent in self.agents:
            # Default observation if packet missing immediately after reset
            agent_obs = data_map.get(agent, {}).get('obs', np.zeros(BASE_STATE_DIM, dtype=np.float32))
            obs[agent] = np.array(agent_obs, dtype=np.float32)
            infos[agent] = {}
            
        return obs, infos

    # Send actions for all active agents
    def step(self, actions: dict[str, np.ndarray]):
        serializable_actions = {agent: acts.tolist() for agent, acts in actions.items()}
        self.data_bridge.send_command("ACTION", serializable_actions)

        # Await the response (observations + rewards)
        data_map = self.observe_all_data()
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            agent_data = data_map.get(agent)
            
            if agent_data:
                observations[agent] = np.array(agent_data.get('obs', np.zeros(BASE_STATE_DIM)), dtype=np.float32)
                rewards[agent] = float(agent_data.get('reward', 0.0))
                terminations[agent] = bool(agent_data.get('terminated', False))
                truncations[agent] = bool(agent_data.get('truncated', False))
            else:
                # Handle missing data (agent died/lag)
                observations[agent] = np.zeros(BASE_STATE_DIM, dtype=np.float32)
                rewards[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False

            infos[agent] = {}

        # Remove dead agents from the loop if necessary
        # self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]

        return observations, rewards, terminations, truncations, infos

# --- TRAINING THREAD ---
def train_model(model_id: str, bridge: DataBridge):
    print(f"[{model_id}] Training thread started.")
    
    env = PettingZooWSEnv(bridge, num_agents=NUM_AGENTS)
    env = ss.frame_stack_v1(env, stack_size=STACK_SIZE)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = SB3VecEnvWrapper(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    stats_path = os.path.join(MODELS_DIR, f"{model_id}_vecnormalize.pkl")
    
    if os.path.exists(model_path):
        try:
            print(f"[{model_id}] Loading existing model...")
            model = SAC.load(model_path, env=env)
            # Load normalization stats if they exist
            if os.path.exists(stats_path):
                print(f"[{model_id}] Loading normalization stats...")
                env = VecNormalize.load(stats_path, env)
        except Exception as e:
            print(f"[{model_id}] Load failed: {e}. Creating new.")
            model = SAC("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, ent_coef="auto")
    else:
        print(f"[{model_id}] Creating new SAC model.")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, ent_coef="auto")

    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODELS_DIR, name_prefix=model_id)
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)
        
        # Save Model
        model.save(os.path.join(MODELS_DIR, model_id))
        # Save Normalization Stats
        env.save(stats_path)
        print(f"[{model_id}] Saved model and normalization stats.")
        
    except Exception as e:
        print(f"[{model_id}] Training Error: {e}")
    finally:
        env.close()

app = FastAPI()

# --- WEBSOCKET ENDPOINT ---vv
@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    await websocket.accept()

    # Initialize the bridge
    bridge = DataBridge(num_agents=NUM_AGENTS)
    
    # Start training logic in a separate thread
    thread_name = f"train_{model_id}"
    thread = threading.Thread(target=train_model, args=(model_id, bridge), name=thread_name, daemon=True)
    thread.start()

    async def sender_task():
        """Reads commands from Bridge (from RL thread) and sends to Roblox"""
        while True:
            cmd = bridge.get_outgoing_command()
            if cmd:
                await websocket.send_json(cmd)
            else:
                await asyncio.sleep(0.01) # Avoid busy loop

    async def receiver_task():
        """Reads JSON from Roblox and routes to Bridge queues"""
        while True:
            # Expected JSON format from Roblox: 
            # { "car_0": {"obs": [...], "reward": X, ...}, "car_1": ... }
            raw_data = await websocket.receive_json()
            
            if isinstance(raw_data, dict):
                for agent_id, agent_data in raw_data.items():
                    bridge.put_incoming_data(agent_id, agent_data)

    try:
        # Run sender and receiver concurrently
        await asyncio.gather(sender_task(), receiver_task())
        
    except (WebSocketDisconnect, Exception) as e:
        print(f"[{model_id}] Connection closed: {e}")
    finally:
        # Ensure thread cleanup if possible, though daemon threads die with main
        print(f"[{model_id}] Websocket session ended.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)