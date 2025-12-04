import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import threading
import queue
import os
from typing import List, Dict, Optional, Any, Union

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import configure
from gymnasium import spaces

# --- CONSTANTS ---
RAYCASTS = 10
NITRO_FUEL_STATE = 1
PREV_ACTIONS = 0 
STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS
ACTION_DIM = 3
NUM_ENVS = 3

# --- HYPERPARAMS ---
TOTAL_TIMESTEPS = 1_000_000 
LEARNING_RATE = 0.0003
BATCH_SIZE = 256
MODELS_DIR = "saved_models"
LOGS_DIR = "sb3_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI()

# --- BRIDGE MECHANISM ---
# This allows the background training thread to talk to the async WebSocket
class DataBridge:
    def __init__(self):
        self.action_queue = queue.Queue(maxsize=1)
        self.obs_queue = queue.Queue(maxsize=1)
        self.connected = False

    def put_actions(self, actions):
        self.action_queue.put(actions)

    def get_actions(self):
        if not self.action_queue.empty():
            return self.action_queue.get()
        return None

    def put_observations(self, obs_data):
        self.obs_queue.put(obs_data)

    def get_observations(self):
        return self.obs_queue.get()

bridges: Dict[str, DataBridge] = {}

# --- CUSTOM VEC ENV ---
class ExternalVecEnv(VecEnv):
    def __init__(self, bridge: DataBridge):
        self.bridge = bridge
        
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        
        super().__init__(NUM_ENVS, observation_space, action_space)

    def step_async(self, actions):
        self.bridge.put_actions(actions)

    def step_wait(self):
        data = self.bridge.get_observations()
        
        obs = data['obs']
        rewards = data['rewards']
        dones = data['dones']
        infos = data['infos']
        
        return obs, rewards, dones, infos

    def reset(self):
        print("Waiting for initial observation from Client...")
        data = self.bridge.get_observations()
        return data['obs']

    def close(self):
        pass

    def get_attr(self, attr_name: str, indices: Any = None) -> List[Any]:
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, **method_kwargs) -> List[Any]:
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> List[bool]:
        return [False] * self.num_envs

# --- TRAINING THREAD ---
def train_model(model_id: str):
    print(f"[{model_id}] Training thread started.")
    bridge = bridges[model_id]
    
    # 1. Create the Custom Env
    env = ExternalVecEnv(bridge)
    
    # 2. Load or Create Model
    model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    
    if os.path.exists(model_path):
        print(f"[{model_id}] Loading existing model.")
        model = SAC.load(model_path, env=env)
    else:
        print(f"[{model_id}] Creating new SAC model.")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            buffer_size=100_000,
            learning_starts=1000,
            tensorboard_log=LOGS_DIR,
            device="auto"
        )

    # 3. Start the standard SB3 loop
    # This will automatically call env.step(), buffer data, and train
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)
        model.save(os.path.join(MODELS_DIR, model_id))
        print(f"[{model_id}] Training finished.")
    except Exception as e:
        print(f"[{model_id}] Training interrupted: {e}")

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    await websocket.accept()
    print(f"--- Client connected: {model_id} ---")
    
    # Initialize Bridge
    if model_id not in bridges:
        bridges[model_id] = DataBridge()
    
    bridge = bridges[model_id]
    bridge.connected = True
    
    # Pre-allocate arrays for receiving data 
    batch_obs = np.zeros((NUM_ENVS, STATE_DIM), dtype=np.float32)
    batch_rewards = np.zeros((NUM_ENVS,), dtype=np.float32)
    batch_dones = np.zeros((NUM_ENVS,), dtype=bool)
    
    # Start Training Thread (if not running)
    # We run this in a separate thread so it doesn't block the WebSocket/FastAPI
    train_thread = threading.Thread(target=train_model, args=(model_id,), daemon=True)
    train_thread.start()

    try:
        while True:
            raw_data = await websocket.receive_json()

            for i, item in enumerate(raw_data):
                batch_obs[i] = item['current_observation']
                batch_rewards[i] = item['reward']
                batch_dones[i] = item['terminated']
            
            # Send to Bridge (Wake up the training thread)
            # We must copy arrays to prevent reference issues if overwritten quickly
            bridge.put_observations({
                'obs': batch_obs.copy(),
                'rewards': batch_rewards.copy(),
                'dones': batch_dones.copy(),
                'infos': [{} for _ in range(NUM_ENVS)]
            })
            
            # Wait for Actions from Training Thread
            # We poll the queue non-blockingly or using asyncio
            actions = None
            while actions is None:
                # We check the queue in a loop with a tiny sleep to allow context switching
                # This bridges the Sync Thread -> Async Loop gap
                if not bridge.action_queue.empty():
                    actions = bridge.action_queue.get()
                else:
                    await asyncio.sleep(0.001)

            await websocket.send_json({"actions": actions.tolist()})

    except WebSocketDisconnect:
        print(f"[{model_id}] Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"Starting Dedicated SB3 Server (Expected Agents: {NUM_ENVS})")
    uvicorn.run(app, host="0.0.0.0", port=8000)