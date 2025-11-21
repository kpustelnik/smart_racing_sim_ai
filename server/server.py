import asyncio
from collections import deque
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Deque, Dict, List
import os
import threading

from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium
from gymnasium import spaces


#total number of floats in the state vector (10 rays)
RAYCASTS = 10
NITRO_FUEL_STATE = 1
PREV_ACTIONS = 0 #3
STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS

#acceleration, steering, nitro_toggle
ACTION_DIM = 3

#hyperparams
BUFFER_SIZE = 200000        #max experience count stored in memory
BATCH_SIZE = 128            
LEARNING_STARTS = 5000      #steps to collect before first training
GAMMA = 0.99                #discount factor for future rewards
TAU = 0.005                 #soft update coefficient
LEARNING_RATE = 0.0003     

TRAIN_FREQUENCY = 100       #train every 100 steps
GRADIENT_STEPS = 100        #how many updates to do when we do train
SAVE_FREQUENCY = 5_000      #save model every N steps
MODELS_DIR = "saved_models"
LOGS_DIR = "sb3_logs"

#best model saving
WINDOW_SIZE = 50             #how many episodes to average over
MIN_EPISODES_BEFORE_SAVE = 5 #dont save "best" until we have a few runs

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
app = FastAPI()

class DummyGymEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)

    def step(self, action): return self.observation_space.sample(), 0, False, False, {}
    def reset(self, seed=None, options=None): return self.observation_space.sample(), {}

class ModelStats:
    def __init__(self):
        self.current_episode_reward = 0.0
        self.recent_scores: Deque[float] = deque(maxlen=WINDOW_SIZE)
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        self.step_count = 0
        self.is_training = False #lock flag

dummy_env = DummyGymEnv()
active_models: Dict[str, SAC] = {}
model_stats: Dict[str, ModelStats] = {}
model_locks: Dict[str, threading.Lock] = {}

def get_or_create_model(model_id: str) -> SAC:
    if model_id in active_models:
        return active_models[model_id]

    model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    if os.path.exists(model_path):
        print(f"--- Loading existing model: {model_id} ---")
        model = SAC.load(model_path, env=dummy_env)
    else:
        print(f"--- Creating NEW model: {model_id} ---")
        model = SAC(
            "MlpPolicy",
            dummy_env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            learning_starts=LEARNING_STARTS,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            policy_kwargs=dict(net_arch=[256, 256])
        )

    log_path = os.path.join(LOGS_DIR, model_id)
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)
    
    active_models[model_id] = model
    model_stats[model_id] = ModelStats()
    model_locks[model_id] = threading.Lock()
    
    return model

async def run_training_background(model: SAC, model_id: str):
    stats = model_stats[model_id]
    lock = model_locks[model_id] 

    try:
        for _ in range(GRADIENT_STEPS):
            # Define a wrapper that acquires the lock in the thread
            def train_step():
                with lock:
                    model.train(gradient_steps=1, batch_size=BATCH_SIZE)
            # Run ONE gradient step in a background thread
            await asyncio.to_thread(train_step)
            # Sleep briefly to yield control back to the event loop (allowing inference to happen)
            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[{model_id}] Training Error: {e}")
    finally:
        stats.is_training = False

@app.websocket("/ws/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    await websocket.accept()
    print(f"--- client connected to websocket for model ID: {model_id} ---")
    
    model = get_or_create_model(model_id)
    stats = model_stats[model_id]
    lock = model_locks[model_id] 

    try:
        while True:
            data = await websocket.receive_json()
            # print(data)

            last_obs = np.array(data['last_observation'], dtype=np.float32).reshape(1, STATE_DIM)
            new_obs = np.array(data['current_observation'], dtype=np.float32).reshape(1, STATE_DIM)
            action = np.array(data['last_action'], dtype=np.float32).reshape(1, ACTION_DIM)
            reward = np.array([data['reward']], dtype=np.float32)
            terminated = np.array([data['terminated']], dtype=bool)

            reward_val = float(data['reward'])
            terminated_val = bool(data['terminated'])

            infos = [{}]
            if not stats.is_training: 
                with lock:
                    model.replay_buffer.add(
                        last_obs, 
                        new_obs, 
                        action, 
                        reward, 
                        terminated, 
                        infos
                    )
            
            stats.step_count += 1
            stats.current_episode_reward += reward_val

            if terminated_val:
                stats.episode_count += 1
                stats.recent_scores.append(stats.current_episode_reward)
                
                #calculate average
                mean_reward = np.mean(stats.recent_scores)
                print(f"[{model_id}] Ep {stats.episode_count} finished. Score: {stats.current_episode_reward:.2f} | Mean (L{len(stats.recent_scores)}): {mean_reward:.2f}")

                #check for new highscore
                if (stats.episode_count >= MIN_EPISODES_BEFORE_SAVE and mean_reward > stats.best_mean_reward):
                    stats.best_mean_reward = mean_reward
                    path = os.path.join(MODELS_DIR, f"{model_id}_best")
                    # PROTECT SAVE
                    with lock:
                        model.save(path)
                    print(f"[{model_id}] NEW BEST MEAN REWARD: Saved to {path}.zip")

                # Reset current counter
                stats.current_episode_reward = 0.0

            if (model.replay_buffer.size() > LEARNING_STARTS and stats.step_count % TRAIN_FREQUENCY == 0):
                if not stats.is_training:
                    stats.is_training = True
                    asyncio.create_task(run_training_background(model, model_id))

            if stats.step_count % SAVE_FREQUENCY == 0:
                save_path = os.path.join(MODELS_DIR, model_id)
                # PROTECT SAVE
                with lock:
                    model.save(save_path)
                print(f"[{model_id}] Periodic Checkpoint Saved.")

            # PROTECT INFERENCE
            with lock:
                action_to_take, _ = model.predict(new_obs, deterministic=False)
            
            await websocket.send_json({"action": action_to_take[0].tolist()})

    except WebSocketDisconnect:
        print("--- client disconnected ---")
        save_path = os.path.join(MODELS_DIR, model_id)
        with lock:
            model.save(save_path)
    except Exception as e:
        print(f"An error occurred: {e}")

@app.get("/")
def root():
    return {"message": "RL server is running", "state_dim": STATE_DIM, "action_dim": ACTION_DIM}

if __name__ == "__main__":
    PORT = 8000

    print("--- Starting RL server ---")
    print(f"Observation (State) Dimension: {STATE_DIM}")
    print(f"Action Dimension: {ACTION_DIM}")
    print(f"Server will start training after {LEARNING_STARTS} steps.")
    print(f"Listening on http://127.0.0.1:8000")
    print(f"WebSocket on http://127.0.0.1:{PORT}/ws/<model_id>")
    
    uvicorn.run(app, host="127.0.0.1", port=PORT)