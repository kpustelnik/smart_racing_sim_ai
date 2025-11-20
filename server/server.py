import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium
from gymnasium import spaces


#total number of floats in the state vector (10 rays)
RAYCASTS = 10
NITRO_FUEL_STATE = 1
PREV_ACTIONS = 3
STATE_DIM = RAYCASTS + NITRO_FUEL_STATE + PREV_ACTIONS

#acceleration, steering, nitro_toggle
ACTION_DIM = 3

#hyperparams
BUFFER_SIZE = 200000        #max experience count stored in memory
BATCH_SIZE = 128            
LEARNING_STARTS = 10000     #steps to collect before first training
GAMMA = 0.99                #discount factor for future rewards
TAU = 0.005                 #soft update coefficient
LEARNING_RATE = 0.0003      


app = FastAPI()
class StepInput(BaseModel):
    last_observation: List[float]    
    last_action: List[float]         
    current_observation: List[float] 
    reward: float                    #reward received for the last action taken
    terminated: bool                       #true if the training ended forcefully (car crashed or sth like that)

class ActionOutput(BaseModel):
    action: List[float]              #[accel, steer, nitro] to perform


class DummyGymEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

    #not used but needed 
    def step(self, action):
        dummy_obs = self.observation_space.sample()
        reward = 0
        terminated = False 
        truncated = False
        info = {}
        return dummy_obs, reward, terminated, truncated, info

    #not used but needed 
    def reset(self, seed=None, options=None):
        dummy_obs = self.observation_space.sample()
        info = {}
        return dummy_obs, info


dummy_env = DummyGymEnv()

#mlp - multilayer perceptron - fully connected NN
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
    policy_kwargs=dict(net_arch=[256, 256]) #hidden layers 
)

replay_buffer = ReplayBuffer(
    BUFFER_SIZE,
    observation_space=dummy_env.observation_space, 
    action_space=dummy_env.action_space,           
    device="auto", 
    n_envs=1
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- client connected to websocket ---")
    
    try:
        while True:
            data = await websocket.receive_json()
            last_obs = np.array(data['last_observation']).reshape(STATE_DIM)
            new_obs = np.array(data['current_observation']).reshape(STATE_DIM)
            action = np.array(data['last_action']).reshape(ACTION_DIM)
            reward = data['reward']
            terminated = data['terminated']
            replay_buffer.add(
                last_obs, new_obs, action, reward, terminated, [{}]
            )

            if replay_buffer.size() > LEARNING_STARTS:
                replay_data = replay_buffer.sample(BATCH_SIZE)
                model.train(
                    replay_data.observations,
                    replay_data.actions,
                    replay_data.next_observations,
                    replay_data.dones,
                    replay_data.rewards,
                )

            action_to_take, _ = model.predict(new_obs, deterministic=False)
            await websocket.send_json({"action": action_to_take.tolist()})

    except WebSocketDisconnect:
        print("--- client disconnected ---")
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
    print(f"WebSocket on http://127.0.0.1:{PORT}/ws")
    
    uvicorn.run(app, host="127.0.0.1", port=PORT)