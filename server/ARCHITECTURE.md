# 3D Racing AI - Server Architecture

## Overview

This server provides a WebSocket-based bridge between a Roblox racing game and Reinforcement Learning (RL) models. It enables training AI agents to drive cars using algorithms like PPO and SAC.

```
┌─────────────────────┐         WebSocket          ┌─────────────────────┐
│                     │◄──────────────────────────►│                     │
│   Roblox Game       │   JSON messages            │   Python Server     │
│   (Luau)            │                            │   (FastAPI)         │
│                     │                            │                     │
│  - Car physics      │                            │  - RL training      │
│  - Observations     │                            │  - Model inference  │
│  - Reward calc      │                            │  - Action selection │
└─────────────────────┘                            └─────────────────────┘
```

---

## Directory Structure

```
server/
├── new.py                 # Main entry point with CLI
├── ARCHITECTURE.md        # This file
└── models/
    ├── __init__.py        # Model registry & exports
    ├── base.py            # TrainingBridge facade + ModelTrainer ABC
    ├── ppo.py             # PPO trainer implementation
    └── sac.py             # SAC trainer implementation
```

---

## Core Components

### 1. Main Server (`new.py`)

The entry point that:
- Parses CLI arguments (`--model-type`, `--port`, etc.)
- Creates FastAPI application
- Manages WebSocket connections
- Spawns training threads

**CLI Usage:**
```bash
python new.py --model-type ppo           # Train with PPO (default)
python new.py -m sac --port 8080         # Train with SAC on custom port
python new.py --list-models              # List available model types
```

### 2. DataBridge (Internal)

Low-level communication layer between WebSocket and training thread.

```
┌───────────────────────────────────────────────────────────────┐
│                         DataBridge                            │
├───────────────────────────────────────────────────────────────┤
│  obs_queues: dict[agent_id, Queue]    # Incoming observations │
│  command_queue: Queue                  # Outgoing commands     │
├───────────────────────────────────────────────────────────────┤
│  put_incoming_data(agent, data)        # WebSocket → Queue    │
│  get_latest_obs(agent) → data          # Queue → Training     │
│  send_command(cmd, env_id, data)       # Training → Queue     │
│  get_outgoing_command() → payload      # Queue → WebSocket    │
└───────────────────────────────────────────────────────────────┘
```

### 3. TrainingBridge (Facade)

Public API for model trainers. Wraps DataBridge with documented, limited methods.

| Method | Description |
|--------|-------------|
| `spawn_agents(env_id, agents)` | Request Roblox to spawn agents |
| `reset_agents(env_id, agents)` | Reset/respawn agents |
| `send_actions(env_id, actions)` | Send action commands to cars |
| `close_environment(env_id)` | Cleanup environment |
| `get_observation(agent)` | Get latest observation (blocking, 10s timeout) |
| `clear_observations()` | Clear all observation queues |

### 4. ModelTrainer (Abstract Base Class)

Base class for all RL trainers. Defines the interface:

```python
class ModelTrainer(ABC):
    HYPERPARAMETERS: Dict[str, Any] = {}
    
    def __init__(self, bridge: TrainingBridge, model_id: str): ...
    
    @abstractmethod
    def create_model(self, env) -> Any: ...
    
    @abstractmethod
    def train(self) -> None: ...
```

### 5. PettingZoo Environment

Multi-agent environment compatible with Stable-Baselines3.

```
PettingZooWSEnv
    │
    ├── Wraps TrainingBridge for Gym-compatible interface
    ├── Manages multiple virtual environments (venvs)
    ├── Handles observation/action spaces
    │
    └── Pipeline:
        env = PettingZooWSEnv(bridge, num_agents=5)
        env = ss.frame_stack_v1(env, stack_size=4)      # Temporal info
        env = ss.pettingzoo_env_to_vec_env_v1(env)      # Vectorize
        env = SB3VecEnvWrapper(env)                      # SB3 compat
        env = VecNormalize(env, norm_obs=True, ...)     # Normalize
```

---

## Communication Protocol

### WebSocket Messages

**Server → Roblox (Commands):**
```json
{
    "command": "SPAWN_AGENTS" | "RESET" | "ACTION" | "CLOSE",
    "envid": "uuid-string",
    "data": { ... }
}
```

**Roblox → Server (Observations):**
```json
{
    "car_0_env_xxx": {
        "obs": [raycast1, ..., raycast10, nitro, velocity, isReverse],
        "reward": 15.5,
        "terminated": false,
        "truncated": false
    },
    "car_1_env_xxx": { ... }
}
```

### Command Types

| Command | Data | Description |
|---------|------|-------------|
| `SPAWN_AGENTS` | `{agents: ["car_0_env_xxx", ...]}` | Create new car agents |
| `RESET` | `{agents: ["car_0_env_xxx", ...]}` | Respawn specified agents |
| `ACTION` | `{car_0_env_xxx: [throttle, steer, nitro], ...}` | Apply actions |
| `CLOSE` | `{}` | Cleanup and destroy agents |

---

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Roblox    │────►│  WebSocket  │────►│ DataBridge  │────►│  Training   │
│   Game      │     │  Receiver   │     │   Queues    │     │   Thread    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ▲                                                           │
       │                                                           │
       │            ┌─────────────┐     ┌─────────────┐            │
       └────────────│  WebSocket  │◄────│ DataBridge  │◄───────────┘
                    │   Sender    │     │   Commands  │
                    └─────────────┘     └─────────────┘

1. Roblox sends observations (raycasts, velocity, rewards)
2. WebSocket receiver routes to per-agent queues
3. Training thread reads observations, runs model inference
4. Model outputs actions → command queue
5. WebSocket sender transmits actions to Roblox
6. Roblox applies actions to cars
7. Loop repeats at ~15 FPS
```

---

## Observation Space

| Index | Value | Range | Description |
|-------|-------|-------|-------------|
| 0-9 | Raycasts | 0-1 | Normalized distance to obstacles |
| 10 | Nitro | 0-1 | Current nitro fuel level |
| 11 | Velocity | -1 to 1 | Normalized speed (forward/backward) |
| 12 | IsReverse | 0 or 1 | Car facing wrong direction |

**With Frame Stacking (STACK_SIZE=4):**
- Final observation shape: `(13 * 4,) = (52,)`
- Provides temporal information for velocity/acceleration inference

---

## Action Space

| Index | Value | Range | Description |
|-------|-------|-------|-------------|
| 0 | Throttle | -1 to 1 | Brake ← → Accelerate |
| 1 | Steering | -1 to 1 | Left ← → Right |
| 2 | Nitro | -1 to 1 | Activated if > 0.5 |

---

## Reward Function

Calculated in Roblox (`CarsAIController.luau`):

```lua
-- Progress reward (main driver)
progressReward = (currentProgress - lastProgress) * 1000

-- Reverse penalty (20x multiplier)
if progressReward < 0 then
    progressReward *= 20
end

-- Speed bonus (exponential scaling)
normalizedSpeed = clamp(velocity / maxVelocity, 0, 1)
speedBonus = (exp(normalizedSpeed * 2) - 1) * 0.5  -- Max ~3.2

-- Final reward
reward = progressReward + speedBonus

-- Termination penalty
if terminated then reward = -10 end
```

| Component | Typical Value | Purpose |
|-----------|---------------|---------|
| Progress | ~2/tick | Main learning signal |
| Speed Bonus | 0-3.2 | Encourages high speed |
| Reverse Penalty | -40 to -60 | Discourages backing up |
| Termination | -10 | Crash/stuck penalty |

---

## Adding a New Model

1. Create file in `models/` (e.g., `td3.py`):

```python
from .base import ModelTrainer, TrainingBridge
from stable_baselines3 import TD3

class TD3Trainer(ModelTrainer):
    """TD3 - Twin Delayed DDPG"""
    
    HYPERPARAMETERS = {
        "total_timesteps": 200_000,
        "learning_rate": 0.001,
        "batch_size": 256,
        "device": "cpu",
    }
    
    def create_model(self, env):
        return TD3("MlpPolicy", env, **self.HYPERPARAMETERS)
    
    def train(self):
        # Setup environment (same pattern as PPO/SAC)
        env = PettingZooWSEnv(self.bridge, ...)
        # ... wrap with supersuit ...
        
        model = self.create_model(env)
        model.learn(total_timesteps=self.HYPERPARAMETERS["total_timesteps"])
        model.save(...)
```

2. Register in `models/__init__.py`:

```python
from .td3 import TD3Trainer

MODEL_REGISTRY = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "td3": TD3Trainer,  # Add here
}
```

3. Use:
```bash
python new.py --model-type td3
```

---

## Threading Model

```
Main Thread (asyncio)
    │
    ├── FastAPI/Uvicorn event loop
    ├── WebSocket sender coroutine
    └── WebSocket receiver coroutine
    
Training Thread (daemon)
    │
    ├── Environment step loop
    ├── Model inference
    └── Gradient updates
    
Communication:
    │
    └── Thread-safe Queue objects in DataBridge
```

**Key Points:**
- Training runs in daemon thread (dies with main process)
- Queues provide thread-safe communication
- `get_latest_obs()` blocks with 10s timeout to prevent deadlock
- Sender/receiver run as concurrent asyncio tasks

---

## Configuration Constants

```python
# Observation dimensions
RAYCASTS = 10
NITRO_FUEL_STATE = 1
VELOCITY_STATE = 1
BASE_STATE_DIM = 12  # (13 with isReverse in newer version)

# Environment
STACK_SIZE = 4       # Frames to stack
ACTION_DIM = 3       # throttle, steering, nitro
NUM_AGENTS = 5       # Cars per virtual environment
NUM_VENVS = 4        # Parallel virtual environments

# Training
TOTAL_TIMESTEPS = 200_000
MODELS_DIR = "saved_models"
LOGS_DIR = "sb3_logs"
```

---

## Saved Files

After training:
```
saved_models/
├── {model_id}.zip                    # Trained model weights
├── {model_id}_vecnormalize.pkl       # Normalization statistics
└── {model_id}_{step}_steps.zip       # Checkpoint saves
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Timeout waiting for obs | Roblox not connected | Ensure game is running and connected |
| "Executed while running last" | Frame rate too fast | Reduce FPS or optimize Roblox code |
| Model not learning | Reward too sparse | Check reward function values |
| Actions not applied | Protocol mismatch | Verify JSON format matches expected |
