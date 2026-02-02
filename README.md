# Smart Racing Sim AI

## Overview

This repository contains a ready to run Roblox Reinforced Learning environment (Gymnasium, Stable Baselines 3, Pettingzoo and Supersuit).
It is composed of the Roblox environment part and Python backend part that communicate with each other via WebSocket 
The WebSocket-based bridge enables training AI agents to drive cars using algorithms like PPO and SAC.
Repository may easily be extended by different models and libraries simply by utilizing the exposed environment API.
Roblox environment has few examples of vehicles added along with few methods of race track generation (the shape and cosmetics are being generated separately).

It is possible to train the agents in parallel even allowing them to collide with each others within a custom virtual environment.
It's also possible to simulate few virtual environment at one time.

```
┌─────────────────────┐         WebSocket          ┌─────────────────────┐
│                     │◄──────────────────────────►│                     │
│   Roblox Game       │   JSON messages            │   Python Server     │
│   (Luau)            │                            │   (FastAPI)         │
│                     │                            │                     │
│  - Car physics      │                            │  - RL training      │
│  - Observations     │                            │  - Model inference  │
│  - Reward calc      │                            │  - Action selection │
│  - Racetrack gen.   │                            └─────────────────────┘
└─────────────────────┘
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

## Core Server Components

### 1. Main Http Server (`main.py`)

The entry point that:
- Parses CLI arguments (`--model-type`, `--port`, etc.)
- Creates FastAPI application
- Manages WebSocket connections
- Spawns training threads

**CLI Usage:**
```bash
pip install -r requirements.txt           # Install required packages

python main.py -h                         # View help
python main.py --model-type ppo           # Train with PPO (default)
python main.py -m sac --port 8080         # Train with SAC on custom port
python main.py --list-models              # List available model types
```

### 2. DataBridge (Internal)

Low-level communication layer between WebSocket and training thread.

```
┌───────────────────────────────────────────────────────────────┐
│                         DataBridge                            │
├───────────────────────────────────────────────────────────────┤
│  obs_queues: dict[agent_id, Queue]    # Incoming observations │
│  command_queue: Queue                  # Outgoing commands    │
├───────────────────────────────────────────────────────────────┤
│  put_incoming_data(agent, data)        # WebSocket → Queue    │
│  get_latest_obs(agent) → data          # Queue → Training     │
│  send_command(cmd, env_id, data)       # Training → Queue     │
│  get_outgoing_command() → payload      # Queue → WebSocket    │
│  clear_data()                          # Clears agents queues │
└───────────────────────────────────────────────────────────────┘
```

### 3. TrainingBridge (Facade)

Public API for model trainers. Wraps DataBridge with documented, limited methods.

| Method | Description |
|--------|-------------|
| `freeze(status)` | Requests Roblox to freeze or unfreeze agents' physics |
| `spawn_agents(env_id, agents)` | Request Roblox to spawn agents (they're respawned if they exist already) |
| `reset_agents(env_id, agents)` | Reset/respawn agents |
| `remove_agents(env_id, agents)` | Remove agents |
| `send_actions(env_id, actions)` | Send action commands to cars |
| `close_environment(env_id)` | Cleanup environment |
| `close_all()` | Cleanups all created environments |
| `update_collisions(status)` | Updates collisions between agents within each virtual environment |
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
    
    # Trains the model
    @abstractmethod
    def train(self) -> None: ...
    
    # Runs the model in inference mode
    @abstractmethod
    def use(self) -> None: ...
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
        env = SB3VecEnvWrapper(env)                     # SB3 compat
        env = VecNormalize(env, norm_obs=True, ...)     # Normalize
```

---

## Communication Protocol

### WebSocket Messages

**Server → Roblox (Commands):**
```json
{
    "command": "SPAWN_AGENTS" | "RESET_AGENTS" | "ACTION" | "CLOSE" | ...,
    "envid": "uuid-venv-string",
    "data": { ... }
}
```

**Roblox → Server (Observations):**
```json
{
    "car_0_env_xxx": {
        "obs": [raycasts..., nitro, velocity, isReverse],
        "reward": 15.5,
        "terminated": false,
        "truncated": false
    },
    "car_1_env_xxx": { ... }
}
```

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
*. When model is being trained internally, the environment is getting frozen temporarily to avoid vehicle simulation when not receiving live actions from backend.
```

## Proposed Observation Space

| Index | Value | Range | Description |
|-------|-------|-------|-------------|
| 0-9 | Raycasts | 0-1 | Normalized distance to obstacles |
| 10 | Nitro | 0-1 | Current nitro fuel level |
| 11 | Velocity | -1 to 1 | Normalized speed (forward/backward) |
| 12 | IsReverse | 0 or 1 | Car facing wrong direction |

**With Frame Stacking (STACK_SIZE=4):**
- Final observation shape: `(13 * 4,) = (52,)`
- Provides temporal information for velocity/acceleration inference

## Proposed Action Space

| Index | Value | Range | Description |
|-------|-------|-------|-------------|
| 0 | Throttle | -1 to 1 | Brake ← → Accelerate |
| 1 | Steering | -1 to 1 | Left ← → Right |
| 2 | Nitro | -1 to 1 | Activated if > 0.5 |

    
# Credits
- https://www.roblox.com/games/215383192/Classic-Racing - car templates and physics
- https://github.com/stravant/roblox-redupe - aligning road templates
- https://github.com/stravant/roblox-resizealign/blob/main/src/main.server.lua - simple aligning road templates
- https://gist.github.com/LukeMS/89dc587abd786f92d60886f4977b1953 - priority queue implementation
- https://github.com/rojo-rbx/rokit - syncing assets with Roblox
and all accompanying libraries

Last but not least - Roblox!