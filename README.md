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
├── main.py                 # Main entry point with CLI
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

Please note that Roblox was historically sticking to basic Http communication. WebSockets are a fresh feature that is still within beta. We have encountered some problems (eg. with memory leaking) with this project.

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

## 6. RL Model Hyperparameters

### PPO (Proximal Policy Optimization)

```python
HYPERPARAMETERS = {
    "learning_rate": 0.0003,
    "batch_size": 64,           # Smaller for on-policy updates
    "n_steps": 1024,            # Steps before each update
    "ent_coef": 0.01,           # Entropy coefficient for exploration
    "net_arch": [256, 256],     # Neural network architecture
}
```

**Characteristics:**
- On-policy algorithm (uses current policy data only)
- Clipped surrogate objective for stable training
- Good for continuous action spaces
- Lower sample efficiency but more stable

### SAC (Soft Actor-Critic)

```python
HYPERPARAMETERS = {
    "learning_rate": 0.0003,
    "batch_size": 256,          # Larger for off-policy
    "buffer_size": 100_000,     # Replay buffer size
    "tau": 0.005,               # Target network update rate
    "ent_coef": "auto",         # Auto-tuned entropy
    "net_arch": [256, 256],
}
```

**Characteristics:**
- Off-policy algorithm (learns from replay buffer)
- Maximum entropy framework for robust exploration
- Higher sample efficiency
- More complex, requires larger batch sizes

### Built-in models
Built-in models are being automatically saved (along with the VecNormalize layers). Furthermore Tensorboard logs are being prepared for further inspection.

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

---

## Roblox Environment

### Directory Structure (game/)

```
game/
├── ServerScriptService/
│   ├── main.server.luau                # Entry point - WebSocket & track generation
│   ├── Car/                            # Vehicle physics & scripting
│   │   ├── CarScripts/
│   │   │   ├── Controller.luau         # Input handling, nitro, speed control
│   │   │   ├── Constants.luau          # Attribute names, physics constants
│   │   │   └── Units.luau              # Unit conversions (studs → mph)
│   │   ├── Recolor.luau                # Environment-based car coloring
│   │   └── WeldAll.luau                # Welding car parts together (Creating proper joints)
│   ├── CarsAIController/
│   │   ├── init.luau                   # WebSocket client & message routing
│   │   ├── CmdHandler.luau             # Command execution (spawn, reset, action)
│   │   ├── CarsController.luau         # Car spawning (preventing collisions), collision groups, observations handling
│   │   └── CheckpointsController.luau  # Checkpoint & reward logic
│   ├── TrackGeneration/
│   │   ├── init.luau                   # Track generator (shape) & finalizer (cosmetics) orchestrator
│   │   ├── types.luau                  # Point type definitions
│   │   ├── Methods/                    # Track generation algorithms
│   │   │   ├── Main.luau               # A*-inspired path generation
│   │   │   └── RemoteSolver.luau       # OR-Tools (deprecated) integration via HTTP
│   │   └── Finalizers/                 # Track cosmetics generation (mesh / tiles / others...)
│   └── Utilities/                      # Helper modules
├── ServerStorage/
│   ├── TrackBase.rbxmx                 # Road segment templates
│   └── CarModels/                      # Vehicle models
└── StarterPlayer/
    └── StarterPlayerScripts/           # Less relevant client-side scripts
```

### Entry Point (`main.server.luau`)

Handles main settings and initializes the modules (generation and environment controller).

```luau
local wsUrl: string = `ws://localhost:8000/ws/{modelId}`  -- WebSocket URL
local TrackGeneration = require(TrackGeneration)          -- Track generator
local CarsAIController = require(CarsAIController)        -- Agent controller

TrackGeneration('Main', 'Main')                           -- Generate track
CarsAIController.Init(carModel, wsUrl)                    -- Start training
```

### WebSocket Command Handler (`CmdHandler.luau`)

Commands received from Python server:

| Command | Data | Description |
|---------|------|-------------|
| `CREATE_ENV` | `{}` | Create a new virtual environment with collision group |
| `SPAWN_AGENTS` | `{agents: string[]}` | Spawn/respawn agents in environment |
| `RESET_AGENTS` | `{agents: string[]}` | Reset agents (teleport to spawn) |
| `REMOVE_AGENTS` | `{agents: string[]}` | Remove agents from environment |
| `ACTION` | `{[agentId]: [throttle, steering, nitro]}` | Apply inputs to cars |
| `FREEZE` | `{status: boolean}` | Anchor/unanchor all car parts |
| `UPDATE_COLLISIONS` | `{enable_collisions: boolean}` | Toggle inter-agent collisions |
| `CLOSE` | `{}` | Cleanup single environment |
| `CLOSE_ALL` | `{}` | Cleanup all environments |

### Car Physics System

The car controller uses Roblox's physics engine with:

- **Throttle/Brake**: Torque applied to wheels based on `throttleInput`
- **Steering**: Wheel hinge angle based on `steeringInput` with speed-based reduction
- **Nitro**: Temporary linear velocity boost (limited fuel, auto-recharge)
- **Speed (Velocity)**

On top of those values the below values may be controlled:
- **Wheels density**
- **Wheels elasticity**
- **Wheels kinetic friction**
- **Wheels slip threshold**
- **Wheels static friction**
Enabling natural drift of the vehicles.
- **Suspension camber**
- **Suspension damping**
- **Suspension length**
- **Suspension stiffness**
- **Steering rack responsiveness**
- **Steering rack speed**
- **Steering rack reduction**
- **Engine acceleration**
- **Engine braking**
- **Engine deceleration**
- **Engine max forward & reverse speed**
- **Engine hand brake, min, max torque** (for wheels correction and to support natural drifting)
- **Nitro acceleration**
- **Nitro max speed**
- **Nitro time & recharge time**
- **Nitro torque**
The vehicles also automatically redress in case they are placed upside-down.

Those settings should cast some light on the vehicles' physics handling. It should be furthermore mentioned that the vehicles utilize built-in physics constraints built into Roblox engine. Those are springs, linear velocities, cylindrical constraints, orientation aligner, prismatic constraints and of course hinge constraints used for connecting pieces together.

On top of that each vehicle has configurable raycasts for observations via a `Config` script put inside the vehicle model.
```luau
local ChassisPart = script.Parent:WaitForChild('Chassis')
return {
	Rays = {
		{
			Source = ChassisPart, -- Source position of the raycast
			Direction = Vector3.new(0, 0, -15) * 25, -- Direction of the raycast
			IncludeNormal = false, -- Whether additional info about raycast normal should be included
			IgnoreCars = true, -- Whether this raycast should ignore other vehicles within same virtual environment
      IgnoreXZRotation = false -- Only account Y-axis rotation of the vehicle (raycasts would be horizontal)
		},
		...
	}
}
```
In future due to difference in track's height it may be beneficial to artificially generate big invisible track walls and exclude the road itself from the raycasts.

### Other agent observations

Other observations are defined within the `CarsController.luau` file (`RetrieveObservation` function). By default the observations consist of the normalized raycasts results, nitro status, velocity and information whether the car is following the right direction (not driving in the opposite way).

### Checkpoints tracking

Checkpoints are used to determine the agents' progress in track. They're also used to count the laps reliably.
Parts representing checkpoints are being placed during the cosmetics generation phase. They are put where nodes have previously been. Agent's position is determined by finding the closest ray between checkpoints' centers and measuring it's relative position on it. Then the distance of all previous checkpoints is appropriately accounted.

### Reward Function

Reward function is being determined within the `CmdHandler.luau` script. It also performs the checkpoints tracking.

```lua
... -- Checkpoint tracking
... -- Truncation handling (if no progress has been made for a while)
local reward = 0

reward += (currentProgress - lastProgress) * 1000
if reward < 0 then reward *= 20 end -- Multiply additionaly by 20 if the reward negative

local normalizedSpeed: number = math.clamp(velocity / maxVelocity, 0, 1)
local speedBonus: number = (math.exp(normalizedSpeed * 2) - 1) * 0.5 -- max ~3.2 at full speed
reward += speedBonus

-- non-linear speed bonus (exponential scaling)
  local carEngine = carData.CarEngine
  local velocityStatus: number = carEngine:GetAttribute('_speed') :: any
  local maxVelocity: number = carEngine:GetAttribute('forwardMaxSpeed') :: any


  if carData.TruncationPending or carData.TerminationPending then
    return -20, totalProgress
  end

  return reward, totalProgress
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

Agent is also automatically being truncated if they do not make any track progress for a given amount of time (or the model gets removed from the map). Agent is being terminated by default if it hits any wall.

### Multi-Environment Isolation

Each virtual environment gets its own **collision group** via PhysicsService:
- Agents in different environments may or may not collide basing on the `UPDATE_COLLISIONS` setting
- Different environments are visually distinguished by highlight colors

---

## Available Car Models

Located in `game/ServerStorage/CarModels/`:

| Model | File | Description |
|-------|------|-------------|
| Bus | `Bus.rbxmx` | Large vehicle with slower handling |
| Car | `Car.rbxmx` | Standard sedan (default for training) |
| Golf Cart | `Golf cart.rbxmx` | Small, nimble vehicle |
| Quad | `Quad.rbxmx` | ATV-style vehicle |
| Tesla | `Telsa.rbxmx` | Electric vehicle|
| Truck | `Truck.rbxmx` | Heavy-duty truck |

Models were found within Roblox freemodels catalogue and modified to apply the same physics.

Model being used may be specified in `main.server.luau`:
```luau
local carModel: string = 'Telsa'  -- Change to any model name
...
```

---

## Track Generation

### Overview

Track generation has two phases:
1. **Generator**: Creates path points (positions in 3D space)
2. **Finalizer**: Converts path to visual road segments

```luau
TrackGeneration('Main', 'Main')  -- Generator: Main, Finalizer: Main
```

### Generators

#### Main Generator (`Methods/Main.luau`)

Uses A*-like pathfinding with constraint satisfaction:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PointsNumber` | 120 | Total candidate points generated |
| `minimumPoints` | 30 | Minimum path length (nodes count) |
| `targetPoints` | 40 | Target path length (nodes count) |
| `minimumDistance` | 130 | Min distance between points |
| `maxDistance` | 500 | Max edge length |
| `pathDistance` | 100 | Min distance of node from existing path |

Constraints:
- No self-intersecting paths
- Angle constraints for drivability (different characteristics - some nodes enforce light turns, while some make it more likely to generate a sharp turn)
- Use prebuilt custom structures such as designed bridges, roundabouts etc.
- Connected loop (start → end)

The search tree is huge and therefore quite often trimmed to find the solution faster. The below principles were taken into account during the generation:
1. No node can repeat in path
2. No direct connection between end and start (also mitigated by minimum path nodes)
3. No intersection with previous path
4. No intersection with not existing path between end and start
5. Respect minimum and maximum nodes distance
6. Do not collide with prebuilt structures
7. Node can't be too close to any path
8. Bending angle should be appropriate based on the directions of the track
9. Make sure that next node may be enforced within prebuilt structures
10. Grade track basing on the total distance (prefer longer tracks) and turn angles
11. Grade track basing on the distance to end over latest point (to create closed-loop more quickly)
12. Measure based on track length (try to achieve the target nodes count)
13. Account the nodes bonuses (eg. to encourage the algorithm to use prebuilt structures)

#### Remote Solver (`Methods/RemoteSolver.luau`)

Offloads path generation to Python server using OR-Tools CP-SAT solver was attempted but the model had too many constraints and therefore could not be reasonably used. We determined that it is better to stick to the current heuristic-driven greedy solution.

### Finishers (Cosmetics generators)

Nodes picked for track generation are being used to further generate the actual track. Catmull-Rom spline helps to create a smooth transition between nodes. Some nodes have pre-set direction (such as pre-built structures) that is being used to determine the some of the spline points. For basic nodes the direction is based on their neighbouring nodes.

Few finishers were implemented as we were trying to achieve the best quality possible:
- **Simple** puts really basic road parts along the spline and connects them using Stravant's align module (that resizes the parts properly to fill the gaps between them).
- **MainTile** uses pre-created `TrackBase` located within `ServerStorage` that is being generated in two directions (forward from the previous checkpoint and backward from the next one). Stravant's redupe module is being utilized to make sure that the tiles stick to each other with at least one side and then to fill gaps taking all sub-parts into account.
- **MainParts** uses similar approach to the MainTile but instead of ensuring connectivity the parts are being placed along the spline just making sure that they do not collide. They are further used to generate a smooth continuous mesh (Also uses the forward and backward approach).
- **Main** best tested approach so far. Spline points are being probed more frequently. Sometimes they in fact overlap but an algorithm is put in place to make sure that no points are allowed to be placed behind already placed ones (Also uses the forward and backward approach).

## Pre-built structuers

Pre-built structures such as bridges, roundabouts etc. may be created and have specific configuration. First of all they may contain some custom textures. Then static `Nodes` may be placed into them

```luau
local data = {}

data.NodeMap = { -- If track goes through any of those nodes, the next (exit from structure) node is determined. This makes it possible to "skip" the track cosmetics generation within this structure
	["1"] = "2",
	["2"] = "1",
	["3"] = "4",
	["4"] = "3"
}
data.OnTrackGenerated = function (obj, objPoints) -- Some changes may be made to the structure basing on the points that ended up being used in track (eg. removing the structure entirely if it was not used at all or hiding some parts of it)
	if #objPoints <= 0 then obj:Destroy() end
end
```

Pre-built structures and terrain generation were out of scope of this projects as this may be easily achieved by simple programs or utilizing existing [Roblox engine tools](https://create.roblox.com/docs/studio/terrain-editor).

### Installation

1. Install Roblox studio
2. Install [Rokit](https://github.com/rojo-rbx/rokit) to install other packages
3. Install packages such as [Rojo](https://github.com/rojo-rbx/rojo) to sync objects from the filesystem
4. Run the game (server will connect via WebSocket)

---
    
# Credits
- https://www.roblox.com/games/215383192/Classic-Racing - car templates and physics
- https://github.com/stravant/roblox-redupe - aligning road templates
- https://github.com/stravant/roblox-resizealign/blob/main/src/main.server.lua - simple aligning road templates
- https://gist.github.com/LukeMS/89dc587abd786f92d60886f4977b1953 - priority queue implementation
- https://github.com/rojo-rbx/rokit - syncing assets with Roblox
and all accompanying libraries

Last but not least - Roblox!