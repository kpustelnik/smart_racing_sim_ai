"""
3D Racing AI Training Server

A modular WebSocket server for training RL agents in a Roblox racing environment.

Usage:
    python new.py --model-type ppo
    python new.py --model-type sac
    python new.py --list-models

Arguments:
    --model-type, -m    Model type to use (ppo, sac, etc.)
    --list-models       List all available model types
    --port, -p          Server port (default: 8000)
    --host              Server host (default: 0.0.0.0)
"""

import argparse
import asyncio
import queue
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from models import get_trainer, list_available_models, TrainingBridge
import threading


# --- CONFIGURATION ---
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
NUM_AGENTS = 5


# --- DATA BRIDGE (internal implementation) ---
class DataBridge:
    """
    Internal bridge for WebSocket <-> Training thread communication.
    
    Use TrainingBridge facade for documented public API.
    """
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.obs_queues: dict[str, queue.Queue] = {}
        self.command_queue: queue.Queue = queue.Queue()

    def get_agent_queue(self, agent: str) -> queue.Queue:
        if agent not in self.obs_queues:
            self.obs_queues[agent] = queue.Queue()
        return self.obs_queues[agent]

    def put_incoming_data(self, agent: str, obs_data):
        q = self.get_agent_queue(agent)
        q.put(obs_data)

    def get_latest_obs(self, agent: str):
        c_queue = self.get_agent_queue(agent)
        latest_data = None
        try:
            while True:
                latest_data = c_queue.get_nowait()
        except queue.Empty:
            pass
        if latest_data is None:
            latest_data = c_queue.get(timeout=10)
        return latest_data

    def send_command(self, command: str, env_id: str, data: Optional[dict] = None):
        payload = {"command": command, "data": data if data else {}, "envid": env_id}
        self.command_queue.put(payload)

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


# --- TRAINING THREAD ---
def train_model(model_type: str, model_id: str, bridge: DataBridge):
    """Start training with the specified model type."""
    trainer_class = get_trainer(model_type)
    training_bridge = TrainingBridge(bridge)
    trainer = trainer_class(training_bridge, model_id)
    trainer.train()


# --- FASTAPI APP FACTORY ---
def create_app(model_type: str) -> FastAPI:
    """Create FastAPI app configured for the specified model type."""
    
    app = FastAPI(
        title="3D Racing AI Training Server",
        description=f"Training server using {model_type.upper()} model",
    )

    @app.websocket("/ws/{model_id}")
    async def websocket_endpoint(websocket: WebSocket, model_id: str):
        await websocket.accept()

        bridge = DataBridge(num_agents=NUM_AGENTS)
        
        thread_name = f"train_{model_id}"
        thread = threading.Thread(
            target=train_model,
            args=(model_type, model_id, bridge),
            name=thread_name,
            daemon=True,
        )
        thread.start()

        async def sender_task():
            """Reads commands from Bridge and sends to Roblox."""
            while True:
                cmd = bridge.get_outgoing_command()
                if cmd:
                    await websocket.send_json(cmd)
                else:
                    await asyncio.sleep(0.01)

        async def receiver_task():
            """Reads JSON from Roblox and routes to Bridge queues."""
            while True:
                raw_data = await websocket.receive_json()
                if isinstance(raw_data, dict):
                    for agent_id, agent_data in raw_data.items():
                        bridge.put_incoming_data(agent_id, agent_data)

        try:
            await asyncio.gather(sender_task(), receiver_task())
        except (WebSocketDisconnect, Exception) as e:
            print(f"[{model_id}] Connection closed: {e}")
        finally:
            print(f"[{model_id}] Websocket session ended.")

    @app.get("/")
    async def root():
        return {
            "status": "running",
            "model_type": model_type,
            "available_models": list_available_models(),
        }

    return app


# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Racing AI Training Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python new.py --model-type ppo
    python new.py -m sac --port 8080
    python new.py --list-models
        """,
    )
    
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        default="ppo",
        help="Model type to use for training (default: ppo)",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model types and exit",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})",
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list_models:
        print("Available model types:")
        for model_name in list_available_models():
            trainer_class = get_trainer(model_name)
            print(f"  - {model_name}: {trainer_class.get_description().strip().split(chr(10))[0]}")
        return
    
    # Validate model type
    try:
        get_trainer(args.model_type)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Starting server with model type: {args.model_type}")
    print(f"Listening on {args.host}:{args.port}")
    
    app = create_app(args.model_type)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()