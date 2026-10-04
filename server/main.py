"""
3D Racing AI Training Server

A modular WebSocket server for training RL agents in a Roblox racing environment.

Usage:
    python main.py --model-type ppo
    python main.py --model-type sac --mode use
    python main.py --list-models

Arguments:
    --model-type, -m    Model type to use (ppo, sac, etc.)
    --mode              Mode: 'train' (default) or 'use' (inference only)
    --list-models       List all available model types
    --port, -p          Server port (default: 8000)
    --host              Server host (default: 0.0.0.0)
"""

import argparse
import asyncio
import queue
import re
import threading
import time
import traceback
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from models import get_trainer, latency, list_available_models, TrainingBridge

# CONFIGURATION
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
OBSERVATION_TIMEOUT = 10.0
PING_INTERVAL = 5.0
COMMAND_POLL_INTERVAL = 0.01
CONTROL_PREFIX = "__"
SHUTDOWN_GRACE = 30.0

MAX_QUEUED_OBSERVATIONS = 256
MAX_TRACKED_AGENTS = 512

SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# DATA BRIDGE (internal implementation)
class DataBridge:
    """
    Internal bridge for WebSocket <-> Training thread communication.

    Use TrainingBridge facade for documented public API.
    """

    def __init__(self):
        self.obs_queues: dict[str, queue.Queue] = {}
        self.command_queue: queue.Queue = queue.Queue()
        self._queues_lock = threading.Lock()
        self._closed = threading.Event()
        self._warned_agent_cap = False

    def _queue_for(self, agent: str, capped: bool) -> Optional[queue.Queue]:
        with self._queues_lock:
            agent_queue = self.obs_queues.get(agent)
            if agent_queue is not None:
                return agent_queue

            if capped and len(self.obs_queues) >= MAX_TRACKED_AGENTS:
                if not self._warned_agent_cap:
                    self._warned_agent_cap = True
                    print(
                        f"[bridge] Ignoring observations for further agents; {len(self.obs_queues)} are already tracked."
                    )
                return None

            agent_queue = queue.Queue(maxsize=MAX_QUEUED_OBSERVATIONS)
            self.obs_queues[agent] = agent_queue
            return agent_queue

    def get_agent_queue(self, agent: str) -> queue.Queue:
        agent_queue = self._queue_for(agent, capped=False)
        assert agent_queue is not None
        return agent_queue

    def _all_queues(self) -> list[queue.Queue]:
        with self._queues_lock:
            return list(self.obs_queues.values())

    def put_incoming_data(self, agent: str, obs_data):
        agent_queue = self._queue_for(agent, capped=True)
        if agent_queue is None:
            return
        while True:
            try:
                agent_queue.put_nowait(obs_data)
                return
            except queue.Full:
                try:
                    agent_queue.get_nowait()
                except queue.Empty:
                    return

    def get_latest_obs(self, agent: str):
        """Newest queued observation, or the next one to arrive if the queue is empty."""
        agent_queue = self.get_agent_queue(agent)

        latest_data = None
        try:
            while True:
                latest_data = agent_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_data is None and not self._closed.is_set():
            try:
                latest_data = agent_queue.get(timeout=OBSERVATION_TIMEOUT)
            except queue.Empty:
                pass

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
        for agent_queue in self._all_queues():
            try:
                while True:
                    agent_queue.get_nowait()
            except queue.Empty:
                pass

    def is_closed(self) -> bool:
        return self._closed.is_set()

    def close(self):
        self._closed.set()
        for agent_queue in self._all_queues():
            try:
                agent_queue.put_nowait(None)
            except queue.Full:
                try:
                    agent_queue.get_nowait()
                    agent_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass


# --- TRAINING THREAD ---
def run_trainer(model_type: str, model_id: str, bridge: DataBridge, mode: str):
    """Run a trainer against the given bridge, either training or inference only."""
    trainer = get_trainer(model_type)(TrainingBridge(bridge), model_id)
    try:
        if mode == "train":
            trainer.train()
        else:
            trainer.use()
    except Exception:
        print(f"[{model_id}] Trainer thread crashed:")
        traceback.print_exc()


# FASTAPI APP FACTORY
def create_app(model_type: str, mode: str = "train") -> FastAPI:
    """Create FastAPI app configured for the specified model type and mode."""

    mode_desc = "Training" if mode == "train" else "Inference"
    app = FastAPI(
        title=f"3D Racing AI {mode_desc} Server",
        description=f"{mode_desc} server using {model_type.upper()} model",
    )

    @app.websocket("/ws/{model_id}")
    async def websocket_endpoint(websocket: WebSocket, model_id: str):
        if not SAFE_MODEL_ID.match(model_id) or model_id in {".", ".."}:
            print(f"Rejected websocket for unusable model id: {model_id!r}")
            await websocket.close(code=1008)
            return

        await websocket.accept()

        bridge = DataBridge()

        thread = threading.Thread(
            target=run_trainer,
            args=(model_type, model_id, bridge, mode),
            name=f"{mode}_{model_id}",
            daemon=True,
        )
        thread.start()

        transport_latency = latency.get_tracker("transport", report_every=20)

        async def sender_task():
            """Reads commands from Bridge and sends to Roblox."""
            loop = asyncio.get_running_loop()
            last_sent: float = loop.time()
            while True:
                cmd = bridge.get_outgoing_command()
                now: float = loop.time()
                if cmd:
                    last_sent = now
                    await websocket.send_json(cmd)
                else:
                    await asyncio.sleep(COMMAND_POLL_INTERVAL)
                if now - last_sent >= PING_INTERVAL:
                    last_sent = now
                    await websocket.send_json(
                        {"command": "PING", "data": {"t": time.perf_counter()}}
                    )

        def handle_control_message(key: str, payload) -> None:
            if key == "__pong" and isinstance(payload, dict):
                sent_at = payload.get("t")
                if isinstance(sent_at, (int, float)):
                    transport_latency.record(time.perf_counter() - sent_at)

        async def receiver_task():
            """Reads JSON from Roblox and routes to Bridge queues."""
            while True:
                raw_data = await websocket.receive_json()
                if isinstance(raw_data, dict):
                    for agent_id, agent_data in raw_data.items():
                        if agent_id.startswith(CONTROL_PREFIX):
                            handle_control_message(agent_id, agent_data)
                            continue
                        bridge.put_incoming_data(agent_id, agent_data)

        tasks = [asyncio.create_task(sender_task()), asyncio.create_task(receiver_task())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
        except WebSocketDisconnect:
            print(f"[{model_id}] Roblox disconnected.")
        except Exception:
            print(f"[{model_id}] Connection closed with an error:")
            traceback.print_exc()
        finally:
            bridge.close()
            await asyncio.to_thread(thread.join, SHUTDOWN_GRACE)
            if thread.is_alive():
                print(f"[{model_id}] Trainer still shutting down after {SHUTDOWN_GRACE:.0f}s.")
            print(f"[{model_id}] Websocket session ended.")

    @app.get("/")
    async def root():
        return {
            "status": "running",
            "model_type": model_type,
            "mode": mode,
            "available_models": list_available_models(),
            "latency_enabled": latency.is_enabled(),
        }

    @app.get("/latency")
    async def latency_report():
        """
        Current bridge delays.
        """
        if not latency.is_enabled():
            return {
                "enabled": False,
                "hint": "restart the server with --latency to collect timings",
            }
        return {"enabled": True, "trackers": latency.snapshot_all()}

    return app


# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Racing AI Training Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --model-type ppo
    python main.py -m sac --port 8080
    python main.py --mode use -m ppo       # Inference only (no training)
    python main.py --list-models
        """,
    )

    parser.add_argument(
        "--model-type", "-m",
        type=str,
        default="ppo",
        help="Model type to use for training (default: ppo)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "use"],
        default="train",
        help="Mode: 'train' for training, 'use' for inference only (default: train)",
    )

    parser.add_argument(
        "--latency",
        action="store_true",
        help="Measure and report Roblox bridge delays (also serves GET /latency)",
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
            summary = get_trainer(model_name).get_description().strip().splitlines()[0]
            print(f"  - {model_name}: {summary}")
        return

    # Validate model type
    try:
        get_trainer(args.model_type)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.latency:
        latency.enable()
        print("Latency reporting enabled; GET /latency for a snapshot.")

    mode_str = "TRAINING" if args.mode == "train" else "INFERENCE (use)"
    print(f"Starting server in {mode_str} mode with model type: {args.model_type}")
    print(f"Listening on {args.host}:{args.port}")

    app = create_app(args.model_type, mode=args.mode)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
