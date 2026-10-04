"""
PPO (Proximal Policy Optimization) Trainer

A policy gradient method that uses clipped surrogate objective for stable training.
Good for continuous action spaces and environments with high variance.
"""

from typing import Any, Optional

from stable_baselines3 import PPO

from .base import ModelTrainer


HYPERPARAMETERS = {
    "total_timesteps": 200_000,
    "learning_rate": 0.0003,
    "batch_size": 4096,
    "n_steps": 2048,
    "n_epochs": 4,
    "ent_coef": 0.01,
    "net_arch": [128, 128],
    "device": "cpu",
    "clip_range": 0.2,
}


class PPOTrainer(ModelTrainer):
    """
    PPO Trainer - Proximal Policy Optimization

    Best for:
    - Continuous action spaces
    - Environments requiring exploration
    - Stable, reliable training
    """

    NAME = "ppo"
    ALGORITHM = PPO

    HYPERPARAMETERS = HYPERPARAMETERS

    LOAD_PARAMETERS = {
        "clip_range": lambda _: HYPERPARAMETERS["clip_range"],
        "ent_coef": HYPERPARAMETERS["ent_coef"],
        "learning_rate": HYPERPARAMETERS["learning_rate"],
        "lr_schedule": lambda _: HYPERPARAMETERS["learning_rate"],
        "batch_size": HYPERPARAMETERS["batch_size"],
        "n_epochs": HYPERPARAMETERS["n_epochs"],
    }

    def create_model(self, env: Any, tensorboard_log: Optional[str] = None) -> PPO:
        """Create a new PPO model with configured hyperparameters."""
        hp = self.HYPERPARAMETERS

        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=hp["n_epochs"],
            ent_coef=hp["ent_coef"],
            clip_range=hp["clip_range"],
            policy_kwargs=dict(net_arch=hp["net_arch"]),
            device=hp["device"],
            tensorboard_log=tensorboard_log,
        )
