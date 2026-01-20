from .base import ModelTrainer, TrainingBridge
from .ppo import PPOTrainer
from .sac import SACTrainer

MODEL_REGISTRY: dict[str, type[ModelTrainer]] = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
}

def get_trainer(model_type: str) -> type[ModelTrainer]:
    """Get a trainer class by model type name."""
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")
    return MODEL_REGISTRY[model_type]

def list_available_models() -> list[str]:
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())

__all__ = [
    "ModelTrainer",
    "TrainingBridge",
    "get_trainer",
    "list_available_models",
    "MODEL_REGISTRY",
]
