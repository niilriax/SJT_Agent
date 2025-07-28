from ..config import BackendConfig
from .base import BACKEND_REGISTRY, IntelligenceBackend, register_backend
from .human import Human
from .openai import OpenAIChat


# Load a backend from a config dictionary
def load_backend(config: BackendConfig):
    try:
        backend_cls = BACKEND_REGISTRY[config.backend_type]
    except KeyError:
        raise ValueError(f"Unknown backend type: {config.backend_type}")

    backend = backend_cls.from_config(config)
    return backend
