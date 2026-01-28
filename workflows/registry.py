from typing import Dict, Type
from .core import Workflow
from tools.logging_config import amd_logger

_REGISTRY: Dict[str, Type[Workflow]] = {}


def register_workflow(name: str):
    """Register a Workflow class under a name."""
    def decorator(cls: Type[Workflow]):
        if name in _REGISTRY:
            amd_logger.critical(f"workflow '{name}' already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_workflow(name: str) -> Type[Workflow] | None:
    wf = _REGISTRY.get(name)
    if wf is None:
        amd_logger.critical(f"Unknown workflow '{name}'. Available: {available_workflows()}")
    return wf


def available_workflows() -> list[str]:
    return sorted(_REGISTRY.keys())
