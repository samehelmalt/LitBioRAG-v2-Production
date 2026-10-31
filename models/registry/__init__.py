"""Model registry: every model role is resolved from configuration, never from code."""

from models.registry.loader import ModelRegistry, RegistryError, adapter
from models.registry.schema import ROLES, Calibration, ModelSpec, Profile

__all__ = [
    "ROLES",
    "Calibration",
    "ModelRegistry",
    "ModelSpec",
    "Profile",
    "RegistryError",
    "adapter",
]
