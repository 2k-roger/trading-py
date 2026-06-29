"""Strategy registry — decorator-based plug-in system."""

from typing import Type

from .base import Strategy

_registry: dict[str, Type[Strategy]] = {}


def register(name: str):
    """Decorator: register a Strategy subclass under *name*."""

    def decorator(cls: Type[Strategy]) -> Type[Strategy]:
        if name in _registry:
            raise ValueError(f"Strategy '{name}' is already registered")
        _registry[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get_strategy(name: str) -> Type[Strategy]:
    """Look up a strategy class by registered name."""
    if name not in _registry:
        available = ', '.join(sorted(_registry))
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {available or '(none)'}"
        )
    return _registry[name]


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_registry)
