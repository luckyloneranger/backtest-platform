"""Strategy registry for discovering and instantiating strategies."""

from strategies.base import Strategy

_STRATEGIES: dict[str, type[Strategy]] = {}


def register(name: str):
    """Decorator to register a strategy class.

    Usage:
        @register("my_strategy")
        class MyStrategy(Strategy):
            ...
    """
    def decorator(cls: type[Strategy]):
        _STRATEGIES[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name and return a new instance."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[name]()


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return list(_STRATEGIES.keys())
