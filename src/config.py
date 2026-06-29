"""Configuration loading from YAML with dict overrides."""

from pathlib import Path
from typing import Optional

import yaml

from .models import BacktestConfig


class ConfigLoader:
    """Loads and merges configuration from YAML and runtime overrides."""

    DEFAULT_PATH = Path(__file__).parent.parent / 'config' / 'default.yaml'

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        overrides: Optional[dict] = None,
    ) -> BacktestConfig:
        """Load config from YAML, merge overrides, return BacktestConfig.

        Args:
            path: Path to YAML config file. Uses default.yaml if None.
            overrides: Dict of values to override on top of file config.

        Returns:
            BacktestConfig with all defaults applied.
        """
        path = path or cls.DEFAULT_PATH
        raw: dict = {}

        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    raw.update(loaded)

        if overrides:
            cls._deep_merge(raw, overrides)

        return BacktestConfig(**raw)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Recursively merge override dict into base dict in-place."""
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                ConfigLoader._deep_merge(base[key], val)
            else:
                base[key] = val
