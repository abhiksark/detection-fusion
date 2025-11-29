import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .defaults import DEFAULT_ENV_PREFIX
from .models import RectificationConfig, StrategyConfig


class ConfigLoader:
    @classmethod
    def from_yaml(cls, path: Path) -> StrategyConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyConfig:
        return StrategyConfig.model_validate(data)

    @classmethod
    def from_env(cls, prefix: str = DEFAULT_ENV_PREFIX) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                result[config_key] = cls._parse_env_value(value)
        return result

    @classmethod
    def _parse_env_value(cls, value: str) -> Any:
        lower = value.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @classmethod
    def load_merged(
        cls,
        yaml_path: Optional[Path] = None,
        env_prefix: Optional[str] = DEFAULT_ENV_PREFIX,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> StrategyConfig:
        base_data: Dict[str, Any] = {}

        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path, "r") as f:
                base_data = yaml.safe_load(f) or {}

        if env_prefix:
            env_data = cls.from_env(env_prefix)
            base_data = cls._deep_merge(base_data, env_data)

        if overrides:
            base_data = cls._deep_merge(base_data, overrides)

        return cls.from_dict(base_data)

    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def load_rectification(cls, path: Path) -> RectificationConfig:
        """Load a RectificationConfig from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RectificationConfig instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return RectificationConfig.model_validate(data)

    @classmethod
    def rectification_from_dict(cls, data: Dict[str, Any]) -> RectificationConfig:
        """Create a RectificationConfig from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            RectificationConfig instance
        """
        return RectificationConfig.model_validate(data)
