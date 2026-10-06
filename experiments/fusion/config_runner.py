"""Public config-driven fusion runner interface."""

from .config_runner_impl import main, parse_args, run

__all__ = ["main", "parse_args", "run"]
