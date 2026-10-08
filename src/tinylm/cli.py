"""Shared argument handling for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse

from tinylm.config import ExperimentConfig, load_config
from tinylm.utils import setup_logging


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults, and keep the docstring's example commands verbatim."""


def make_parser(description: str, config_required: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=_HelpFormatter)
    parser.add_argument(
        "--config", required=config_required, help="Path to an experiment YAML config."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.max_steps=100 (repeatable).",
    )
    return parser


def load(args: argparse.Namespace) -> ExperimentConfig:
    setup_logging()
    return load_config(args.config, args.overrides)
