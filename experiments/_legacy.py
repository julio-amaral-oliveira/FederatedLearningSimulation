"""Compatibility adapters for the historical flat experiment entrypoints."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


def run_legacy_module(module_name: str, argv: Sequence[str] | None = None):
    """Run a legacy module while exposing a namespaced CLI entrypoint."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [module_name, *argv]
    try:
        return runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = original_argv
