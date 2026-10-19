"""Standalone DSPy prompt-optimization lab.

The package intentionally imports no DSPy, provider, or UI modules here.
Importing :mod:`dspy_lab` must stay lightweight and must never perform network
I/O or construct an LM.
"""

from __future__ import annotations

import os

__all__ = ["__version__"]

__version__ = "0.1.0"

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
