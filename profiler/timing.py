"""
Core timing primitives for JAXTPC profiling.

Provides TimingResult, time_function with proper
JAX GPU synchronization via jax.block_until_ready().
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class TimingResult:
    """Container for timing results."""
    name: str
    times_ms: List[float] = field(default_factory=list)
    description: str = ""

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms)) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms)) if self.times_ms else 0.0

    def __repr__(self):
        return f"{self.name}: {self.mean_ms:.2f} +/- {self.std_ms:.2f} ms (n={len(self.times_ms)})"

    def to_dict(self):
        d = {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "n_runs": len(self.times_ms),
            "times_ms": self.times_ms,
        }
        if self.description:
            d["description"] = self.description
        if hasattr(self, 'enabled_components'):
            d["enabled_components"] = self.enabled_components
        return d


def sync_result(result):
    """Recursively call jax.block_until_ready() on JAX arrays in result."""
    if isinstance(result, jnp.ndarray):
        jax.block_until_ready(result)
    elif isinstance(result, (list, tuple)):
        for item in result:
            sync_result(item)
    elif isinstance(result, dict):
        for value in result.values():
            sync_result(value)


def time_function(func, *args, sync=True, **kwargs):
    """
    Time a single function call with proper JAX synchronization.

    Returns (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if sync:
        sync_result(result)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


