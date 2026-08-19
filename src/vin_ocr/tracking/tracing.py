"""
MLflow tracing for the VIN recognition app - optional, loud, never fatal.

Pattern per MLflow's own instrumentation skill
(mlflow/assistant/skills/instrumenting-with-mlflow-tracing): no autolog
integration exists for a custom paddle inference stack, so custom code uses
the ``@mlflow.trace`` decorator on root operations and external calls, and
skips granular utilities.

Design constraints honoured here:

- The inference pipeline must import and run WITHOUT mlflow installed
  (tracing lives in the optional [tracking] extra). ``traced()`` is a
  no-op passthrough until :func:`enable_tracing` succeeds.
- The tracking URI follows the project convention: ``MLFLOW_TRACKING_URI``
  when set, else the repo-local SQLite store used by the training runs -
  the same resolver the run tracker uses, so traces land next to the runs
  they belong with.
- Failure to enable is reported loudly and returns False; it never breaks
  the app. Silent no-ops are how this repo's historical bugs hid.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Experiment traces are recorded under, matching the training runs.
DEFAULT_EXPERIMENT = "vin_finetune"

_enabled = False


def is_enabled() -> bool:
    """Whether tracing has been switched on for this process."""
    return _enabled


def enable_tracing(experiment: str = DEFAULT_EXPERIMENT) -> bool:
    """
    Turn on MLflow tracing for this process.

    Resolves the tracking URI (``MLFLOW_TRACKING_URI`` env, else the repo's
    SQLite store), selects the experiment, and arms the ``traced()``
    decorators.

    Returns:
        True when tracing is active; False when mlflow is unavailable or
        setup failed (reported via logger.warning - the app continues
        untraced).
    """
    global _enabled
    try:
        import mlflow

        from .run import resolve_tracking_uri
    except ImportError as exc:
        logger.warning(
            "MLflow tracing unavailable (%s); install the [tracking] extra "
            "to record traces. The app continues untraced.", exc,
        )
        return False

    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip() or resolve_tracking_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
    except Exception as exc:
        logger.warning(
            "MLflow tracing setup failed (%s: %s); the app continues "
            "untraced.", type(exc).__name__, exc,
        )
        return False

    _enabled = True
    logger.info("MLflow tracing enabled -> %s (experiment %r)", uri, experiment)
    return True


def disable_tracing() -> None:
    """Turn tracing off (used by tests; safe to call anytime)."""
    global _enabled
    _enabled = False


def traced(name: Optional[str] = None, span_type: str = "CHAIN") -> Callable:
    """
    Decorator: record the function as an MLflow span when tracing is on.

    A no-op passthrough (zero overhead beyond one flag check) until
    :func:`enable_tracing` succeeds, so decorated modules stay importable
    and fast without mlflow.

    Args:
        name: Span name; defaults to the function's qualified name.
        span_type: MLflow SpanType name - 'CHAIN' for root operations,
            'TOOL' for external engine calls, 'PARSER' for postprocessing.
    """
    def decorator(fn: Callable) -> Callable:
        traced_fn: Optional[Callable] = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced_fn
            if not _enabled:
                return fn(*args, **kwargs)
            if traced_fn is None:
                import mlflow
                traced_fn = mlflow.trace(
                    fn, name=name or fn.__qualname__, span_type=span_type
                )
            return traced_fn(*args, **kwargs)

        return wrapper

    return decorator
