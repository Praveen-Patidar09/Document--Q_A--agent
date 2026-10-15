"""Opt-in, content-bearing pipeline traces for local debugging.

Enable with DOCQA_DIAGNOSTICS=true. Trace records go through Python logging and are
disabled by default; callers should avoid enabling them in shared log environments.
"""

import logging

from app.config import settings

logger = logging.getLogger("app.pipeline_trace")


def trace(stage: str, **fields) -> None:
    if settings.diagnostics:
        logger.info("TRACE %s %s", stage, fields)
