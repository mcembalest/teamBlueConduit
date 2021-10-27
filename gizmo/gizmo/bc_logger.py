#!/usr/bin/env python3
"""
The BlueConduit Logging Module
==============================

Maintainer: ac@blueconduit.com
Purpose: Advanced and custom log configuration, environment aware and
_beautiful!_
"""

import structlog
import os


# ENV VARS
DEVELOPMENT_ENV_FLAG_STR = "development"
PRODUCTION_ENV_FLAG_STR = "production"
ENV_FLAG = os.getenv("ENV_FLAG", DEVELOPMENT_ENV_FLAG_STR)


def get_simple_logger(*args, **kwargs):
    """A simple logger, based on structlog. Passes all arguments to
    structlog. Use instead of the base python logger."""

    # Configure JSON output logging for production log viewers
    if ENV_FLAG == PRODUCTION_ENV_FLAG_STR:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    return structlog.get_logger(*args, **kwargs)
