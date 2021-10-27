#!/usr/bin/env python3
"""
Run this script after `export ENV_FLAG=production` or `export
ENV_FLAG=development`. With "production", print only warnings and errors, ignore
info and debug, format as json. With "development", print all levels, colorize
output, format nicely.
"""

from gizmo.bc_logger import get_simple_logger, ENV_FLAG


if __name__ == "__main__":
    print("Logging for mode: %s\n" % ENV_FLAG)

    log = get_simple_logger(__name__)

    # Log at various levels.
    log.error("an error", key="val")
    log.warning("a warning", key="val")
    log.info("some info", key="val")
    log.debug("some debug info", key="val")
