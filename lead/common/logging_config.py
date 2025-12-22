"""
Central logging configuration for the LEAD project.

This module provides a single point to configure logging for all components.
Import and call setup_logging() at the start of your application.
"""

import logging
import os
import sys


# ANSI color codes for terminal output
class ColorCodes:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels and uses relative paths."""

    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.BRIGHT_BLACK,
        logging.INFO: ColorCodes.BRIGHT_BLUE,
        logging.WARNING: ColorCodes.BRIGHT_YELLOW,
        logging.ERROR: ColorCodes.BRIGHT_RED,
        logging.CRITICAL: ColorCodes.BOLD + ColorCodes.RED,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get current working directory for relative path calculation
        self.cwd = os.getcwd()

    def format(self, record):
        # Save original values
        original_levelname = record.levelname
        original_pathname = record.pathname

        # Convert absolute path to relative path
        try:
            record.pathname = os.path.relpath(record.pathname, self.cwd)
        except ValueError:
            # If relpath fails (e.g., different drives on Windows), keep absolute
            pass

        # Add color to level name
        if record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.pathname = original_pathname

        return result


def setup_logging(level=None, format_string=None):
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Can be set via LEAD_LOG_LEVEL environment variable.
               Defaults to INFO.
        format_string: Custom format string for log messages.
                      Defaults to a standard format with timestamp.

    Returns:
        The root logger configured with the specified settings.
    """
    # Determine log level
    if level is None:
        # Check environment variable first
        env_level = os.environ.get("LEAD_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default format with timestamp, level, file path, line number, and message
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s"

    # Configure root logger with basic config first
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Get root logger
    logger = logging.getLogger()

    # Replace handler with colored formatter if outputting to terminal
    if sys.stdout.isatty():
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create new handler with colored formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Suppress third-party loggers - only show warnings and errors
    # This makes only your 'lead.*' code logs visible at INFO level
    third_party_loggers = [
        "matplotlib",
        "PIL",
        "urllib3",
        "asyncio",
        "srunner",
        "leaderboard",
        "agents",
        "carla",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set root logger to WARNING by default, so only lead.* modules show INFO
    # This prevents unknown third-party modules from cluttering output
    logging.getLogger().setLevel(logging.WARNING)

    # Enable your lead modules at the specified level
    logging.getLogger("lead").setLevel(level)

    logger.info("Logging configured: level=%s for 'lead' modules", logging.getLevelName(level))

    return logger


def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        A logger instance
    """
    return logging.getLogger(name)


# Convenience function to set log level at runtime
def set_log_level(level):
    """
    Change the log level at runtime.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as string or int
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.getLogger().setLevel(level)
    logging.info("Log level changed to: %s", logging.getLevelName(level))
