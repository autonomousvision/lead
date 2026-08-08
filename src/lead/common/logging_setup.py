"""Central logging configuration for the LEAD project."""

import logging
import os
import sys

from lightning.pytorch.utilities import rank_zero_only


# ANSI color codes for terminal output
class ColorCodes:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Colors
    RED = "\033[31m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"


class LeadFormatter(logging.Formatter):
    """Formatter that shortens paths to the working directory, optionally colored."""

    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.BRIGHT_BLACK,
        logging.INFO: ColorCodes.BRIGHT_BLUE,
        logging.WARNING: ColorCodes.BRIGHT_YELLOW,
        logging.ERROR: ColorCodes.BRIGHT_RED,
        logging.CRITICAL: ColorCodes.BOLD + ColorCodes.RED,
    }

    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color
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
        if self.use_color and record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.pathname = original_pathname

        return result


class RankFilter(logging.Filter):
    """Stamp every record with its process rank; drop INFO/DEBUG from non-zero ranks.

    Every DDP rank runs the same code, so an unfiltered INFO log gets printed
    once per rank into what's typically one merged log stream. WARNING and
    above still pass from every rank -- a problem on rank 3 is worth seeing
    even though its INFO chatter isn't.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # rank_zero_only.rank and LogRecord.rank_tag are both attached
        # dynamically (by lightning and this filter, respectively), invisible to
        # their declared types, hence the dict-style access instead of attributes.
        # Rank 0 is the norm, so only the other ranks' records get tagged.
        rank = getattr(rank_zero_only, "rank", 0)
        record.__dict__["rank_tag"] = "" if rank == 0 else f"[rank {rank}] "
        return rank == 0 or record.levelno >= logging.WARNING


def setup_logging(level: str | int | None = None, format_string: str | None = None):
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Can be set via LEAD_LOG_LEVEL environment variable.
               Defaults to INFO.
        format_string: Custom format string for log messages.
                      Defaults to a standard format with timestamp.
    """
    # Determine log level
    if level is None:
        # Check environment variable first
        env_level = os.environ.get("LEAD_LOG_LEVEL", "INFO").upper()
        resolved_level: int = getattr(logging, env_level, logging.INFO)
    elif isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = level

    # Default format with timestamp, level, rank (non-zero only), file path,
    # line number, and message
    if format_string is None:
        format_string = (
            "%(asctime)s [%(levelname)s] %(rank_tag)s"
            "[%(pathname)s:%(lineno)d] %(message)s"
        )

    # Configure root logger with basic config first
    logging.basicConfig(
        level=resolved_level,
        format=format_string,
        datefmt="%y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Get root logger
    logger = logging.getLogger()

    # Replace the handler so paths are shortened whether or not this is a
    # terminal; only the color depends on that.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        LeadFormatter(
            fmt=format_string,
            datefmt="%y-%m-%d %H:%M:%S",
            use_color=sys.stdout.isatty(),
        ),
    )
    logger.addHandler(handler)

    # Drop INFO/DEBUG from non-zero ranks at the handler, so it applies no
    # matter which logger a record came from (a Logger-level filter would
    # only catch records logged directly through that logger, not ones
    # propagating up from lead.* children).
    for handler in logger.handlers:
        handler.addFilter(RankFilter())

    # Suppress third-party loggers - only show warnings and errors
    for logger_name in [
        "matplotlib",
        "PIL",
        "urllib3",
        "asyncio",
        "srunner",
        "leaderboard",
        "agents",
        "carla",
        "timm",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set root logger to WARNING by default, so only lead.* modules show INFO
    logging.getLogger().setLevel(logging.INFO)

    # Enable your lead modules at the specified level
    logging.getLogger("lead").setLevel(resolved_level)

    logger.info(
        "Logging configured: level=%s for 'lead' modules",
        logging.getLevelName(resolved_level),
    )
