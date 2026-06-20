import logging
import os.path
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def _add_request_id(_, __, event_dict: dict) -> dict:
    request_id = event_dict.get("request_id")
    if request_id is None:
        # When contextvars merge hasn't added request_id yet, default to '-'
        event_dict["request_id"] = "-"
    return event_dict


def configure_logging(log_level: str = "INFO", env: str = "development") -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True,  # override any pre-configured handlers (e.g., uvicorn defaults)
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if env == "development":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def setup_logger(
    log_level: str = "INFO",
    log_dir: str | Path = "data/logs",
    env: str = "development",
) -> None:
    log_level = log_level.upper()
    log_dir = Path(log_dir)

    # Shared processors for both console and file
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Setup file logging with configurable path (Docker-friendly)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "logger.log"

    # Create rotating file handler (10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    # Use ProcessorFormatter for file to render as JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Use ProcessorFormatter for console to render as colored text
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
        ]
        + shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure root logger with both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party libraries unless explicitly re-enabled
    logging.getLogger("python_multipart.multipart").setLevel(logging.INFO)
    logging.getLogger("watchfiles.main").setLevel(logging.INFO)
