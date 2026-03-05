from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from traceback import format_exception
from typing import Any

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
_BASE_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(  # noqa: UP017
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }

        for field_name, value in record.__dict__.items():
            if field_name in _BASE_LOG_RECORD_FIELDS or field_name == "event":
                continue
            if value is not None:
                payload[field_name] = value

        if record.exc_info is not None:
            exception_type, exception_value, exception_traceback = record.exc_info
            if exception_type is not None and exception_value is not None:
                payload["exception_type"] = exception_type.__name__
                payload["exception_message"] = str(exception_value)
                payload["exception_traceback"] = "".join(
                    format_exception(exception_type, exception_value, exception_traceback)
                )

        return json.dumps(payload, default=str)


def configure_logging(level_name: str) -> None:
    level = _LOG_LEVELS[level_name]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers.clear()
    uvicorn_access_logger.propagate = False
    uvicorn_access_logger.disabled = True

    for logger_name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)
