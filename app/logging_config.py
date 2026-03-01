"""Production logging configuration."""

import json
import logging
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """JSON log formatter for production/cloud."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(log_format: str = "text") -> None:
    """Configure root logger. Call at startup."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.removeHandler(root.handlers[0])
    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    root.addHandler(handler)
