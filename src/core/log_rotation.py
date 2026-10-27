"""Log rotation module.

Provides daily log file rotation with automatic cleanup of expired archives.

Features:
- Rotates log at midnight to `backend-YYYY-MM-DD.log`
- Automatically deletes logs older than retention_days (configurable)
- Safe for concurrent access (handles missing files gracefully)
"""

import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


logger = logging.getLogger(__name__)

# Log file naming pattern
LOG_ARCHIVE_PATTERN = re.compile(r"^backend-(\d{4}-\d{2}-\d{2})\.log$")
DEFAULT_RETENTION_DAYS = 7


def clean_old_logs(logs_dir: Path, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Delete archived log files older than retention_days.

    Scans the logs directory for files matching `backend-YYYY-MM-DD.log`,
    parses the date, and removes those exceeding the retention period.

    Args:
        logs_dir: Directory containing log files.
        retention_days: Number of days to keep logs.

    Returns:
        Number of deleted files.
    """
    if not logs_dir.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    for f in logs_dir.iterdir():
        if not f.is_file():
            continue
        match = LOG_ARCHIVE_PATTERN.match(f.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d")
            if file_date < cutoff:
                f.unlink()
                deleted += 1
                logger.info("Deleted expired log: %s", f.name)
        except ValueError:
            # Filename matches pattern but date is invalid — skip
            continue
        except OSError as e:
            logger.warning("Failed to delete %s: %s", f.name, e)

    if deleted > 0:
        logger.info("Log cleanup: deleted %d expired files (retention=%d days)", deleted, retention_days)

    return deleted


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """TimedRotatingFileHandler that names rotated files `backend-YYYY-MM-DD.log`.

    Overrides the default suffixExt to produce clean date-based filenames
    instead of appending `.2026-08-04` to the base filename.
    """

    def __init__(self, logs_dir: Path, retention_days: int = DEFAULT_RETENTION_DAYS, **kwargs):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

        current_log = logs_dir / "backend.log"

        super().__init__(
            filename=str(current_log),
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
            **kwargs,
        )

        # Override namer to produce `backend-YYYY-MM-DD.log` instead of
        # `backend.log.1`
        self.namer = self._namer

    def _namer(self, default_name: str) -> str:
        """Rename rotated file to `backend-YYYY-MM-DD.log`."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return str(self.logs_dir / f"backend-{yesterday}.log")

    def doRollover(self):
        """Perform rollover and then clean up expired logs."""
        super().doRollover()
        try:
            clean_old_logs(self.logs_dir, self.retention_days)
        except Exception as e:
            logger.warning("Log cleanup after rollover failed: %s", e)


def setup_logging(logs_dir: Path, retention_days: int = DEFAULT_RETENTION_DAYS) -> None:
    """Configure application-wide logging with daily rotation.

    This replaces the default logging config and should be called once at
    application startup (before any other module imports logger).

    Args:
        logs_dir: Directory for log files.
        retention_days: Days to keep old logs.
    """
    # Clean expired logs on startup
    deleted = clean_old_logs(logs_dir, retention_days)
    if deleted:
        print(f"已清理 {deleted} 个过期日志文件")

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates on restart
    root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = DailyRotatingFileHandler(logs_dir, retention_days)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ["urllib3", "watchfiles", "shutil"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    current_log = logs_dir / "backend.log"
    logger.info("Logging initialized: %s (retention=%d days)", current_log, retention_days)
