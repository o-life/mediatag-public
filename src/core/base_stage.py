"""Base class for stage processors with checkpoint/resume support.

Each processing stage inherits from BaseStageProcessor and implements
unit-level processing with atomic status tracking in the database.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Dict

logger = logging.getLogger(__name__)


class BaseStageProcessor(ABC):
    """Abstract base class for stage processors.

    Provides:
    - Atomic unit-level processing with status tracking
    - Resume from last completed unit
    - Progress callback support
    - Error handling with retry

    Subclasses must implement:
    - get_processing_units(task_id)
    - get_unit_status(unit_id)
    - update_unit_status(unit_id, status, error_msg)
    - process_unit(unit_id)
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name

    @abstractmethod
    def get_processing_units(self, task_id: int) -> List[Any]:
        """Return all processing units for a task.

        Each unit should have an identifier usable by get_unit_status()
        and process_unit().
        """
        ...

    @abstractmethod
    def get_unit_status(self, unit_id: Any) -> str:
        """Return the processing status of a unit (pending/processing/completed/failed)."""
        ...

    @abstractmethod
    def update_unit_status(self, unit_id: Any, status: str, error_msg: Optional[str] = None):
        """Update the processing status of a unit."""
        ...

    @abstractmethod
    def process_unit(self, unit_id: Any) -> bool:
        """Process a single unit. Returns True on success."""
        ...

    def process_all(
        self,
        task_id: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_retries: int = 0,
    ) -> Dict[str, Any]:
        """Process all units for a task, with resume support.

        Args:
            task_id: Task identifier
            progress_callback: Optional callback(current, total, message)
            max_retries: Number of retries for failed units (0 = no retry)

        Returns:
            Dict with total, completed, failed, skipped counts
        """
        units = self.get_processing_units(task_id)
        total = len(units)

        if total == 0:
            logger.info("[%s] No units to process for task %d", self.stage_name, task_id)
            return {"total": 0, "completed": 0, "failed": 0, "skipped": 0}

        logger.info("[%s] Processing %d units for task %d", self.stage_name, total, task_id)

        completed = 0
        failed = 0
        skipped = 0

        for i, unit_id in enumerate(units):
            status = self.get_unit_status(unit_id)

            if status == "completed":
                skipped += 1
                continue

            self.update_unit_status(unit_id, "processing")

            success = False
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    result = self.process_unit(unit_id)
                    if result:
                        success = True
                        break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "[%s] Unit %s attempt %d/%d failed: %s",
                        self.stage_name, unit_id, attempt + 1, max_retries + 1, e,
                    )
                    time.sleep(0.1 * (attempt + 1))

            if success:
                self.update_unit_status(unit_id, "completed")
                completed += 1
            else:
                self.update_unit_status(unit_id, "failed", last_error)
                failed += 1

            if progress_callback:
                progress_callback(
                    i + 1, total,
                    f"{self.stage_name}: {completed} done, {failed} failed",
                )

        logger.info(
            "[%s] Done: %d total, %d completed, %d failed, %d skipped",
            self.stage_name, total, completed, failed, skipped,
        )

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        }
