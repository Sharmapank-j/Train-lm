"""Simple asyncio-based in-process job queue for Train-LM.

Supports: queued, running, completed, failed, cancelled states.
Each job is a coroutine factory.  Workers pick up queued jobs and run them.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger("train_lm.jobs")


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = "generic"
    state: JobState = JobState.queued
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    _task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "state": self.state.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class JobQueue:
    """Singleton async job queue backed by asyncio tasks."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)

    def submit(
        self,
        coro_factory: Callable[[Job], Awaitable[dict[str, Any]]],
        job_type: str = "generic",
    ) -> Job:
        job = Job(job_type=job_type)
        self._jobs[job.id] = job
        job._task = asyncio.create_task(self._dispatch(job, coro_factory))
        logger.info("job.submitted", extra={"job_id": job.id, "job_type": job_type})
        return job

    async def _dispatch(self, job: Job, coro_factory: Callable[[Job], Awaitable[dict[str, Any]]]) -> None:
        async with self._semaphore:
            job.state = JobState.running
            job.started_at = datetime.now(UTC)
            try:
                result = await coro_factory(job)
                job.result = result or {}
                job.state = JobState.completed
                job.progress = 1.0
            except asyncio.CancelledError:
                job.state = JobState.cancelled
            except Exception as exc:  # noqa: BLE001
                job.state = JobState.failed
                job.error = str(exc)
                logger.exception("job.failed", extra={"job_id": job.id})
            finally:
                job.completed_at = datetime.now(UTC)
                logger.info("job.finished", extra={"job_id": job.id, "state": job.state})

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job._task and not job._task.done():
            job._task.cancel()
            job.state = JobState.cancelled
            return True
        return False

    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        return [
            j.to_dict()
            for j in self._jobs.values()
            if job_type is None or j.job_type == job_type
        ]


# Global singleton
_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    global _queue
    if _queue is None:
        from app.config.settings import settings
        _queue = JobQueue(max_workers=settings.max_concurrent_jobs)
    return _queue


__all__ = ["Job", "JobQueue", "JobState", "get_job_queue"]
