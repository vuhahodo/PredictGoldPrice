from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional
import uuid


@dataclass
class JobRecord:
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: Dict[str, JobRecord] = {}

    def create(self) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobRecord(status="pending")
        return job_id

    def set_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = JobRecord(status="running")

    def set_done(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = JobRecord(status="done", result=result)

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            self._jobs[job_id] = JobRecord(status="error", error=message)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)
