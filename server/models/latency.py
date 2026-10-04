import threading
import time
from collections import deque
from typing import Dict, Optional

WINDOW = 2000

_enabled = False
_trackers: Dict[str, "LatencyTracker"] = {}
_registry_lock = threading.Lock()


def enable(value: bool = True) -> None:
    global _enabled
    _enabled = value


def is_enabled() -> bool:
    return _enabled


def get_tracker(name: str, report_every: int = 0) -> "LatencyTracker":
    with _registry_lock:
        tracker = _trackers.get(name)
        if tracker is None:
            tracker = LatencyTracker(name, report_every=report_every)
            _trackers[name] = tracker
        return tracker


def snapshot_all() -> Dict[str, dict]:
    with _registry_lock:
        trackers = list(_trackers.values())
    return {tracker.name: tracker.snapshot() for tracker in trackers}


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class LatencyTracker:
    def __init__(self, name: str, report_every: int = 0):
        self.name = name
        self.report_every = report_every
        self._samples: deque[float] = deque(maxlen=WINDOW)
        self._lock = threading.Lock()
        self._total = 0
        self._since_report = 0
        self._timeouts = 0

    def record(self, seconds: float) -> None:
        if not _enabled:
            return
        with self._lock:
            self._samples.append(seconds)
            self._total += 1
            self._since_report += 1
            due = self.report_every > 0 and self._since_report >= self.report_every
            if due:
                self._since_report = 0
        if due:
            print(self.format())

    def record_timeout(self, count: int = 1) -> None:
        if not _enabled:
            return
        with self._lock:
            self._timeouts += count

    def snapshot(self) -> dict:
        with self._lock:
            samples = sorted(self._samples)
            total = self._total
            timeouts = self._timeouts

        if not samples:
            return {"samples": 0, "total": total, "timeouts": timeouts}

        return {
            "samples": len(samples),
            "total": total,
            "timeouts": timeouts,
            "mean_ms": 1000 * sum(samples) / len(samples),
            "p50_ms": 1000 * _percentile(samples, 0.50),
            "p90_ms": 1000 * _percentile(samples, 0.90),
            "p99_ms": 1000 * _percentile(samples, 0.99),
            "max_ms": 1000 * samples[-1],
        }

    def format(self) -> str:
        stats = self.snapshot()
        if not stats.get("samples"):
            return f"[latency:{self.name}] no samples yet"
        return (
            f"[latency:{self.name}] n={stats['samples']} "
            f"mean={stats['mean_ms']:.1f}ms "
            f"p50={stats['p50_ms']:.1f}ms "
            f"p90={stats['p90_ms']:.1f}ms "
            f"p99={stats['p99_ms']:.1f}ms "
            f"max={stats['max_ms']:.1f}ms "
            f"timeouts={stats['timeouts']}"
        )


class Stopwatch:
    def __init__(self, tracker: "LatencyTracker"):
        self._tracker = tracker
        self._start: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        if self._start is not None:
            self._tracker.record(time.perf_counter() - self._start)
        return None
