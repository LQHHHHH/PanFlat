"""Lightweight timestamped progress reporting to stderr (never alters output files)."""

from __future__ import annotations

import datetime
import sys
import threading

_PROGRESS_LOCK = threading.Lock()


def log_progress(message: str) -> None:
    """Emit a timestamped progress line to stderr and flush immediately."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [panflat] {message}\n"
    with _PROGRESS_LOCK:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except OSError:
            pass
