"""Lightweight timestamped progress reporting to stderr (never alters output files)."""

from __future__ import annotations

import datetime
import os
import sys
import threading

_PROGRESS_LOCK = threading.Lock()


def _acquire_progress_lock_before_fork() -> None:
    _PROGRESS_LOCK.acquire()


def _release_progress_lock_in_parent() -> None:
    _PROGRESS_LOCK.release()


def _reset_progress_lock_in_child() -> None:
    global _PROGRESS_LOCK
    _PROGRESS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_acquire_progress_lock_before_fork,
        after_in_parent=_release_progress_lock_in_parent,
        after_in_child=_reset_progress_lock_in_child,
    )


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
