from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TextIO, TypeVar

_T = TypeVar("_T")

_enabled = True


def log(msg: str) -> None:
    """Print an info message if logging is enabled."""
    if _enabled:
        print(msg, flush=True)


def warn(msg: str) -> None:
    """Print a warning message if logging is enabled."""
    if _enabled:
        print(f"Warning: {msg}", flush=True)


def disable_logs() -> None:
    """Suppress all lensboy log output."""
    global _enabled
    _enabled = False


def enable_logs() -> None:
    """Re-enable lensboy log output."""
    global _enabled
    _enabled = True


def _set_cpp_log_level(level: str) -> None:
    """Set the spdlog log level for the C++ backend.

    Args:
        level: One of "trace", "debug", "info", "warn", "error", "critical", "off".
    """
    import lensboy.lensboy_bindings as lbb

    lbb.set_log_level(level)


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _fmt_int(n: int) -> str:
    # 1234567 -> 1,234,567 (no locale dependency)
    s = str(int(n))
    out = []
    for i, ch in enumerate(reversed(s)):
        if i and i % 3 == 0:
            out.append(",")
        out.append(ch)
    return "".join(reversed(out))


@dataclass
class Progress:
    total: int
    desc: str = "doing thing"
    file: TextIO = sys.stdout

    # update control
    min_interval_s: float = 0.08  # don't redraw faster than this
    min_iters: int = 1  # redraw at least every N updates
    smoothing: float = 0.3  # for rate/ETA if you later want it

    # formatting
    percent_decimals: int = 1
    show_percent_when_unknown_total: bool = False

    # internal state
    n: int = 0
    _start_t: float = 0.0
    _last_draw_t: float = 0.0
    _last_draw_n: int = 0
    _last_line_len: int = 0
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.total is None:
            raise ValueError("total must be an int (use total=0 if unknown)")
        if self.total < 0:
            raise ValueError(f"total must be >= 0, got {self.total}")
        self._start_t = time.time()
        self._last_draw_t = 0.0
        self._last_draw_n = 0
        self._last_line_len = 0
        self._closed = False

    def __enter__(self) -> Progress:
        self.draw(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, inc: int = 1) -> None:
        if self._closed:
            return
        if inc < 0:
            raise ValueError("inc must be >= 0")
        self.n += inc
        if self.total > 0 and self.n > self.total:
            self.n = self.total
        self.draw()

    def set(self, n: int) -> None:
        if self._closed:
            return
        if n < 0:
            raise ValueError("n must be >= 0")
        self.n = n
        if self.total > 0:
            self.n = min(n, self.total)
        self.draw(force=True)

    def draw(self, *, force: bool = False) -> None:
        if self._closed or not _enabled:
            return

        now = time.time()
        iters_since = self.n - self._last_draw_n
        time_since = now - self._last_draw_t

        if not force:
            if iters_since < self.min_iters and time_since < self.min_interval_s:
                return
            if time_since < self.min_interval_s:
                return

        line = self._format_line()

        # in-place redraw: carriage return + pad to clear previous longer line
        pad = ""
        if self._last_line_len > len(line):
            pad = " " * (self._last_line_len - len(line))

        try:
            self.file.write("\r" + line + pad)
            self.file.flush()
        except Exception:
            self._closed = True
            return

        self._last_line_len = max(self._last_line_len, len(line))
        self._last_draw_t = now
        self._last_draw_n = self.n

    def _format_line(self) -> str:
        completed = self.n
        total = self.total

        if total > 0:
            pct = 100.0 * completed / total
            pct = _clamp(pct, 0.0, 100.0)
            pct_str = f"{pct:.{self.percent_decimals}f}%"
            return f"{self.desc}: {_fmt_int(completed)}/{_fmt_int(total)} ({pct_str})"
        else:
            if self.show_percent_when_unknown_total:
                return f"{self.desc}: {_fmt_int(completed)}/? (?)"
            return f"{self.desc}: {_fmt_int(completed)}/?"

    def close(self) -> None:
        if self._closed:
            return
        self.draw(force=True)
        if _enabled:
            try:
                self.file.write("\n")
                self.file.flush()
            except Exception:
                pass
        self._closed = True


def progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str = "doing thing",
    **kwargs,
) -> Iterator[_T]:
    """Wrap an iterable and yield items while updating progress.

    Args:
        iterable: Items to iterate over.
        total: Total count (inferred from iterable length if None).
        desc: Description shown in the progress line.
        **kwargs: Extra arguments forwarded to Progress.
    """
    iterable = list(iterable)
    if total is None:
        total = len(iterable)

    with Progress(total=total, desc=desc, **kwargs) as p:
        for item in iterable:
            yield item
            p.update(1)
