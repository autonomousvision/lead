#!/usr/bin/env python
"""Report statistics on "Time per step" values parsed from expert stdout logs.

Scans every ``*.log`` file in ``$PY123D_DATA_ROOT/stdout`` (see ``.env``) for
lines of the form::

    ... Step: 40, Time per step: 213.52 ms

and prints min / max / mean / median / histogram of the time-per-step values,
both per file and aggregated over all files. Also saves a PNG histogram to
the PY123D data root.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from lead.common.dotenv import REPO_ROOT, read_dotenv

TIME_RE = re.compile(r"Time per step:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
OUTLIER_MAX_MS = 1000.0  # drop startup/stall spikes above this

DATA_ROOT = REPO_ROOT / read_dotenv("PY123D_DATA_ROOT")
LOG_DIR = DATA_ROOT / "stdout"
PLOT_PATH = DATA_ROOT / "time_per_step_histogram.png"


def parse_file(path: Path) -> list[float]:
    """Return all time-per-step values (ms) found in ``path``."""
    values: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = TIME_RE.search(line)
            if match:
                values.append(float(match.group(1)))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def print_summary(label: str, values: list[float]) -> None:
    """Print a one-block summary for ``values`` under ``label``."""
    if not values:
        print(f"{label}: no 'Time per step' entries found")
        return
    s = summarize(values)
    print(
        f"{label}: n={s['count']}  "
        f"min={s['min']:.2f}  max={s['max']:.2f}  "
        f"mean={s['mean']:.2f}  median={s['median']:.2f}  "
        f"std={s['std']:.2f}  (ms)",
    )


def print_histogram(values: list[float], bins: int = 20, width: int = 50) -> None:
    """Print an ASCII histogram of ``values``."""
    lo, hi = min(values), max(values)
    if hi == lo:
        print(f"all values == {lo:.2f} ms (n={len(values)})")
        return
    span = hi - lo
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / span * bins), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    print(f"\nHistogram ({bins} bins, {lo:.1f}–{hi:.1f} ms, n={len(values)}):")
    for i, c in enumerate(counts):
        edge_lo = lo + span * i / bins
        edge_hi = lo + span * (i + 1) / bins
        bar = "#" * round(c / peak * width)
        print(f"  [{edge_lo:7.1f}, {edge_hi:7.1f}) {c:6d} {bar}")


def print_ranking(title: str, per_file: list[dict], key: str) -> None:
    """Print all files sorted by ``key`` (descending)."""
    unit = "s" if key == "total_s" else "ms"
    print(f"\n{title} (highest first):")
    for r in sorted(per_file, key=lambda r: r[key], reverse=True):
        print(f"  {r[key]:12.2f} {unit}  n={r['n']:6d}  {r['name']}")


def save_plot(values: list[float], path: Path, bins: int = 50) -> None:
    """Save a PNG histogram of ``values`` to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax.axvline(s["median"], color="#c44e52", linestyle="--", label=f"median {s['median']:.1f} ms")
    ax.axvline(s["mean"], color="#dd8452", linestyle="--", label=f"mean {s['mean']:.1f} ms")
    ax.set_xlabel("Time per step (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Time per step  (n={s['count']}, min={s['min']:.1f}, max={s['max']:.1f} ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved histogram to {path}")


def main() -> None:
    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        raise SystemExit(f"no *.log files found in {LOG_DIR}")

    all_values: list[float] = []
    dropped = 0
    per_file: list[dict] = []  # {name, total_s, median_ms, n}
    for path in log_files:
        values = parse_file(path)
        kept = [v for v in values if v <= OUTLIER_MAX_MS]
        dropped += len(values) - len(kept)
        values = kept
        all_values.extend(values)
        per_file.append(
            {
                "name": path.name,
                "total_s": sum(values) / 1000.0,
                "median_ms": statistics.median(values) if values else 0.0,
                "n": len(values),
            },
        )
        print_summary(path.name, values)

    print("-" * 60)
    print(f"(dropped {dropped} values > {OUTLIER_MAX_MS:.0f} ms as outliers)")
    print_summary(f"ALL ({len(log_files)} files)", all_values)

    print_ranking("Files by total run time (sum of step times)", per_file, "total_s")
    print_ranking("Files by median time per step", per_file, "median_ms")

    if all_values:
        print_histogram(all_values)
        save_plot(all_values, PLOT_PATH)


if __name__ == "__main__":
    main()
