#!/usr/bin/env python3
"""Production-style Ciot benchmark runner.

No external dependencies. Runs the Ciot executable multiple times, captures the
CSV suite output, and writes a timestamped CSV file suitable for commits/issues.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import subprocess
from pathlib import Path
from typing import cast


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}"
        )
    return proc.stdout


def parse_csv_lines(text: str) -> list[dict[str, str]]:
    lines = [
        line
        for line in text.splitlines()
        if line and not line.startswith("ciot_bench_v1")
    ]
    reader = csv.DictReader(lines)
    return list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Ciot benchmark suite and write CSV"
    )
    _ = parser.add_argument("--root", default=".", help="Ciot repository root")
    _ = parser.add_argument(
        "--bin",
        default=None,
        help="Path to ciot executable; default: bin/ciot.exe on Windows, bin/ciot otherwise",
    )
    _ = parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    root_arg = cast(str, args.root)
    bin_arg = cast(str | None, args.bin)
    out_arg = cast(str | None, args.out)

    root = Path(root_arg).resolve()
    exe = (
        Path(bin_arg)
        if bin_arg
        else root / "bin" / ("ciot.exe" if os.name == "nt" else "ciot")
    )
    if not exe.exists():
        raise SystemExit(f"missing executable: {exe}")

    backend = run([str(exe), "--backend"], root).strip()
    suite = run([str(exe), "--bench-suite"], root)
    rows = parse_csv_lines(suite)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_arg) if out_arg else root / "data" / f"ciot_bench_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "machine",
        "system",
        "processor",
        "backend",
        "name",
        "rows",
        "cols",
        "iters",
        "repeats",
        "min_ms",
        "median_ms",
        "p95_ms",
        "max_ms",
        "median_logical_Gop_s",
        "checksum",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": stamp,
                    "machine": platform.node(),
                    "system": platform.platform(),
                    "processor": platform.processor(),
                    "backend": backend,
                    "name": row.get("name", ""),
                    "rows": row.get("rows", ""),
                    "cols": row.get("cols", ""),
                    "iters": row.get("iters", ""),
                    "repeats": row.get("repeats", ""),
                    "min_ms": row.get("min_ms", ""),
                    "median_ms": row.get("median_ms", ""),
                    "p95_ms": row.get("p95_ms", ""),
                    "max_ms": row.get("max_ms", ""),
                    "median_logical_Gop_s": row.get("median_logical_Gop_s", ""),
                    "checksum": row.get("checksum", ""),
                }
            )

    print(f"backend={backend}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
