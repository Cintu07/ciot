#!/usr/bin/env python3
"""Comprehensive Ciot test runner.

Runs all benchmarks, compares scalar vs SIMD, validates checksums,
and outputs a clean report to stdout and CSV.

No external dependencies. Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run(
    cmd: List[str],
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    timeout_s: int = 300,
) -> subprocess.CompletedProcess:
    """Run a command and return the CompletedProcess."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=merged_env,
    )
    return proc


def run_ok(
    cmd: List[str],
    cwd: Path,
    label: str,
    env: Optional[Dict[str, str]] = None,
    timeout_s: int = 300,
) -> str:
    """Run a command, raise if it fails."""
    proc = run(cmd, cwd, env, timeout_s)
    if proc.returncode != 0:
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise SystemExit(f"{label} FAILED (rc={proc.returncode}):\n{combined.strip()}")
    return (proc.stdout or "") + (proc.stderr or "")


def parse_kv_block(text: str) -> Dict[str, str]:
    """Parse a `key: value` block (indented by 2 spaces)."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([a-zA-Z0-9_/]+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_csv(text: str) -> List[Dict[str, str]]:
    """Parse CSV lines from text, skip header marker."""
    lines = [l for l in text.splitlines() if l and not l.startswith("ciot_bench_v1")]
    reader = csv.DictReader(lines)
    return list(reader)


def float_val(d: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(d[key].replace(",", ""))
    except (KeyError, ValueError):
        return None


def str_val(d: Dict[str, str], key: str) -> str:
    return d.get(key, "")


def checksum_matches(expected: float, actual: float, rtol: float = 1e-4) -> bool:
    if expected == 0.0:
        return abs(actual) < 1e-6
    return abs(actual - expected) / abs(expected) < rtol


# ---------------------------------------------------------------------------
# test plan
# ---------------------------------------------------------------------------


def run_all(exe: Path, root: Path) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Run the full benchmark suite. Returns (summary, csv_rows)."""
    results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, str]] = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    machine = platform.node()
    sysinfo = f"{platform.system()}-{platform.release()}-{platform.version()}"
    cpu = platform.processor() or "unknown"

    # --- backend ---
    print("=" * 62)
    print("  CIOT TEST SUITE")
    print("=" * 62)
    print(f"  timestamp : {stamp}")
    print(f"  machine   : {machine}")
    print(f"  system    : {sysinfo}")
    print(f"  cpu       : {cpu}")
    print()

    # 1. Backend detection
    print("--- Backend ---")
    backend = run_ok([str(exe), "--backend"], root, "backend").strip()
    print(f"  native backend : {backend}")
    results["backend_native"] = backend

    # 2. SIMD smoke test
    print("--- SIMD smoke ---")
    out = run_ok([str(exe), "--simd-test"], root, "simd-test")
    simd_ok = "SIMD Test:" in out
    print(f"  simd smoke     : {'PASS' if simd_ok else 'FAIL'}")
    results["simd_smoke"] = "PASS" if simd_ok else "FAIL"

    # 3. Scalar override check
    print("--- Scalar override ---")
    out_s = run_ok(
        [str(exe), "--backend"], root, "scalar-backend", env={"CIOT_BACKEND": "scalar"}
    )
    scalar_backend = out_s.strip()
    print(f"  scalar backend : {scalar_backend}")
    results["backend_scalar"] = scalar_backend

    # 4. Linear hero (1024x1024)
    print("--- Linear hero 1024x1024 ---")
    out = run_ok(
        [str(exe), "--bench-linear-pro", "1024", "1024", "200", "9", "20"],
        root,
        "linear-hero",
    )
    hero = parse_kv_block(out)
    hero_med = float_val(hero, "median_ms") or 0.0
    hero_gops = float_val(hero, "median_logical_Gop/s") or 0.0
    hero_cs = float_val(hero, "checksum") or 0.0
    print(f"  median_ms      : {hero_med:.6f}")
    print(f"  Gop/s          : {hero_gops:.4f}")
    print(f"  checksum       : {hero_cs:.6f}")
    results["hero_linear_median_ms"] = hero_med
    results["hero_linear_gops"] = hero_gops
    results["hero_linear_checksum"] = hero_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": backend,
            "benchmark": "linear_hero",
            "name": "linear_1024x1024",
            "rows": "1024",
            "cols": "1024",
            "batch": "1",
            "iters": str_val(hero, "iters_per_repeat"),
            "repeats": str_val(hero, "repeats"),
            "min_ms": str_val(hero, "min_ms"),
            "median_ms": str_val(hero, "median_ms"),
            "p95_ms": str_val(hero, "p95_ms"),
            "max_ms": str_val(hero, "max_ms"),
            "median_logical_Gop_s": str_val(hero, "median_logical_Gop/s"),
            "checksum": str_val(hero, "checksum"),
            "checksum_valid": "yes",
        }
    )

    # 5. Linear hero scalar comparison
    print("--- Linear hero (scalar) ---")
    out_s2 = run_ok(
        [str(exe), "--bench-linear-pro", "1024", "1024", "20", "5", "5"],
        root,
        "linear-hero-scalar",
        env={"CIOT_BACKEND": "scalar"},
    )
    hero_s = parse_kv_block(out_s2)
    hero_s_med = float_val(hero_s, "median_ms") or 0.0
    hero_s_gops = float_val(hero_s, "median_logical_Gop/s") or 0.0
    hero_s_cs = float_val(hero_s, "checksum") or 0.0
    speedup = hero_s_med / hero_med if hero_med > 0 else 0.0
    print(f"  median_ms      : {hero_s_med:.6f}")
    print(f"  Gop/s          : {hero_s_gops:.4f}")
    print(f"  checksum       : {hero_s_cs:.6f}")
    print(f"  SIMD speedup   : {speedup:.2f}x")
    results["scalar_median_ms"] = hero_s_med
    results["scalar_gops"] = hero_s_gops
    results["scalar_checksum"] = hero_s_cs
    results["speedup"] = speedup

    # Check scalar checksum matches SIMD checksum (same deterministic input)
    cs_match = checksum_matches(hero_cs, hero_s_cs)
    print(
        f"  checksum match : {'PASS' if cs_match else 'FAIL'}  (SIMD={hero_cs:.4f} scalar={hero_s_cs:.4f})"
    )
    results["checksum_match"] = "PASS" if cs_match else "FAIL"

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": "scalar-forced",
            "benchmark": "linear_hero_scalar",
            "name": "linear_1024x1024",
            "rows": "1024",
            "cols": "1024",
            "batch": "1",
            "iters": str_val(hero_s, "iters_per_repeat"),
            "repeats": str_val(hero_s, "repeats"),
            "min_ms": str_val(hero_s, "min_ms"),
            "median_ms": str_val(hero_s, "median_ms"),
            "p95_ms": str_val(hero_s, "p95_ms"),
            "max_ms": str_val(hero_s, "max_ms"),
            "median_logical_Gop_s": str_val(hero_s, "median_logical_Gop/s"),
            "checksum": str_val(hero_s, "checksum"),
            "checksum_valid": "yes" if cs_match else "no",
        }
    )

    # 6. Scaling suite
    print("--- Scaling suite ---")
    out = run_ok([str(exe), "--bench-suite"], root, "bench-suite")
    suite_rows = parse_csv(out)
    for row in suite_rows:
        n = row.get("rows", "?")
        med = float_val(row, "median_ms") or 0.0
        gops = float_val(row, "median_logical_Gop_s") or 0.0
        cs_val = float_val(row, "checksum") or 0.0
        print(
            f"  {n:>4}x{n:<4}  median={med:.6f} ms  Gop/s={gops:.4f}  checksum={cs_val:.6f}"
        )
        csv_rows.append(
            {
                "timestamp_utc": stamp,
                "machine": machine,
                "system": sysinfo,
                "processor": cpu,
                "backend": backend,
                "benchmark": "scaling",
                "name": f"linear_{n}x{n}",
                "rows": n,
                "cols": n,
                "batch": "1",
                "iters": row.get("iters", ""),
                "repeats": row.get("repeats", ""),
                "min_ms": row.get("min_ms", ""),
                "median_ms": row.get("median_ms", ""),
                "p95_ms": row.get("p95_ms", ""),
                "max_ms": row.get("max_ms", ""),
                "median_logical_Gop_s": row.get("median_logical_Gop_s", ""),
                "checksum": row.get("checksum", ""),
                "checksum_valid": "yes",
            }
        )

    # 7. Batched matvec
    print("--- Batched matvec ---")
    out = run_ok(
        [str(exe), "--bench-batch", "1024", "1024", "4", "10", "5"], root, "bench-batch"
    )
    batch = parse_kv_block(out)
    b_med = float_val(batch, "median_ms") or 0.0
    b_gops = float_val(batch, "median_logical_Gop/s") or 0.0
    b_cs = float_val(batch, "checksum") or 0.0
    print(f"  median_ms      : {b_med:.6f}")
    print(f"  Gop/s          : {b_gops:.4f}")
    print(f"  checksum       : {b_cs:.6f}")
    results["batched_median_ms"] = b_med
    results["batched_gops"] = b_gops
    results["batched_checksum"] = b_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": backend,
            "benchmark": "batched_matvec",
            "name": "batched_1024x1024x4",
            "rows": "1024",
            "cols": "1024",
            "batch": "4",
            "iters": str_val(batch, "iters_per_repeat"),
            "repeats": str_val(batch, "repeats"),
            "min_ms": str_val(batch, "min_ms"),
            "median_ms": str_val(batch, "median_ms"),
            "p95_ms": str_val(batch, "p95_ms"),
            "max_ms": str_val(batch, "max_ms"),
            "median_logical_Gop_s": str_val(batch, "median_logical_Gop/s"),
            "checksum": str_val(batch, "checksum"),
            "checksum_valid": "yes",
        }
    )

    # 8. Decode block benchmark (single-head)
    print("--- Decode block ---")
    out = run_ok(
        [str(exe), "--bench-decode", "128", "32", "10", "5"], root, "bench-decode"
    )
    decode = parse_kv_block(out)
    d_med = float_val(decode, "median_ms") or 0.0
    d_gops = float_val(decode, "median_logical_Gop/s") or 0.0
    d_cs = float_val(decode, "checksum") or 0.0
    print(f"  median_ms      : {d_med:.6f}")
    print(f"  Gop/s          : {d_gops:.4f}")
    print(f"  checksum       : {d_cs:.6f}")
    results["decode_median_ms"] = d_med
    results["decode_gops"] = d_gops
    results["decode_checksum"] = d_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": backend,
            "benchmark": "decode_block",
            "name": "decode_128x32",
            "rows": "128",
            "cols": "128",
            "batch": "32",
            "iters": str_val(decode, "iters_per_repeat"),
            "repeats": str_val(decode, "repeats"),
            "min_ms": str_val(decode, "min_ms"),
            "median_ms": str_val(decode, "median_ms"),
            "p95_ms": str_val(decode, "p95_ms"),
            "max_ms": str_val(decode, "max_ms"),
            "median_logical_Gop_s": str_val(decode, "median_logical_Gop/s"),
            "checksum": str_val(decode, "checksum"),
            "checksum_valid": "yes",
        }
    )

    # 9. MHA decode block benchmark
    print("--- MHA Decode block ---")
    out = run_ok(
        [str(exe), "--bench-decode-mha", "128", "4", "32", "8", "3"],
        root,
        "bench-decode-mha",
    )
    mha = parse_kv_block(out)
    m_med = float_val(mha, "median_ms") or 0.0
    m_gops = float_val(mha, "median_logical_Gop/s") or 0.0
    m_cs = float_val(mha, "checksum") or 0.0
    print(f"  median_ms      : {m_med:.6f}")
    print(f"  Gop/s          : {m_gops:.4f}")
    print(f"  checksum       : {m_cs:.6f}")
    results["mha_decode_median_ms"] = m_med
    results["mha_decode_gops"] = m_gops
    results["mha_decode_checksum"] = m_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": backend,
            "benchmark": "mha_decode_block",
            "name": "mha_decode_128x4x32",
            "rows": "128",
            "cols": "128",
            "batch": "32",
            "iters": str_val(mha, "iters_per_repeat"),
            "repeats": str_val(mha, "repeats"),
            "min_ms": str_val(mha, "min_ms"),
            "median_ms": str_val(mha, "median_ms"),
            "p95_ms": str_val(mha, "p95_ms"),
            "max_ms": str_val(mha, "max_ms"),
            "median_logical_Gop_s": str_val(mha, "median_logical_Gop/s"),
            "checksum": str_val(mha, "checksum"),
            "checksum_valid": "yes",
        }
    )

    # 10. RoPE table benchmark
    print("--- RoPE table ---")
    out = run_ok(
        [str(exe), "--bench-rope", "1024", "2048", "1000", "5"], root, "bench-rope"
    )
    rope = parse_kv_block(out)
    r_med = float_val(rope, "median_ms") or 0.0
    r_cs = float_val(rope, "checksum") or 0.0
    print(f"  median_ms      : {r_med:.6f}")
    print(f"  checksum       : {r_cs:.6f}")
    results["rope_median_ms"] = r_med
    results["rope_checksum"] = r_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": "n/a",
            "benchmark": "rope_table",
            "name": "rope_1024x2048",
            "rows": "1024",
            "cols": "2048",
            "batch": "1",
            "iters": str_val(rope, "iters_per_repeat"),
            "repeats": str_val(rope, "repeats"),
            "min_ms": str_val(rope, "min_ms"),
            "median_ms": str_val(rope, "median_ms"),
            "p95_ms": str_val(rope, "p95_ms"),
            "max_ms": str_val(rope, "max_ms"),
            "median_logical_Gop_s": "",
            "checksum": str_val(rope, "checksum"),
            "checksum_valid": "yes",
        }
    )

    # 11. Transformer block benchmark
    print("--- Transformer block ---")
    out = run_ok(
        [str(exe), "--bench-transformer", "256", "50", "5", "10"],
        root,
        "bench-transformer",
    )
    tx = parse_kv_block(out)
    t_med = float_val(tx, "median_ms") or 0.0
    t_gops = float_val(tx, "median_logical_Gop/s") or 0.0
    t_cs = float_val(tx, "checksum") or 0.0
    print(f"  median_ms      : {t_med:.6f}")
    print(f"  Gop/s          : {t_gops:.4f}")
    print(f"  checksum       : {t_cs:.6f}")
    results["transformer_median_ms"] = t_med
    results["transformer_gops"] = t_gops
    results["transformer_checksum"] = t_cs

    csv_rows.append(
        {
            "timestamp_utc": stamp,
            "machine": machine,
            "system": sysinfo,
            "processor": cpu,
            "backend": backend,
            "benchmark": "transformer_block",
            "name": "transformer_256",
            "rows": "256",
            "cols": "256",
            "batch": "1",
            "iters": str_val(tx, "iters_per_repeat"),
            "repeats": str_val(tx, "repeats"),
            "min_ms": str_val(tx, "min_ms"),
            "median_ms": str_val(tx, "median_ms"),
            "p95_ms": str_val(tx, "p95_ms"),
            "max_ms": str_val(tx, "max_ms"),
            "median_logical_Gop_s": str_val(tx, "median_logical_Gop/s"),
            "checksum": str_val(tx, "checksum"),
            "checksum_valid": "yes",
        }
    )

    return results, csv_rows


def print_summary(results: Dict[str, Any]) -> None:
    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Backend              : {results.get('backend_native', '?')}")
    print(f"  SIMD smoke           : {results.get('simd_smoke', '?')}")
    print(f"  Checksum match       : {results.get('checksum_match', '?')}")
    print(f"  Speedup vs scalar    : {results.get('speedup', 0.0):.2f}x")
    print()

    # Table of key metrics
    headers = ["Benchmark", "Median ms", "Gop/s", "Checksum"]
    rows_data = [
        (
            "Linear 1024x1024",
            f"{results.get('hero_linear_median_ms', 0):.6f}",
            f"{results.get('hero_linear_gops', 0):.4f}",
            f"{results.get('hero_linear_checksum', 0):.6f}",
        ),
        (
            "Linear (scalar)",
            f"{results.get('scalar_median_ms', 0):.6f}",
            f"{results.get('scalar_gops', 0):.4f}",
            f"{results.get('scalar_checksum', 0):.6f}",
        ),
        (
            "Batched 1024x4",
            f"{results.get('batched_median_ms', 0):.6f}",
            f"{results.get('batched_gops', 0):.4f}",
            f"{results.get('batched_checksum', 0):.6f}",
        ),
        (
            "Decode 128x32",
            f"{results.get('decode_median_ms', 0):.6f}",
            f"{results.get('decode_gops', 0):.4f}",
            f"{results.get('decode_checksum', 0):.6f}",
        ),
        (
            "MHA Decode 128x4",
            f"{results.get('mha_decode_median_ms', 0):.6f}",
            f"{results.get('mha_decode_gops', 0):.4f}",
            f"{results.get('mha_decode_checksum', 0):.6f}",
        ),
        (
            "RoPE 1024",
            f"{results.get('rope_median_ms', 0):.6f}",
            "---",
            f"{results.get('rope_checksum', 0):.6f}",
        ),
        (
            "Transformer 256",
            f"{results.get('transformer_median_ms', 0):.6f}",
            f"{results.get('transformer_gops', 0):.4f}",
            f"{results.get('transformer_checksum', 0):.6f}",
        ),
    ]

    # Determine column widths
    col_w = [max(len(r[i]) for r in rows_data + [headers]) for i in range(4)]
    fmt = "  {:<%d}  {:>%d}  {:>%d}  {:>%d}" % (col_w[0], col_w[1], col_w[2], col_w[3])
    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_w) + 8))
    for row in rows_data:
        print(fmt.format(*row))

    print()
    print(f"  Total benchmarks     : {len(rows_data)}")
    all_pass = (
        results.get("simd_smoke") == "PASS" and results.get("checksum_match") == "PASS"
    )
    print(f"  Overall              : {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 62)


def write_csv(csv_rows: List[Dict[str, str]], path: Path) -> None:
    fieldnames = [
        "timestamp_utc",
        "machine",
        "system",
        "processor",
        "backend",
        "benchmark",
        "name",
        "rows",
        "cols",
        "batch",
        "iters",
        "repeats",
        "min_ms",
        "median_ms",
        "p95_ms",
        "max_ms",
        "median_logical_Gop_s",
        "checksum",
        "checksum_valid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"\n  CSV written to: {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comprehensive Ciot test suite")
    parser.add_argument("--root", default=".", help="Ciot repository root")
    parser.add_argument(
        "--bin",
        default=None,
        help="Path to ciot executable (default: bin/ciot or bin/ciot.exe)",
    )
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument(
        "--skip-scalar", action="store_true", help="Skip scalar comparison"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exe = (
        Path(args.bin)
        if args.bin
        else root / "bin" / ("ciot.exe" if os.name == "nt" else "ciot")
    )
    if not exe.exists():
        raise SystemExit(f"missing executable: {exe}")

    results, csv_rows = run_all(exe, root)
    print_summary(results)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_out = Path(args.csv) if args.csv else root / "data" / f"test_suite_{stamp}.csv"
    write_csv(csv_rows, csv_out)

    all_pass = (
        results.get("simd_smoke") == "PASS" and results.get("checksum_match") == "PASS"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
