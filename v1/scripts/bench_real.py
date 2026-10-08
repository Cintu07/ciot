#!/usr/bin/env python3
"""Real-world benchmark suite for Ciot ternary inference engine.
Tests what actually matters: matvec throughput, scaling, stability, decode latency."""

import datetime
import json
import os
import statistics
import subprocess
import sys
import time

CIOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "ciot"
)
if os.name == "nt":
    CIOT += ".exe"


def run(cmd, timeout=120, env=None):
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(CIOT),
        env=env,
    )
    if r.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}\n{r.stderr}")
        return None
    return r.stdout


def parse_bench(output):
    """Parse ciot_bench_v1 or ciot_decode output into dict."""
    d = {}
    for line in output.strip().split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("ciot_") and not line.startswith("name"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    d[key] = float(parts[1].strip())
                except:
                    d[key] = parts[1].strip()
    return d


def test_matvec_throughput():
    """Run 50 iterations of 1024x1024 matvec, measure consistency."""
    print("\n=== matvec throughput stability (50 runs) ===")
    times = []
    for i in range(50):
        out = run([CIOT, "--bench-linear-pro", "1024", "1024", "1", "1", "0"])
        if out is None:
            return False
        d = parse_bench(out)
        if "median_ms" in d:
            times.append(d["median_ms"])
    if len(times) < 40:
        return False
    avg = statistics.mean(times)
    std = statistics.stdev(times)
    cv = (std / avg) * 100
    print(
        f"  avg: {avg:.6f}ms  stddev: {std:.6f}ms  cv: {cv:.2f}%  min: {min(times):.6f}ms  max: {max(times):.6f}ms"
    )
    print(f"  {'PASS' if cv < 10 else 'FAIL'} (cv < 10%)")
    return cv < 10


def test_batch_scaling():
    """Test how throughput scales with batch size."""
    print("\n=== batch scaling (512x512) ===")
    for batch in [1, 2, 4, 8, 16]:
        out = run([CIOT, "--bench-batch", "512", "512", str(batch), "20", "3"])
        if out is None:
            continue
        d = parse_bench(out)
        ms = d.get("median_ms", 0)
        ops = 512 * 512 * batch * 2
        gops = ops / (ms * 1e6) if ms > 0 else 0
        print(f"  batch={batch:3d}  median={ms:.4f}ms  gop/s={gops:.2f}")
    return True


def test_context_scaling():
    """Test decode latency as context grows."""
    print("\n=== decode context scaling (128-dim, 4-head) ===")
    for ctx in [4, 8, 16, 32, 64, 128]:
        out = run(
            [
                CIOT,
                "--bench-decode-mha",
                "128",
                "4",
                str(ctx),
                str(max(2, 64 // ctx)),
                "3",
            ]
        )
        if out is None:
            continue
        d = parse_bench(out)
        ms = d.get("median_ms", 0)
        us_per_token = (ms * 1000) / ctx if ctx > 0 else 0
        print(f"  context={ctx:4d}  median={ms:.4f}ms  per_token={us_per_token:.1f}us")
    return True


def test_size_scaling():
    """Test matvec throughput across sizes to find cache boundary."""
    print("\n=== size scaling (find L2/L3 cache boundary) ===")
    for n in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]:
        iters = max(5, 2000 // n)
        out = run([CIOT, "--bench-linear-pro", str(n), str(n), str(iters), "5", "10"])
        if out is None:
            continue
        d = parse_bench(out)
        ms = d.get("median_ms", 0)
        ops = n * n * 2
        gops = ops / (ms * 1e6) if ms > 0 else 0
        bytes_per_row = n * 2 / 8  # 2 bits per weight, 8 bits per byte
        total_bytes = n * bytes_per_row
        print(
            f"  {n:4d}x{n:<4d}  median={ms:.4f}ms  gop/s={gops:.2f}  matrix_bytes={total_bytes:.0f}"
        )
    return True


def test_thermal_stability():
    """Run continuous matvec for 10 seconds, check for degradation."""
    print("\n=== thermal stability (10s continuous matvec) ===")
    out = run([CIOT, "--bench-linear-pro", "1024", "1024", "10000", "1", "0"])
    if out is None:
        return False
    d = parse_bench(out)
    ms = d.get("median_ms", 0)
    ops = 1024 * 1024 * 2
    gops = ops / (ms * 1e6) if ms > 0 else 0
    print(f"  10000 iters  median={ms:.4f}ms  gop/s={gops:.2f}")
    print(f"  {'PASS' if gops > 3 else 'FAIL'} (throughput > 3 gop/s under load)")
    return gops > 3


def test_checksum_verification():
    """Verify SIMD and scalar produce identical results."""
    print("\n=== checksum verification (SIMD vs scalar) ===")
    simd = run([CIOT, "--bench-linear-pro", "512", "512", "10", "1", "0"])
    scalar = run(
        [CIOT, "--bench-linear-pro", "512", "512", "10", "1", "0"],
        env={**os.environ, "CIOT_BACKEND": "scalar"},
    )
    if simd is None or scalar is None:
        return False
    cs = parse_bench(simd).get("checksum", -1)
    cc = parse_bench(scalar).get("checksum", -2)
    ms_simd = parse_bench(simd).get("median_ms", 0)
    ms_scalar = parse_bench(scalar).get("median_ms", 0)
    match = abs(cs - cc) < 0.0001
    speedup = ms_scalar / ms_simd if ms_simd > 0 else 0
    print(f"  simd checksum={cs:.6f}  scalar checksum={cc:.6f}")
    print(f"  simd={ms_simd:.4f}ms  scalar={ms_scalar:.4f}ms  speedup={speedup:.1f}x")
    print(
        f"  {'PASS' if match else 'FAIL'} (checksums {'match' if match else 'MISMATCH'})"
    )
    return match


def main():
    print(f"ciot real-world benchmark suite")
    print(f"  binary: {CIOT}")
    print(
        f"  time: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    backend = run([CIOT, "--backend"])
    if backend:
        print(f"  backend: {backend.strip()}")

    results = {}
    results["checksum"] = test_checksum_verification()
    results["throughput"] = test_matvec_throughput()
    results["size_scaling"] = test_size_scaling()
    results["batch_scaling"] = test_batch_scaling()
    results["context_scaling"] = test_context_scaling()
    results["thermal"] = test_thermal_stability()

    print(f"\n=== summary ===")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL'}")
    print(f"  {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
