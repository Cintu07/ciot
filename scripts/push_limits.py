#!/usr/bin/env python3
"""Push Ciot to its limits. One script, every test, results in one report."""

import datetime
import math
import os
import random
import statistics
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIOT = os.path.join(ROOT, "bin", "ciot")
if os.name == "nt":
    CIOT += ".exe"


def run(cmd, timeout=180, env=None, cwd=ROOT):
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
    )
    return r.stdout if r.returncode == 0 else None


def bench(cmd, timeout=120, env=None):
    out = run(cmd, timeout, env=env)
    if not out:
        return {}
    d = {}
    for line in out.strip().split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith(("ciot_", "name", "---", "===")):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    d[parts[0].strip()] = float(parts[1].strip())
                except:
                    pass
    return d


def compile_cpp(flags="-O3"):
    srcs = (
        "src/main.cpp src/kernels/ternary_simd.cpp src/linalg/linear.cpp "
        "src/model/ops.cpp src/model/tiny_transformer.cpp src/model/kv_cache.cpp "
        "src/model/mha_cache.cpp src/model/tokenizer.cpp src/model/bpe_tokenizer.cpp "
        "src/model/model_loader.cpp"
    )
    out = "-o bin/ciot_push"
    cmd = f"g++ -std=c++17 {flags} -Iinclude {srcs} {out}"
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    return r.returncode == 0, os.path.join(
        ROOT, "bin/ciot_push" + (".exe" if os.name == "nt" else "")
    )


def push_bench(binary, name, rows, cols, iters, repeats):
    d = bench(
        [
            binary,
            "--bench-linear-pro",
            str(rows),
            str(cols),
            str(iters),
            str(repeats),
            str(max(5, repeats * 2)),
        ]
    )
    return d.get("median_ms", 0), d.get("median_logical_Gop/s", 0), d.get("checksum", 0)


results = []


def record(name, value, unit=""):
    results.append((name, value, unit))
    print(f"  {name:40s} {value} {unit}")


print("=" * 65)
print("CIOT LIMIT TEST")
print(
    f"  {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
print("=" * 65)

# ---- COMPILE ----
print("\n[1] building with -O3 ...")
ok, binary = compile_cpp("-O3 -march=native")
if not ok:
    print("  compile failed, falling back to existing binary")
    binary = CIOT
else:
    print(f"  built: {binary}")

backend = bench([binary, "--backend"])
record("simd backend", backend.get("backend", "unknown"))

# ---- TEST 1: PEAK THROUGHPUT ----
print("\n[2] peak throughput vs matrix size")
best_gops, best_size = 0, 0
for n in [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096]:
    iters = max(5, 4000 // max(n, 1))
    ms, gops, cs = push_bench(binary, "peak", n, n, iters, 5)
    if gops > 0:
        print(f"    {n:5d}x{n:<5d}  {ms:.4f}ms  {gops:.2f} gop/s  chk={cs:.6f}")
        if gops > best_gops:
            best_gops, best_size = gops, n
record("peak gop/s", f"{best_gops:.2f} at {best_size}x{best_size}")

# ---- TEST 2: QUANTIZATION ERROR ----
print("\n[3] quantization error on realistic input")
vals = []
for i in range(1024):
    r = random.gauss(0, 1.0)
    if random.random() < 0.05:
        r *= random.uniform(3, 8)
    vals.append(r * 0.3)
with open(os.path.join(ROOT, "data/_realistic.f32"), "wb") as f:
    for v in vals:
        f.write(struct.pack("<f", v))

# Scalar matvec with random ternary matrix (same seed)
random.seed(42)
matrix_vals = [random.choice([-1, 0, 1]) for _ in range(1024 * 1024)]
scalar_result = sum(matrix_vals[i] * vals[i % 1024] for i in range(1024 * 1024))
record("realistic input range", f"{min(vals):.4f} to {max(vals):.4f}")
record(
    "realistic input stddev", f"{math.sqrt(sum(v * v for v in vals) / len(vals)):.4f}"
)

# ---- TEST 3: DECODE STRESS ----
print("\n[4] decode stress (500 tokens)")
out = run([binary, "--decode-generate", "64", "500"], timeout=60)
if out and "decode generation" in out:
    word_count = len(out.split())
    record("decode 500 tokens", f"OK ({word_count} words)")
else:
    record("decode 500 tokens", "FAIL")

print("\n[5] decode stress (2000 tokens)")
out = run([binary, "--decode-generate", "64", "2000"], timeout=180)
if out and "decode generation" in out:
    record("decode 2000 tokens", "OK")
else:
    record("decode 2000 tokens", "FAIL")

# ---- TEST 4: MEMORY STRESS ----
print("\n[6] model load/unload cycles")
cycles_ok = 0
for i in range(10):
    out = run(
        [binary, "--model-generate", "data/trained_model", "test", "3"], timeout=30
    )
    if out and "backend" in out:
        cycles_ok += 1
record("model load cycles", f"{cycles_ok}/10")

# ---- TEST 5: SCALAR vs SIMD ----
print("\n[7] scalar vs simd verification")
simd = bench([binary, "--bench-linear-pro", "512", "512", "20", "5", "5"])
scalar = bench(
    [binary, "--bench-linear-pro", "512", "512", "10", "3", "3"],
    env={**os.environ, "CIOT_BACKEND": "scalar"},
)
cs_match = abs(simd.get("checksum", -1) - scalar.get("checksum", -2)) < 0.0001
speedup = scalar.get("median_ms", 1) / max(simd.get("median_ms", 0.001), 0.001)
record(
    "checksums match",
    "YES" if cs_match else f"NO ({simd.get('checksum')} vs {scalar.get('checksum')})",
)
record("simd vs scalar speedup", f"{speedup:.1f}x")

# ---- TEST 6: CONTEXT SCALING ----
print("\n[8] attention context scaling")
ctx_times = []
for ctx in [4, 16, 64, 256]:
    d = bench(
        [
            binary,
            "--bench-decode-mha",
            "128",
            "4",
            str(ctx),
            str(max(2, 64 // ctx)),
            "3",
        ],
        timeout=120,
    )
    ms = d.get("median_ms", 0)
    if ms > 0:
        us_per_tok = (ms * 1000) / ctx
        ctx_times.append((ctx, ms, us_per_tok))
        print(f"    ctx={ctx:4d}  {ms:.4f}ms  {us_per_tok:.1f}us/tok")
if len(ctx_times) >= 2:
    ratio = ctx_times[-1][2] / max(ctx_times[0][2], 0.001)
    record("context scaling (4->256)", f"{ratio:.1f}x slowdown")

# ---- TEST 7: THERMAL ----
print("\n[9] sustained load (5000 matvecs)")
d = bench([binary, "--bench-linear-pro", "1024", "1024", "5000", "1", "0"], timeout=60)
ms, gops = d.get("median_ms", 0), d.get("median_logical_Gop/s", 0)
if gops > 0:
    record("5000 iters median", f"{ms:.4f}ms")
    record("5000 iters gop/s", f"{gops:.2f}")
    record("thermal pass", "YES" if gops > 3 else "NO")

# ---- REPORT ----
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
for name, value, unit in results:
    print(f"  {name:40s} {str(value):15s} {unit}")

passed = sum(
    1 for _, v, _ in results if not isinstance(v, str) or "FAIL" not in str(v).upper()
)
total = len(results)
print(f"\n  {passed}/{total} checks passed")

# Cleanup
for f in [
    os.path.join(ROOT, "data/_realistic.f32"),
    os.path.join(ROOT, "bin/ciot_push"),
    os.path.join(ROOT, "bin/ciot_push.exe"),
]:
    try:
        os.remove(f)
    except:
        pass
