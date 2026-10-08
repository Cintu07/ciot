#!/usr/bin/env python3
"""
Ciot ternary weight packer.

Zero-heavy-dependency rule: this script uses only the Python standard library.
Input format is raw little-endian float32 in row-major order.
Output format:
  magic      8 bytes  b"CIOTBIT1"
  rows       uint32
  cols       uint32
  blocks64   uint32
  row_scale  rows * float32
  pos_bits   rows * blocks64 * uint64
  neg_bits   rows * blocks64 * uint64
"""

from __future__ import annotations

import argparse
import array
import math
import os
import struct
import sys
from typing import Iterable, List, Tuple

MAGIC = b"CIOTBIT1"


def read_f32(path: str, expected: int) -> array.array:
    values = array.array("f")
    with open(path, "rb") as f:
        values.fromfile(f, expected)
    if len(values) != expected:
        raise ValueError(f"expected {expected} float32 values, got {len(values)}")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def row_scale(row: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in row:
        total += abs(value)
        count += 1
    if count == 0:
        return 1.0
    scale = total / count
    return scale if scale > 1.0e-12 else 1.0


def quantize_row_error_compensated(row: List[float]) -> Tuple[float, List[int]]:
    scale = row_scale(row)
    threshold = 0.5 * scale
    carry = 0.0
    quantized: List[int] = []

    for value in row:
        adjusted = value + carry
        if adjusted > threshold:
            q = 1
        elif adjusted < -threshold:
            q = -1
        else:
            q = 0
        carry = adjusted - (q * scale)
        quantized.append(q)

    return scale, quantized


def pack_bits(q: List[int], cols: int, blocks64: int) -> Tuple[List[int], List[int]]:
    pos = [0] * blocks64
    neg = [0] * blocks64
    for c in range(cols):
        block = c >> 6
        bit = c & 63
        if q[c] > 0:
            pos[block] |= 1 << bit
        elif q[c] < 0:
            neg[block] |= 1 << bit
    return pos, neg


def pack_file(input_path: str, output_path: str, rows: int, cols: int) -> None:
    expected = rows * cols
    values = read_f32(input_path, expected)
    blocks64 = (cols + 63) // 64

    scales: List[float] = []
    all_pos: List[int] = []
    all_neg: List[int] = []
    total_abs_error = 0.0

    for r in range(rows):
        start = r * cols
        row = list(values[start:start + cols])
        scale, q = quantize_row_error_compensated(row)
        pos, neg = pack_bits(q, cols, blocks64)
        scales.append(scale)
        all_pos.extend(pos)
        all_neg.extend(neg)
        for original, quant in zip(row, q):
            total_abs_error += abs(original - (quant * scale))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", rows, cols, blocks64))
        f.write(struct.pack(f"<{len(scales)}f", *scales))
        f.write(struct.pack(f"<{len(all_pos)}Q", *all_pos))
        f.write(struct.pack(f"<{len(all_neg)}Q", *all_neg))

    mae = total_abs_error / expected if expected else math.nan
    print(f"wrote {output_path}")
    print(f"rows={rows} cols={cols} blocks64={blocks64} mean_abs_error={mae:.8f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack raw float32 weights into Ciot ternary .bits")
    parser.add_argument("input", help="raw little-endian float32 file, row-major")
    parser.add_argument("rows", type=int)
    parser.add_argument("cols", type=int)
    parser.add_argument("output", help="output .bits path")
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("rows and cols must be positive")

    pack_file(args.input, args.output, args.rows, args.cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
