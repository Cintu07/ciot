#!/usr/bin/env python3
"""Create deterministic raw float32 weights for Ciot pack/loader benchmarks."""

from __future__ import annotations

import argparse
import math
import os
import struct
from typing import cast


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write raw little-endian float32 matrix weights"
    )
    _ = parser.add_argument("rows", type=int)
    _ = parser.add_argument("cols", type=int)
    _ = parser.add_argument("output")
    args = parser.parse_args()

    rows = cast(int, args.rows)
    cols = cast(int, args.cols)
    output = cast(str, args.output)
    if rows <= 0 or cols <= 0:
        raise SystemExit("rows and cols must be positive")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as f:
        for r in range(rows):
            for c in range(cols):
                v = math.sin((r * 131 + c * 17) * 0.013) * 0.25
                _ = f.write(struct.pack("<f", v))

    print(f"wrote {output} rows={rows} cols={cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
