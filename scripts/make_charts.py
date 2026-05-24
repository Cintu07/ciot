#!/usr/bin/env python3
"""Generate clean, professional SVG benchmark and architecture figures."""

import datetime as dt
import math
import os

FONT = "system-ui, -apple-system, sans-serif"


def svg_hero(width=900, height=440):
    bars = [
        ("scalar", 3.058, "#c0392b", "3.06 ms"),
        ("arm neon", 0.535, "#27ae60", "0.54 ms"),
    ]
    start_x, bar_w, max_bar = 220, 440, 4.0
    bar_h, gap, base_y = 72, 56, 140

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
    ]

    svg += [
        f'<text x="450" y="44" text-anchor="middle" fill="#e6edf3" font-family="{FONT}" font-size="22" font-weight="600">1024x1024 ternary matvec latency</text>',
        f'<text x="450" y="70" text-anchor="middle" fill="#8b949e" font-family="{FONT}" font-size="13">snapdragon x  /  arm neon  /  -O3  /  zero dependencies  /  warmup 20 repeats 9</text>',
        f'<line x1="60" y1="88" x2="840" y2="88" stroke="#21262d" stroke-width="1"/>',
    ]

    for yi in range(5):
        val = (5 - yi) * 1.0
        x = start_x + (val / max_bar) * bar_w
        svg += [
            f'<line x1="{x}" y1="{base_y - 10}" x2="{x}" y2="{base_y + 2 * (bar_h + gap) - bar_h / 2}" stroke="#21262d" stroke-width="0.5" stroke-dasharray="4 4"/>',
            f'<text x="{x}" y="{base_y - 16}" text-anchor="middle" fill="#484f58" font-family="{FONT}" font-size="12">{val:.1f} ms</text>',
        ]

    for i, (name, val, color, label) in enumerate(bars):
        y = base_y + i * (bar_h + gap)
        w = (val / max_bar) * bar_w
        svg += [
            f'<text x="{start_x - 16}" y="{y + bar_h / 2 + 5}" text-anchor="end" fill="#e6edf3" font-family="{FONT}" font-size="18" font-weight="500">{name}</text>',
            f'<rect x="{start_x}" y="{y}" width="{w}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>',
            f'<text x="{start_x + w + 16}" y="{y + bar_h / 2 + 5}" fill="#e6edf3" font-family="{FONT}" font-size="18" font-weight="600">{label}</text>',
        ]

    speedup = bars[0][1] / bars[1][1]
    foot_y = base_y + 2 * (bar_h + gap) + 20
    svg += [
        f'<line x1="60" y1="{foot_y}" x2="840" y2="{foot_y}" stroke="#21262d" stroke-width="1"/>',
        f'<text x="450" y="{foot_y + 36}" text-anchor="middle" fill="#27ae60" font-family="{FONT}" font-size="19" font-weight="600">simd is {speedup:.1f}x faster. checksums match between scalar and simd.</text>',
        f'<text x="450" y="{foot_y + 60}" text-anchor="middle" fill="#484f58" font-family="{FONT}" font-size="12">scalar checksum = simd checksum = 1.951170  |  github.com/Cintu07/ciot  |  {dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}</text>',
    ]

    svg.append("</svg>")
    return "\n".join(svg)


def svg_table(width=900, height=480):
    rows = [
        ("benchmark", "median ms", "gop/s", "checksum", "status"),
        ("linear 1024x1024", "0.54", "3.92", "1.951170", "ok"),
        ("linear (scalar ref)", "3.06", "0.69", "1.951170", "ok"),
        ("batched 4x1024", "1.84", "4.57", "7.804690", "ok"),
        ("decode 128x32", "1.27", "4.97", "-604.644", "ok"),
        ("mha decode 128x4h", "1.31", "4.80", "0.000000", "ok"),
        ("rope 1024-dim", "0.00045", "--", "-8.003250", "ok"),
        ("transformer 256", "0.15", "5.14", "62.525900", "ok"),
    ]

    cols = [36, 240, 390, 516, 686, 786]
    row_h, start_y = 44, 100

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="450" y="40" text-anchor="middle" fill="#e6edf3" font-family="{FONT}" font-size="20" font-weight="600">production benchmarks (7/7 pass)</text>',
        f'<text x="450" y="64" text-anchor="middle" fill="#8b949e" font-family="{FONT}" font-size="12">snapdragon x  /  arm neon  /  -O3  /  checksum verified</text>',
    ]

    for ri, row in enumerate(rows):
        y = start_y + ri * (row_h + 2)
        bg = "#161b22" if ri == 0 else ("#0d1117" if ri % 2 == 0 else "#161b22")
        svg.append(
            f'<rect x="20" y="{y}" width="860" height="{row_h}" rx="4" fill="{bg}"/>'
        )
        for ci, cell in enumerate(row):
            c = "#8b949e" if ri == 0 else ("#27ae60" if ci == 4 else "#e6edf3")
            w = "600" if ri == 0 else "400"
            svg.append(
                f'<text x="{cols[ci]}" y="{y + row_h - 14}" fill="{c}" font-family="{FONT}" font-size="14" font-weight="{w}">{cell}</text>'
            )

    foot_y = start_y + len(rows) * (row_h + 2) + 18
    svg += [
        f'<line x1="40" y1="{foot_y}" x2="860" y2="{foot_y}" stroke="#21262d" stroke-width="1"/>',
        f'<text x="450" y="{foot_y + 28}" text-anchor="middle" fill="#27ae60" font-family="{FONT}" font-size="15" font-weight="500">simd speedup: 5.7x over scalar  |  checksum match: verified  |  total benchmarks: 7</text>',
        f'<text x="450" y="{foot_y + 52}" text-anchor="middle" fill="#484f58" font-family="{FONT}" font-size="11">github.com/Cintu07/ciot  |  {dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}</text>',
    ]

    svg.append("</svg>")
    return "\n".join(svg)


def svg_architecture(width=960, height=640):
    box_color = "#58a6ff"
    weight_color = "#d2a8ff"
    compute_color = "#27ae60"
    output_color = "#f0883e"
    arrow_color = "#30363d"
    text_color = "#e6edf3"

    stages = [
        ("text input", 60, 460, box_color, 180, 44),
        ("tokenizer\nword-level lookup", 60, 340, output_color, 180, 56),
        ("embedding\nvocab x dim (.bits)", 60, 200, weight_color, 180, 56),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="480" y="36" text-anchor="middle" fill="#e6edf3" font-family="{FONT}" font-size="20" font-weight="600">ciot inference pipeline</text>',
        f'<text x="480" y="60" text-anchor="middle" fill="#8b949e" font-family="{FONT}" font-size="12">all weights stored as packed ternary .bits  |  zero heavy dependencies</text>',
    ]

    # Draw left column
    for text, x, y, color, w, h in stages:
        lines = text.split("\n")
        sw, sh = w, h
        svg.append(
            f'<rect x="{x}" y="{y}" width="{sw}" height="{sh}" rx="6" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.5" stroke-opacity="0.4"/>'
        )
        for li, line in enumerate(lines):
            svg.append(
                f'<text x="{x + sw / 2}" y="{y + sh / 2 - (len(lines) - 1) * 9 + li * 18}" text-anchor="middle" fill="{color}" font-family="{FONT}" font-size="13" font-weight="500">{line}</text>'
            )

    # Arrows left column
    for y1, y2 in [(424, 398), (304, 278), (164, 538)]:
        svg.append(
            f'<line x1="150" y1="{y1}" x2="150" y2="{y2}" stroke="{arrow_color}" stroke-width="2"/>'
        )

    # Transformer layers - center
    tx, ty, tw, th = 290, 100, 400, 440
    svg.append(
        f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" rx="8" fill="{compute_color}" fill-opacity="0.08" stroke="{compute_color}" stroke-width="2" stroke-opacity="0.5"/>'
    )
    svg.append(
        f'<text x="{tx + tw / 2}" y="{ty + 28}" text-anchor="middle" fill="{compute_color}" font-family="{FONT}" font-size="15" font-weight="600">transformer layers (1 to n)</text>'
    )

    # Layer internals
    layer_items = [
        ("rmsnorm", 144),
        ("q/k/v projection", 184),
        ("rope rotation", 224),
        ("multi-head attention", 264),
        ("kv cache append", 304),
        ("output projection", 344),
        ("residual add", 384),
        ("rmsnorm", 424),
        ("feedforward (relu)", 464),
        ("residual add", 504),
    ]
    for text, yoff in layer_items:
        svg.append(
            f'<text x="{tx + 20}" y="{ty + yoff}" fill="{text_color}" font-family="{FONT}" font-size="12">{text}</text>'
        )

    # Right column
    r_stages = [
        ("lm head\nproject to vocab (.bits)", 740, 340, weight_color, 180, 56),
        ("token decode\nargmax selection", 740, 220, output_color, 180, 44),
        ("text output", 740, 100, box_color, 180, 44),
    ]
    for text, x, y, color, w, h in r_stages:
        lines = text.split("\n")
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.5" stroke-opacity="0.4"/>'
        )
        for li, line in enumerate(lines):
            svg.append(
                f'<text x="{x + w / 2}" y="{y + h / 2 - (len(lines) - 1) * 9 + li * 18}" text-anchor="middle" fill="{color}" font-family="{FONT}" font-size="13" font-weight="500">{line}</text>'
            )

    # Arrows right column
    for y1, y2 in [(304, 278), (184, 164)]:
        svg.append(
            f'<line x1="830" y1="{y1}" x2="830" y2="{y2}" stroke="{arrow_color}" stroke-width="2"/>'
        )

    # Horizontal arrows
    svg.append(
        f'<line x1="240" y1="300" x2="290" y2="300" stroke="{arrow_color}" stroke-width="2"/>'
    )
    svg.append(
        f'<line x1="690" y1="300" x2="740" y2="300" stroke="{arrow_color}" stroke-width="2"/>'
    )

    svg.append(
        f'<text x="480" y="590" text-anchor="middle" fill="#484f58" font-family="{FONT}" font-size="12">simd backends: arm neon / avx2 / avx-512  |  scalar reference path included  |  github.com/Cintu07/ciot</text>'
    )
    svg.append(
        f'<text x="480" y="615" text-anchor="middle" fill="#484f58" font-family="{FONT}" font-size="11">{dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(base), "data")

    charts = [
        ("bench_hero.svg", svg_hero()),
        ("bench_table.svg", svg_table()),
        ("architecture.svg", svg_architecture()),
    ]

    for name, svg in charts:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
