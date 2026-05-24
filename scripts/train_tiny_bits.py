#!/usr/bin/env python3
"""Train a tiny token-transition model and write Ciot .bits.

This is intentionally tiny: it learns strongest next-token transitions from text,
then stores them as a ternary transition matrix. It is not an LLM, but it is a
real `.bits` artifact created from data, useful for end-to-end loader/generator
smoke tests without heavy dependencies.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from collections import Counter, defaultdict
from typing import cast

MAGIC = b"CIOTBIT1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_']+|[.,!?;:]", text.lower())


def write_bits(
    tokens: list[str], bits_path: str, vocab_path: str, vocab_limit: int
) -> None:
    counts: Counter[str] = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(max(2, vocab_limit))]
    if "<unk>" not in vocab:
        vocab.append("<unk>")
    index = {word: i for i, word in enumerate(vocab)}
    unk = index["<unk>"]
    n = len(vocab)
    blocks64 = (n + 63) // 64

    transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for a, b in zip(tokens, tokens[1:]):
        src = index.get(a, unk)
        dst = index.get(b, unk)
        transitions[src][dst] += 1

    pos = [0] * (n * blocks64)
    neg = [0] * (n * blocks64)
    scales = [4.0] * n

    for src in range(n):
        if transitions[src]:
            dst = transitions[src].most_common(1)[0][0]
        else:
            dst = (src + 1) % n
        word_index = dst * blocks64 + (src >> 6)
        pos[word_index] |= 1 << (src & 63)

    os.makedirs(os.path.dirname(bits_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
    with open(bits_path, "wb") as f:
        _ = f.write(MAGIC)
        _ = f.write(struct.pack("<III", n, n, blocks64))
        _ = f.write(struct.pack(f"<{len(scales)}f", *scales))
        _ = f.write(struct.pack(f"<{len(pos)}Q", *pos))
        _ = f.write(struct.pack(f"<{len(neg)}Q", *neg))

    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in vocab:
            _ = f.write(word + "\n")

    print(f"wrote {bits_path}")
    print(f"wrote {vocab_path}")
    print(f"vocab={n} blocks64={blocks64} training_tokens={len(tokens)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train tiny transition .bits model from text"
    )
    _ = parser.add_argument("text", help="training text file")
    _ = parser.add_argument("bits", help="output .bits path")
    _ = parser.add_argument("vocab", help="output vocab path")
    _ = parser.add_argument("--vocab-limit", type=int, default=64)
    args = parser.parse_args()

    text_path = cast(str, args.text)
    bits_path = cast(str, args.bits)
    vocab_path = cast(str, args.vocab)
    vocab_limit = cast(int, args.vocab_limit)

    with open(text_path, "r", encoding="utf-8") as f:
        tokens = tokenize(f.read())
    if len(tokens) < 2:
        raise SystemExit("need at least two tokens")
    write_bits(tokens, bits_path, vocab_path, vocab_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
