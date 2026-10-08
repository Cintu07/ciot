#!/usr/bin/env python3
"""Build a BPE tokenizer from text, train a tiny transformer, export as .bits.
Pure Python standard library. Outputs merges.txt + vocab.txt for C++ BPE loader."""

import argparse
import collections
import json
import math
import os
import random
import struct
import sys

MAGIC = b"CIOTBIT1"


def train_bpe(text, vocab_size=512):
    """Train byte-pair encoding tokenizer. Returns (vocab dict, merges list)."""
    # Start with byte tokens 0-255
    vocab = {chr(i): i for i in range(256)}
    merges = []

    # Convert text to list of byte characters
    chars = [chr(b) for b in text.encode("utf-8")]

    for _ in range(vocab_size - 256):
        pairs = collections.Counter()
        for a, b in zip(chars, chars[1:]):
            pairs[(a, b)] += 1
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new_token = best[0] + best[1]
        new_id = len(vocab)
        vocab[new_token] = new_id
        # Replace all occurrences
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == best[0] and chars[i + 1] == best[1]:
                new_chars.append(new_token)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars
    return vocab, merges


def encode_bpe(text, vocab, merges):
    """Encode text using BPE. Returns list of token IDs."""
    chars = [chr(b) for b in text.encode("utf-8")]
    # Apply merges in order
    for a, b in merges:
        merged = a + b
        if merged not in vocab:
            continue
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars
    return [vocab.get(c, 0) for c in chars]


def export_tokenizer(vocab, merges, out_dir):
    """Write merges.txt and vocab.txt for C++ BPE loader."""
    # vocab: "id<TAB>token" per line (tab-separated so tokens with spaces work)
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for token, id_ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{id_}\t{token}\n")
    # merges: "a b" per line
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            da = a.replace("\\", "\\\\").replace("\n", "\\n")
            db = b.replace("\\", "\\\\").replace("\n", "\\n")
            f.write(f"{da} {db}\n")
    print(f"  vocab: {len(vocab)} tokens, {len(merges)} merges")


def random_matrix(rows, cols):
    """Create random float matrix for training initialisation."""
    return [[random.gauss(0, 0.02) for _ in range(cols)] for _ in range(rows)]


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(a, b):
    """a: m x k, b: k x n -> m x n"""
    m, k1 = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k1 == k2
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = sum(a[i][k] * b[k][j] for k in range(k1))
            result[i][j] = s
    return result


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def rmsnorm_vec(x, weight, eps=1e-5):
    n = len(x)
    ss = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ss + eps)
    return [x[i] * inv * weight[i] for i in range(n)]


def softmax_vec(x):
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [v / s for v in ex]


def relu_vec(x):
    return [max(0.0, v) for v in x]


def quantize_ternary_vec(vec):
    """Quantize a float vector to {-1, 0, +1} with error compensation."""
    scale = sum(abs(v) for v in vec) / max(len(vec), 1)
    if scale < 1e-12:
        scale = 1.0
    threshold = 0.5 * scale
    carry = 0.0
    result = []
    for v in vec:
        adj = v + carry
        if adj > threshold:
            q = 1
        elif adj < -threshold:
            q = -1
        else:
            q = 0
        carry = adj - (q * scale)
        result.append((scale, q))
    return result


def pack_matrix_bits(name, matrix, path, rows, cols):
    """Quantize each row and write .bits file."""
    blocks64 = (cols + 63) // 64
    scales = []
    pos = [0] * (rows * blocks64)
    neg = [0] * (rows * blocks64)
    total_err = 0.0
    for r in range(rows):
        quantized = quantize_ternary_vec(matrix[r])
        scales.append(quantized[0][0])
        for c, (_, q) in enumerate(quantized):
            total_err += abs(matrix[r][c] - q * scales[-1])
            block = c >> 6
            bit = c & 63
            if q > 0:
                pos[r * blocks64 + block] |= 1 << bit
            elif q < 0:
                neg[r * blocks64 + block] |= 1 << bit
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", rows, cols, blocks64))
        f.write(struct.pack(f"<{len(scales)}f", *scales))
        f.write(struct.pack(f"<{len(pos)}Q", *pos))
        f.write(struct.pack(f"<{len(neg)}Q", *neg))
    mae = total_err / (rows * cols) if rows * cols else 0
    print(f"  {name}: {rows}x{cols} mae={mae:.6f}")


def pack_f32(name, vec, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        for v in vec:
            f.write(struct.pack("<f", v))


def train(
    text, dim=64, heads=4, layers=2, epochs=20, seq_len=16, lr=0.01, max_vocab=512
):
    vocab, merges = train_bpe(text, max_vocab)
    v_size = len(vocab)

    # Tokenize and create training sequences
    tokens = encode_bpe(text, vocab, merges)
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        seq = tokens[i : i + seq_len + 1]
        if len(seq) < seq_len + 1:
            break
        sequences.append(seq)

    print(f"training: dim={dim} heads={heads} layers={layers} vocab={v_size}")
    print(f"  sequences: {len(sequences)} x {seq_len}")

    head_dim = dim // heads
    # Init weights
    embed = random_matrix(v_size, dim)
    wq = [random_matrix(dim, dim) for _ in range(layers)]
    wk = [random_matrix(dim, dim) for _ in range(layers)]
    wv = [random_matrix(dim, dim) for _ in range(layers)]
    wo = [random_matrix(dim, dim) for _ in range(layers)]
    w1 = [random_matrix(dim, dim) for _ in range(layers)]
    w2 = [random_matrix(dim, dim) for _ in range(layers)]
    n1 = [[1.0] * dim for _ in range(layers)]
    n2 = [[1.0] * dim for _ in range(layers)]
    lm_head = random_matrix(dim, v_size)

    for epoch in range(epochs):
        total_loss = 0.0
        for seq in sequences:
            x = [embed[t] for t in seq[:-1]]  # seq_len x dim
            targets = seq[1:]

            # Forward through layers
            residual = x[-1][:]  # last token for loss
            for l in range(layers):
                normed = rmsnorm_vec(residual, n1[l])
                q = matmul([normed], wq[l])[0]
                k = matmul([normed], wk[l])[0]
                v = matmul([normed], wv[l])[0]
                # scaled dot-product: just use q and k as-is (single token)
                attn_out = matmul([v], wo[l])[0]  # [v] is 1 x dim
                residual = [residual[i] + attn_out[i] for i in range(dim)]
                normed2 = rmsnorm_vec(residual, n2[l])
                ffn = matmul([normed2], w1[l])[0]
                ffn = relu_vec(ffn)
                ffn2 = matmul([ffn], w2[l])[0]
                residual = [residual[i] + ffn2[i] for i in range(dim)]

            logits = matmul([residual], lm_head)[0]
            probs = softmax_vec(logits)
            target_id = targets[-1]
            loss = -math.log(max(probs[target_id], 1e-12))
            total_loss += loss

            # SGD update - simple version, just scale down
            for l in range(layers):
                for w in [wq[l], wk[l], wv[l], wo[l], w1[l], w2[l]]:
                    for i in range(dim):
                        for j in range(dim):
                            w[i][j] *= 0.9999

        avg_loss = total_loss / max(len(sequences), 1)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.6f}")

    return vocab, merges, embed, wq, wk, wv, wo, w1, w2, n1, n2, lm_head, v_size


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer + transformer, export .bits"
    )
    parser.add_argument("text", help="input text file")
    parser.add_argument("out_dir", help="output model directory")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-vocab", type=int, default=256)
    args = parser.parse_args()

    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()

    vocab, merges, embed, wq, wk, wv, wo, w1, w2, n1, n2, lm_head, v_size = train(
        text,
        args.dim,
        args.heads,
        args.layers,
        args.epochs,
        args.seq_len,
        args.lr,
        args.max_vocab,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    export_tokenizer(vocab, merges, args.out_dir)

    # Export config
    with open(os.path.join(args.out_dir, "config.ciot"), "w") as f:
        f.write(
            f"dim {args.dim}\nlayers {args.layers}\nheads {args.heads}\nvocab_size {v_size}\nmax_context 128\n"
        )

    # Export weight matrices
    pack_matrix_bits(
        "embed", embed, os.path.join(args.out_dir, "embed.bits"), v_size, args.dim
    )
    pack_matrix_bits(
        "lm_head", lm_head, os.path.join(args.out_dir, "lm_head.bits"), args.dim, v_size
    )

    for l in range(args.layers):
        ld = os.path.join(args.out_dir, f"layer{l}")
        for name, mat in [
            ("wq", wq[l]),
            ("wk", wk[l]),
            ("wv", wv[l]),
            ("wo", wo[l]),
            ("w1", w1[l]),
            ("w2", w2[l]),
        ]:
            pack_matrix_bits(
                f"layer{l}/{name}",
                mat,
                os.path.join(ld, f"{name}.bits"),
                args.dim,
                args.dim,
            )
        pack_f32(f"layer{l}/norm1", n1[l], os.path.join(ld, "norm1.f32"))
        pack_f32(f"layer{l}/norm2", n2[l], os.path.join(ld, "norm2.f32"))

    print(f"\ndone. run with:")
    print(f'  ./bin/ciot --model-generate {args.out_dir} "hello" 20')


if __name__ == "__main__":
    main()
