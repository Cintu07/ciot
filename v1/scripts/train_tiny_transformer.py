#!/usr/bin/env python3
"""Train a tiny 2-layer transformer from text, quantize to ternary, export as .bits.

Pure Python stdlib only: math, struct, random, collections, argparse, os, re, sys.

Produces a model directory compatible with `ciot --model-generate <dir>`.

Example:
  python scripts/train_tiny_transformer.py data/input.txt data/tiny_model --dim 64 --heads 2 --epochs 20
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Matrix operations (list-of-lists, row-major)
# ============================================================================


def zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0] * cols for _ in range(rows)]


def random_matrix(rows: int, cols: int, scale: float = 0.02) -> List[List[float]]:
    """Xavier-like init: uniform [-scale, +scale]."""
    return [
        [(random.random() * 2.0 - 1.0) * scale for _ in range(cols)]
        for _ in range(rows)
    ]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """A: MxK, B: KxN -> MxN."""
    M, K = len(A), len(A[0])
    KB, N = len(B), len(B[0])
    assert K == KB, f"matmul shape mismatch: ({M},{K}) x ({KB},{N})"
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        Ai = A[i]
        Ci = C[i]
        for k in range(K):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            for j in range(N):
                Ci[j] += aik * Bk[j]
    return C


def matmul_add(
    A: List[List[float]], B: List[List[float]], C: List[List[float]]
) -> List[List[float]]:
    """A @ B + C, returns new matrix."""
    M, K = len(A), len(A[0])
    KB, N = len(B), len(B[0])
    assert K == KB
    D = [row[:] for row in C]  # copy C
    for i in range(M):
        Ai = A[i]
        Di = D[i]
        for k in range(K):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            for j in range(N):
                Di[j] += aik * Bk[j]
    return D


def transpose(A: List[List[float]]) -> List[List[float]]:
    M, N = len(A), len(A[0])
    return [[A[i][j] for i in range(M)] for j in range(N)]


def add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]


def sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]


def scale(s: float, A: List[List[float]]) -> List[List[float]]:
    return [[v * s for v in row] for row in A]


def elem_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] * B[i][j] for j in range(cols)] for i in range(rows)]


def sum_cols(A: List[List[float]]) -> List[float]:
    """Sum over rows (axis=0) -> list of length cols."""
    if not A:
        return []
    cols = len(A[0])
    result = [0.0] * cols
    for row in A:
        for j in range(cols):
            result[j] += row[j]
    return result


def sum_rows(A: List[List[float]]) -> List[float]:
    """Sum over cols (axis=1) -> list of length rows."""
    return [sum(row) for row in A]


def hadamard(a: List[float], b: List[float]) -> List[float]:
    return [a[i] * b[i] for i in range(len(a))]


def vec_scale(s: float, v: List[float]) -> List[float]:
    return [x * s for x in v]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(len(a))]


def outer(a: List[float], b: List[float]) -> List[List[float]]:
    """Outer product: a (M) x b (N) -> MxN."""
    return [[ai * bj for bj in b] for ai in a]


def matvec(M: List[List[float]], v: List[float]) -> List[float]:
    """Matrix-vector multiply: M (RxC) @ v (C) -> (R)."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def vec_add_to(target: List[float], source: List[float]) -> None:
    for i in range(len(target)):
        target[i] += source[i]


def vec_scale_add(target: List[float], scale: float, source: List[float]) -> None:
    for i in range(len(target)):
        target[i] += scale * source[i]


def clip_grad_norm(grads: List[List[float]], max_norm: float) -> float:
    total = 0.0
    for row in grads:
        for v in row:
            total += v * v
    norm = math.sqrt(total)
    if norm > max_norm:
        s = max_norm / norm
        for row in grads:
            for j in range(len(row)):
                row[j] *= s
    return norm


# ============================================================================
# Tokenizer
# ============================================================================


def tokenize(text: str) -> List[str]:
    """Simple word-level tokenizer: lowercase, split on word boundaries."""
    return re.findall(r"[a-zA-Z0-9_']+|[.,!?;:\-]", text.lower())


def build_vocab(
    tokens: List[str], max_vocab: int = 256
) -> Tuple[List[str], Dict[str, int]]:
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(max_vocab)]
    if "<unk>" not in vocab:
        vocab.insert(0, "<unk>")
    word2id = {w: i for i, w in enumerate(vocab)}
    return vocab, word2id


# ============================================================================
# RMSNorm
# ============================================================================


class RMSNorm:
    """RMS normalization: out = x / rms(x) * weight."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim  # learnable scale

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """x: (L, dim) -> (L, dim)."""
        L = len(x)
        out = zeros(L, self.dim)
        for i in range(L):
            xi = x[i]
            sq_sum = sum(v * v for v in xi)
            rms = math.sqrt(sq_sum / self.dim + self.eps)
            inv_rms = 1.0 / rms
            oi = out[i]
            w = self.weight
            for j in range(self.dim):
                oi[j] = xi[j] * inv_rms * w[j]
        return out

    def backward(
        self,
        grad_out: List[List[float]],
        x: List[List[float]],
        normed: List[List[float]],
    ) -> Tuple[List[List[float]], List[float]]:
        """Returns (grad_x, grad_weight)."""
        L = len(x)
        dim = self.dim
        w = self.weight
        grad_x = zeros(L, dim)
        grad_w = [0.0] * dim

        for i in range(L):
            xi = x[i]
            gi = grad_out[i]
            ni = normed[i]

            # grad_weight = sum_i grad_out_i * (x_i / rms)
            sq_sum = sum(v * v for v in xi)
            rms = math.sqrt(sq_sum / dim + self.eps)
            inv_rms = 1.0 / rms

            for j in range(dim):
                grad_w[j] += gi[j] * xi[j] * inv_rms

            # grad_x
            # grad_x_j = w_j * grad_out_j / rms - x_j * sum_k(grad_out_k * w_k * x_k) / (dim * rms^3)
            sum_grad = sum(gi[k] * w[k] * xi[k] for k in range(dim))
            denom = dim * rms * rms * rms

            for j in range(dim):
                grad_x[i][j] = gi[j] * w[j] * inv_rms - xi[j] * sum_grad / denom

        return grad_x, grad_w


# ============================================================================
# Multi-Head Self-Attention (causal)
# ============================================================================


class MultiHeadAttention:
    def __init__(self, dim: int, num_heads: int):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        scale = 0.02
        self.Wq = random_matrix(dim, dim, scale)
        self.Wk = random_matrix(dim, dim, scale)
        self.Wv = random_matrix(dim, dim, scale)
        self.Wo = random_matrix(dim, dim, scale)

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """(L, dim) -> list of heads [(L, head_dim), ...]."""
        L = len(x)
        heads = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            head = [[x[i][start + j] for j in range(self.head_dim)] for i in range(L)]
            heads.append(head)
        return heads

    def _merge_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """[(L, head_dim), ...] -> (L, dim)."""
        L = len(heads[0])
        out = zeros(L, self.dim)
        for h in range(self.num_heads):
            start = h * self.head_dim
            hh = heads[h]
            for i in range(L):
                oi = out[i]
                hi = hh[i]
                for j in range(self.head_dim):
                    oi[start + j] = hi[j]
        return out

    def forward(
        self, x: List[List[float]], causal_mask: bool = True
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """x: (L, dim). Returns (output, cache)."""
        L = len(x)
        dim = self.dim
        hdim = self.head_dim
        nh = self.num_heads

        # Linear projections
        Q = matmul(x, transpose(self.Wq))  # (L, dim)
        K = matmul(x, transpose(self.Wk))
        V = matmul(x, transpose(self.Wv))

        # Split into heads
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scale_factor = 1.0 / math.sqrt(hdim)
        attn_heads = []

        for h in range(nh):
            qh = Qh[h]  # (L, hdim)
            kh = Kh[h]
            vh = Vh[h]

            # scores = q @ k^T / sqrt(d)
            scores = matmul(qh, transpose(kh))  # (L, L)
            for i in range(L):
                si = scores[i]
                for j in range(L):
                    si[j] *= scale_factor

            # Causal mask
            if causal_mask:
                for i in range(L):
                    si = scores[i]
                    for j in range(i + 1, L):
                        si[j] = -1e9

            # Softmax
            for i in range(L):
                si = scores[i]
                mx = max(si)
                exp_sum = 0.0
                for j in range(L):
                    si[j] = math.exp(si[j] - mx)
                    exp_sum += si[j]
                inv = 1.0 / exp_sum if exp_sum > 0 else 1.0
                for j in range(L):
                    si[j] *= inv

            # Weighted sum: attn_h = scores @ V
            attn_h = matmul(scores, vh)  # (L, hdim)
            attn_heads.append(attn_h)

        # Merge heads
        attn = self._merge_heads(attn_heads)  # (L, dim)

        # Output projection
        out = matmul(attn, transpose(self.Wo))

        cache = {
            "x": x,
            "Q": Q,
            "K": K,
            "V": V,
            "attn": attn,
            "Qh": Qh,
            "Kh": Kh,
            "Vh": Vh,
        }
        return out, cache

    def backward(
        self, grad_out: List[List[float]], cache: Dict[str, Any]
    ) -> Dict[str, List[List[float]]]:
        """Returns grads for Wq, Wk, Wv, Wo and grad_x."""
        x = cache["x"]
        Q = cache["Q"]
        K = cache["K"]
        V = cache["V"]
        attn = cache["attn"]
        Qh = cache["Qh"]
        Kh = cache["Kh"]
        Vh = cache["Vh"]

        L = len(x)
        dim = self.dim
        hdim = self.head_dim
        nh = self.num_heads
        scale_factor = 1.0 / math.sqrt(hdim)

        # grad_out through Wo
        # out = attn @ Wo^T -> grad_attn = grad_out @ Wo
        grad_attn = matmul(grad_out, self.Wo)  # (L, dim)
        grad_Wo = matmul(transpose(grad_out), attn)  # (dim, dim)

        # Split grad_attn into heads
        grad_attn_heads = self._split_heads(grad_attn)

        grad_Qh = []
        grad_Kh = []
        grad_Vh = []

        for h in range(nh):
            qh = Qh[h]  # (L, hdim)
            kh = Kh[h]
            vh = Vh[h]
            gattn_h = grad_attn_heads[h]  # (L, hdim)

            # Recompute causal softmax forward for this head
            # (could cache probabilities but let's recompute for simplicity)
            scores = matmul(qh, transpose(kh))
            for i in range(L):
                si = scores[i]
                for j in range(L):
                    si[j] *= scale_factor
            # Apply causal mask
            for i in range(L):
                for j in range(i + 1, L):
                    scores[i][j] = -1e9
            # Softmax
            probs = zeros(L, L)
            for i in range(L):
                si = scores[i]
                mx = max(si)
                exp_sum = 0.0
                for j in range(L):
                    probs[i][j] = math.exp(si[j] - mx)
                    exp_sum += probs[i][j]
                inv = 1.0 / exp_sum if exp_sum > 0 else 1.0
                for j in range(L):
                    probs[i][j] *= inv

            # attn_h = probs @ vh
            # grad_probs = gattn_h @ vh^T  (L, L)
            grad_probs = matmul(gattn_h, transpose(vh))  # (L, L)

            # grad_vh = probs^T @ gattn_h  (L, hdim)
            grad_vh = matmul(transpose(probs), gattn_h)

            # Softmax backward: grad_scores = probs * (grad_probs - sum(probs * grad_probs, dim=-1))
            grad_scores = zeros(L, L)
            for i in range(L):
                row_p = probs[i]
                row_gp = grad_probs[i]
                # sum over j of p_ij * grad_probs_ij
                spg = sum(row_p[j] * row_gp[j] for j in range(L))
                gsi = grad_scores[i]
                for j in range(L):
                    gsi[j] = row_p[j] * (row_gp[j] - spg)

            # Apply causal mask: positions where p=0 (upper triangle) have grad 0
            for i in range(L):
                for j in range(i + 1, L):
                    grad_scores[i][j] = 0.0

            # Scale back
            for i in range(L):
                gsi = grad_scores[i]
                for j in range(L):
                    gsi[j] *= scale_factor

            # grad_Qh: grad_scores @ kh  (L, hdim)
            grad_qh = matmul(grad_scores, kh)
            # grad_Kh: grad_scores^T @ qh  (L, hdim)
            grad_kh = matmul(transpose(grad_scores), qh)

            grad_Qh.append(grad_qh)
            grad_Kh.append(grad_kh)
            grad_Vh.append(grad_vh)

        # Merge head gradients
        grad_Q = self._merge_heads(grad_Qh)
        grad_K = self._merge_heads(grad_Kh)
        grad_V = self._merge_heads(grad_Vh)

        # Back through linear projections
        # Q = x @ Wq^T -> grad_x_q = grad_Q @ Wq, grad_Wq = grad_Q^T @ x
        grad_x_q = matmul(grad_Q, self.Wq)
        grad_Wq = matmul(transpose(grad_Q), x)

        grad_x_k = matmul(grad_K, self.Wk)
        grad_Wk = matmul(transpose(grad_K), x)

        grad_x_v = matmul(grad_V, self.Wv)
        grad_Wv = matmul(transpose(grad_V), x)

        # Total grad_x from attention
        grad_x = add(add(grad_x_q, grad_x_k), grad_x_v)

        return {
            "Wq": grad_Wq,
            "Wk": grad_Wk,
            "Wv": grad_Wv,
            "Wo": grad_Wo,
            "x": grad_x,
        }


# ============================================================================
# Feed-Forward Network
# ============================================================================


class FeedForward:
    def __init__(self, dim: int, hidden_mult: int = 4):
        self.dim = dim
        self.hidden_dim = dim * hidden_mult
        scale = 0.02
        self.W1 = random_matrix(self.hidden_dim, dim, scale)
        self.W2 = random_matrix(dim, self.hidden_dim, scale)

    def forward(self, x: List[List[float]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """x: (L, dim). Returns (output, cache)."""
        # ffn = relu(x @ W1^T)
        hidden = matmul(x, transpose(self.W1))  # (L, hidden_dim)
        activated = [[max(0.0, v) for v in row] for row in hidden]
        out = matmul(activated, transpose(self.W2))  # (L, dim)
        cache = {"x": x, "hidden": hidden, "activated": activated}
        return out, cache

    def backward(
        self, grad_out: List[List[float]], cache: Dict[str, Any]
    ) -> Dict[str, List[List[float]]]:
        x = cache["x"]
        hidden = cache["hidden"]
        activated = cache["activated"]

        L = len(x)
        hid_dim = self.hidden_dim

        # out = activated @ W2^T  (L, dim) = (L, hid_dim) @ (hid_dim, dim)
        # grad_activated = grad_out @ W2  (L, hid_dim) = (L, dim) @ (dim, hid_dim)
        grad_activated = matmul(grad_out, self.W2)  # (L, hid_dim)
        grad_W2 = matmul(transpose(grad_out), activated)  # (dim, hid_dim)

        # ReLU backward
        grad_hidden = zeros(L, hid_dim)
        for i in range(L):
            ghi = grad_hidden[i]
            gai = grad_activated[i]
            for j in range(hid_dim):
                ghi[j] = gai[j] if hidden[i][j] > 0 else 0.0

        # hidden = x @ W1^T  (L, hid_dim) = (L, dim) @ (dim, hid_dim)
        # grad_x = grad_hidden @ W1  (L, dim) = (L, hid_dim) @ (hid_dim, dim)
        grad_x = matmul(grad_hidden, self.W1)  # (L, dim)
        grad_W1 = matmul(transpose(grad_hidden), x)  # (hid_dim, dim)

        return {"W1": grad_W1, "W2": grad_W2, "x": grad_x}


# ============================================================================
# Transformer Layer
# ============================================================================


class TransformerLayer:
    def __init__(self, dim: int, num_heads: int):
        self.dim = dim
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: List[List[float]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """x: (L, dim). Returns (output, cache)."""
        # Pre-norm attention
        normed1 = self.norm1.forward(x)
        attn_out, attn_cache = self.attn.forward(normed1)
        x1 = add(x, attn_out)

        # Pre-norm FFN
        normed2 = self.norm2.forward(x1)
        ffn_out, ffn_cache = self.ffn.forward(normed2)
        x2 = add(x1, ffn_out)

        cache = {
            "x": x,
            "normed1": normed1,
            "attn_out": attn_out,
            "attn_cache": attn_cache,
            "x1": x1,
            "normed2": normed2,
            "ffn_out": ffn_out,
            "ffn_cache": ffn_cache,
        }
        return x2, cache

    def backward(
        self, grad_out: List[List[float]], cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        x = cache["x"]
        x1 = cache["x1"]
        normed2 = cache["normed2"]
        ffn_cache = cache["ffn_cache"]

        # ---- FFN path ----
        # x2 = x1 + ffn_out, grad_out = dL/d(x2)
        # grad flows to ffn_out directly
        ffn_grads = self.ffn.backward(grad_out, ffn_cache)
        # ffn_grads["x"] = gradient w.r.t. ffn input = normed2
        grad_normed2 = ffn_grads["x"]

        # Back through norm2: normed2 = norm2(x1)
        grad_x1_norm2, grad_norm2_w = self.norm2.backward(grad_normed2, x1, normed2)

        # grad_x1 = grad_out (residual) + grad_x1_norm2 (through norm2->ffn)
        grad_x1 = add(grad_out, grad_x1_norm2)

        # ---- Attention path ----
        # x1 = x + attn_out, grad_x1 flows to attn_out
        attn_grads = self.attn.backward(grad_x1, cache["attn_cache"])
        # attn_grads["x"] = gradient w.r.t. attn input = normed1
        grad_normed1 = attn_grads["x"]

        # Back through norm1: normed1 = norm1(x)
        grad_x_norm1, grad_norm1_w = self.norm1.backward(
            grad_normed1, x, cache["normed1"]
        )

        # grad_x = grad_x1 (residual x->x1) + grad_x_norm1 (through norm1->attn)
        grad_x = add(grad_x1, grad_x_norm1)

        return {
            "attn": attn_grads,
            "ffn": ffn_grads,
            "norm1_weight": grad_norm1_w,
            "norm2_weight": grad_norm2_w,
            "x": grad_x,
        }


# ============================================================================
# Full Transformer Model
# ============================================================================


class TinyTransformer:
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_context: int = 64,
        rope_theta: float = 10000.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_context = max_context
        self.rope_theta = rope_theta

        scale = 0.02
        # Embedding: vocab_size x dim
        self.embed = random_matrix(vocab_size, dim, scale)
        # LM head: dim x vocab_size (tied with embed or separate)
        self.lm_head = random_matrix(dim, vocab_size, scale)

        self.layers = [TransformerLayer(dim, num_heads) for _ in range(num_layers)]

    def forward(self, token_ids: List[int]) -> Tuple[List[List[float]], List[Dict]]:
        """Forward pass. token_ids length = L. Returns (logits, layer_caches)."""
        L = len(token_ids)

        # Embedding lookup
        x = [self.embed[tid][:] for tid in token_ids]  # (L, dim)

        # Optional: add learned positional embedding (simple version)
        # For simplicity, we use a small learned pos matrix
        # Actually let's skip explicit position embeddings for now -
        # the attention pattern alone can learn sequence statistics

        layer_caches = []
        for layer in self.layers:
            x, cache = layer.forward(x)
            layer_caches.append(cache)

        # LM head: logits = x @ lm_head^T
        logits = matmul(x, self.lm_head)  # (L, vocab_size)
        return logits, layer_caches

    def compute_loss(
        self, logits: List[List[float]], targets: List[int]
    ) -> Tuple[float, List[List[float]]]:
        """Cross-entropy loss. Returns (loss, grad_logits)."""
        L = len(logits)
        loss = 0.0
        grad_logits = zeros(L, self.vocab_size)

        for i in range(L):
            li = logits[i]
            mx = max(li)
            exp_sum = 0.0
            for j in range(self.vocab_size):
                li[j] = math.exp(li[j] - mx)
                exp_sum += li[j]
            inv_sum = 1.0 / exp_sum if exp_sum > 0 else 1.0

            # probs
            probs = [v * inv_sum for v in li]

            # loss
            target = targets[i]
            p = max(probs[target], 1e-15)
            loss += -math.log(p)

            # gradient
            gli = grad_logits[i]
            for j in range(self.vocab_size):
                gli[j] = probs[j] - (1.0 if j == target else 0.0)

        return loss / L, grad_logits

    def backward(
        self,
        grad_logits: List[List[float]],
        token_ids: List[int],
        layer_caches: List[Dict],
    ) -> Dict[str, Any]:
        """Full backward pass. Returns gradients for all parameters."""
        L = len(token_ids)

        # LM head grad
        # logits = x @ lm_head^T, where x is output of last layer
        # Actually we need x from the last layer's output
        last_layer_output = None
        for cache in reversed(layer_caches):
            # The output of each layer is stored in the next layer's cache as "x"
            # For the last layer, its output IS what was used for lm_head
            pass

        # We need the final hidden states. Let's get them from the forward pass again.
        # Actually, we'll compute them during backward from the gradient.

        # grad_lm_head = grad_logits^T @ x_last
        # grad_x_last = grad_logits @ lm_head

        # But x_last = layer_caches[-1]'s output... we need to store it.
        # The last cache's output IS x2 from the last forward call.
        # Let me restructure: store the final x in the return.

        # Actually, we know logits = x_final @ lm_head^T, where x_final is the
        # output of the last layer. We need x_final for the lm_head gradient.
        # Let's get x_final from the last layer cache: it's the x1 + ffn_out = x2.
        # The last layer cache stores all that info.

        # For simplicity, let me reconstruct the forward pass to get x_final,
        # or better, let me restructure to return it.

        # Since I already run forward and know the exact path, I'll get x_final from:
        last_cache = layer_caches[-1]
        x1 = last_cache["x1"]
        ffn_out = last_cache["ffn_out"]
        x_final = add(x1, ffn_out)  # This is what went into lm_head

        # logits = x_final @ lm_head  where lm_head: (dim, vocab_size)
        # grad_lm_head = x_final^T @ grad_logits  -> (dim, vocab_size)
        grad_lm_head = matmul(transpose(x_final), grad_logits)

        # grad_x_final = grad_logits @ lm_head^T  -> (L, dim)
        grad_x = matmul(grad_logits, transpose(self.lm_head))

        # Backward through layers (reversed)
        all_grads = {"lm_head": grad_lm_head, "embed": zeros(self.vocab_size, self.dim)}

        for layer_idx in reversed(range(self.num_layers)):
            layer_grads = self.layers[layer_idx].backward(
                grad_x, layer_caches[layer_idx]
            )
            all_grads[f"layer{layer_idx}"] = layer_grads
            grad_x = layer_grads["x"]

        # grad_embed: accumulate from token positions
        # grad_x now is gradient w.r.t. the embedding output
        grad_embed = all_grads["embed"]
        for i in range(L):
            tid = token_ids[i]
            for j in range(self.dim):
                grad_embed[tid][j] += grad_x[i][j]

        return all_grads


# ============================================================================
# Trainer
# ============================================================================


def create_sequences(
    tokens: List[int], seq_len: int
) -> List[Tuple[List[int], List[int]]]:
    """Create (input, target) pairs by sliding window over tokens.
    input: tokens[i:i+seq_len], target: tokens[i+1:i+seq_len+1]"""
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        inp = tokens[i : i + seq_len]
        tgt = tokens[i + 1 : i + seq_len + 1]
        if len(inp) == seq_len and len(tgt) == seq_len:
            sequences.append((inp, tgt))
    return sequences


def train_model(
    model: TinyTransformer,
    tokens: List[int],
    epochs: int = 20,
    seq_len: int = 16,
    lr: float = 0.01,
    grad_clip: float = 1.0,
) -> List[float]:
    """Train the model. Returns list of epoch losses."""
    sequences = create_sequences(tokens, seq_len)
    if not sequences:
        print("Warning: no sequences created from tokens")
        return []

    print(f"Training on {len(sequences)} sequences of length {seq_len}")
    print(
        f"Vocab: {model.vocab_size}, Dim: {model.dim}, "
        f"Layers: {model.num_layers}, Heads: {model.num_heads}"
    )

    losses = []

    for epoch in range(epochs):
        random.shuffle(sequences)
        epoch_loss = 0.0
        n_batches = 0

        for inp, tgt in sequences:
            # Forward
            logits, layer_caches = model.forward(inp)
            loss, grad_logits = model.compute_loss(logits, tgt)

            # Backward
            grads = model.backward(grad_logits, inp, layer_caches)

            # Apply gradients with SGD
            _apply_grads_sgd(model, grads, lr, grad_clip)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.6f}")

    return losses


def _apply_grads_sgd(
    model: TinyTransformer, grads: Dict, lr: float, grad_clip: float
) -> None:
    """Apply gradients with simple SGD. Updates model weights in place."""

    # LM head
    if "lm_head" in grads:
        clip_grad_norm(grads["lm_head"], grad_clip)
        W = model.lm_head
        GW = grads["lm_head"]
        for i in range(len(W)):
            for j in range(len(W[i])):
                W[i][j] -= lr * GW[i][j]

    # Embedding
    if "embed" in grads:
        clip_grad_norm(grads["embed"], grad_clip)
        W = model.embed
        GW = grads["embed"]
        for i in range(len(W)):
            for j in range(len(W[i])):
                W[i][j] -= lr * GW[i][j]

    # Layers
    for layer_idx in range(model.num_layers):
        key = f"layer{layer_idx}"
        if key not in grads:
            continue
        lg = grads[key]
        layer = model.layers[layer_idx]

        # Attention weights
        attn = layer.attn
        for wname in ["Wq", "Wk", "Wv", "Wo"]:
            if wname in lg["attn"]:
                clip_grad_norm(lg["attn"][wname], grad_clip)
                W = getattr(attn, wname)
                GW = lg["attn"][wname]
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        W[i][j] -= lr * GW[i][j]

        # FFN weights
        ffn = layer.ffn
        for wname in ["W1", "W2"]:
            if wname in lg["ffn"]:
                clip_grad_norm(lg["ffn"][wname], grad_clip)
                W = getattr(ffn, wname)
                GW = lg["ffn"][wname]
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        W[i][j] -= lr * GW[i][j]

        # Norm weights
        for wname in ["norm1_weight", "norm2_weight"]:
            if wname in lg:
                gw = lg[wname]
                w = getattr(layer, wname.replace("_weight", ""))
                # w is either norm1.weight or norm2.weight (list[float])
                nw = layer.norm1 if "norm1" in wname else layer.norm2
                target = nw.weight
                gn = math.sqrt(sum(v * v for v in gw))
                if gn > grad_clip:
                    s = grad_clip / gn
                    gw = [v * s for v in gw]
                for j in range(len(target)):
                    target[j] -= lr * gw[j]


# ============================================================================
# Ternary Quantization
# ============================================================================


def row_scale(row: List[float]) -> float:
    total = sum(abs(v) for v in row)
    avg = total / len(row) if row else 1.0
    return avg if avg > 1e-12 else 1.0


def quantize_matrix(
    matrix: List[List[float]],
) -> Tuple[List[float], List[int], List[int], float]:
    """Quantize a matrix to ternary with error compensation.
    Returns (scales, pos_bits, neg_bits, mean_abs_error).
    matrix: rows x cols.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    blocks64 = (cols + 63) // 64

    scales = []
    all_pos = []
    all_neg = []
    total_abs_error = 0.0

    for r in range(rows):
        row = matrix[r]
        scale = row_scale(row)
        threshold = 0.5 * scale
        carry = 0.0
        quantized = []

        for v in row:
            adjusted = v + carry
            if adjusted > threshold:
                q = 1
            elif adjusted < -threshold:
                q = -1
            else:
                q = 0
            carry = adjusted - (q * scale)
            quantized.append(q)

        scales.append(scale)

        # Pack into 64-bit blocks
        pos = [0] * blocks64
        neg = [0] * blocks64
        for c in range(cols):
            block = c >> 6
            bit = c & 63
            if quantized[c] > 0:
                pos[block] |= 1 << bit
            elif quantized[c] < 0:
                neg[block] |= 1 << bit

        all_pos.extend(pos)
        all_neg.extend(neg)

        for original, q in zip(row, quantized):
            total_abs_error += abs(original - (q * scale))

    mae = total_abs_error / (rows * cols) if rows * cols > 0 else 0.0
    return scales, all_pos, all_neg, mae


# ============================================================================
# .bits file writer
# ============================================================================

MAGIC = b"CIOTBIT1"


def write_bits_file(
    path: str, matrix: List[List[float]], rows: int, cols: int
) -> float:
    """Write a single .bits file. Returns mean abs quantization error."""
    scales, pos, neg, mae = quantize_matrix(matrix)
    blocks64 = (cols + 63) // 64

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", rows, cols, blocks64))
        f.write(struct.pack(f"<{len(scales)}f", *scales))
        f.write(struct.pack(f"<{len(pos)}Q", *pos))
        f.write(struct.pack(f"<{len(neg)}Q", *neg))

    return mae


def write_f32_file(path: str, data: List[float]) -> None:
    """Write raw float32 binary file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data)}f", *data))


# ============================================================================
# Model export
# ============================================================================


def export_model(model: TinyTransformer, vocab: List[str], out_dir: str) -> None:
    """Export trained model as .bits directory compatible with Ciot model loader."""
    os.makedirs(out_dir, exist_ok=True)
    dim = model.dim
    vs = model.vocab_size

    total_mae = 0.0
    file_count = 0

    # config.ciot
    config_path = os.path.join(out_dir, "config.ciot")
    with open(config_path, "w") as f:
        f.write(f"dim {dim}\n")
        f.write(f"num_layers {model.num_layers}\n")
        f.write(f"num_heads {model.num_heads}\n")
        f.write(f"vocab_size {vs}\n")
        f.write(f"max_context {model.max_context}\n")
    print(f"  wrote {config_path}")

    # vocab.txt
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print(f"  wrote {vocab_path}")

    # embed.bits: vocab_size x dim
    path = os.path.join(out_dir, "embed.bits")
    mae = write_bits_file(path, model.embed, vs, dim)
    total_mae += mae
    file_count += 1
    print(f"  wrote {path}  (mae={mae:.6f})")

    # lm_head.bits: Ciot expects (vocab_size x dim) so matvec(lm_head, x[dim]) -> logits[vocab_size]
    # model.lm_head is (dim, vocab_size); store its transpose as (vocab_size, dim)
    path = os.path.join(out_dir, "lm_head.bits")
    mae = write_bits_file(path, transpose(model.lm_head), vs, dim)
    total_mae += mae
    file_count += 1
    print(f"  wrote {path}  (mae={mae:.6f})")

    # Per-layer files
    for l_idx, layer in enumerate(model.layers):
        layer_dir = os.path.join(out_dir, f"layer{l_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        # wq.bits: dim x dim
        path = os.path.join(layer_dir, "wq.bits")
        mae = write_bits_file(path, layer.attn.Wq, dim, dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # wk.bits
        path = os.path.join(layer_dir, "wk.bits")
        mae = write_bits_file(path, layer.attn.Wk, dim, dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # wv.bits
        path = os.path.join(layer_dir, "wv.bits")
        mae = write_bits_file(path, layer.attn.Wv, dim, dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # wo.bits
        path = os.path.join(layer_dir, "wo.bits")
        mae = write_bits_file(path, layer.attn.Wo, dim, dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # w1.bits: hidden_dim x dim
        path = os.path.join(layer_dir, "w1.bits")
        mae = write_bits_file(path, layer.ffn.W1, layer.ffn.hidden_dim, dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # w2.bits: dim x hidden_dim
        path = os.path.join(layer_dir, "w2.bits")
        mae = write_bits_file(path, layer.ffn.W2, dim, layer.ffn.hidden_dim)
        total_mae += mae
        file_count += 1
        print(f"  wrote {path}  (mae={mae:.6f})")

        # norm1.f32
        path = os.path.join(layer_dir, "norm1.f32")
        write_f32_file(path, layer.norm1.weight)
        print(f"  wrote {path}")

        # norm2.f32
        path = os.path.join(layer_dir, "norm2.f32")
        write_f32_file(path, layer.norm2.weight)
        print(f"  wrote {path}")

    avg_mae = total_mae / max(file_count, 1)
    print(f"\n  Total files: {file_count} weight files + config + vocab + norm files")
    print(f"  Average quantization MAE: {avg_mae:.6f}")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a tiny transformer, quantize to ternary, export as .bits"
    )
    parser.add_argument("text", help="Input text file for training")
    parser.add_argument("out_dir", help="Output model directory")
    parser.add_argument(
        "--dim", type=int, default=64, help="Model dimension (default: 64)"
    )
    parser.add_argument(
        "--heads", type=int, default=2, help="Number of attention heads (default: 2)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of transformer layers (default: 2)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=16, help="Training sequence length (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--max-vocab", type=int, default=256, help="Max vocabulary size (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    dim = args.dim
    num_heads = args.heads
    num_layers = args.layers
    epochs = args.epochs
    seq_len = args.seq_len
    lr = args.lr
    max_vocab = args.max_vocab

    if dim % num_heads != 0:
        print(f"Error: dim ({dim}) must be divisible by num_heads ({num_heads})")
        return 1

    # Read text
    print(f"Reading text from: {args.text}")
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()
    raw_tokens = tokenize(text)
    print(f"Raw tokens: {len(raw_tokens)}")

    if len(raw_tokens) < seq_len * 2:
        print(f"Error: need at least {seq_len * 2} tokens (got {len(raw_tokens)})")
        return 1

    # Build vocabulary
    vocab, word2id = build_vocab(raw_tokens, max_vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Convert tokens to IDs
    token_ids = [word2id.get(t, word2id.get("<unk>", 0)) for t in raw_tokens]

    # Build model
    print(
        f"\nBuilding model: dim={dim}, heads={num_heads}, "
        f"layers={num_layers}, vocab={vocab_size}"
    )
    model = TinyTransformer(
        vocab_size=vocab_size, dim=dim, num_layers=num_layers, num_heads=num_heads
    )

    # Count parameters
    total_params = 0
    total_params += len(model.embed) * len(model.embed[0])
    total_params += len(model.lm_head) * len(model.lm_head[0])
    for layer in model.layers:
        total_params += len(layer.attn.Wq) * len(layer.attn.Wq[0])
        total_params += len(layer.attn.Wk) * len(layer.attn.Wk[0])
        total_params += len(layer.attn.Wv) * len(layer.attn.Wv[0])
        total_params += len(layer.attn.Wo) * len(layer.attn.Wo[0])
        total_params += len(layer.ffn.W1) * len(layer.ffn.W1[0])
        total_params += len(layer.ffn.W2) * len(layer.ffn.W2[0])
    print(f"Total parameters: {total_params}")

    # Train
    print(f"\nStarting training ({epochs} epochs)...")
    losses = train_model(model, token_ids, epochs=epochs, seq_len=seq_len, lr=lr)

    if losses:
        print(f"\nFinal loss: {losses[-1]:.6f}")
        print(f"Losses: {', '.join(f'{l:.4f}' for l in losses)}")

    # Export
    print(f"\nExporting model to: {args.out_dir}")
    export_model(model, vocab, args.out_dir)

    print(f"\nDone. Run with:")
    print(f'  ./bin/ciot --model-generate {args.out_dir} "hello world" 10')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
