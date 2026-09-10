"""A small character-level decoder-only transformer written in NumPy.

Everything here is hand-written: the forward pass, the backward pass, the
Adam optimizer, greedy decoding and the save/load format. There is no
autograd. The model is deliberately tiny (2 layers, d=64 by default) so it
trains on a laptop CPU in a few minutes.

Architecture (pre-LayerNorm, GPT style, with a causal local-window input
embedding so that each position also sees the few characters before it):

    x = tok_emb[tokens] + sum_k prev_emb_k[tokens shifted by k] + pos_emb[positions]
    for each layer:
        x = x + MHA(LN(x))        # causal multi-head self-attention
        x = x + MLP(LN(x))        # two-layer MLP with GELU
    logits = LN(x) @ W_out + b_out
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Character vocabulary: printable ASCII plus newline, with three specials.
PAD = 0
EOS = 1
UNK = 2
_CHARS = "\n" + "".join(chr(c) for c in range(32, 127))
CHAR_TO_ID: Dict[str, int] = {ch: i + 3 for i, ch in enumerate(_CHARS)}
ID_TO_CHAR: Dict[int, str] = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(_CHARS) + 3


def encode(text: str) -> List[int]:
    """Map a string to token ids (unknown characters become UNK)."""
    return [CHAR_TO_ID.get(ch, UNK) for ch in text]


def decode(ids: Sequence[int]) -> str:
    """Map token ids back to a string, stopping at EOS and skipping PAD."""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i == PAD:
            continue
        out.append(ID_TO_CHAR.get(int(i), ""))
    return "".join(out)


@dataclass(frozen=True)
class Config:
    """Model hyperparameters."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_len: int = 128
    vocab_size: int = VOCAB_SIZE
    n_prev: int = 3  # how many preceding characters feed each position's embedding


def _gelu(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tanh-approximation GELU. Returns (output, derivative)."""
    c = np.float32(np.sqrt(2.0 / np.pi))
    k = np.float32(0.044715)
    x2 = x * x
    t = np.tanh(c * x * (np.float32(1.0) + k * x2))
    half_x = np.float32(0.5) * x
    y = half_x + half_x * t
    du = c * (np.float32(1.0) + np.float32(3.0) * k * x2)
    dy = np.float32(0.5) + np.float32(0.5) * t + half_x * (np.float32(1.0) - t * t) * du
    return y, dy


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd)


def _layer_norm_backward(dy: np.ndarray, cache, g: np.ndarray):
    xhat, rstd = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def _scatter_rows(ids: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    """Sum ``rows`` into an (n, d) array by ``ids`` (a fast embedding gradient)."""
    onehot = np.zeros((n, ids.shape[0]), dtype=rows.dtype)
    onehot[ids, np.arange(ids.shape[0])] = 1.0
    return onehot @ rows


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Transformer:
    """Decoder-only transformer with a hand-written backward pass.

    Parameters live in ``self.params`` (a dict of float32 arrays) so that
    saving, loading and optimizer bookkeeping are trivial.
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d, h, f = cfg.d_model, cfg.n_heads, cfg.d_ff
        assert d % h == 0, "d_model must be divisible by n_heads"
        p: Dict[str, np.ndarray] = {}

        def w(shape, scale):
            return (rng.standard_normal(shape) * scale).astype(np.float32)

        p["tok_emb"] = w((cfg.vocab_size, d), 0.02)
        for k in range(1, cfg.n_prev + 1):
            p[f"prev{k}_emb"] = w((cfg.vocab_size, d), 0.02)
        p["pos_emb"] = w((cfg.max_len, d), 0.02)
        for i in range(cfg.n_layers):
            p[f"l{i}.ln1_g"] = np.ones(d, np.float32)
            p[f"l{i}.ln1_b"] = np.zeros(d, np.float32)
            p[f"l{i}.w_qkv"] = w((d, 3 * d), 0.02)
            p[f"l{i}.b_qkv"] = np.zeros(3 * d, np.float32)
            p[f"l{i}.w_o"] = w((d, d), 0.02 / np.sqrt(2 * cfg.n_layers))
            p[f"l{i}.b_o"] = np.zeros(d, np.float32)
            p[f"l{i}.ln2_g"] = np.ones(d, np.float32)
            p[f"l{i}.ln2_b"] = np.zeros(d, np.float32)
            p[f"l{i}.w_fc"] = w((d, f), 0.02)
            p[f"l{i}.b_fc"] = np.zeros(f, np.float32)
            p[f"l{i}.w_proj"] = w((f, d), 0.02 / np.sqrt(2 * cfg.n_layers))
            p[f"l{i}.b_proj"] = np.zeros(d, np.float32)
        p["lnf_g"] = np.ones(d, np.float32)
        p["lnf_b"] = np.zeros(d, np.float32)
        p["w_out"] = w((d, cfg.vocab_size), 0.02)
        p["b_out"] = np.zeros(cfg.vocab_size, np.float32)
        self.params = p

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: np.ndarray) -> Tuple[np.ndarray, list]:
        """Compute logits for a batch of token ids of shape (B, T).

        Returns (logits of shape (B, T, V), cache for backward).
        """
        p, cfg = self.params, self.cfg
        B, T = tokens.shape
        assert T <= cfg.max_len, f"sequence length {T} exceeds max_len {cfg.max_len}"
        d, h = cfg.d_model, cfg.n_heads
        dh = d // h
        x = p["tok_emb"][tokens] + p["pos_emb"][:T][None, :, :]
        shifted = []
        for k in range(1, cfg.n_prev + 1):
            sh = np.full_like(tokens, PAD)
            sh[:, k:] = tokens[:, :-k]
            shifted.append(sh)
            x = x + p[f"prev{k}_emb"][sh]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        caches = []
        for i in range(cfg.n_layers):
            pre = f"l{i}."
            ln1, ln1c = _layer_norm(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
            qkv = ln1 @ p[pre + "w_qkv"] + p[pre + "b_qkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(dh))
            scores = np.where(mask, np.float32(-1e9), scores)
            att = _softmax(scores)
            ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            attn_out = ctx @ p[pre + "w_o"] + p[pre + "b_o"]
            x2 = x + attn_out
            ln2, ln2c = _layer_norm(x2, p[pre + "ln2_g"], p[pre + "ln2_b"])
            hid = ln2 @ p[pre + "w_fc"] + p[pre + "b_fc"]
            act, dact = _gelu(hid)
            mlp_out = act @ p[pre + "w_proj"] + p[pre + "b_proj"]
            x_new = x2 + mlp_out
            caches.append((ln1, ln1c, q, k, v, att, ctx, x2, ln2, ln2c, act, dact))
            x = x_new
        lnf, lnfc = _layer_norm(x, p["lnf_g"], p["lnf_b"])
        logits = lnf @ p["w_out"] + p["b_out"]
        return logits, [tokens, shifted, caches, lnf, lnfc]

    # ----------------------------------------------------------------- backward
    def loss_and_grads(
        self, tokens: np.ndarray, targets: np.ndarray, loss_mask: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Masked cross-entropy loss and gradients for every parameter.

        ``tokens``, ``targets`` and ``loss_mask`` all have shape (B, T). The
        loss is the ``loss_mask``-weighted average of per-position
        cross-entropies (a weight of 0 excludes a position entirely).
        """
        p, cfg = self.params, self.cfg
        logits, (tokens, shifted, caches, lnf, lnfc) = self.forward(tokens)
        B, T, V = logits.shape
        d, h = cfg.d_model, cfg.n_heads
        dh = d // h
        probs = _softmax(logits)
        n_tok = max(float(loss_mask.sum()), 1.0)
        logp = np.log(np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0] + 1e-12)
        loss = float(-(logp * loss_mask).sum() / n_tok)

        grads: Dict[str, np.ndarray] = {}
        dlogits = probs
        np.put_along_axis(dlogits, targets[..., None], np.take_along_axis(dlogits, targets[..., None], -1) - 1.0, -1)
        dlogits = dlogits * (loss_mask[..., None] / n_tok).astype(np.float32)

        grads["w_out"] = lnf.reshape(-1, d).T @ dlogits.reshape(-1, V)
        grads["b_out"] = dlogits.reshape(-1, V).sum(0)
        dlnf = dlogits @ p["w_out"].T
        dx, grads["lnf_g"], grads["lnf_b"] = _layer_norm_backward(dlnf, lnfc, p["lnf_g"])

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        for i in reversed(range(cfg.n_layers)):
            pre = f"l{i}."
            ln1, ln1c, q, k, v, att, ctx, x2, ln2, ln2c, act, dact = caches[i]
            # MLP block
            dmlp = dx
            grads[pre + "w_proj"] = act.reshape(-1, cfg.d_ff).T @ dmlp.reshape(-1, d)
            grads[pre + "b_proj"] = dmlp.reshape(-1, d).sum(0)
            dact_out = dmlp @ p[pre + "w_proj"].T
            dhid = dact_out * dact
            grads[pre + "w_fc"] = ln2.reshape(-1, d).T @ dhid.reshape(-1, cfg.d_ff)
            grads[pre + "b_fc"] = dhid.reshape(-1, cfg.d_ff).sum(0)
            dln2 = dhid @ p[pre + "w_fc"].T
            dx2, grads[pre + "ln2_g"], grads[pre + "ln2_b"] = _layer_norm_backward(dln2, ln2c, p[pre + "ln2_g"])
            dx2 = dx2 + dx
            # Attention block
            dattn = dx2
            grads[pre + "w_o"] = ctx.reshape(-1, d).T @ dattn.reshape(-1, d)
            grads[pre + "b_o"] = dattn.reshape(-1, d).sum(0)
            dctx = (dattn @ p[pre + "w_o"].T).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
            datt = dctx @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ dctx
            dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
            dscores = np.where(mask, 0.0, dscores) / np.float32(np.sqrt(dh))
            dq = dscores @ k
            dk = dscores.transpose(0, 1, 3, 2) @ q
            dqkv = np.concatenate(
                [
                    dq.transpose(0, 2, 1, 3).reshape(B, T, d),
                    dk.transpose(0, 2, 1, 3).reshape(B, T, d),
                    dv.transpose(0, 2, 1, 3).reshape(B, T, d),
                ],
                axis=-1,
            )
            grads[pre + "w_qkv"] = ln1.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
            grads[pre + "b_qkv"] = dqkv.reshape(-1, 3 * d).sum(0)
            dln1 = dqkv @ p[pre + "w_qkv"].T
            dx1, grads[pre + "ln1_g"], grads[pre + "ln1_b"] = _layer_norm_backward(dln1, ln1c, p[pre + "ln1_g"])
            dx = dx1 + dx2

        grads["pos_emb"] = np.zeros_like(p["pos_emb"])
        grads["pos_emb"][:T] = dx.sum(0)
        flat_dx = dx.reshape(-1, d)
        grads["tok_emb"] = _scatter_rows(tokens.reshape(-1), flat_dx, cfg.vocab_size)
        for k, sh in enumerate(shifted, start=1):
            grads[f"prev{k}_emb"] = _scatter_rows(sh.reshape(-1), flat_dx, cfg.vocab_size)
        return loss, {k_: g.astype(np.float32) for k_, g in grads.items()}

    # ---------------------------------------------------------------- inference
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 6) -> List[str]:
        """Greedy decoding. Prompts are bucketed by length so no padding is needed."""
        outputs = [""] * len(prompts)
        by_len: Dict[int, List[int]] = {}
        for idx, s in enumerate(prompts):
            by_len.setdefault(len(s), []).append(idx)
        for length, idxs in by_len.items():
            if length + max_new_tokens > self.cfg.max_len:
                raise ValueError(f"prompt of length {length} too long for max_len {self.cfg.max_len}")
            toks = np.array([encode(prompts[i]) for i in idxs], dtype=np.int64)
            gen = np.zeros((len(idxs), 0), dtype=np.int64)
            done = np.zeros(len(idxs), dtype=bool)
            for _ in range(max_new_tokens):
                logits, _cache = self.forward(np.concatenate([toks, gen], axis=1))
                nxt = logits[:, -1, :].argmax(-1)
                nxt = np.where(done, EOS, nxt)
                gen = np.concatenate([gen, nxt[:, None]], axis=1)
                done |= nxt == EOS
                if done.all():
                    break
            for row, i in enumerate(idxs):
                outputs[i] = decode(gen[row])
        return outputs

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Save weights plus config to an .npz file."""
        meta = np.array([f"{k}={v}" for k, v in asdict(self.cfg).items()])
        np.savez(path, __config__=meta, **self.params)

    @classmethod
    def load(cls, path: str) -> "Transformer":
        """Load a model saved with :meth:`save`."""
        data = np.load(path)
        kv = dict(item.split("=", 1) for item in data["__config__"].tolist())
        cfg = Config(**{k: int(v) for k, v in kv.items()})
        model = cls(cfg, seed=0)
        for k in model.params:
            model.params[k] = data[k].astype(np.float32)
        return model


class Adam:
    """Plain Adam with bias correction and optional global-norm gradient clipping."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3, betas=(0.9, 0.98), eps: float = 1e-8, clip: float = 1.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, betas[0], betas[1], eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float | None = None) -> float:
        """Apply one update in place. Returns the pre-clip gradient norm."""
        lr = self.lr if lr is None else lr
        self.t += 1
        norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        for k, g in grads.items():
            g = g * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= (lr * mhat / (np.sqrt(vhat) + self.eps)).astype(np.float32)
        return norm
