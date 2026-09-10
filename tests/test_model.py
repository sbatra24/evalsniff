import numpy as np

from model import EOS, Adam, Config, Transformer, decode, encode


def _tiny():
    return Config(d_model=16, n_heads=2, n_layers=2, d_ff=32, max_len=24)


def test_encode_decode_roundtrip():
    text = "Question: What is 47 + 8?\nAnswer:"
    assert decode(encode(text)) == text
    assert decode(encode("ab") + [EOS] + encode("cd")) == "ab"


def test_gradients_match_finite_differences():
    cfg = _tiny()
    model = Transformer(cfg, seed=1)
    rng = np.random.default_rng(0)
    for k in model.params:  # float64 and larger weights so the check is meaningful
        model.params[k] = model.params[k].astype(np.float64) * 5 + 0.01 * rng.standard_normal(model.params[k].shape)
    B, T = 2, 7
    tok = rng.integers(3, cfg.vocab_size, (B, T))
    tgt = rng.integers(3, cfg.vocab_size, (B, T))
    mask = (rng.random((B, T)) > 0.3).astype(np.float64)
    _, grads = model.loss_and_grads(tok, tgt, mask)
    eps = 1e-5
    checked = 0
    for name, P in model.params.items():
        for _ in range(3):
            idx = tuple(int(rng.integers(0, s)) for s in P.shape)
            if name.endswith("_emb") and name != "pos_emb":
                idx = (int(tok[0, int(rng.integers(0, T))]), idx[1])
            old = P[idx]
            P[idx] = old + eps
            lp, _ = model.loss_and_grads(tok, tgt, mask)
            P[idx] = old - eps
            lm, _ = model.loss_and_grads(tok, tgt, mask)
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[name][idx]) < 1e-4 * (1 + abs(num)), (name, idx, num, grads[name][idx])
            checked += 1
    assert checked > 50


def test_training_step_reduces_loss_and_model_can_memorise():
    cfg = _tiny()
    model = Transformer(cfg, seed=0)
    opt = Adam(model.params, lr=5e-3)
    seqs = ["ab>x", "cd>y", "ef>z"]
    ids = [encode(s) + [EOS] for s in seqs]
    x = np.array([i[:-1] for i in ids])
    y = np.array([i[1:] for i in ids])
    m = np.array([[0, 0, 1, 1]] * 3, dtype=np.float32)
    first, _ = model.loss_and_grads(x, y, m)
    for _ in range(150):
        loss, grads = model.loss_and_grads(x, y, m)
        opt.step(model.params, grads)
    assert loss < first * 0.1
    assert model.generate(["ab>", "cd>", "ef>"], max_new_tokens=3) == ["x", "y", "z"]


def test_save_and_load_roundtrip(tmp_path):
    cfg = _tiny()
    model = Transformer(cfg, seed=3)
    path = tmp_path / "m.npz"
    model.save(str(path))
    loaded = Transformer.load(str(path))
    assert loaded.cfg == cfg
    assert set(loaded.params) == set(model.params)
    for k in model.params:
        assert np.array_equal(loaded.params[k], model.params[k])
    prompts = ["hello", "hi there"]
    assert loaded.generate(prompts, 4) == model.generate(prompts, 4)


def test_generate_is_batch_invariant():
    model = Transformer(_tiny(), seed=5)
    prompts = ["ab", "abc", "xyz", "q"]
    together = model.generate(prompts, 3)
    separate = [model.generate([p], 3)[0] for p in prompts]
    assert together == separate
