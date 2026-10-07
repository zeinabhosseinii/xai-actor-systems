"""Verify everything in surrogate.py that does not need a trained network.

WHY THIS EXISTS. torch cannot be installed in the environment these files were
written in (no outbound network), so `surrogate.py --traces
traces_v5_safety.json` has never been executed. That is stated plainly rather
than papered over. But the parts most likely to be silently wrong need no
gradients at all:

  * the INDEX REMAP. `causal` and `resp` are indices into the unfiltered step
    list; the model sees a filtered list. Dropping TURNON/TURNOFF shifts every
    later index. A remap bug would score attention against the wrong positions
    and be indistinguishable from a bad model.
  * the PER-STEP TARGET. "violated by step t" is folded here from message types
    only, in a second implementation, and must agree with both the simulator's
    recorded temperature and the trace label.
  * the BASELINES AND METRICS. They are reimplemented in surrogate.py over a
    different data structure than safety.py uses, so the two must be
    cross-checked against each other. Agreement to the tenth of a percent is
    what makes the reported headroom trustworthy.

torch is replaced by the smallest stub that lets the module import: `nn.Module`
as a base class and `torch.no_grad()` as a decorator. No tensor is created, so
nothing about the model itself is verified here -- only the data path, the
baselines and the metrics.

Run:  python3 v5_verify.py
"""

import json
import os
import sys
import types


def fake_torch():
    """The minimum that makes `import surrogate` succeed without torch."""
    t = types.ModuleType("torch")

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    t.no_grad = lambda: _NoGrad()
    t.set_grad_enabled = lambda _f: _NoGrad()
    t.manual_seed = lambda _s: None
    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *_a, **_k):
            pass

    nn.Module = Module
    for name in ("Embedding", "LSTM", "Linear", "Sequential", "Tanh"):
        setattr(nn, name, lambda *a, **k: None)
    nn.utils = types.SimpleNamespace(clip_grad_norm_=lambda *a, **k: None)
    F = types.ModuleType("torch.nn.functional")
    t.nn = nn
    nn.functional = F
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = F


fake_torch()
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import surrogate as S            # noqa: E402  (must follow the stub)

FAIL = []


def check(name, got, want, tol=0.05, pct=True):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print("   %-52s %-12s %s" % (name, _fmt(got, pct),
                                 "ok" if ok else "EXPECTED %s" % _fmt(want, pct)))
    if not ok:
        FAIL.append(name)


def _fmt(v, pct=True):
    if not isinstance(v, float):
        return str(v)
    return "%.1f%%" % v if pct else "%.2f" % v


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "traces_v5_safety.json"
    print("=== 1. target detection, including the refusal on the stale file ===")
    for p, want in (("traces_v4.json", "latch"),
                    ("traces_v5_safety.json", "refused"),
                    ("traces_v5_safety.json", "safety")):
        if not os.path.exists(p):
            print("   %-24s -> absent, skipped" % p)
            continue
        try:
            got = S.detect_target(p)
        except SystemExit as e:
            got = "refused"
            note = str(e).splitlines()[0]
        else:
            note = ""
        check("%s" % p, got, want)
        if note:
            print("      (%s...)" % note[:66])

    print("\n=== 2. data path (asserts inside build_safety must all pass) ===")
    recs, vocab = S.build_safety(path)
    print("   vocabulary            : %s" % ", ".join(sorted(vocab, key=vocab.get)))
    print("   records               : %d" % len(recs))
    lens = {len(r[0]) for r in recs}
    print("   sequence lengths      : %d .. %d" % (min(lens), max(lens)))
    check("COOL is in the vocabulary", "COOL" in vocab, True)
    check("TURNON is not", "TURNON" not in vocab, True)
    check("payload channel is all zero by default",
          all(f[1] == 0.0 for r in recs for f in r[1]), True)

    # the remap: every causal index must land on a DOOROPENED in the FILTERED list
    bad = sum(1 for r in recs for i in r[3] if r[4][i] != "DOOROPENED")
    check("causal indices land on DOOROPENED after remap", bad, 0)
    badr = sum(1 for r in recs for i in r[6]["resp"] if r[4][i] != "DOOROPENED")
    check("resp indices land on DOOROPENED after remap", badr, 0)
    # the per-step target must be monotone and agree with the label
    nm = sum(1 for r in recs if any(r[2][i] > r[2][i + 1] for i in range(len(r[2]) - 1)))
    check("per-step target is monotone (never un-violates)", nm, 0)
    bl = sum(1 for r in recs if r[6]["label"] != (r[2][-1] > .5))
    check("last step equals the trace label", bl, 0)

    print("\n=== 3. cross-check the baselines against safety.py ===")
    blob = json.load(open(path))
    S.CFG = dict(blob["config"])
    viol = [r for r in recs if r[6]["label"]]
    elig = [r for r in viol if r[3]]
    n = len(recs)
    check("violation rate", 100 * len(viol) / n, 54.4)
    check("over-determined share of violations",
          100 * (len(viol) - len(elig)) / len(viol), 36.7)
    check("mean |causal| over eligible",
          sum(len(r[3]) for r in elig) / len(elig), 4.5, pct=False)

    def ranked(fn):
        t1 = f1 = ex = 0.0
        for r in elig:
            o, c = fn(r), set(r[3])
            top = set(o[:len(c)])
            t1 += o[0] in c
            f1 += 2 * len(top & c) / (len(top) + len(c))
            ex += top == c
        k = len(elig)
        return 100 * t1 / k, 100 * f1 / k, 100 * ex / k

    for nm_, fn, want in (("recency", S.rank_recency, (9.4, 9.7, 1.0)),
                          ("DOOROPENED latest first", S.rank_door_late, (64.7, 83.8, 64.7)),
                          ("DOOROPENED earliest first", S.rank_door_early, (68.8, 74.9, 68.8)),
                          ("oracle", S.rank_oracle, (100.0, 100.0, 100.0))):
        got = ranked(fn)
        for lbl, g, w in zip(("Top-1", "set-F1", "exact"), got, want):
            check("%s %s" % (nm_, lbl), g, w)

    # cardinality-free: the rule that looks at nothing
    p = f1 = ex = 0.0
    for r in elig:
        Sd, c = set(S._doors(r)), set(r[3])
        p += len(Sd & c) / len(Sd)
        f1 += 2 * len(Sd & c) / (len(Sd) + len(c))
        ex += Sd == c
    k = len(elig)
    check("'every DOOROPENED' F1  (the metric trap)", 100 * f1 / k, 83.0)
    check("'every DOOROPENED' exact (the honest floor)", 100 * ex / k, 39.9)

    # responsibility floors and ceiling
    flat = best = 0.0
    for r in viol:
        rr = r[6]["resp"]
        flat += sum(rr.values()) / len(rr)
        best += max(rr.values())
    check("blame of a random opening (floor)", 100 * flat / len(viol), 68.5)
    check("blame of the best opening (ceiling)", 100 * best / len(viol), 80.6)

    print("\n=== 4. the metrics, fed explanations whose score is known a priori ===")
    # A metric is only useful if it scores a known answer correctly. Feed the
    # attention code path three hand-built maps whose scores are already known
    # from section 3, and require it to reproduce them.

    def amap_of(fn):
        """id(record) -> attention row that puts equal mass on fn(record)."""
        out = {}
        for r in viol:
            w = [0.0] * len(r[0])
            sel = fn(r) or S._doors(r)
            for i in sel:
                w[i] = 1.0 / len(sel)
            out[id(r)] = w
        return out

    def cardinality_free(am):
        return 100 * sum(1 for r in elig
                         if S._att_set(am[id(r)]) == set(r[3])) / len(elig)

    gt = amap_of(lambda r: list(r[3]))
    check("ground truth as attention, exact-set (handed |c|)",
          ranked(lambda r: S._att_order(gt[id(r)]))[2], 100.0)
    check("ground truth as attention, cardinality-free exact",
          cardinality_free(gt), 100.0)
    # mass spread over every opening: must land on the section-3 floor exactly,
    # which is what proves the two code paths measure the same thing
    allc = amap_of(lambda r: S._doors(r))
    check("attention on every opening -> the 39.9% floor",
          cardinality_free(allc), 39.9)
    # uniform over the whole trace: no position is above uniform, by definition
    flat = {id(r): [1.0 / len(r[0])] * len(r[0]) for r in viol}
    check("uniform attention names nothing (threshold sanity)",
          sum(1 for r in elig if S._att_set(flat[id(r)])), 0)

    print("\n=== 5. the latch path is intact as a negative control ===")
    # build() and build_safety() must produce the SAME tuple shape, because
    # collate / run_epoch / final_attention / occlude are shared. If the latch
    # path drifted, the v4-vs-v5 ablation would no longer be one variable.
    if os.path.exists("traces_v4.json"):
        lrecs, lvocab = S.build("traces_v4.json")
        check("latch records built", len(lrecs) > 0, True)
        check("same tuple width as safety records", len(lrecs[0]), len(recs[0]))
        check("latch extra field is an empty dict", lrecs[0][6], {})
        check("latch vocabulary excludes COOL (it is an EFFECT there)",
              "COOL" not in lvocab, True)
    else:
        print("   traces_v4.json absent, skipped")

    print("\n=== 6. what this harness does NOT verify ===")
    print("   Nothing above creates a tensor. The model class, the attention")
    print("   decomposition assert, the training loop, faithfulness and occlude")
    print("   are all UNTESTED -- torch cannot be installed here (proxy 403).")
    print("   Run locally:")
    print("     python3 -m venv ~/venvs/xai && . ~/venvs/xai/bin/activate")
    print("     python3 -m pip install torch")
    print("     python3 surrogate.py --traces traces_v5_safety.json --read values")
    print("     python3 surrogate.py --traces traces_v5_safety.json --read hidden")
    print("     python3 surrogate.py --traces traces_v5_safety.json --keep-payload")
    print("     python3 surrogate.py --traces traces_v4.json   # negative control")

    print("\n%s" % ("ALL CHECKS PASSED" if not FAIL else
                    "FAILED: %s" % ", ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
