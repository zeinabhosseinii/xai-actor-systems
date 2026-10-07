#!/usr/bin/env python3
"""
surrogate.py -- stage 2 of the v4 pipeline: the surrogate and its explanation.

Reads traces_v4.json (from `pipeline.py --sample N`) and answers the question
the nineteen earlier experiments could not: does attention over a surrogate
recover the messages that actually caused compressorOn to change?

FOUR DEFECTS OF THE OLD SETUP, AND WHAT REPLACES EACH
  1. NO HELD-OUT SET. run_masked_turnon_experiment.py used
     `indices = list(range(len(data)))` for both fitting and scoring, so
     exp #17's 100% was training accuracy. Here: a seeded 70/15/15 split BY
     TRACE, and every number printed is on the test split.

  2. THE OBJECTIVE GAVE ATTENTION NO GRADIENT. `bce(comp_logit, comp_y)` on the
     FINAL compressorOn is satisfied by ANY attention distribution, because the
     last hidden state already contains the answer. Here the target is PER STEP
     -- compressorOn after every message -- and the prediction at step t may
     only read positions <= t. Attention now has a job at every step, and the
     decisive position is usually not the most recent one.

  3. POSITION AND LENGTH WERE LEAKING. `pos = i/(T-1)` went straight into the
     input, and in v3 temperature was monotone in position, so
     rank(temperature) == rank(position). Here there is no positional feature
     and no temperature feature; and because the effect messages are dropped
     (see 4) EVERY sequence is exactly 20 long, so length cannot leak either.

  4. THE EFFECT MESSAGES WERE IN THE INPUT. TURNON/TURNOFF are *consequences*
     of the flip and COOL is a consequence of TURNON. A model that sees them
     learns `compressorOn == (#TURNON > #TURNOFF)`, which is nearly exact and
     bypasses the thermostat logic entirely -- a shortcut, not a surrogate.
     Dropped by default; --keep-effects reproduces the leak so you can measure
     how large it was rather than assert it.

WHY THIS IS NOW A REAL LEARNING PROBLEM
    pipeline.py's oracle check proves compressorOn is a function of the
    REPORTTEMP payload subsequence alone: a two-state latch,
        false --[payload > 7]--> true --[payload <= 3]--> false.
    Three consequences, and all three are measured below.
      * TICK/TRYOPEN/DOOROPENED carry zero information, so a faithful
        explanation must give them ~no mass. Reported as "distractor mass".
      * Position is irrelevant, so the recency baseline should now collapse.
      * Payload-alone is INSUFFICIENT: the same payload means different things
        depending on the latch, so any per-position scorer is capped below
        100%. That cap is what v3 did not have, and it is the room the
        surrogate has to earn its keep in.

THE ONE ARCHITECTURAL CHOICE THAT DECIDES EVERYTHING  (--read)
    values  (default) the context vector is a weighted sum of PER-POSITION
            vectors carrying no recurrent state. The query still comes from the
            LSTM, so attention may condition on the latch, but the answer has to
            be FETCHED from a position. Attending to the wrong place costs
            accuracy, so attention becomes load-bearing.
    hidden  the context vector is a weighted sum of LSTM hidden states, as in
            your original model. h_t alone already answers the question, so
            putting all the mass on t is free -- this is the recency degeneracy,
            kept as a control. Expect comparable accuracy and much worse
            explanations. That contrast is the paper's result, stated as an
            architectural claim rather than as "attention is unfaithful".

WHAT TO READ IN THE OUTPUT
    Two numbers decide whether stage 2 worked.
      * test per-step accuracy must clearly beat the "always off" majority
        rate, otherwise the surrogate has not learned the latch and its
        explanations are meaningless regardless of how they score.
      * agreement with the causal set must beat BOTH the recency baseline and
        the best per-position payload rule. Beating ground truth alone proves
        nothing -- rule E already does that for free.
    Comprehensiveness / sufficiency are reported separately because they measure
    a different thing: whether the explanation is faithful to THIS MODEL, not
    whether it matches the semantics. A high ground-truth score with near-zero
    comprehensiveness means attention is describing something the model does not
    actually use.

USAGE
    # v4, the latch target -- kept only as the negative control now
    python3 pipeline.py --sample 4000 --out traces_v4.json
    python3 surrogate.py --traces traces_v4.json

    # v5, the safety target -- the one with headroom
    python3 safety.py --lazy --preset --sample 3000 --out traces_v5_safety.json
    python3 surrogate.py --traces traces_v5_safety.json                 # values
    python3 surrogate.py --traces traces_v5_safety.json --read hidden   # control
    python3 surrogate.py --traces traces_v5_safety.json --keep-payload  # the leak

    The target is auto-detected from the file. Requires torch; everything else is
    stdlib. CPU is fine: sequences are 20-40 long and 3000 traces train in
    well under a minute.

WHY THE LATCH TARGET IS ONLY A CONTROL NOW
    Measured after this file was written, before it was ever run: the final
    compressorOn equals the direction of the LAST threshold crossing, 100.0% in
    every configuration tried, and the whole flip set equals the
    direction-alternation points of the crossing subsequence (rule G), also
    100.0% exactly. Both are stateless, zero-parameter and need a two-token
    window. So on the latch target a surrogate reaches ceiling from one late
    token and attending anywhere earlier reduces the loss by exactly zero -- the
    recency behaviour was the OPTIMAL POLICY for the task as posed, not a defect
    of attention. Run it to reproduce that, not to beat it. The safety target
    (--traces traces_v5_safety.json) is where the numbers can move.
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

HI, LO = 7, 3
CAUSE = ["TICK", "TRYOPEN", "DOOROPENED", "REPORTTEMP"]
EFFECT = ["TURNON", "TURNOFF", "COOL"]

# ------------------------------------------------------------------- v5 target
# For the safety property `temperature > SAFE_CAP` the cause/effect split is
# DIFFERENT, and getting it wrong makes the task either unsolvable or trivial.
#
#   * COOL IS A CAUSE HERE, not an effect. Temperature is the fold of exactly
#     three messages -- TICK (+1, capped and budgeted), DOOROPENED (+3, uncapped)
#     and COOL (-10, floored). Drop COOL and the model cannot know the
#     temperature at all, so the target becomes unobservable rather than hard.
#     It was an effect on the latch target because there it was a CONSEQUENCE of
#     the thing being predicted; here it is one of the three inputs to it.
#   * TURNON/TURNOFF stay out. They are Controller-to-Compressor plumbing, they
#     never touch the temperature, and TURNON is redundant with the COOL it
#     sends. --keep-effects puts them back so the size of that leak is measured
#     rather than asserted.
#   * THE PAYLOAD IS OFF BY DEFAULT, which is the opposite of the latch setup.
#     `max REPORTTEMP payload > SAFE_CAP` predicts the label 76.3% of the time
#     against a 54.4% majority, because a report is a direct reading of the
#     variable the property is about. It is also REDUNDANT: the message TYPE
#     sequence alone determines the temperature exactly (safety.check_replay,
#     3000/3000). So the payload buys the model nothing it cannot compute and
#     hands it most of the label. --keep-payload reproduces the leak.
SAFETY_CAUSE = ["TICK", "TRYOPEN", "DOOROPENED", "COOL", "REPORTTEMP"]
SAFETY_EFFECT = ["TURNON", "TURNOFF"]
CFG = {}


def _peak_prefix(steps, cfg):
    """-> per-step running peak temperature, folded over the message types only.

    Mirrors safety.temp_path, including the two traps: the tick budget (a
    trace holds max_ticks + 1 TICKs and the last is a no-op) and the tick guard
    (a TICK at tick_cap changes nothing). Asserted against the recorded
    `temperature` field in build_safety, so a drift here cannot go unnoticed.
    """
    t = pk = cfg["init_temp"]
    used = 0
    out = []
    for s in steps:
        m = s["message"]
        if m == "TICK":
            if used < cfg["max_ticks"]:
                used += 1
                if t < cfg["tick_cap"]:
                    t += 1
        elif m == "DOOROPENED":
            t += cfg["door_delta"]
            if cfg.get("door_cap") is not None:
                t = min(cfg["door_cap"], t)
        elif m == "COOL":
            t = max(cfg["cool_floor"], t - cfg["cool_delta"])
        pk = max(pk, t)
        out.append((t, pk))
    return out


def build_safety(path, keep_effects=False, use_payload=False):
    """Records for the safety target, in the same tuple shape as build().

    target[i] is "the property has ALREADY been violated by message i" -- a
    monotone step function that turns on at the first breach. Per-step
    supervision is what gives the causal attention a job at every position, and
    the prediction at step i may still only read positions <= i, so the final
    step's prediction IS the trace label.

    THE INDEX REMAP IS THE DANGEROUS PART. `causal` and `resp` in the JSON are
    indices into the UNFILTERED step list. Dropping TURNON/TURNOFF shifts every
    later index, so they are remapped through `keep` and the result is asserted
    to still land on DOOROPENED messages. An off-by-one here would silently
    score attention against the wrong positions and look like a bad model.
    """
    global CFG
    blob = json.load(open(path))
    if not isinstance(blob, dict) or "config" not in blob:
        raise SystemExit("%s has no embedded config; regenerate with safety.py" % path)
    CFG = dict(blob["config"])
    if "safe_cap" not in CFG:
        raise SystemExit(
            "%s predates the safe_cap fix -- its 'target' field is wrong.\n"
            "Regenerate: python3 safety.py --lazy --preset --sample 3000 --out %s"
            % (path, path))
    cap = CFG["safe_cap"]
    print("target from %s: %s  (safe_cap=%d, tick_cap=%d)"
          % (path, CFG.get("target"), cap, CFG["tick_cap"]))
    tokens = SAFETY_CAUSE + (SAFETY_EFFECT if keep_effects else [])
    vocab = {t: i + 1 for i, t in enumerate(tokens)}
    out, bad = [], 0
    for rec in blob["traces"]:
        steps = rec["steps"]
        path_t = _peak_prefix(steps, CFG)
        if any(a != s["temperature"] for (a, _p), s in zip(path_t, steps)):
            bad += 1
            continue
        keep = [i for i, s in enumerate(steps) if s["message"] in vocab]
        if not keep:
            continue
        pos = {old: new for new, old in enumerate(keep)}
        ids = [vocab[steps[i]["message"]] for i in keep]
        feats = []
        for i in keep:
            p = steps[i]["payload"]
            feats.append([0.0, 0.0] if p is None else
                         [1.0, (p - cap) / 5.0 if use_payload else 0.0])
        tgt = [1.0 if path_t[i][1] > cap else 0.0 for i in keep]
        caus = [pos[i] for i in rec["causal"] if i in pos]
        assert all(steps[i]["message"] == "DOOROPENED" for i in rec["causal"]), \
            "a causal index does not point at a DOOROPENED -- ground truth is misaligned"
        assert len(caus) == len(rec["causal"]), "a causal position was filtered away"
        resp = {pos[i]: v for i, v in rec.get("resp", []) if i in pos}
        extra = {"resp": resp, "label": bool(rec["label"]),
                 "ncool": sum(1 for s in steps if s["message"] == "COOL"),
                 "maxpay": max([s["payload"] for s in steps
                                if s["message"] == "REPORTTEMP"] or [0]),
                 "min_repair": [pos[i] for i in rec.get("min_repair", []) if i in pos]}
        assert bool(rec["label"]) == (tgt[-1] > 0.5), \
            "the per-step target disagrees with the trace label at the last step"
        out.append((ids, feats, tgt, caus, [steps[i]["message"] for i in keep],
                    [steps[i]["payload"] for i in keep], extra))
    if bad:
        raise SystemExit("temperature replay failed on %d traces -- the fold in "
                         "_peak_prefix disagrees with the simulator" % bad)
    print("temperature replay: %d/%d traces reproduced exactly"
          % (len(out), len(blob["traces"])))
    return out, vocab


# --------------------------------------------------------------------- data
def build(path, keep_effects=False, use_payload=True):
    """-> (records, vocab). A record is
       (ids, feats, target, causal, messages, payloads).

    target[i] is compressorOn AFTER message i, so the model is a sequence
    labeller and every position is supervised. causal is the list of positions
    where target changed -- the multi-label ground truth. Dropping the effect
    messages loses nothing, because compressorOn only ever changes at a
    REPORTTEMP, which is retained.

    Payload is centred on the UP threshold and scaled by the gap between the two
    thresholds, so `payload > 7` and `payload <= 3` are both simple sign tests
    on one channel. This is deliberately generous: the point of v4 is that
    knowing the payload is still not enough without the latch. `payloads` keeps
    the raw values so the baselines stay comparable under --no-payload.
    """
    global HI, LO
    blob = json.load(open(path))
    if isinstance(blob, dict):          # pipeline embeds the config it used
        HI, LO = blob["config"]["hi"], blob["config"]["lo"]
        raw = blob["traces"]
        print("thresholds from %s: HI=%d LO=%d" % (path, HI, LO))
    else:                               # older flat list, assume shipped 7/3
        raw = blob
    tokens = CAUSE + (EFFECT if keep_effects else [])
    vocab = {t: i + 1 for i, t in enumerate(tokens)}
    out = []
    for rec in raw:
        steps = [s for s in rec["steps"] if s["message"] in vocab]
        if not steps:
            continue
        ids = [vocab[s["message"]] for s in steps]
        feats = []
        for s in steps:
            if s["payload"] is None:
                feats.append([0.0, 0.0])
            else:
                feats.append([1.0, (s["payload"] - HI) / (HI - LO)
                              if use_payload else 0.0])
        tgt = [1.0 if s["compressorOn"] else 0.0 for s in steps]
        caus, prev = [], 0.0
        for i, v in enumerate(tgt):
            if v != prev:
                caus.append(i)
            prev = v
        out.append((ids, feats, tgt, caus, [s["message"] for s in steps],
                    [s["payload"] for s in steps], {}))
    return out, vocab


def collate(batch, device):
    T = max(len(b[0]) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    ft = torch.zeros(B, T, 2)
    tg = torch.zeros(B, T)
    mk = torch.zeros(B, T)
    for i, b in enumerate(batch):
        n = len(b[0])
        ids[i, :n] = torch.tensor(b[0], dtype=torch.long)
        ft[i, :n] = torch.tensor(b[1], dtype=torch.float)
        tg[i, :n] = torch.tensor(b[2], dtype=torch.float)
        mk[i, :n] = 1.0
    return ids.to(device), ft.to(device), tg.to(device), mk.to(device)


# -------------------------------------------------------------------- model
class Surrogate(nn.Module):
    """LSTM + additive attention, but CAUSAL and supervised at every step.

    a[b, t, u] is how much the prediction at step t leans on position u <= t, so
    the explanation of the final compressorOn is a[b, T-1, :] -- a well-defined
    object rather than a by-product of pooling.

    WHY THE READOUT IS EXACTLY DECOMPOSABLE
        logit_t = w . sum_u a[t,u] src_u + b = sum_u a[t,u] (w . src_u) + b,
        so contrib[t,u] = a[t,u] * (w . src_u) is not a saliency heuristic -- it
        is the actual additive share position u contributes to the logit, and the
        shares sum to the logit exactly. That identity is asserted at runtime.
        This is why signed contribution is reported alongside raw attention:
        attention says where the model looked, contribution says which way that
        look pushed the answer, and only the second can distinguish an up-flip
        from a down-flip.

    WHY val() IS NONLINEAR
        With a single linear layer, w . val(x_u) would be affine in the payload,
        so it could not express "positive above 7, negative at or below 3, and
        zero in between" -- payload 5 would be scored as a weak down-flip. The
        latch needs both thresholds, so val() gets one hidden layer. The
        decomposition above survives, because the nonlinearity is inside the
        per-position term.
    """

    def __init__(self, vocab_size, read="values", embed_dim=32, hidden_dim=64,
                 n_feat=2):
        super().__init__()
        self.read = read
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + n_feat, hidden_dim, batch_first=True)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.val = nn.Sequential(nn.Linear(embed_dim + n_feat, hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, ids, ft, mk, kmask=None):
        x = torch.cat([self.emb(ids), ft], dim=-1)               # B,T,E+F
        h, _ = self.lstm(x)                                      # B,T,H
        # additive attention: e[b,t,u] = w . tanh(Wk h_u + Wq h_t)
        e = self.score(torch.tanh(self.key(h).unsqueeze(1)
                                  + self.query(h).unsqueeze(2))).squeeze(-1)
        T = ids.size(1)
        causal = torch.tril(torch.ones(T, T, device=ids.device)).unsqueeze(0)
        km = mk if kmask is None else kmask
        valid = causal * km.unsqueeze(1) * mk.unsqueeze(2)
        # A fully masked query row becomes a constant -1e9 vector, whose softmax
        # is uniform rather than NaN (-1e9 is finite, unlike -inf). Such rows are
        # padding, or an occlusion that removed everything, and are excluded from
        # every loss and metric -- so uniform is harmless and NaN-free.
        a = torch.softmax(e.masked_fill(valid <= 0, -1e9), dim=-1)
        src = h if self.read == "hidden" else self.val(x)         # B,T,H
        logit = self.head(torch.bmm(a, src)).squeeze(-1)          # B,T
        contrib = a * (src @ self.head.weight.view(-1)).unsqueeze(1)
        return logit, a, contrib


# ------------------------------------------------------------------ training
def batched(recs, bs, shuffle, rng, device):
    idx = list(range(len(recs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([recs[j] for j in idx[i:i + bs]], device)


def run_epoch(model, recs, bs, device, rng, opt=None):
    model.train(opt is not None)
    tot = seen = hit = 0.0
    with torch.set_grad_enabled(opt is not None):
        for ids, ft, tg, mk in batched(recs, bs, opt is not None, rng, device):
            logit, _a, _c = model(ids, ft, mk)
            loss = F.binary_cross_entropy_with_logits(logit, tg, reduction="none")
            loss = (loss * mk).sum() / mk.sum()
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            w = float(mk.sum())
            tot += float(loss.detach()) * w
            seen += w
            hit += float((((logit > 0).float() == tg).float() * mk).sum())
    return tot / max(seen, 1), hit / max(seen, 1)


# ----------------------------------------------------------------- baselines
def rank_recency(rec):
    return list(range(len(rec[0]) - 1, -1, -1))


def rank_payload(rec):
    """Best possible PER-POSITION scorer: test this payload against both
    thresholds and ignore all state. Capped below 100% by construction in v4,
    because whether payload 8 flips anything depends on the latch. Reads the raw
    payloads, so --no-payload does not silently weaken it."""
    msgs, pay = rec[4], rec[5]
    rt = [i for i, m in enumerate(msgs) if m == "REPORTTEMP"]
    cross = [i for i in rt if pay[i] is not None
             and (pay[i] > HI or pay[i] <= LO)]
    seen = set(cross)
    return (cross + [i for i in rt if i not in seen]
            + [i for i in range(len(msgs)) if i not in seen and i not in set(rt)])


def rank_latch(rec):
    """Oracle: simulate the two-state latch and return exactly the flip
    positions. Must score 100% -- it is a sanity check on the ground truth,
    not a competitor."""
    tgt, caus = rec[2], rec[3]
    return caus + [i for i in range(len(tgt)) if i not in set(caus)]


# ------------------------------------------------- baselines, safety target
def _doors(rec):
    return [i for i, m in enumerate(rec[4]) if m == "DOOROPENED"]


def rank_door_late(rec):
    d = _doors(rec)
    return d[::-1] + [i for i in range(len(rec[4]) - 1, -1, -1) if i not in set(d)]


def rank_door_early(rec):
    d = _doors(rec)
    return d + [i for i in range(len(rec[4])) if i not in set(d)]


def rank_oracle(rec):
    c = rec[3]
    return list(c) + [i for i in range(len(rec[4])) if i not in set(c)]


def prediction_floors(recs, tag=""):
    """What the LABEL is worth to a stateless reader of the same input.

    NEVER print surrogate accuracy without this. On the latch target the
    equivalent line read 100.0%, and that single number is what nineteen
    experiments were unknowingly competing with. Here the floor is 76.3% and it
    comes from the payload channel, which is why the channel is off by default.
    """
    n = len(recs)
    if not n:
        return 0.0
    y = [r[6]["label"] for r in recs]
    pos = sum(y)

    def acc(p):
        return 100 * sum(a == b for a, b in zip(p, y)) / n

    cap = CFG["safe_cap"]
    mp = [r[6]["maxpay"] for r in recs]
    nc = [r[6]["ncool"] for r in recs]
    bt = max(range(0, 40), key=lambda th: acc([m > th for m in mp]))
    bc = max(range(0, 8), key=lambda th: acc([c <= th for c in nc]))
    print("=== stateless floors for the LABEL on the %s split ===" % tag)
    print("   majority class                  %.1f%%" % (100 * max(pos, n - pos) / n))
    print("   #COOL <= %-2d                      %.1f%%" % (bc, acc([c <= bc for c in nc])))
    print("   max REPORTTEMP payload > %-2d      %.1f%%   (payload channel is %s)"
          % (bt, acc([m > bt for m in mp]),
             "IN THE INPUT -- leaking" if any(f[1] for r in recs for f in r[1])
             else "off"))
    print("   -> the surrogate has to beat the LAST of these, not the first.")
    unseen = 100 * sum(1 for r, v in zip(recs, y) if v and r[6]["maxpay"] <= cap) \
        / max(pos, 1)
    print("      safe_cap=%d, so a report above %d is a literal sighting of the"
          % (cap, cap))
    print("      breach; it is invisible in every report on %.1f%% of violations,"
          % unseen)
    print("      because reports are stale snapshots taken only at ticks.")
    print()
    return acc([m > bt for m in mp])


# ---------------------------------------------------------------- evaluation
def agree(order, caus):
    """-> (top1, set-F1, rank of the best-ranked true causal position)."""
    top = set(order[:len(caus)])
    f1 = 2 * len(top & set(caus)) / (len(top) + len(caus))
    pos = {p: i for i, p in enumerate(order)}
    return float(order[0] in caus), f1, 1.0 + min(pos[c] for c in caus)


@torch.no_grad()
def final_attention(model, recs, device, bs=256):
    """For every record, the explanation of the FINAL compressorOn:
    (attention row, signed contribution row, p(compressorOn), logit)."""
    model.eval()
    out = []
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        ids, ft, _tg, mk = collate(chunk, device)
        logit, a, con = model(ids, ft, mk)
        for j, rec in enumerate(chunk):
            n = len(rec[0])
            out.append((a[j, n - 1, :n].cpu().tolist(),
                        con[j, n - 1, :n].cpu().tolist(),
                        float(torch.sigmoid(logit[j, n - 1])),
                        float(logit[j, n - 1])))
    return out


@torch.no_grad()
def occlude(model, recs, sets, device, invert=False, bs=256):
    """Re-run with the given positions removed (invert: keep ONLY them).

    Removal means the token becomes PAD, its features go to zero, and it is
    masked out as an attention key -- the evidence is gone, but the sequence
    geometry is untouched, so nothing shifts position.
    """
    model.eval()
    ps = []
    for i in range(0, len(recs), bs):
        chunk, S = recs[i:i + bs], sets[i:i + bs]
        ids, ft, _tg, mk = collate(chunk, device)
        km = mk.clone()
        for j in range(len(chunk)):
            for u in range(len(chunk[j][0])):
                if (u not in S[j]) if invert else (u in S[j]):
                    ids[j, u] = 0
                    ft[j, u] = 0.0
                    km[j, u] = 0.0
        logit, _a, _c = model(ids, ft, mk, kmask=km)
        for j, rec in enumerate(chunk):
            ps.append(float(torch.sigmoid(logit[j, len(rec[0]) - 1])))
    return ps


def report(model, test, device, rng):
    """Everything that decides whether stage 2 worked."""
    elig = [r for r in test if r[3]]
    if not elig:
        print("!! no test trace ever flipped compressorOn -- nothing to explain.")
        return [], []
    att = final_attention(model, elig, device)
    n = len(elig)
    bias = float(model.head.bias.detach().view(-1)[0])
    err = max(abs(sum(c) + bias - lg) for _w, c, _p, lg in att)
    print("=== explanation of the FINAL compressorOn, test split only ===")
    print("test traces with >=1 flip : %d/%d" % (n, len(test)))
    print("flips per trace           : %s" % sorted({len(r[3]) for r in elig}))
    print("decomposition residual    : %.2e  (sum of contributions + bias vs logit)"
          % err)
    print("   -> a residual at float32 noise level means the per-position shares")
    print("      below are exact, not an approximation of the model.")
    print()

    def by_contrib(c, p):
        sign = 1.0 if p >= 0.5 else -1.0
        return sorted(range(len(c)), key=lambda u: -sign * c[u])

    rankings = [
        ("attention magnitude a[T-1]",
         [sorted(range(len(w)), key=lambda u: -w[u]) for w, _c, _p, _l in att]),
        ("signed contribution a[T-1] * (w . src)",
         [by_contrib(c, p) for _w, c, p, _l in att]),
        ("recency (latest position first)", [rank_recency(r) for r in elig]),
        ("payload rule (per-position, both thresholds)",
         [rank_payload(r) for r in elig]),
        ("latch oracle (sanity: must be 100%)", [rank_latch(r) for r in elig]),
    ]
    print("   %-43s %9s %9s %8s" % ("", "Top-1", "set-F1", "rank"))
    for name, orders in rankings:
        m = [agree(o, r[3]) for o, r in zip(orders, elig)]
        print("   %-43s %7.1f%% %7.1f%% %8.2f"
              % (name, 100 * sum(x[0] for x in m) / n,
                 100 * sum(x[1] for x in m) / n,
                 sum(x[2] for x in m) / n))
    print("   -> the model must beat BOTH recency and the payload rule. Beating")
    print("      ground truth alone proves nothing: the payload rule does that")
    print("      for free, with no parameters and no training.")
    print()

    # where the mass actually goes
    dm = cm = 0.0
    for (w, _c, _p, _l), r in zip(att, elig):
        tot = sum(w) or 1.0
        dm += sum(w[i] for i, m in enumerate(r[4]) if m != "REPORTTEMP") / tot
        cm += sum(w[i] for i in r[3]) / tot
    print("attention mass on REPORTTEMP positions      : %.1f%%"
          % (100 * (1 - dm / n)))
    print("attention mass on the true causal positions : %.1f%%" % (100 * cm / n))
    print("   -> TICK/TRYOPEN/DOOROPENED carry zero information about")
    print("      compressorOn, so distractor mass is a fidelity failure with no")
    print("      excuse: there is nothing there to explain.")
    print()
    return elig, att


def report_safety(model, test, device, rng):
    """The v5 explanation report. Three metrics, because one of them is a trap.

      TOP-K HANDED |c|      generous: the rule is told how many openings to name,
                            and the causal set is a PREFIX of the openings on
                            68.8% of traces, so getting the count right is most
                            of the task. Reported for comparability only.
      CARDINALITY-FREE SET  the model must decide how many. Attention's set is
                            the positions above uniform (1/n), the only threshold
                            that needs no tuning. The floor is a rule that names
                            every door opening while looking at nothing at all:
                            83.0% F1, 39.9% exact. So F1 is not a result on this
                            benchmark -- exact-set is.
      BLAME OF THE TOP PICK responsibility (Chockler-Halpern) of the first
                            opening the explanation names. Defined on EVERY
                            violating trace, including the 36.7% where no single
                            opening is necessary and the set metrics have nothing
                            to score against. Floor = mean blame of a random
                            opening, ceiling = mean of the per-trace maximum.
    """
    viol = [r for r in test if r[6]["label"]]
    elig = [r for r in viol if r[3]]
    if not viol:
        print("!! no test trace violates the property -- nothing to explain.")
        return [], []
    att = final_attention(model, viol, device)
    bias = float(model.head.bias.detach().view(-1)[0])
    err = max(abs(sum(c) + bias - lg) for _w, c, _p, lg in att)
    print("=== explanation of the FINAL violation flag, test split only ===")
    print("test traces violating     : %d/%d" % (len(viol), len(test)))
    print("   of those, with >=1 necessary opening: %d (%.1f%%); the rest are"
          % (len(elig), 100 * len(elig) / len(viol)))
    print("   over-determined, so only the blame metric applies to them.")
    print("decomposition residual    : %.2e" % err)
    print()
    _safety_tables(viol, elig, att)
    _safety_mass(viol, att)
    return viol, att


def _att_order(w):
    return sorted(range(len(w)), key=lambda u: -w[u])


def _att_set(w):
    """The positions attention puts ABOVE UNIFORM.

    A threshold has to come from somewhere, and 1/n over the n positions of THIS
    trace is the only one that is not a tuned hyper-parameter: it is exactly
    "more mass than if the model had no preference". Tuning the threshold against
    the ground truth would be fitting the metric, which is the failure mode this
    whole audit exists to avoid.

    `w` is already truncated to the true length by `final_attention` (a[j, n-1,
    :n]), so len(w) is the sequence length and no padding dilutes the threshold.

    THE VERSION BEFORE THIS ONE USED 1/live, live = the count of nonzero
    weights, and it was silently broken in the worst possible direction:
    an explanation that spreads its mass evenly over exactly the right k
    positions has every weight equal to 1/k, so `x > 1/live` is false everywhere
    and the metric returned the EMPTY SET for a perfect answer. v5_verify.py
    caught it by handing the metric the ground truth and getting 0.0%. A metric
    that scores the ideal answer at zero cannot be used to reject a model.
    """
    thr = 1.0 / max(len(w), 1)
    return {u for u, x in enumerate(w) if x > thr}


def _safety_tables(viol, elig, att):
    amap = {id(r): w for r, (w, _c, _p, _l) in zip(viol, att)}
    named = [
        ("attention a[T-1]", lambda r: _att_order(amap[id(r)])),
        ("recency (latest position first)", rank_recency),
        ("DOOROPENED, latest first", rank_door_late),
        ("DOOROPENED, earliest first", rank_door_early),
        ("oracle (sanity: must be 100%)", rank_oracle),
    ]
    k = max(len(elig), 1)
    print("   %-38s %8s %8s %8s" % ("ranked, HANDED |c|", "Top-1", "set-F1", "exact"))
    for nm, fn in named:
        t1 = f1 = ex = 0.0
        for r in elig:
            o, c = fn(r), set(r[3])
            top = set(o[:len(c)])
            t1 += o[0] in c
            f1 += 2 * len(top & c) / (len(top) + len(c))
            ex += top == c
        print("   %-38s %7.1f%% %7.1f%% %7.1f%%"
              % (nm, 100 * t1 / k, 100 * f1 / k, 100 * ex / k))
    print()
    sets = [("attention above uniform", lambda r: _att_set(amap[id(r)])),
            ("every DOOROPENED (looks at nothing)", lambda r: set(_doors(r))),
            ("the last DOOROPENED only",
             lambda r: {_doors(r)[-1]} if _doors(r) else set())]
    print("   %-38s %8s %8s %8s" % ("cardinality-free set", "prec", "F1", "exact"))
    for nm, fn in sets:
        p = f1 = ex = 0.0
        for r in elig:
            S, c = fn(r), set(r[3])
            p += len(S & c) / max(len(S), 1)
            f1 += 2 * len(S & c) / max(len(S) + len(c), 1)
            ex += S == c
        print("   %-38s %7.1f%% %7.1f%% %7.1f%%"
              % (nm, 100 * p / k, 100 * f1 / k, 100 * ex / k))
    print()
    print("   %-38s %8s %8s %8s" % ("blame of the first named opening",
                                    "rule", "random", "best"))
    for nm, fn in named[:-1]:
        got = flat = best = 0.0
        for r in viol:
            rr, d = r[6]["resp"], set(_doors(r))
            if not d or not rr:
                continue
            o = [i for i in fn(r) if i in d]
            got += rr.get(o[0], 0.0) if o else 0.0
            flat += sum(rr.values()) / len(rr)
            best += max(rr.values())
        n = max(len(viol), 1)
        print("   %-38s %7.1f%% %7.1f%% %7.1f%%"
              % (nm, 100 * got / n, 100 * flat / n, 100 * best / n))
    print("   -> a rule at or below the random column is guessing. On the shipped")
    print("      preset NO stateless rule beats it, which is the point of v5.")
    print()


def _safety_mass(viol, att):
    warm = {"TICK", "DOOROPENED", "COOL"}
    dm = cm = 0.0
    for (w, _c, _p, _l), r in zip(att, viol):
        tot = sum(w) or 1.0
        dm += sum(w[i] for i, m in enumerate(r[4]) if m in warm) / tot
        cm += sum(w[i] for i in r[3]) / tot
    n = max(len(viol), 1)
    print("attention mass on temperature-changing messages : %.1f%%" % (100 * dm / n))
    print("attention mass on the necessary openings        : %.1f%%" % (100 * cm / n))
    print("   -> TRYOPEN and REPORTTEMP cannot change the temperature, so mass")
    print("      there is a fidelity failure with no excuse. Note the contrast with")
    print("      the latch target, where REPORTTEMP was the ONLY informative")
    print("      message: same model, same traces, opposite distractors.")
    print()


def faithfulness(model, elig, att, device, rng):
    """Is the attention map faithful to THIS MODEL, as opposed to matching the
    semantics? A high ground-truth score with near-zero comprehensiveness means
    attention describes something the model does not actually rely on."""
    print("=== faithfulness to the model (occlusion), test split only ===")
    ks = [len(r[3]) for r in elig]
    topk = [set(sorted(range(len(w)), key=lambda u: -w[u])[:k])
            for (w, _c, _p, _l), k in zip(att, ks)]
    rnd = [set(rng.sample(range(len(r[0])), k)) for r, k in zip(elig, ks)]
    orac = [set(r[3]) for r in elig]
    p_full = [p for _w, _c, p, _l in att]
    lab = [p >= 0.5 for p in p_full]

    def pc(ps):                    # probability of the FULL-input prediction
        return [p if l else 1.0 - p for p, l in zip(ps, lab)]

    base = pc(p_full)
    print("   %-27s %13s %12s" % ("explanation", "comprehens.", "sufficiency"))
    for name, S in (("attention top-k", topk), ("random k positions", rnd),
                    ("true causal set", orac)):
        comp = pc(occlude(model, elig, S, device))
        suff = pc(occlude(model, elig, S, device, invert=True))
        print("   %-27s %13.3f %12.3f"
              % (name,
                 sum(b - c for b, c in zip(base, comp)) / len(base),
                 sum(b - s for b, s in zip(base, suff)) / len(base)))
    print("   -> comprehensiveness: how far the prediction collapses when the")
    print("      explanation is deleted (higher is better). Sufficiency: how far")
    print("      it degrades when ONLY the explanation is kept (lower is better).")
    print("      If attention's comprehensiveness is no better than random-k, the")
    print("      map is decoration whatever it scores against ground truth.")
    print()


def show_one(elig, att, k=1):
    """One trace printed as a column, for eyeballing and for the paper figure."""
    for (w, c, p, _l), r in list(zip(att, elig))[:k]:
        print("=== example test trace (the final explanation) ===")
        print("   p(target at end) = %.3f, true = %d, ground-truth positions %s"
              % (p, int(r[2][-1]), r[3]))
        print("   %2s %-11s %-5s %-4s %-9s %s" %
              ("i", "message", "pay", "y", "contrib", "attention"))
        for i, m in enumerate(r[4]):
            pay = "" if r[5][i] is None else "%d" % r[5][i]
            tail = ""
            if i in r[3]:
                tail = "  <-- CAUSAL"
            elif r[6].get("resp", {}).get(i):
                tail = "  (blame %.2f)" % r[6]["resp"][i]
            print("   %2d %-11s %-5s %-4s %+9.3f %s%s"
                  % (i, m, pay, "1" if r[2][i] else "0", c[i],
                     "#" * int(round(40 * w[i])), tail))
        print()


def detect_target(path):
    """latch or safety, read off the file rather than off a flag the user forgot.

    The two schemas differ in more than a field name: the safety file carries
    `label`, `resp` and `safe_cap`, its causal positions are DOOROPENED rather
    than REPORTTEMP, and COOL moves from the effect set to the cause set. Running
    the wrong path would train happily and report meaningless numbers, so this is
    detected, printed, and overridable rather than assumed.
    """
    blob = json.load(open(path))
    if isinstance(blob, dict) and "safe_cap" in blob.get("config", {}):
        return "safety"
    if isinstance(blob, dict) and blob.get("traces") and \
            "label" in blob["traces"][0]:
        raise SystemExit(
            "%s looks like a safety file but has no safe_cap in its config, so its\n"
            "'target' field is the stale one (it says temperature>tick_cap, which is\n"
            "wrong). Regenerate it:\n"
            "  python3 safety.py --lazy --preset --sample 3000 --out %s" % (path, path))
    return "latch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces_v4.json")
    ap.add_argument("--target", choices=["auto", "latch", "safety"], default="auto")
    ap.add_argument("--read", choices=["values", "hidden"], default="values")
    ap.add_argument("--keep-effects", action="store_true",
                    help="latch: put TURNON/TURNOFF/COOL back in the input (the "
                         "leak). safety: put TURNON/TURNOFF back (COOL is always "
                         "in, it is one of the three messages that move the "
                         "temperature)")
    ap.add_argument("--no-payload", action="store_true",
                    help="latch: zero the payload channel; the task should collapse")
    ap.add_argument("--keep-payload", action="store_true",
                    help="safety: put the payload channel back. It is OFF by "
                         "default because max-payload>safe_cap alone predicts the "
                         "label 76.3%% against a 54.4%% majority, and the message "
                         "types already determine the temperature exactly")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="path for the trained weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tgt = detect_target(a.traces) if a.target == "auto" else a.target
    print("=== target: %s%s ===" % (tgt, "  (auto-detected)" if a.target == "auto" else ""))
    if tgt == "safety":
        recs, vocab = build_safety(a.traces, a.keep_effects, a.keep_payload)
    else:
        recs, vocab = build(a.traces, a.keep_effects, not a.no_payload)
    if not recs:
        print("no usable traces in %s -- run pipeline.py --sample N first" % a.traces)
        return 1
    rng.shuffle(recs)
    n1, n2 = int(.70 * len(recs)), int(.85 * len(recs))
    tr, va, te = recs[:n1], recs[n1:n2], recs[n2:]
    steps = sum(len(r[0]) for r in te)
    off = sum(1 for r in te for v in r[2] if v == 0.0)

    print("=== setup ===")
    print("vocabulary             : %s" % ", ".join(sorted(vocab, key=vocab.get)))
    print("traces train/val/test  : %d / %d / %d" % (len(tr), len(va), len(te)))
    print("sequence length        : %d .. %d"
          % (min(len(r[0]) for r in recs), max(len(r[0]) for r in recs)))
    print("read mode              : %s" % a.read)
    if tgt == "safety":
        print("payload channel        : %s"
              % ("ON (leaks ~76% of the label)" if a.keep_payload else "off (default)"))
        print("effect messages        : %s"
              % ("TURNON/TURNOFF in the input" if a.keep_effects
                 else "TURNON/TURNOFF dropped; COOL kept (it moves the temperature)"))
        print("per-step target        : has the property been violated by step t")
    else:
        print("payload channel        : %s"
              % ("OFF (ablation)" if a.no_payload else "on"))
        print("effect messages        : %s"
              % ("IN THE INPUT (leak)" if a.keep_effects else "dropped"))
        print("per-step target        : compressorOn after step t")
    print("always-0 per-step      : %.1f%%  <- the accuracy floor"
          % (100 * off / max(steps, 1)))
    print()
    if tgt == "safety":
        floor = prediction_floors(te, "test")

    model = Surrogate(len(vocab) + 1, a.read).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state, bad, patience = 1e9, None, 0, 10
    print("=== training (per-step BCE, causal attention) ===")
    for ep in range(1, a.epochs + 1):
        trl, tra = run_epoch(model, tr, a.batch, device, rng, opt)
        vl, vac = run_epoch(model, va, a.batch, device, rng)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0:
            print("   epoch %3d   train %.4f / %.4f    val %.4f / %.4f"
                  % (ep, trl, tra, vl, vac))
        if bad >= patience:
            print("   early stop at epoch %d (best val loss %.4f)" % (ep, best))
            break
    if best_state:
        model.load_state_dict(best_state)
    print()

    tl, tac = run_epoch(model, te, a.batch, device, rng)
    fa = final_attention(model, te, device)
    fin = sum(1 for (_w, _c, p, _l), r in zip(fa, te) if (p >= .5) == (r[2][-1] > .5))
    print("=== surrogate accuracy (test split) ===")
    print("per-step accuracy  : %.2f%%   (BCE %.4f)" % (100 * tac, tl))
    print("final-step accuracy: %.2f%%   (%d/%d traces)"
          % (100 * fin / len(te), fin, len(te)))
    if tgt == "safety":
        acc = 100 * fin / len(te)
        print("   -> the only comparison that means anything is against the best")
        print("      stateless floor above (%.1f%%), not the majority class."
              % floor)
        print("      %s the payload rule by %+.1f points."
              % ("BEATS" if acc > floor else "does NOT beat", acc - floor))
        print("      The exact ceiling is 100%: the temperature is a deterministic")
        print("      fold of the message types, so a replay of the fold is perfect.")
        print("      This number measures how close the network got to that fold.")
    else:
        print("   -> if this is not clearly above the always-off floor, the surrogate")
        print("      has not learned the latch and its explanations mean nothing.")
        print("      And note: the LAST-CROSSING rule gets 100.0% here with no state")
        print("      at all, so this number is not evidence of anything by itself.")
    print()

    if tgt == "safety":
        viol, att = report_safety(model, te, device, rng)
        elig = [r for r in viol if r[3]]
        if elig:
            keep = [i for i, r in enumerate(viol) if r[3]]
            faithfulness(model, elig, [att[i] for i in keep], device, rng)
            show_one(elig, [att[i] for i in keep])
    else:
        elig, att = report(model, te, device, rng)
        if elig:
            faithfulness(model, elig, att, device, rng)
            show_one(elig, att)
    if a.save:
        torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                    "read": a.read, "target": tgt, "config": CFG}, a.save)
        print("wrote %s" % a.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
