#!/usr/bin/env python3
"""
verify_payload_rules.py

Settles, from traces_parsed_with_payload.csv alone and with ZERO learned
parameters, what the decisive-message problem in this dataset actually is.

Answers, in order:
  0. are there message types outside the 5-token vocab (the IDLE question)?
  1. is compressorOn really a write-once latch (at most one flip, ever)?
  2. is gt_step exactly "first processed reportTemp with payload > 7"?
  3. where do REPORTTEMP steps sit in the trace, and where does gt_step sit?
  4. how do zero-parameter rules score vs the reported neural numbers?
  5. what does the "neutral" payload feature value 0.0 decode to after
     z-scoring (i.e. what do #17/#19 actually tell the net at non-REPORTTEMP
     steps)?
"""
import numpy as np
import pandas as pd
from collections import Counter

CSV = "/sessions/busy-pensive-wozniak/mnt/uploads/traces_parsed_with_payload.csv"
COMP = "controllerRebec.Controller.compressorOn"
VOCAB = {"TICK", "TRYOPEN", "DOOROPENED", "REPORTTEMP", "TURNON"}
THRESH = 7

REPORTED = {
    "baseline attention (#1)": (21.2, 62.1, 1.33),
    "exp #11 featurized": (21.2, 62.1, 1.33),
    "exp #17 payload LSTM": (None, None, None),
    "exp #19 masked TURNON": (43.0, 78.9, 0.82),
}


def tobool(v):
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


df = pd.read_csv(CSV)
df[COMP] = df[COMP].apply(tobool)
df = df.sort_values(["trace_id", "step"])

seen = set(df["message"].unique())
print("=== 0. token inventory ===")
print("message types present :", sorted(seen))
print("outside 5-token vocab :", sorted(seen - VOCAB) or "none")
print("TURNOFF ever emitted  :", "TURNOFF" in seen, "(dead branch check)")
print()

traces = []
for tid, g in df.groupby("trace_id"):
    msgs = g["message"].tolist()
    comp = g[COMP].tolist()
    pay = [None if pd.isna(p) else int(p) for p in g["reporttemp_payload"].tolist()]
    gt, prev, nflip = None, False, 0
    for i, v in enumerate(comp):
        if v != prev:
            nflip += 1
            if v == comp[-1]:
                gt = i
        prev = v
    rt = [i for i, m in enumerate(msgs) if m == "REPORTTEMP"]
    traces.append(dict(tid=tid, msgs=msgs, comp=comp, pay=pay, gt=gt,
                       nflip=nflip, L=len(msgs), rt=rt))

N = len(traces)
elig = [t for t in traces if t["gt"] is not None]
n = len(elig)
nTrue = sum(t["comp"][-1] for t in traces)

print("=== 1. latch check ===")
print("traces=%d   eligible(gt is not None)=%d   never-flip=%d" % (N, n, N - n))
print("max flips observed in any single trace : %d" % max(t["nflip"] for t in traces))
print("final compressorOn True                : %d/%d = %.2f%%  <- majority baseline"
      % (nTrue, N, 100 * nTrue / N))
print("gt_step is always a REPORTTEMP         :",
      all(t["msgs"][t["gt"]] == "REPORTTEMP" for t in elig))
print()

print("=== 2. is gt_step exactly 'first processed reportTemp with payload > %d'? ===" % THRESH)
agree, disagree, examples = 0, 0, []
for t in elig:
    first_over = next((i for i in t["rt"] if t["pay"][i] is not None
                       and t["pay"][i] > THRESH), None)
    if first_over == t["gt"]:
        agree += 1
    else:
        disagree += 1
        if len(examples) < 3:
            examples.append((t["tid"], t["gt"], first_over,
                             [(i, t["pay"][i]) for i in t["rt"]]))
print("agreement: %d/%d = %.2f%%" % (agree, n, 100 * agree / n))
for tid, gt, fo, pays in examples:
    print("  counterexample trace %s: gt=%s  first>%d=%s  payloads=%s"
          % (tid, gt, THRESH, fo, pays))
# does any non-flipping trace contain a payload > THRESH? (would break the rule)
bad = [t["tid"] for t in traces if t["gt"] is None
       and any(t["pay"][i] is not None and t["pay"][i] > THRESH for i in t["rt"])]
print("never-flip traces that DID see a payload > %d : %d %s"
      % (THRESH, len(bad), bad[:5]))
print()

print("=== 3. trace shape: are all REPORTTEMPs pushed to the tail? ===")
tail_only = 0
for t in traces:
    if t["rt"] and all(m == "REPORTTEMP" or m == "TURNON"
                       for m in t["msgs"][t["rt"][0]:]):
        tail_only += 1
print("traces where nothing but REPORTTEMP/TURNON follows the 1st REPORTTEMP:"
      " %d/%d = %.1f%%" % (tail_only, N, 100 * tail_only / N))
print("length distribution      :", sorted(Counter(t["L"] for t in traces).items()))
print("REPORTTEMPs per trace    :", sorted(Counter(len(t["rt"]) for t in traces).items()))
relpos = [t["gt"] / max(t["L"] - 1, 1) for t in elig]
print("gt_step normalised position: mean=%.3f  median=%.3f  min=%.3f"
      % (float(np.mean(relpos)), float(np.median(relpos)), float(np.min(relpos))))
rtidx = [t["rt"].index(t["gt"]) for t in elig]
print("gt_step's index WITHIN the REPORTTEMP subsequence:",
      sorted(Counter(rtidx).items()))
print("  -> fraction where gt is the 2nd REPORTTEMP : %.1f%%"
      % (100 * sum(i == 1 for i in rtidx) / n))
print("  -> fraction where gt is the LAST REPORTTEMP: %.1f%%"
      % (100 * sum(t["rt"].index(t["gt"]) == len(t["rt"]) - 1 for t in elig) / n))
print()

print("=== 4. zero-parameter rules (no training, no parameters) ===")


def report(label, rank_of_gt):
    r = [rank_of_gt(t) for t in elig]
    r = [x for x in r if x is not None]
    print("  %-44s Top-1=%5.1f%%  Top-2=%5.1f%%  mean rank=%.2f"
          % (label, 100 * sum(x == 0 for x in r) / len(r),
             100 * sum(x <= 1 for x in r) / len(r), float(np.mean(r))))


# A: recency over every position
report("A  recency over all positions",
       lambda t: t["L"] - 1 - t["gt"])
# B: REPORTTEMP first, then recency  (what #19 provably reduces to)
report("B  REPORTTEMP first, then recency",
       lambda t: sum(1 for i in range(t["gt"] + 1, t["L"])
                     if t["msgs"][i] == "REPORTTEMP"))
# C: TURNON excluded from the pool, then recency
report("C  TURNON excluded, then recency",
       lambda t: sum(1 for i in range(t["gt"] + 1, t["L"])
                     if t["msgs"][i] != "TURNON"))
# D: the true semantics -- threshold, earliest first
report("D  payload > %d, earliest first  <- SEMANTICS" % THRESH,
       lambda t: sum(1 for i in t["rt"]
                     if t["pay"][i] is not None and t["pay"][i] > THRESH
                     and i < t["gt"]))
# E: highest payload so far (their symbolic heuristic)
report("E  running-max payload, earliest first",
       lambda t: sum(1 for i in t["rt"] if i < t["gt"] and t["pay"][i] is not None
                     and t["pay"][i] >= max(
                         [t["pay"][j] for j in t["rt"] if j <= i
                          and t["pay"][j] is not None] or [0])))
# F: uniform guess among REPORTTEMP candidates (informed random, expected Top-1)
print("  %-44s Top-1=%5.1f%%  (expected value, no ranking)"
      % ("F  uniform random over REPORTTEMPs",
         100 * float(np.mean([1.0 / len(t["rt"]) for t in elig]))))
print()

print("reported neural results, for comparison:")
for k, v in REPORTED.items():
    if v[0] is None:
        print("  %-44s (not recorded)" % k)
    else:
        print("  %-44s Top-1=%5.1f%%  Top-2=%5.1f%%  mean rank=%.2f" % (k, *v))
print()

print("=== 5. what does the 'neutral' payload feature 0.0 actually mean? ===")
pmean = df["reporttemp_payload"].dropna().astype(float).mean()
pstd = df["reporttemp_payload"].dropna().astype(float).std()
print("payload mean=%.3f  std=%.3f  (over %d REPORTTEMP events)"
      % (pmean, pstd, int(df["reporttemp_payload"].notna().sum())))
print("payload value distribution:",
      sorted(Counter(df["reporttemp_payload"].dropna().astype(int)).items()))
print("#17/#19 feed 0.0 at every non-REPORTTEMP step; after z-scoring that")
print("decodes to payload = %.2f, which is %s the decision threshold of %d."
      % (pmean, "ABOVE" if pmean > THRESH else "below", THRESH))
print("So TICK / TRYOPEN / DOOROPENED / TURNON steps are each told")
print("'my temperature is %.1f' -- i.e. trigger-worthy -- with no has_payload"
      % pmean)
print("flag to distinguish them. This is not a neutral fill value.")
print()

print("=== 6. truncation ===")
sent = []
for t in traces:
    n_tick = sum(1 for m in t["msgs"] if m == "TICK")
    n_rt = len(t["rt"])
    sent.append((n_tick, n_rt))
print("mean reportTemps SENT (=TICK steps) per trace : %.2f"
      % float(np.mean([a for a, _ in sent])))
print("mean reportTemps PROCESSED per trace          : %.2f"
      % float(np.mean([b for _, b in sent])))
print("traces ending with unprocessed reportTemps in the queue: %d/%d = %.1f%%"
      % (sum(1 for a, b in sent if a > b), N,
         100 * sum(1 for a, b in sent if a > b) / N))
print("traces with a flip but NO TURNON step (TURNON sent, never scheduled): %d"
      % sum(1 for t in elig if "TURNON" not in t["msgs"]))
print()
print("prefix redundancy: distinct 5-step prefixes among %d traces = %d"
      % (N, len({tuple(t["msgs"][:5]) for t in traces})))
