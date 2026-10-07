#!/usr/bin/env python3
"""
verify_rule_equivalence.py

Decisive test of the claim that the reported attention metrics are the
statistics of pure lexical+positional rules, using all THREE reported
numbers per experiment (Top-1, Top-2, mean rank) rather than just one.

Rules tested, each with no learned parameters:
  A  rank all positions by recency (latest first)
  B  rank REPORTTEMP positions above all others, then by recency
  C  exclude TURNON, rank remaining positions by recency   <- what #19 did
"""
import pandas as pd
import numpy as np

CSV = "/sessions/busy-pensive-wozniak/mnt/uploads/traces_parsed.csv"
COMP = "controllerRebec.Controller.compressorOn"

REPORTED = {
    "baseline attention": (21.2, 62.1, 1.33),
    "exp #11 featurized": (21.2, 62.1, 1.33),
    "exp #19 masked TURNON": (43.0, 78.9, 0.82),
}


def tobool(v):
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


df = pd.read_csv(CSV)
df[COMP] = df[COMP].apply(tobool)
df = df.sort_values(["trace_id", "step"])

recs = []
for tid, g in df.groupby("trace_id"):
    comp, msgs = g[COMP].tolist(), g["message"].tolist()
    gt, prev = None, False
    for i, v in enumerate(comp):
        if v != prev and v == comp[-1]:
            gt = i
        prev = v
    if gt is not None:
        recs.append((msgs, gt))

N = len(recs)


def score(rank_fn, label):
    ranks = [rank_fn(msgs, gt) for msgs, gt in recs]
    ranks = [r for r in ranks if r is not None]
    t1 = 100 * sum(r == 0 for r in ranks) / len(ranks)
    t2 = 100 * sum(r <= 1 for r in ranks) / len(ranks)
    mr = float(np.mean(ranks))
    print("  %-46s Top-1=%5.1f%%  Top-2=%5.1f%%  mean rank=%.2f" % (label, t1, t2, mr))
    return t1, t2, mr


def rank_A(msgs, gt):
    # recency over every position: rank == distance from the end
    return len(msgs) - 1 - gt


def rank_B(msgs, gt):
    # all REPORTTEMP above everything else, then recency within REPORTTEMP
    return sum(1 for i in range(gt + 1, len(msgs)) if msgs[i] == "REPORTTEMP")


def rank_C(msgs, gt):
    # TURNON removed from the candidate pool, then recency
    return sum(1 for i in range(gt + 1, len(msgs)) if msgs[i] != "TURNON")


print("scorable traces: %d\n" % N)
print("zero-parameter rules computed from the CSV alone:")
a = score(rank_A, "A  recency over all positions")
b = score(rank_B, "B  REPORTTEMP first, then recency")
c = score(rank_C, "C  TURNON excluded, then recency")
print()
print("reported model results:")
for k, v in REPORTED.items():
    print("  %-46s Top-1=%5.1f%%  Top-2=%5.1f%%  mean rank=%.2f" % (k, *v))
print()

print("match check (all three statistics must agree):")
for name, (t1, t2, mr) in REPORTED.items():
    for rlabel, r in (("A", a), ("B", b), ("C", c)):
        ok = abs(t1 - r[0]) < 0.1 and abs(t2 - r[1]) < 0.1 and abs(mr - r[2]) < 0.01
        if ok:
            print("  %-24s == rule %s   (exact on Top-1, Top-2 AND mean rank)" % (name, rlabel))
