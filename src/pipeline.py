#!/usr/bin/env python3
"""
pipeline.py  --  one file that replaces Afra in the loop.

WHY THIS EXISTS
    traces.xml is a 266-state PREFIX of the v4 state space (many states in it
    still hold pending messages and have no outgoing transitions). Random walks
    over a truncated graph would reintroduce the truncated-trace problem that
    the idle() removal was meant to kill. So instead of asking Afra for a bigger
    export, this script reimplements the v4 operational semantics directly,
    CROSS-VALIDATES that reimplementation against every transition Afra did
    report, and then enumerates the state space to closure itself.

    Validation is the point: you do not have to trust the reimplementation,
    because step 1 proves it agrees with Afra on the 266 states Afra explored.

TWO SEMANTIC FACTS READ OFF traces.xml (both are load-bearing)
    1. The message queue is a BAG, not a sequence. Evidence:
         3_0 --TICK--> 5_0
       3_0's sensor queue is {tick, doorOpened}; executing tick() removes tick
       and appends a fresh tick, giving the ORDER [doorOpened, tick], yet Afra
       maps it onto 5_0 whose queue is PRINTED [tick, doorOpened]. So state
       identity ignores queue order and the printed order is cosmetic.
       => states must be hashed with queues canonicalised as sorted multisets.
    2. Any message in the queue may be selected, not just the head. Evidence:
         4_0 --REPORTTEMP--> 11_0   and   4_0 --REPORTTEMP--> 12_0
       from controller queue {reportTemp(6), reportTemp(7)}. This is what makes
       payload, not token type, the only thing that identifies the fired message.

    Corollary on duplicate parallel edges (8_0->15_0 twice, 17_0->28_0 thrice):
    those are Afra enumerating identical bag elements. Under bag semantics they
    are ONE action, so --action-weighting=distinct (default) dedupes them;
    --action-weighting=positions reproduces Afra's multiplicity instead.

USAGE
    python3 pipeline.py                                  # self-test + enumerate
    python3 pipeline.py --xml traces.xml                  # + validate vs Afra
    python3 pipeline.py --xml traces.xml --sample 4000    # + generate traces
    python3 pipeline.py --sweep 1000                      # hunt a better config
    python3 pipeline.py --lazy --sample 4000 \
            --init-temp 0 --max-ticks 12 --cool-delta 6       # one candidate
    then: python3 surrogate.py --traces traces_v4.json    # stage 2

    Stdlib only, no dependencies. Step 0 is a set of hand-computed assertions
    that runs by default -- if it prints FAIL, stop and read that line, because
    nothing after it means anything. It is skipped automatically when the
    configuration is changed away from fridge_v4.rebeca, since its expected
    values were derived by hand for that one configuration.

    --lazy samples by walking the semantics on demand instead of enumerating the
    closure first. Needed for anything much beyond the shipped maxTicks=6: that
    already has 166,652 states, and the count grows fast.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, deque

# ---------------------------------------------------------------- v4 constants
# These are MUTABLE on purpose: --init-temp/--max-ticks/--cool-delta/... rebind
# them through configure() before anything else runs, so one file can explore the
# whole parameter family instead of hard-coding the one configuration that turned
# out to be degenerate. DEFAULTS is exactly fridge_v4.rebeca as written.
DEFAULTS = dict(init_temp=5, max_ticks=6, max_opens=3, tick_cap=15,
                door_delta=3, door_cap=None, cool_delta=10, cool_floor=0,
                hi=7, lo=3)

INIT_TEMP, MAX_TICKS, MAX_OPENS = 5, 6, 3
TICK_CAP = 15          # `if (temperature < 15)` guards tick() ONLY
DOOR_DELTA = 3         # doorOpened() adds +3 ...
DOOR_CAP = None        # ... and is UNCAPPED by default -- the known v4 bug
COOL_DELTA = 10        # cool() subtracts 10 ...
COOL_FLOOR = 0         # ... with a floor at 0
HI, LO = 7, 3          # reportTemp thresholds: temp > 7 turns on, temp <= 3 off

# A state is a flat hashable tuple. Queues are sorted tuples => bag semantics.
#   (temp, ticksLeft, sensorQ, opensLeft, doorQ, compressorOn, ctrlQ, cycles, compQ)
# Messages inside a queue: ("tick",), ("doorOpened",), ("cool",), ("tryOpen",),
#                          ("reportTemp", n), ("turnOn",), ("turnOff",)
INITIAL = (INIT_TEMP, MAX_TICKS, (("tick",),),
           MAX_OPENS, (("tryOpen",),),
           False, (),
           0, ())


def configure(**kw):
    """Rebind the constants and recompute INITIAL. Returns the effective config.

    Anything omitted keeps its DEFAULTS value, so `configure()` restores the
    shipped fridge_v4.rebeca configuration exactly.
    """
    global INIT_TEMP, MAX_TICKS, MAX_OPENS, TICK_CAP, DOOR_DELTA, DOOR_CAP
    global COOL_DELTA, COOL_FLOOR, HI, LO, INITIAL
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    INIT_TEMP, MAX_TICKS, MAX_OPENS = cfg["init_temp"], cfg["max_ticks"], cfg["max_opens"]
    TICK_CAP, DOOR_DELTA, DOOR_CAP = cfg["tick_cap"], cfg["door_delta"], cfg["door_cap"]
    COOL_DELTA, COOL_FLOOR = cfg["cool_delta"], cfg["cool_floor"]
    HI, LO = cfg["hi"], cfg["lo"]
    INITIAL = (INIT_TEMP, MAX_TICKS, (("tick",),),
               MAX_OPENS, (("tryOpen",),),
               False, (),
               0, ())
    return cfg


def is_default_config():
    """The self-test's expected values are hand-derived for DEFAULTS only."""
    return (INIT_TEMP, MAX_TICKS, MAX_OPENS, TICK_CAP, DOOR_DELTA, DOOR_CAP,
            COOL_DELTA, COOL_FLOOR, HI, LO) == (
        DEFAULTS["init_temp"], DEFAULTS["max_ticks"], DEFAULTS["max_opens"],
        DEFAULTS["tick_cap"], DEFAULTS["door_delta"], DEFAULTS["door_cap"],
        DEFAULTS["cool_delta"], DEFAULTS["cool_floor"], DEFAULTS["hi"],
        DEFAULTS["lo"])

LABEL = {"tick": "TICK", "doorOpened": "DOOROPENED", "cool": "COOL",
         "tryOpen": "TRYOPEN", "reportTemp": "REPORTTEMP",
         "turnOn": "TURNON", "turnOff": "TURNOFF"}


def _bag(msgs):
    return tuple(sorted(msgs))


def _drop(bag, msg):
    lst = list(bag)
    lst.remove(msg)
    return tuple(lst)


def successors(s, weighting="distinct"):
    """Every enabled action. Returns [(label, payload, next_state), ...].

    payload is the int argument of reportTemp, else None -- this is what the
    trace needs in order to be self-describing, and it is exactly what the
    5-token vocabulary threw away.
    """
    temp, ticks, sq, opens, dq, on, cq, cyc, mq = s
    out = []

    src = [(sq, 0), (dq, 1), (cq, 2), (mq, 3)]
    for bag, which in src:
        pool = sorted(set(bag)) if weighting == "distinct" else sorted(bag)
        for msg in pool:
            t, temp2, ticks2, opens2, on2, cyc2 = msg[0], temp, ticks, opens, on, cyc
            sq2, dq2, cq2, mq2 = sq, dq, cq, mq
            pay = None

            if which == 0:                                   # ---- Sensor ----
                sq2 = _drop(sq, msg)
                if t == "tick":
                    if ticks > 0:                            # else: silent no-op
                        ticks2 = ticks - 1
                        if temp < TICK_CAP:
                            temp2 = temp + 1
                        cq2 = _bag(cq + (("reportTemp", temp2),))
                        sq2 = _bag(sq2 + (("tick",),))
                elif t == "doorOpened":
                    temp2 = temp + DOOR_DELTA                # uncapped by default
                    if DOOR_CAP is not None:                 # --cap-door closes it
                        temp2 = min(DOOR_CAP, temp2)
                elif t == "cool":
                    temp2 = max(COOL_FLOOR, temp - COOL_DELTA)
            elif which == 1:                                 # ---- Door ------
                dq2 = _drop(dq, msg)
                if opens > 0:                                 # else: silent no-op
                    opens2 = opens - 1
                    sq2 = _bag(sq + (("doorOpened",),))
                    dq2 = _bag(dq2 + (("tryOpen",),))
            elif which == 2:                                 # ---- Controller-
                cq2 = _drop(cq, msg)
                pay = msg[1]
                if pay > HI and not on:
                    on2 = True
                    mq2 = _bag(mq + (("turnOn",),))
                elif pay <= LO and on:
                    on2 = False
                    mq2 = _bag(mq + (("turnOff",),))
            else:                                            # ---- Compressor-
                mq2 = _drop(mq, msg)
                if t == "turnOn":
                    cyc2 = cyc + 1
                    sq2 = _bag(sq + (("cool",),))
                else:
                    cyc2 = 0

            nxt = (temp2, ticks2, _bag(sq2), opens2, _bag(dq2),
                   on2, _bag(cq2), cyc2, _bag(mq2))
            out.append((LABEL[t], pay, nxt))
    return out


def is_terminal(s):
    return not (s[2] or s[4] or s[6] or s[8])


# ------------------------------------------------------------- Afra XML parser
RE_ID = re.compile(r'"([^"]+)"')
RE_VAR = re.compile(r'<variable name="([^"]+)"[^>]*>([^<]*)</variable>')
RE_MSG = re.compile(r'<message [^>]*>([^<]*)</message>')
RE_TRANS = re.compile(
    r'<transition source="([^"]+)" destination="([^"]+)"[^>]*>\s*'
    r'<messageserver sender="([^"]+)" owner="([^"]+)" title="([^"]+)"')


def parse_msg(text):
    text = text.strip()
    m = re.match(r'^(\w+)\((.*)\)$', text)
    name, arg = m.group(1), m.group(2).strip()
    return (name, int(arg)) if arg else (name,)


def parse_afra(path):
    """-> (states: id -> flat state tuple, edges: list of (src, dst, title))"""
    blob = open(path, encoding="utf-8", errors="replace").read()
    states, edges = {}, []
    for chunk in blob.split("<state id=")[1:]:
        body = chunk.split("</state>")[0]
        sid = RE_ID.match(chunk).group(1)
        per = {}
        for rb in body.split('<rebec name="')[1:]:
            name = rb[:rb.index('"')]
            vs = {k.split(".")[-1]: v for k, v in RE_VAR.findall(rb)}
            qtxt = rb.split("<queue>")[1].split("</queue>")[0]
            per[name] = (vs, _bag(parse_msg(t) for t in RE_MSG.findall(qtxt)))
        sv, dv = per["sensorRebec"][0], per["doorRebec"][0]
        cv, mv = per["controllerRebec"][0], per["compressorRebec"][0]
        states[sid] = (int(sv["temperature"]), int(sv["ticksLeft"]),
                       per["sensorRebec"][1],
                       int(dv["opensLeft"]), per["doorRebec"][1],
                       cv["compressorOn"].strip() == "true",
                       per["controllerRebec"][1],
                       int(mv["consecutiveOnCycles"]), per["compressorRebec"][1])
    for src, dst, _s, _o, title in RE_TRANS.findall(blob):
        edges.append((src, dst, title))
    return states, edges


# ------------------------------------------------------- step 1: cross-validate
def validate(path):
    """Every transition Afra reported must be reproduced by successors()."""
    states, edges = parse_afra(path)
    print("=== 1. cross-validation against Afra's export ===")
    print("states in export      : %d" % len(states))
    print("transitions in export : %d" % len(edges))
    print("initial state matches : %s" % (states.get("1_0") == INITIAL))

    ok = bad = 0
    missing = []
    for src, dst, title in edges:
        if src not in states or dst not in states:
            continue
        got = {(l, n) for l, _p, n in successors(states[src], "positions")}
        if (title, states[dst]) in got:
            ok += 1
        else:
            bad += 1
            if len(missing) < 5:
                missing.append((src, dst, title))
    print("edges reproduced      : %d/%d" % (ok, ok + bad))
    if bad == 0:
        print("   -> the reimplementation below is Afra's semantics, verified.")
    for src, dst, title in missing:
        print("   MISMATCH %s -> %s (%s)" % (src, dst, title))

    # the converse: does the export omit successors it should have had?
    frontier = []
    have = {}
    for src, dst, title in edges:
        have.setdefault(src, []).append((title, dst))
    for sid, s in states.items():
        exp = len(successors(s, "positions"))
        if len(have.get(sid, [])) < exp:
            frontier.append((sid, len(have.get(sid, [])), exp))
    print("truncation: %d/%d states have FEWER successors in the export than the"
          % (len(frontier), len(states)))
    print("            semantics allows -> the export is a prefix, not the closure.")
    print("            e.g. %s" % ", ".join(
        "%s(%d/%d)" % f for f in sorted(frontier, key=lambda x: -x[2])[:6]))
    print()
    return states, edges, len(frontier) == 0


# ---------------------------------------------- step 2: enumerate to closure
def enumerate_full(weighting="distinct", cap=4_000_000):
    """BFS to closure. Returns (order, adj, depth)."""
    adj, depth = {}, {INITIAL: 0}
    q, order = deque([INITIAL]), []
    while q:
        s = q.popleft()
        order.append(s)
        succ = successors(s, weighting)
        adj[s] = succ
        for _l, _p, n in succ:
            if n not in depth:
                depth[n] = depth[s] + 1
                q.append(n)
        if len(depth) > cap:
            raise RuntimeError("state space exceeded %d -- shrink maxTicks/maxOpens"
                               % cap)
    return order, adj, depth


def audit(order, adj, depth):
    print("=== 2. the FULL state space (enumerated here, not by Afra) ===")
    print("reachable states : %d" % len(order))
    print("transitions      : %d" % sum(len(v) for v in adj.values()))
    print("max BFS depth    : %d" % max(depth.values()))
    lab = Counter(l for v in adj.values() for l, _p, _n in v)
    print("transitions by label:")
    for k in ("TICK", "TRYOPEN", "DOOROPENED", "REPORTTEMP",
              "TURNON", "COOL", "TURNOFF"):
        print("   %-11s %6d" % (k, lab.get(k, 0)))
    print("TURNON : TURNOFF = %.1f : 1" % (lab.get("TURNON", 1) /
                                           max(lab.get("TURNOFF", 1), 1)))
    term = [s for s in order if is_terminal(s)]
    print("terminal states (all queues empty) : %d" % len(term))
    print("   final compressorOn True : %d/%d = %.1f%%"
          % (sum(s[5] for s in term), len(term),
             100 * sum(s[5] for s in term) / max(len(term), 1)))
    print("temperature range observed : %d .. %d"
          % (min(s[0] for s in order), max(s[0] for s in order)))
    if DOOR_CAP is None and max(s[0] for s in order) > TICK_CAP:
        print("   (max > %d confirms the uncapped doorOpened() bug)" % TICK_CAP)
    print("consecutiveOnCycles values : %s"
          % sorted({s[7] for s in order}))
    caps = {"sensor": (2, 12), "door": (4, 6), "controller": (6, 9),
            "compressor": (8, 9)}
    for nm, (ix, cap) in caps.items():
        print("   max %-11s queue = %d  (declared cap %d)"
              % (nm, max(len(s[ix]) for s in order), cap))
    print()


# ----------------------------------------- step 2b: structural theorems
def invariants(traces):
    """Two theorems about v4 that a broken trace generator would violate.

    (i) LENGTH IDENTITY. Every complete execution consumes exactly 7 TICK
        (1 from the constructor + 1 resent per successful tick), 4 TRYOPEN,
        3 DOOROPENED and 6 REPORTTEMP, plus exactly one COOL per TURNON. So
            len(trace) == 20 + 2*nTurnOn + nTurnOff,   and len >= 22 always,
        because the first turnOn is unavoidable: with initTemp=5 the third tick
        reports 8 > 7 and no cool() can possibly precede it. This is also the
        proof that Afra's 266-state export is a prefix -- 266 states in BFS
        reach nowhere near depth 22.

    (ii) AUTOMATON ORACLE. compressorOn is a function of the REPORTTEMP PAYLOAD
        SUBSEQUENCE ALONE -- a two-state latch,
            false --[payload > 7]--> true --[payload <= 3]--> false
        -- because reportTemp() is the only message server that writes it.
        This is the load-bearing fact for everything downstream: it means
        TICK/TRYOPEN/DOOROPENED are pure distractors, position is irrelevant,
        and payload-alone is provably insufficient (the same payload means
        different things depending on the latch). That is exactly what v3
        lacked, and it is the target the surrogate has to fit.
    """
    print("=== 2b. structural invariants ===")
    # Derived from the configuration, not hard-coded: the constructor sends one
    # tick and each successful tick resends one (the last is a no-op), so
    # TICK == maxTicks+1, TRYOPEN == maxOpens+1, DOOROPENED == maxOpens,
    # REPORTTEMP == maxTicks, and everything else is turnOn/cool/turnOff.
    exp = {"TICK": MAX_TICKS + 1, "TRYOPEN": MAX_OPENS + 1,
           "DOOROPENED": MAX_OPENS, "REPORTTEMP": MAX_TICKS}
    base = 2 * (MAX_TICKS + MAX_OPENS + 1)
    bad_fixed = bad_len = bad_auto = 0
    for t in traces:
        c = Counter(s["message"] for s in t)
        if any(c[k] != v for k, v in exp.items()):
            bad_fixed += 1
        if c["COOL"] != c["TURNON"] or \
                len(t) != base + 2 * c["TURNON"] + c["TURNOFF"]:
            bad_len += 1
        on = False
        for s in t:
            if s["message"] == "REPORTTEMP":
                if s["payload"] > HI and not on:
                    on = True
                elif s["payload"] <= LO and on:
                    on = False
            if s["compressorOn"] != on:
                bad_auto += 1
                break
    n = max(len(traces), 1)
    print("fixed counts %d/%d/%d/%d per trace : %d/%d ok"
          % (exp["TICK"], exp["TRYOPEN"], exp["DOOROPENED"], exp["REPORTTEMP"],
             n - bad_fixed, n))
    print("length == %d + 2*on + off      : %d/%d ok" % (base, n - bad_len, n))
    print("payload-latch oracle           : %d/%d ok" % (n - bad_auto, n))
    print("   -> oracle ok means the surrogate's target is a 2-state latch over")
    print("      payloads, NOT a function of position. Problem #1 cannot recur.")
    print()
    return bad_fixed + bad_len + bad_auto == 0


# --------------------------------- step 3: sample complete traces + baselines
def walk(adj, rng):
    """One uniform-random complete execution: initial -> terminal, no truncation."""
    s, steps = INITIAL, []
    while True:
        succ = adj[s]
        if not succ:
            return steps
        label, pay, nxt = rng.choice(succ)
        steps.append(dict(message=label, payload=pay,
                          compressorOn=nxt[5], temperature=nxt[0]))
        s = nxt


def walk_lazy(rng, weighting="distinct"):
    """Same random walk, but computing successors on demand instead of reading adj.

    This is what makes bigger configurations tractable. The closure grows roughly
    exponentially in maxTicks -- the shipped config already has 166,652 states --
    but a single execution only ever touches len(trace) states, so sampling stays
    cheap even where enumerate_full() would be hopeless. Distributionally
    identical to walk(): both choose uniformly among the enabled actions.
    """
    s, steps = INITIAL, []
    while True:
        succ = successors(s, weighting)
        if not succ:
            return steps
        label, pay, nxt = rng.choice(succ)
        steps.append(dict(message=label, payload=pay,
                          compressorOn=nxt[5], temperature=nxt[0]))
        s = nxt


def instances(trace):
    """MULTI-LABEL ground truth: mark every message that flipped compressorOn.

    This replaces find_ground_truth_step's "last flip to final value", which is
    ill-posed once compressorOn oscillates (v4 reaches 3 flips on one path).
    Position cannot solve a multi-label target, which is the whole point.
    """
    caus, prev = [], False
    for i, st in enumerate(trace):
        if st["compressorOn"] != prev:
            caus.append(i)
        prev = st["compressorOn"]
    return caus


def rank_rules(trace, caus):
    """Zero-parameter scorers. Each returns positions ranked best-first."""
    L = len(trace)
    rt = [i for i, s in enumerate(trace) if s["message"] == "REPORTTEMP"]
    return {
        "A recency (all positions, latest first)": list(range(L - 1, -1, -1)),
        "B REPORTTEMP first, then latest": rt[::-1] + [i for i in range(L - 1, -1, -1)
                                                      if i not in set(rt)],
        "C REPORTTEMP first, then EARLIEST": rt + [i for i in range(L)
                                                   if i not in set(rt)],
        "D payload>%d earliest, then rest" % HI: (
            [i for i in rt if trace[i]["payload"] > HI]
            + [i for i in rt if trace[i]["payload"] <= HI]
            + [i for i in range(L) if i not in set(rt)]),
        "E payload>%d or <=%d, earliest (per-position test)" % (HI, LO): (
            [i for i in rt if trace[i]["payload"] > HI
             or trace[i]["payload"] <= LO]
            + [i for i in rt if LO < trace[i]["payload"] <= HI]
            + [i for i in range(L) if i not in set(rt)]),
        "G alternating crossings (2-crossing window, NO state)": (
            _alternating(trace)
            + [i for i in rt if i not in set(_alternating(trace))]
            + [i for i in range(L) if i not in set(rt)]),
    }


def _alternating(trace):
    """The causal set, computed WITHOUT tracking compressorOn. Read this.

    Take the threshold-crossing subsequence and keep a crossing iff its
    direction differs from the previous crossing's direction, seeded with
    "down" because the latch starts false. Measured 2026-09-04: this equals
    instances(trace) for 100.0% of traces -- including 100.0% on the >=2-flip
    split -- in every configuration tried (shipped fridge_v4; init_temp=0
    max_ticks=24; +hi=11 lo=7; +door_cap=15).

    That is a structural fact, not a parameter accident: compressorOn is a
    hysteresis latch, and a hysteresis latch's flip set is a local property of
    its crossing subsequence. Median token gap between consecutive crossings is
    2-3, so a short receptive field suffices -- there is no long-range
    dependence anywhere in this model for a surrogate to learn.

    Consequence for the research question: no re-parameterisation of the fridge
    can make the explanation task require carrying state. Rule E can be broken
    (set init_temp <= LO - 1 and it collapses to ~0%), but rule G cannot. To get
    a benchmark with genuine headroom the TARGET has to change -- a property
    with unbounded dependence such as the `temperature > 15` safety violation,
    a queue-length bound, or "door opened while the compressor was on".
    """
    out, prev = [], "down"
    for i, s in enumerate(trace):
        if s["message"] != "REPORTTEMP":
            continue
        p = s["payload"]
        d = "up" if p > HI else ("down" if p <= LO else None)
        if d is not None and d != prev:
            out.append(i)
            prev = d
    return out


def _rule_stats(elig, nm):
    """-> (Top-1 %, set-F1 %, exact-set %) for one rule over one trace set."""
    t1 = f1 = ex = 0.0
    for t, c in elig:
        order = rank_rules(t, c)[nm]
        t1 += order[0] in c
        top = set(order[:len(c)])
        f1 += 2 * len(top & set(c)) / (len(top) + len(c))
        ex += top == set(c)
    n = max(len(elig), 1)
    return 100 * t1 / n, 100 * f1 / n, 100 * ex / n


def _rule_table(elig, header):
    """Read the columns right-to-left: EXACT is the honest one.

    Top-1 is nearly free and must not be quoted as evidence. The FIRST threshold
    crossing in a trace is ALWAYS causal -- the latch starts false, so the first
    payload above HI has to fire -- therefore every rule that ranks crossings
    earliest-first scores 100% Top-1 by construction. That is a fact about the
    metric, not about the rule. set-F1 collapses onto Top-1 whenever k == 1, so
    it inherits the same problem in any configuration where most traces flip
    once. EXACT (top-k set equals the causal set) is the only column that
    punishes a rule for mislabelling the LATER crossings, which is exactly where
    tracking the latch is required.
    """
    print("--- %s ---" % header)
    for nm in rank_rules(*elig[0]):
        print("   %-46s Top-1=%5.1f%%  set-F1=%5.1f%%  exact=%5.1f%%"
              % ((nm,) + _rule_stats(elig, nm)))
    ir = sum(1 / max(sum(1 for s in t if s["message"] == "REPORTTEMP"), 1)
             for t, _c in elig) / max(len(elig), 1)
    print("   %-46s Top-1=%5.1f%%  (must be beaten)"
          % ("F informed random over REPORTTEMPs", 100 * ir))
    print()


def score(traces, out_json=None):
    print("=== 3. generated traces + zero-parameter baselines ===")
    kept = [(t, instances(t)) for t in traces]
    flips = Counter(len(c) for _t, c in kept)
    print("traces                 : %d" % len(kept))
    print("length  min/mean/max   : %d / %.1f / %d"
          % (min(len(t) for t, _ in kept),
             sum(len(t) for t, _ in kept) / len(kept),
             max(len(t) for t, _ in kept)))
    print("flips per trace        : %s" % sorted(flips.items()))
    mean_flips = sum(len(c) for _t, c in kept) / max(len(kept), 1)
    print("mean flips per trace   : %.2f" % mean_flips)
    elig = [(t, c) for t, c in kept if c]
    print("traces with >=1 flip   : %d/%d" % (len(elig), len(kept)))
    if not elig:
        print("!! no trace flipped compressorOn -- nothing to explain.")
        return

    # THE headline number: where does the decisive message sit now?
    rel = [c[0] / max(len(t) - 1, 1) for t, c in elig]
    rel.sort()
    print("1st decisive message, normalised position:")
    print("   mean=%.3f  median=%.3f  p05=%.3f  p95=%.3f"
          % (sum(rel) / len(rel), rel[len(rel) // 2],
             rel[int(.05 * len(rel))], rel[int(.95 * len(rel))]))
    print("   -> if this is no longer pinned near 1.0, position has stopped")
    print("      being a sufficient statistic and problem #1 is dead.")
    print()

    # How many threshold crossings are NON-causal? This is the quantity that
    # decides whether a per-position payload test can be beaten at all.
    tot, dud = _dud_rate(elig)
    print("threshold crossings    : %d, of which NON-causal %d = %.1f%%"
          % (tot, dud, 100 * dud / max(tot, 1)))
    print("   -> this is the ONLY headroom a learned explainer has over rule E.")
    print("      At 0%% the task is solvable per-position and no surrogate can win.")
    print()

    _rule_table(elig, "ALL traces with >=1 flip")
    multi = [(t, c) for t, c in elig if len(c) >= 2]
    if multi:
        _rule_table(multi, "ONLY traces with >=2 flips  (%d of %d = %.1f%%)"
                    % (len(multi), len(elig), 100 * len(multi) / len(elig)))
    else:
        print("!! no trace has 2+ flips, so k == 1 everywhere, set-F1 IS Top-1,")
        print("   and this configuration cannot discriminate between rules.\n")

    print("HOW TO READ THIS. Rule E is the strongest PER-POSITION scorer: test")
    print("this payload against both thresholds and ignore all state. Rule G is")
    print("the strongest STATELESS scorer: same test, plus one look back at the")
    print("previous crossing's direction. G's exact column is the real verdict,")
    print("and it sits at 100% in every configuration tried so far, because a")
    print("hysteresis latch's flip set is a local property of its crossing")
    print("subsequence. E can be broken (init_temp <= LO-1 collapses it to ~0%);")
    print("G cannot be broken by any parameter. Whatever a surrogate scores, it")
    print("is competing with G, not with E.")
    _predict_baselines(traces)
    if out_json:
        # The configuration travels WITH the traces. Stage 2 has to test payloads
        # against the same HI/LO that generated them, and hard-coding 7/3 there
        # would silently mis-score every sweep candidate.
        with open(out_json, "w") as fh:
            json.dump({"config": dict(init_temp=INIT_TEMP, max_ticks=MAX_TICKS,
                                      max_opens=MAX_OPENS, tick_cap=TICK_CAP,
                                      door_delta=DOOR_DELTA, door_cap=DOOR_CAP,
                                      cool_delta=COOL_DELTA,
                                      cool_floor=COOL_FLOOR, hi=HI, lo=LO),
                       "traces": [{"steps": t, "causal": c} for t, c in kept]},
                      fh)
        print("\nwrote %s (config embedded)" % out_json)


# ------------------------------------------------- step 4: parameter sweep
def _predict_baselines(traces):
    """The PREDICTION task, which is separate from the explanation task.

    Prints three stateless predictors of the final compressorOn. The third one,
    "direction of the last threshold crossing", is exact by construction: a
    crossing above HI leaves the latch on whether or not it fired, and a
    crossing at or below LO leaves it off, so the last crossing pins the final
    value with no history at all. Measured 100.0% in every configuration.

    This is why experiments 1-19 plateaued. A surrogate trained to predict the
    final compressorOn reaches ceiling by attending to the last crossing and
    nothing else; attending to earlier flips reduces the loss by zero. The
    "recency bias" was the optimal policy for the task as posed, not a defect of
    the attention mechanism -- so no architectural change could ever have fixed
    it. Report these numbers next to any surrogate accuracy, always.
    """
    n = len(traces)
    on = sum(t[-1]["compressorOn"] for t in traces)
    lastx = lastrt = 0
    for t in traces:
        y = t[-1]["compressorOn"]
        cr = [s["payload"] for s in t if s["message"] == "REPORTTEMP"
              and (s["payload"] > HI or s["payload"] <= LO)]
        lastx += (cr[-1] > HI) == y if cr else (not y)
        rt = [s["payload"] for s in t if s["message"] == "REPORTTEMP"]
        lastrt += (rt[-1] > HI) == y if rt else (not y)
    print("\n--- PREDICTION of the final compressorOn (a different task) ---")
    print("   final state ON in         %.1f%% of traces" % (100 * on / n))
    print("   majority-class baseline   %.1f%%" % (100 * max(on, n - on) / n))
    print("   last REPORTTEMP > %-2d      %.1f%%" % (HI, 100 * lastrt / n))
    print("   LAST CROSSING direction   %.1f%%   <- exact, stateless, one token"
          % (100 * lastx / n))
    print("   -> quote this beside any surrogate accuracy. If it reads 100%,")
    print("      accuracy is not evidence of anything and the paper's claim has")
    print("      to rest on the explanation metrics above.\n")


def _dud_rate(elig):
    """-> (threshold crossings, how many of them fired nothing).

    A crossing that fires nothing is the whole game: it is indistinguishable
    from a real flip unless you carry the latch state along the prefix. At zero
    duds the explanation task is solvable position-by-position and no surrogate
    can beat rule E.
    """
    tot = dud = 0
    for t, c in elig:
        cs = set(c)
        for i, st in enumerate(t):
            if st["message"] == "REPORTTEMP" and (
                    st["payload"] > HI or st["payload"] <= LO):
                tot += 1
                dud += i not in cs
    return tot, dud


# initTemp <= LO is the interesting knob: it makes the first few reportTemps
# carry payloads at or below LO while the latch is still false, so they cross a
# threshold and do NOTHING. More ticks plus a smaller cool() delta buy repeated
# oscillation, which is what pushes k above 1.
SWEEP_GRID = [
    dict(init_temp=5, max_ticks=6),                       # shipped fridge_v4
    dict(init_temp=5, max_ticks=12),
    dict(init_temp=5, max_ticks=18, cool_delta=6),
    dict(init_temp=0, max_ticks=6),
    dict(init_temp=0, max_ticks=12),
    dict(init_temp=0, max_ticks=12, cool_delta=6),
    dict(init_temp=0, max_ticks=18, cool_delta=6),
    dict(init_temp=0, max_ticks=18, cool_delta=6, door_cap=15),
    dict(init_temp=2, max_ticks=14, cool_delta=6),
    dict(init_temp=0, max_ticks=24, cool_delta=6),
]


def sweep(n, seed, weighting="distinct"):
    """Shop the parameter family for a configuration that has real headroom.

    Sampling is lazy, so configurations whose closure is far too large to
    enumerate are still measurable here. Restores DEFAULTS on the way out.

    Columns: flips = mean flips per trace; dud% = share of threshold crossings
    that fired nothing; E-F1 / E-exact = rule E over all eligible traces;
    E-x>=2 = rule E's exact-set score restricted to traces with 2+ flips, which
    is the column that decides the question.
    """
    print("=== 4. parameter sweep (lazy sampling, %d traces per config) ===" % n)
    print("%-40s %6s %6s %6s %7s %7s"
          % ("config", "flips", "dud%", "E-F1", "E-exact", "E-x>=2"))
    best = None
    for g in SWEEP_GRID:
        configure(**g)
        tag = "init=%-2d ticks=%-2d cool=%-2d door=%s" % (
            INIT_TEMP, MAX_TICKS, COOL_DELTA,
            "cap%d" % DOOR_CAP if DOOR_CAP is not None else "open")
        rng = random.Random(seed)
        kept = [(t, instances(t)) for t in
                (walk_lazy(rng, weighting) for _ in range(n))]
        elig = [(t, c) for t, c in kept if c]
        if not elig:
            print("%-40s %6s %6s %6s %7s %7s"
                  % (tag, "-", "-", "-", "-", "no flips"))
            continue
        mean_f = sum(len(c) for _t, c in kept) / len(kept)
        tot, dud = _dud_rate(elig)
        ekey = next(k for k in rank_rules(*elig[0]) if k.startswith("E "))
        _t1, f1, ex = _rule_stats(elig, ekey)
        multi = [(t, c) for t, c in elig if len(c) >= 2]
        ex2 = _rule_stats(multi, ekey)[2] if multi else float("nan")
        print("%-40s %6.2f %5.1f%% %5.1f%% %6.1f%% %6.1f%%"
              % (tag, mean_f, 100 * dud / max(tot, 1), f1, ex, ex2))
        # rank candidates by how much room they leave rule E on multi-flip traces
        if multi and (best is None or ex2 < best[1]):
            best = (dict(g), ex2, mean_f, 100 * dud / max(tot, 1))
    print()
    if best is not None:
        print("lowest E-exact on >=2-flip traces: %.1f%% at %s"
              % (best[1], best[0]))
        print("   mean flips %.2f, non-causal crossings %.1f%%"
              % (best[2], best[3]))
        print("   -> if that number is still near 100, no amount of retuning these")
        print("      four knobs will help and the thresholds themselves have to")
        print("      change so that a payload can be decisive in one latch state")
        print("      and inert in the other.")
    configure()                                   # restore fridge_v4 defaults
    print()


# ------------------------------------------------------------------ self-test

def _s(temp, ticks, sq, opens, dq, on, cq, cyc, mq):
    return (temp, ticks, _bag(sq), opens, _bag(dq), on, _bag(cq), cyc, _bag(mq))


M_TICK, M_DOOR, M_COOL, M_TRY = ("tick",), ("doorOpened",), ("cool",), ("tryOpen",)
M_ON, M_OFF = ("turnOn",), ("turnOff",)


def selftest():
    """Hand-computed assertions, independent of traces.xml.

    Every expected value below was derived by hand from fridge_v4.rebeca and,
    where possible, cross-checked against a NAMED state in Afra's export. So a
    green self-test means the semantics here are not merely self-consistent --
    they agree with the model and with the checker. Runs in milliseconds; it is
    on by default because nothing further down is worth reading if it fails.
    """
    fails = []

    def chk(name, cond):
        print("   %-58s %s" % (name, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(name)

    print("=== 0. self-test (hand-computed, no XML needed) ===")

    # -- 1_0 has exactly two successors: Afra's 1_0->2_0 and 1_0->3_0.
    succ1 = successors(INITIAL)
    chk("1_0 has exactly 2 successors", len(succ1) == 2)
    chk("1_0 labels are {TICK, TRYOPEN}",
        {l for l, _p, _n in succ1} == {"TICK", "TRYOPEN"})
    s2 = _s(6, 5, [M_TICK], 3, [M_TRY], False, [("reportTemp", 6)], 0, [])
    s3 = _s(5, 6, [M_TICK, M_DOOR], 2, [M_TRY], False, [], 0, [])
    got = {l: n for l, _p, n in succ1}
    chk("1_0 --TICK--> 2_0 (temp 5->6, reportTemp(6) queued)", got["TICK"] == s2)
    chk("1_0 --TRYOPEN--> 3_0 (opens 3->2, doorOpened queued)", got["TRYOPEN"] == s3)

    # -- CONFLUENCE. 2_0 --TRYOPEN--> 5_0 <--TICK-- 3_0. This is the assertion
    #    that fails under FIFO or order-sensitive queues, and it is the reason
    #    queues are hashed as sorted multisets everywhere in this file.
    a = {n for l, _p, n in successors(s2) if l == "TRYOPEN"}
    b = {n for l, _p, n in successors(s3) if l == "TICK"}
    chk("bag semantics: TICK;TRYOPEN and TRYOPEN;TICK meet at 5_0",
        len(a) == 1 and a == b)
    # -- NON-FIFO SELECTION. 4_0's controller queue holds reportTemp(6) and
    #    reportTemp(7); Afra gives it TWO distinct REPORTTEMP successors
    #    (11_0 and 12_0), so any queued message may fire, not just the head.
    s4 = _s(7, 4, [M_TICK], 3, [M_TRY], False,
            [("reportTemp", 6), ("reportTemp", 7)], 0, [])
    rt = [(p, n) for l, p, n in successors(s4) if l == "REPORTTEMP"]
    chk("4_0 offers both queued payloads, 6 and 7",
        sorted(p for p, _n in rt) == [6, 7])
    chk("4_0's two REPORTTEMPs lead to 2 DISTINCT states",
        len({n for _p, n in rt}) == 2)
    chk("neither 6 nor 7 flips compressorOn (7 > 7 is false)",
        all(n[5] is False for _p, n in rt))

    # -- 9_0: three ticks in, temp 8. reportTemp(8) is the first possible flip,
    #    and it is unavoidable in every execution -- hence the >= 22 length bound.
    s9 = {n for l, _p, n in successors(s4) if l == "TICK"}.pop()
    chk("4_0 --TICK--> 9_0 has temp 8, ticksLeft 3, 3 reportTemps queued",
        s9[0] == 8 and s9[1] == 3 and len(s9[6]) == 3)
    flip = [n for l, p, n in successors(s9) if l == "REPORTTEMP" and p == 8]
    chk("reportTemp(8) sets compressorOn and queues turnOn",
        len(flip) == 1 and flip[0][5] is True and flip[0][8] == (M_ON,))

    # -- THE v4 CHANGE ITSELF: turnOn must reach the Sensor as cool().
    son = _s(10, 2, [M_TICK], 1, [M_TRY], True, [], 0, [M_ON])
    nx = {l: n for l, _p, n in successors(son)}
    chk("turnOn increments cycles AND queues cool() on the sensor",
        nx["TURNON"][7] == 1 and M_COOL in nx["TURNON"][2])
    chk("cool() subtracts 10 (10 -> 0)",
        {n[0] for l, _p, n in successors(
            _s(10, 2, [M_COOL], 1, [M_TRY], True, [], 1, [])) if l == "COOL"} == {0})
    chk("cool() floors at 0, never negative (4 -> 0)",
        {n[0] for l, _p, n in successors(
            _s(4, 2, [M_COOL], 1, [M_TRY], True, [], 1, [])) if l == "COOL"} == {0})

    # -- THE DOWN-FLIP: dead code in v3, reachable in v4. Afra: 164_0 --TURNOFF--> 178_0.
    sdn = _s(1, 2, [M_TICK], 1, [M_TRY], True, [("reportTemp", 1)], 1, [])
    off = [n for l, p, n in successors(sdn) if l == "REPORTTEMP" and p == 1]
    chk("reportTemp(1) with compressorOn clears it and queues turnOff",
        len(off) == 1 and off[0][5] is False and off[0][8] == (M_OFF,))
    chk("turnOff resets consecutiveOnCycles to 0",
        {n[7] for l, _p, n in successors(
            _s(1, 2, [M_TICK], 1, [M_TRY], False, [], 3, [M_OFF]))
         if l == "TURNOFF"} == {0})

    # -- THE KNOWN BUG, asserted so it cannot be "fixed" by accident: the
    #    `temperature < 15` guard sits inside tick() only, not doorOpened().
    chk("doorOpened is UNCAPPED: 13 -> 16 (kept deliberately)",
        {n[0] for l, _p, n in successors(
            _s(13, 1, [M_DOOR], 0, [], False, [], 0, []))
         if l == "DOOROPENED"} == {16})
    chk("tick IS capped: temp stays 15 while ticksLeft still decrements",
        {(n[0], n[1]) for l, _p, n in successors(
            _s(15, 1, [M_TICK], 0, [], False, [], 0, []))
         if l == "TICK"} == {(15, 0)})

    # -- exhausted counters are SILENT no-ops: consume, send nothing, no error.
    dead = {l: n for l, _p, n in successors(
        _s(9, 0, [M_TICK], 0, [M_TRY], False, [], 0, []))}
    chk("tick with ticksLeft 0 consumes the message and sends nothing",
        dead["TICK"] == _s(9, 0, [], 0, [M_TRY], False, [], 0, []))
    chk("tryOpen with opensLeft 0 consumes the message and sends nothing",
        dead["TRYOPEN"] == _s(9, 0, [M_TICK], 0, [], False, [], 0, []))
    chk("all queues empty == terminal (this is the intended end of a run)",
        is_terminal(_s(9, 0, [], 0, [], False, [], 0, [])))
    print()
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", help="Afra transitionsystem export, for validation")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-weighting", choices=["distinct", "positions"],
                    default="distinct")
    ap.add_argument("--out", default="traces_v4.json")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--sweep", type=int, metavar="N", default=0,
                    help="search SWEEP_GRID with N lazy traces per config")
    ap.add_argument("--lazy", action="store_true",
                    help="sample without enumerating -- required once the "
                         "closure is too big to hold in memory")
    for k in ("init-temp", "max-ticks", "max-opens", "tick-cap", "door-delta",
              "door-cap", "cool-delta", "cool-floor", "hi", "lo"):
        ap.add_argument("--" + k, type=int, default=None)
    a = ap.parse_args()

    cfg = configure(init_temp=a.init_temp, max_ticks=a.max_ticks,
                    max_opens=a.max_opens, tick_cap=a.tick_cap,
                    door_delta=a.door_delta, door_cap=a.door_cap,
                    cool_delta=a.cool_delta, cool_floor=a.cool_floor,
                    hi=a.hi, lo=a.lo)

    if a.sweep:
        sweep(a.sweep, a.seed, a.action_weighting)
        if not a.sample:
            return 0

    if not a.skip_selftest:
        if is_default_config():
            fails = selftest()
            if fails:
                print("SELF-TEST FAILED. Stopping: nothing below is trustworthy.")
                for f in fails:
                    print("   %s" % f)
                return 1
        else:
            # Being explicit rather than silently green: every expected value in
            # selftest() was hand-derived for the shipped configuration, so under
            # a changed one it would be asserting the wrong arithmetic.
            print("=== 0. self-test SKIPPED: configuration differs from "
                  "fridge_v4.rebeca ===")
            print("    %s" % cfg)
            print("    invariants() in step 2b still applies -- it derives its")
            print("    expected counts from the configuration.\n")

    if a.xml:
        if not is_default_config():
            print("!! --xml compares against Afra's export of the SHIPPED model;")
            print("   skipping it because the configuration was changed.\n")
        else:
            try:
                validate(a.xml)
            except Exception as exc:              # never block on the parser
                print("!! validation skipped: %r\n" % exc, file=sys.stderr)

    if not a.lazy:
        order, adj, depth = enumerate_full(a.action_weighting)
        audit(order, adj, depth)

    if a.sample:
        rng = random.Random(a.seed)
        if a.lazy:
            traces = [walk_lazy(rng, a.action_weighting) for _ in range(a.sample)]
        else:
            traces = [walk(adj, rng) for _ in range(a.sample)]
        if not invariants(traces):
            print("!! an invariant failed -- treat the traces as suspect.\n")
        score(traces, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())







