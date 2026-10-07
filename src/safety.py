"""Stage 1b: a target that actually requires carrying state.

WHY. `compressorOn` is unusable. It is a hysteresis latch, and rule G in
pipeline (keep a threshold crossing iff its direction differs from the
previous crossing's) reproduces its flip set 100.0% exactly in every
configuration, with a median token gap of 2-3. Its final value equals the
direction of the last crossing, also 100.0%. So both the prediction task and
the explanation task have zero-parameter stateless closed forms, and no
re-parameterisation of the fridge changes that.

THE REPLACEMENT is the safety property `temperature > TICK_CAP` -- did the
fridge ever get warmer than its own tick ceiling? Three reasons it is the right
target:

  1. It is a PROPERTY, not a state variable, which is what model checkers
     actually verify. The paper's framing improves.
  2. `tick()` is guarded by `if (temperature < 15)`, so TICK can never push the
     temperature past the ceiling. Only the UNCAPPED `doorOpened()` +3 can. The
     violation is therefore a genuine interaction between two actors that never
     synchronise -- exactly the thing actor-based verification exists for.
  3. Base rate on the shipped fridge_v4 configuration is 48.8%, so the label is
     balanced with NO change to fridge_v4.rebeca. The 166,652-state enumeration
     and the 363/363 Afra cross-validation both stay valid.

GROUND TRUTH is counterfactual and needs no expert: a message is causal iff
deleting it from the processed sequence removes the violation. That is
well-defined here because temperature evolves deterministically from the
message sequence alone (+1 capped for TICK, +DOOR_DELTA for DOOROPENED,
-COOL_DELTA floored for COOL, nothing for the rest), which `temp_path` asserts
against the recorded trajectory.

The cap is what makes this hard rather than diffuse: a TICK processed while the
temperature already sits at the ceiling is a no-op, so it is NOT causal even
though it looks identical to every other TICK. Deciding which TICKs count
requires knowing the temperature at that moment, which requires accumulating
every TICK, DOOROPENED and COOL from the start of the run. There is no local
window that does it.

TWO DEFECTS FOUND WHEN THIS FILE WAS AUDITED BEFORE STAGE 2 WAS WIRED TO IT,
both now fixed here rather than downstream.

  (i) SINGLE-DELETION NECESSITY IS SILENT ON A THIRD OF THE POSITIVES. 599 of
      1631 violating traces (36.7%) are OVER-DETERMINED: no single door opening
      is necessary, because removing any one still leaves enough warming behind.
      `instances_safety` returned [] for all of them, so a third of the positive
      class had no explanation to score against. The fix is
      `responsibility()` -- the Chockler-Halpern degree of responsibility,
      r(i) = 1/(1+k) where k is the fewest OTHER door openings you must delete
      before deleting i becomes decisive. r(i)=1 recovers single-deletion
      necessity exactly, r(i)=1/2 means "critical once one other is gone", and
      r(i)=0 means i is never decisive under any contingency. It is total: every
      door opening in every violating trace gets a number. Measured values are
      exactly {0, 1/4, 1/3, 1/2, 1}. This is the same notion Chockler, Halpern
      and Kupferman used for coverage in model checking, which is the right
      literature for this paper to sit in.

      Computing it exactly is cheap because the repair predicate is MONOTONE:
      deleting more door openings can never make the fridge hotter. That is not
      obvious -- a lower temperature lets the `temperature < 15` guard fire
      again, so deletions can buy back tick increments -- but the increments can
      never overshoot: temp <= 14 implies temp+1 <= 15 <= the undeleted run's
      temperature. `check_monotone` asserts it at runtime.

 (ii) set-F1 IS MET BY A RULE THAT IGNORES THE TRACE. Answering "every door
      opening is causal", with no parameters and no look at the trace, scores
      83.0% set-F1 -- against 83.8% for the best ranked stateless rule. set-F1
      is therefore uninformative here, exactly as Top-1 was on the latch target.
      Only EXACT-SET match separates them (39.9% vs 68.8%), and even that hands
      every ranked rule the true cardinality |c| for free. Both the constant
      rule and the cardinality caveat are printed by `report` now, so the
      generous metric cannot be quoted by accident.

THE PREDICTION TASK LEAKS 76.3% THROUGH THE PAYLOAD, and that has to be said
next to any surrogate accuracy. `max REPORTTEMP payload > 20` predicts the label
76.3% of the time (majority 54.4%). It is not 100% because a REPORTTEMP is a
STALE snapshot under bag semantics and reports only happen at ticks, so a peak
reached at a door opening and then cooled away is invisible: only 56.5% of
violations are ever visible in a report. The number of COOLs leaks a further,
much weaker 60.0%. Both are printed as floors. The message TYPE sequence alone
determines the temperature exactly (payload is a redundant stale copy of it),
which is why stage 2 drops the payload channel by default for this target.

Run:  python3 safety.py --sample 3000            # shipped config
      python3 safety.py --sample 3000 --lazy --preset --out traces.json
"""

import argparse
import itertools
import random
import sys

sys.path.insert(0, ".")
import pipeline as P

# The safety ceiling is DELIBERATELY separate from TICK_CAP. If they are equal,
# tick() pins the temperature at exactly the ceiling and then ANY door opening
# breaches it -- which is why the first version of this file found "the last door
# opening" scoring 73.9%. Setting SAFE_CAP above TICK_CAP means a breach needs
# door openings to ACCUMULATE faster than cool() can undo them, which is a
# property of the interleaving rather than of a counter.
SAFE_CAP = None          # set by set_cap(); defaults to TICK_CAP + DOOR_DELTA


# Found by grid search over (max_opens, max_ticks, SAFE_CAP), scoring
# |50 - violation rate| + best-stateless-rule exact match. At this setting the
# label is balanced at 51.7%, the causal set averages 4.5 of the 6 door
# openings, and the best zero-parameter stateless rule reaches 81.9% set-F1 /
# 70.5% exact -- so for the first time in this project there is a 30-point gap
# for a learned explainer to earn, instead of rule G's flat 100%.
PRESET = dict(max_opens=6, max_ticks=12)
PRESET_CAP = 20


def set_cap(v=None):
    global SAFE_CAP
    SAFE_CAP = P.TICK_CAP + P.DOOR_DELTA if v is None else v
    return SAFE_CAP


def step_temp(temp, label):
    """One temperature update, mirroring successors() in pipeline exactly.

    NOT sufficient on its own for TICK -- see temp_path, which carries the tick
    budget. Kept separate only for DOOROPENED and COOL, which are unconditional.
    """
    if label == "DOOROPENED":
        t = temp + P.DOOR_DELTA
        return t if P.DOOR_CAP is None else min(P.DOOR_CAP, t)
    if label == "COOL":
        return max(P.COOL_FLOOR, temp - P.COOL_DELTA)
    return temp


def temp_path(trace, skip=None):
    """Replay the temperature from the message sequence, optionally dropping some.

    Returns the peak temperature reached. `skip` is an index, a set of indices,
    or None.

    THE TICK BUDGET MATTERS. `tick()` is `if (ticksLeft > 0) { ...; self.tick() }`
    so a trace contains MAX_TICKS + 1 TICK messages and the LAST one is a no-op:
    it is processed, it appears in the log, and it changes nothing. A fold that
    increments on every TICK disagrees with the simulator on 83% of traces --
    which is exactly what check_replay caught on the first run of this file.

    `skip` is restricted to DOOROPENED by every caller, deliberately. Deleting
    a TICK is not a valid counterfactual: TICKs are chained (each one sends the
    next) and bounded by a budget, so removing one from the log would shift the
    budget and promote the trailing no-op into a real increment. Door openings are
    the only contingent warming events -- Door decides them independently, and
    "the door opened one fewer time" is a question the model can actually answer.
    """
    if skip is None:
        skip = ()
    elif isinstance(skip, int):
        skip = (skip,)
    temp, peak, used = P.INIT_TEMP, P.INIT_TEMP, 0
    for i, st in enumerate(trace):
        if i in skip:
            continue
        if st["message"] == "TICK":
            if used < P.MAX_TICKS:
                used += 1
                if temp < P.TICK_CAP:
                    temp += 1
        else:
            temp = step_temp(temp, st["message"])
        peak = max(peak, temp)
    return peak


def doors(trace):
    return [i for i, st in enumerate(trace) if st["message"] == "DOOROPENED"]


def repair_lattice(trace):
    """{subset of door openings -> does deleting it remove the violation}.

    Exact and exhaustive. 2**MAX_OPENS subsets, 64 at the preset, so this costs
    less than the single-deletion loop it replaces did per trace. Monotonicity
    (see check_monotone) means the answer is an up-set, so minimal repairs are
    well defined and `responsibility` below is exact rather than a search.
    """
    d = doors(trace)
    if len(d) > 16:
        raise ValueError("2**%d subsets is too many; use --max-opens <= 16" % len(d))
    return {frozenset(S): temp_path(trace, frozenset(S)) <= SAFE_CAP
            for k in range(len(d) + 1)
            for S in itertools.combinations(d, k)}


def responsibility(trace, lat=None):
    """Chockler-Halpern degree of responsibility for each door opening.

    r(i) = 1 / (1 + k), where k is the size of the smallest CONTINGENCY S -- a
    set of other door openings -- such that deleting S alone does NOT repair the
    violation but deleting S together with i DOES. So i is decisive once k other
    openings are out of the way. r(i) = 1 means i is necessary on its own (that
    is exactly instances_safety), r(i) = 0 means no contingency ever makes i
    decisive.

    Why this and not the single-deletion set: it is TOTAL. 36.7% of violating
    traces have no necessary opening at all, and for those the single-deletion
    ground truth is the empty set, i.e. no explanation to score an explainer
    against. Responsibility gives every opening in every violating trace a
    number, and it degrades gracefully to necessity when necessity exists.
    """
    d = doors(trace)
    lat = repair_lattice(trace) if lat is None else lat
    if lat[frozenset()]:                      # no violation: nothing to blame
        return {i: 0.0 for i in d}
    out = {}
    for i in d:
        others = [j for j in d if j != i]
        r = 0.0
        for k in range(len(others) + 1):
            hit = any(not lat[frozenset(S)] and lat[frozenset(S) | {i}]
                      for S in itertools.combinations(others, k))
            if hit:
                r = 1.0 / (1 + k)
                break
        out[i] = r
    return out


def min_repair(trace, lat=None):
    """-> (one minimum-cardinality repair set, how many are tied at that size).

    The sharpest form of the question: what is the SMALLEST set of door openings
    whose absence would have kept the fridge safe. Sizes measured at the preset:
    1 in 63%, 2 in 32%, 3 in 5%, 4 once. The tie count matters -- it is usually
    above 1, so "the" minimal repair is not unique and any metric that demands a
    single set has to say which one it means.
    """
    lat = repair_lattice(trace) if lat is None else lat
    ok = [S for S, good in lat.items() if good and S]
    if not ok:
        return frozenset(), 0
    m = min(len(S) for S in ok)
    tied = [S for S in ok if len(S) == m]
    return sorted(tied, key=sorted)[0], len(tied)


def check_monotone(traces, n=60):
    """Deleting MORE door openings must never make the fridge hotter.

    Not obvious: a deletion lowers the temperature, which can re-enable the
    `temperature < 15` guard and buy back tick increments. It can never overshoot,
    because temp <= 14 implies temp + 1 <= 15 <= the undeleted temperature, so the
    pointwise ordering survives every message type. Asserted rather than argued.
    """
    bad = 0
    for t in traces[:n]:
        d = doors(t)
        base = temp_path(t)
        for i in d:
            if temp_path(t, i) > base:
                bad += 1
            for j in d:
                if j != i and temp_path(t, {i, j}) > temp_path(t, i):
                    bad += 1
    print("monotonicity of deletion: %s (checked %d traces)"
          % ("ok" if not bad else "VIOLATED %d times" % bad, min(n, len(traces))))
    return bad == 0


def check_replay(traces):
    """The fold above must reproduce the trajectory the simulator recorded."""
    bad = 0
    for t in traces:
        temp, used = P.INIT_TEMP, 0
        for st in t:
            if st["message"] == "TICK":
                if used < P.MAX_TICKS:
                    used += 1
                    if temp < P.TICK_CAP:
                        temp += 1
            else:
                temp = step_temp(temp, st["message"])
            if temp != st["temperature"]:
                bad += 1
                break
    print("temperature replay      : %d/%d ok" % (len(traces) - bad, len(traces)))
    return bad == 0


def violated(trace):
    return temp_path(trace) > SAFE_CAP


def instances_safety(trace):
    """Counterfactually necessary DOOROPENEDs: delete one, does the breach survive?

    Equal by construction to {i : responsibility(trace)[i] == 1.0}, and kept as a
    separate O(#doors) loop only because it is called on every trace while the
    2**#doors lattice is not. `report` asserts the two agree.

    EMPTY ON 36.7% OF VIOLATIONS at the preset, because those violations are
    over-determined -- see responsibility(), which is the total replacement.

    Deleting a COOL can only make the fridge warmer, so a COOL is never necessary
    FOR a violation. TICKs are excluded for the budget reason in temp_path.
    """
    if not violated(trace):
        return []
    return [i for i, st in enumerate(trace)
            if st["message"] == "DOOROPENED"
            and temp_path(trace, skip=i) <= SAFE_CAP]


WARM = ("TICK", "DOOROPENED")


def first_breach(trace):
    """Index of the message that first pushes the temperature past the ceiling.

    Always a DOOROPENED: tick() is guarded by `if (temperature < 15)` so a TICK
    can reach the ceiling but never pass it. STATEFUL by necessity -- it folds
    the whole prefix. This is also a second, sharper target than the binary
    label: "which door opening was the one that broke it" is a single position,
    and picking it out requires knowing the running temperature.
    """
    temp, used = P.INIT_TEMP, 0
    for i, st in enumerate(trace):
        if st["message"] == "TICK":
            if used < P.MAX_TICKS:
                used += 1
                if temp < P.TICK_CAP:
                    temp += 1
        else:
            temp = step_temp(temp, st["message"])
        if temp > SAFE_CAP:
            return i
    return None


def rank_rules(trace, _caus=None):
    """Zero-parameter scorers for the safety target, ranked best-first."""
    L = len(trace)
    door = [i for i, s in enumerate(trace) if s["message"] == "DOOROPENED"]
    tick = [i for i, s in enumerate(trace) if s["message"] == "TICK"]
    rest = [i for i in range(L) if s_not_warm(trace, i)]
    fb = first_breach(trace)
    pre = [i for i in door if fb is not None and i <= fb]
    return {
        "A recency (all positions, latest first)": list(range(L - 1, -1, -1)),
        "B DOOROPENED, LATEST first": door[::-1] + tick[::-1] + rest,
        "C DOOROPENED, earliest first": door + tick + rest,
        "D TICK first, then DOOROPENED": tick + door + rest,
        "S* stateful: DOOROPENEDs up to 1st breach": (
            pre + [i for i in door if i not in set(pre)] + tick + rest),
    }


def s_not_warm(trace, i):
    return trace[i]["message"] not in WARM


def set_rules(trace):
    """Cardinality-FREE set predictions -> {name: set of positions}.

    `stats` below hands every ranked rule the true |causal set|, which is a large
    gift. These rules do not get it: they have to decide how many openings to
    name. The constant rule is the one that matters -- it looks at nothing at all,
    and it is the reason set-F1 must never be the headline number here.
    """
    d = doors(trace)
    fb = first_breach(trace)
    return {
        "K1 every DOOROPENED (looks at nothing)": set(d),
        "K2 DOOROPENED up to the first breach": {i for i in d
                                                 if fb is not None and i <= fb},
        "K3 the last DOOROPENED only": {d[-1]} if d else set(),
    }


def set_stats(elig, nm):
    """-> (precision, recall, F1, exact) for a cardinality-free set rule."""
    p = r = f = e = 0.0
    for t, c in elig:
        S, C = set_rules(t)[nm], set(c)
        inter = len(S & C)
        p += inter / max(len(S), 1)
        r += inter / max(len(C), 1)
        f += 2 * inter / max(len(S) + len(C), 1)
        e += S == C
    n = max(len(elig), 1)
    return 100 * p / n, 100 * r / n, 100 * f / n, 100 * e / n


def resp_stats(viol, nm):
    """How much blame does a rule's FIRST pick carry?

    Defined on every violating trace, including the 36.7% where no opening is
    necessary and the exact-set metrics have nothing to compare against. The
    floor is the mean responsibility of a random opening; the ceiling is the mean
    of the per-trace maximum. A rule that scores at the floor is guessing.
    """
    got = best = flat = 0.0
    for t, r in viol:
        d = doors(t)
        if not d:
            continue
        order = [i for i in rank_rules(t)[nm] if i in set(d)]
        got += r[order[0]] if order else 0.0
        best += max(r.values())
        flat += sum(r.values()) / len(d)
    n = max(len(viol), 1)
    return 100 * got / n, 100 * flat / n, 100 * best / n


def leak_baselines(traces):
    """What the LABEL can be had for, with no state and no training.

    Mandatory alongside any surrogate accuracy. On the latch target the
    equivalent number was 100.0%, which is what invalidated nineteen
    experiments; here it is 76.3%, which is a floor rather than a ceiling -- but
    it is a high floor, and it comes entirely from the payload channel.
    """
    n = len(traces)
    y = [violated(t) for t in traces]
    pos = sum(y)

    def acc(pred):
        return 100 * sum(p == q for p, q in zip(pred, y)) / max(n, 1)

    mx = [max([s["payload"] for s in t if s["message"] == "REPORTTEMP"] or [0])
          for t in traces]
    nc = [sum(1 for s in t if s["message"] == "COOL") for t in traces]
    bt = max(range(0, 40), key=lambda th: acc([m > th for m in mx]))
    bc = max(range(0, 8), key=lambda th: acc([c <= th for c in nc]))
    seen = sum(1 for t, v in zip(traces, y)
               if v and any(s["payload"] is not None and s["payload"] > SAFE_CAP
                            for s in t))
    print("\n--- what the LABEL leaks to a stateless reader (print this next to")
    print("    any surrogate accuracy, always) ---")
    print("   majority class                    %.1f%%" % (100 * max(pos, n - pos) / n))
    print("   #COOL <= %-2d (bag of counts)       %.1f%%" % (bc, acc([c <= bc for c in nc])))
    print("   max REPORTTEMP payload > %-2d       %.1f%%   <- the payload channel"
          % (bt, acc([m > bt for m in mx])))
    print("   a report literally shows the breach in %.1f%% of violations, so the"
          % (100 * seen / max(pos, 1)))
    print("   payload rule cannot reach 100%: reports are stale snapshots taken")
    print("   only at ticks, and a peak that is cooled away between two ticks is")
    print("   never reported. The message TYPE sequence alone fixes the")
    print("   temperature exactly, so stage 2 drops the payload by default.")


def stats(elig, nm):
    """-> (Top-1 %, set-F1 %, exact-set %). Same metric as pipeline."""
    t1 = f1 = ex = 0.0
    for t, c in elig:
        order = rank_rules(t)[nm]
        t1 += order[0] in c
        top = set(order[:len(c)])
        f1 += 2 * len(top & set(c)) / (len(top) + len(c))
        ex += top == set(c)
    n = max(len(elig), 1)
    return 100 * t1 / n, 100 * f1 / n, 100 * ex / n


def report(traces, out_json=None):
    check_replay(traces)
    check_monotone(traces)
    viol = [t for t in traces if violated(t)]
    print("safety violations       : %d/%d = %.1f%%  (balanced target if ~50%%)"
          % (len(viol), len(traces), 100 * len(viol) / len(traces)))
    print("peak temperature        : max %d over %d traces (ceiling %d)"
          % (max(temp_path(t) for t in traces), len(traces), SAFE_CAP))
    if not viol:
        print("!! nothing violates -- raise max_opens or door_delta.")
        return
    elig = [(t, instances_safety(t)) for t in viol]
    elig = [(t, c) for t, c in elig if c]
    print("over-determined violations: %d/%d = %.1f%% have NO necessary opening"
          % (len(viol) - len(elig), len(viol),
             100 * (len(viol) - len(elig)) / len(viol)))
    print("   -> single-deletion necessity is SILENT on those. responsibility()")
    print("      below is the total replacement; it is what stage 2 scores against.")
    sizes = [len(c) for _t, c in elig]
    warm = [sum(1 for s in t if s["message"] in WARM) for t, _c in elig]
    print("causal set size         : mean %.1f, min %d, max %d"
          % (sum(sizes) / len(sizes), min(sizes), max(sizes)))
    print("warming messages/trace  : mean %.1f, of which causal %.1f%%"
          % (sum(warm) / len(warm),
             100 * sum(sizes) / max(sum(warm), 1)))
    print("   -> the complement are NO-OP warming messages: TICKs processed")
    print("      while the temperature already sat at the ceiling. Telling them")
    print("      apart is what forces a model to accumulate the whole prefix.")
    print()
    print("--- RANKED rules, each HANDED the true |causal set| (generous) ---")
    for nm in rank_rules(elig[0][0]):
        print("   %-46s Top-1=%5.1f%%  set-F1=%5.1f%%  exact=%5.1f%%"
              % ((nm,) + stats(elig, nm)))
    ir = sum(1 / max(w, 1) for w in warm) / len(warm)
    print("   %-46s Top-1=%5.1f%%  (must be beaten)"
          % ("R informed random over warming messages", 100 * ir))
    print()
    print("--- CARDINALITY-FREE set rules, which must decide how many to name ---")
    print("   %-46s %5s %5s %5s %5s" % ("", "prec", "rec", "F1", "exact"))
    for nm in set_rules(elig[0][0]):
        print("   %-46s %4.1f%% %4.1f%% %4.1f%% %4.1f%%" % ((nm,) + set_stats(elig, nm)))
    print("   -> K1 looks at NOTHING and still reaches ~83% F1, because |causal|")
    print("      averages 4.5 of 6 openings. set-F1 is therefore not a result on")
    print("      this benchmark; exact-set is, and K1's exact rate is the floor.")
    pre = sum(1 for t, c in elig if c == doors(t)[:len(c)])
    print("   the causal set is exactly a PREFIX of the door openings on %.1f%% of"
          % (100 * pre / len(elig)))
    print("   traces, which is exactly rule C's and S*'s exact score. Once the")
    print("   CARDINALITY is given, the identities come for free -- so the stateful")
    print("   rule buys nothing over the stateless one here, and the honest claim is")
    print("   that NEITHER exceeds %.1f%% while the trace-ignoring floor is %.1f%%."
          % (100 * pre / len(elig), set_stats(elig, "K1 every DOOROPENED (looks at nothing)")[3]))
    print("   The open %.1f%% are the traces where the necessary openings are NOT the"
          % (100 - 100 * pre / len(elig)))
    print("   earliest ones. That residual is the benchmark.")
    print()
    print("--- RESPONSIBILITY, defined on all %d violating traces ---" % len(viol))
    rs = [(t, responsibility(t)) for t in viol]
    agreed = sum(1 for t, r in rs
                 if {i for i, v in r.items() if v == 1.0} == set(instances_safety(t)))
    print("   {r == 1} equals the single-deletion set on %d/%d traces  <- must be all"
          % (agreed, len(rs)))
    hist = {}
    for _t, r in rs:
        for v in r.values():
            hist[round(v, 3)] = hist.get(round(v, 3), 0) + 1
    print("   responsibility histogram over all door openings: %s"
          % sorted(hist.items()))
    mr = [min_repair(t) for t in viol]
    msz = {}
    for S, k in mr:
        msz[len(S)] = msz.get(len(S), 0) + 1
    print("   smallest repair size    : %s" % sorted(msz.items()))
    print("   tied smallest repairs   : mean %.1f (so 'the' minimal repair is not"
          % (sum(k for _S, k in mr) / len(mr)))
    print("                             unique -- any single-set metric must say")
    print("                             which one it means)")
    print("   %-46s %6s %6s %6s" % ("blame carried by the rule's first pick",
                                    "rule", "random", "best"))
    for nm in rank_rules(viol[0]):
        print("   %-46s %5.1f%% %5.1f%% %5.1f%%" % ((nm,) + resp_stats(rs, nm)))
    print()
    print("HOW TO READ THIS. S* is the only rule that folds the prefix, so the")
    print("gap between S* and the best stateless rule is the headroom a learned")
    print("explainer has. Unlike rule G on the latch target, no stateless rule")
    print("here can reach 100%: the no-op TICKs are indistinguishable from real")
    print("ones without knowing the running temperature.")
    leak_baselines(traces)
    breach_task(viol)
    if out_json:
        import json
        with open(out_json, "w") as fh:
            json.dump({"config": dict(init_temp=P.INIT_TEMP, max_ticks=P.MAX_TICKS,
                                      max_opens=P.MAX_OPENS, tick_cap=P.TICK_CAP,
                                      door_delta=P.DOOR_DELTA, door_cap=P.DOOR_CAP,
                                      cool_delta=P.COOL_DELTA,
                                      cool_floor=P.COOL_FLOOR, hi=P.HI, lo=P.LO,
                                      safe_cap=SAFE_CAP,
                                      target="temperature>%d" % SAFE_CAP),
                       "traces": [_dump(t) for t in traces]}, fh)
        print("\nwrote %s (config embedded, label = violation)" % out_json)


def _dump(t):
    """One trace as stage 2 consumes it.

    `causal` is single-deletion necessity, kept because every number reported so
    far is against it. `resp` is the total ground truth -- responsibility for
    every door opening, present even on the 36.7% of violations where `causal` is
    empty. `min_repair` is the sharpest form and `n_min_repair` says how many sets
    tie for it, so a downstream metric cannot silently pretend it is unique.
    """
    v = violated(t)
    r = responsibility(t) if v else {}
    S, k = min_repair(t) if v else (frozenset(), 0)
    return {"steps": t, "label": v, "causal": instances_safety(t),
            "resp": sorted([i, round(x, 6)] for i, x in r.items()),
            "min_repair": sorted(S), "n_min_repair": k,
            "first_breach": first_breach(t) if v else None,
            "peak": temp_path(t)}


def breach_task(viol):
    """The sharper, single-position target: WHICH door opening broke it?

    Reported separately because it is a clean argmax task, directly comparable
    to the old "which message flipped compressorOn" framing, but with no
    stateless closed form. The baselines are the only three position-only
    answers available: the first door opening, the last one, or a coin toss
    among them.
    """
    nth, first, last, tot = {}, 0, 0, 0
    for t in viol:
        door = [i for i, s in enumerate(t) if s["message"] == "DOOROPENED"]
        fb = first_breach(t)
        if fb is None or fb not in door:
            continue
        k = door.index(fb)
        nth[k + 1] = nth.get(k + 1, 0) + 1
        first += k == 0
        last += k == len(door) - 1
        tot += 1
    if not tot:
        return
    print("\n--- sharper target: WHICH door opening caused the breach ---")
    print("   breach is the k-th door opening: %s"
          % sorted(nth.items()))
    print("   baseline 'the FIRST door opening'  %.1f%%" % (100 * first / tot))
    print("   baseline 'the LAST door opening'   %.1f%%" % (100 * last / tot))
    print("   baseline random among openings     %.1f%%"
          % (100 * sum(1 / len([i for i, s in enumerate(t)
                                if s["message"] == "DOOROPENED"])
                       for t in viol) / len(viol)))
    print("   -> a stateless model has to guess; the answer depends on the")
    print("      temperature at that instant, i.e. on the whole prefix.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lazy", action="store_true",
                    help="sample without enumerating (needed above max_ticks~8)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--safe-cap", type=int, default=None)
    ap.add_argument("--preset", action="store_true",
                    help="the grid-searched hard configuration (see PRESET)")
    for k in ("init-temp", "max-ticks", "max-opens", "tick-cap", "door-delta",
              "door-cap", "cool-delta", "cool-floor", "hi", "lo"):
        ap.add_argument("--" + k, type=int, default=None)
    a = ap.parse_args()
    if a.preset:
        for k, v in PRESET.items():
            if getattr(a, k) is None:
                setattr(a, k, v)
        if a.safe_cap is None:
            a.safe_cap = PRESET_CAP
    cfg = P.configure(**{k: getattr(a, k) for k in
                         ("init_temp", "max_ticks", "max_opens", "tick_cap",
                          "door_delta", "door_cap", "cool_delta", "cool_floor",
                          "hi", "lo")})
    set_cap(a.safe_cap)
    print("=== safety target: temperature > %d  (tick cap %d) ==="
          % (SAFE_CAP, P.TICK_CAP))
    print("config: %s\n" % cfg)
    rng = random.Random(a.seed)
    if a.lazy:
        traces = [P.walk_lazy(rng) for _ in range(a.sample)]
    else:
        order, adj, depth = P.enumerate_full()
        print("state space: %d states, max depth %d\n" % (len(order), depth))
        traces = [P.walk(adj, rng) for _ in range(a.sample)]
    report(traces, a.out)


if __name__ == "__main__":
    main()
