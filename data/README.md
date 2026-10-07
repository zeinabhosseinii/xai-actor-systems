# Data

The 3,000-trace dataset (`traces_v5_safety.json`, ~10 MB) is not stored in git.
Regenerate it from the repository root:

    python3 src/safety.py --lazy --preset --sample 3000 --out data/traces_v5_safety.json

Then train / evaluate the surrogate:

    python3 src/surrogate.py --traces data/traces_v5_safety.json
    python3 src/surrogate.py --traces data/traces_v5_safety.json --read hidden   # control
    python3 src/surrogate.py --traces data/traces_v5_safety.json --keep-payload  # leak check

Note: `src/safety.py` imports `pipeline`, so run scripts from inside `src/`
or add `src/` to PYTHONPATH.
