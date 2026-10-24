# Legacy Exploratory Experiments

This folder preserves earlier exploratory work that predates the audited RW3
baseline. These files may use an older feature table, a sample-level split, or
post-processing experiments. They are retained for learning and traceability,
but they must not be presented as results from the current leakage-audited
baseline.

Current baseline files remain in `src` and `data/processed`:

- `src/build_feature_table.py`
- `src/check_reference_interval_split.py`
- `src/train_svr_baseline.py`
- `src/inspect_rw3.py`
- `data/processed/rw3_features_4p0_to_4p2_audit.csv`

The current baseline uses the first five seconds of 4.0-4.2 V-starting RW3
random-walk discharge segments, IRWST only, and a reference-interval-aware
chronological split.
