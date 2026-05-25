# Final thesis results — auto-generated

- Generated: `2026-05-22T09:11:27`
- Repo commit: `4743c3d3e384e4f322a5231cd9c42c49ab04ea8a`
- Bootstrap replicates: `n_boot=1000`
- Configs fully covered (5/5 folds in both cohorts): `11/12`

## What's in here

- `per_config/<arm>/<pipeline>/<model>/` — per-config artefacts: pooled
  predictions, per-fold metrics, confusion matrices.
- `tables/` — LaTeX-ready table fragments (`\input{...}` from `Results.tex`)
  alongside CSV companions.
- `figures/` — PDF figures cited by `Results.tex`.
- `coverage_report.md` — which (arm, model, cohort, fold) is present.

## Aggregation

Per-fold patient predictions are pooled into a single union-of-folds
vector per (config, cohort), and metrics are computed once on that union
with stratified non-parametric bootstrap CIs (resampling within each
true-label class). This matches the methodology in Section 4.7 of the
thesis: "the union of the five disjoint per-fold test partitions covers
the full cohort exactly once".

Per-fold variance (mean ± SE across the five fold point estimates) is
reported separately in `tables/table_per_fold_variance.tex`.

## Regeneration

```bash
python scripts/build_final_results.py
```

Re-run after the gap-fill scripts finish to refresh the tree in place.
