# Data dictionary

## Full-model study manifests

Each `configs/<study>.json` is a list of complete `StudyConfig` records. A configuration is identified by the first twenty hexadecimal characters of the SHA-256 hash of its sorted JSON representation. The same identifier is written to its summary and directory name. Identifiers are local reproducibility keys, not digital object identifiers.

`dt`, `duration`, and `delay` are seconds. `delay_steps=round(delay/dt)` must be an exact grid index. `M` is register size. `geometry_quantum` selects full off-diagonal geometry blocks. `geometry_initial` is delta, uniform, diagonal Gibbs or coherent Gibbs. `initial_population` is p in the pure product state [sqrt(1−p)|0>+sqrt(p)|1>]⊗6. `normalized_proposals` divides the base jump rate by the maximum configuration degree. `trace_tol` is an admissibility guard, not an integration tolerance. `store_states` controls retention of complete matter-density trajectories; all source and discrepancy histories are always stored.

## Stored arrays: trajectory.npz

| Key | Meaning |
|---|---|
| time | State-sample times, including both endpoints |
| trace_distance | Half the trace norm of the two matter-state differences |
| source_fr, source_nl | Instantaneous modular source at each state sample |
| effective_source_fr, effective_source_nl | Input that would be used by the update starting at that sample; the final sample is diagnostic only |
| geometry_population | Normalized pointer populations at state samples |
| closure_defect | Distance between the next finite-register marginal and a baseline step from the current register marginal with the same register-side input; one entry per physical step |
| geometry_configurations | Curvature configurations, each summing to 4π |
| configuration_codes, adjacency | Discrete configuration identifiers and proposal graph |
| rho_fr, rho_nl | Optional complete 64×64 matter density trajectories |

The one-step closure defect is a diagnostic evaluation and not an additional persistent model trajectory. Its source-buffer use is read-only.

## Summary statistics

`peak_early` and `area_early` use [0,0.1] s. `peak_full` and `area_full` use the full horizon. `area_active` uses [0.14,T] s for the original physical-delay studies and [0.014,0.1] s for the short-lag studies, including their refinement. Areas use the trapezoidal rule and have units seconds. Trace distances and peaks are dimensionless. Paired contrasts subtract zero delay within the same fold, mode and coherence flag before averaging. Standard deviations use the three deterministic folds; they are descriptive, not significance tests.

`first_active_source_*` is the first update time at which the effective source differs from its initialization above a relative numerical activity guard. Final sample values are excluded from this activation test. Buffer-index correctness is separately checked against direct array indexing, without an activity threshold.

## Reduced topology results

The graph engine reports root-mean-square node-amplitude difference from its zero-delay path. Product and dense quantum engines report trace distance to their own zero-delay paths. `area` integrates the relevant statistic on [0,0.4] s. The records include `model_level`, `statistic`, topology, delay audits and parameters. These quantities must not be pooled with full-model integrated discrepancy.

## Provenance and execution

`provenance/release_validation.json` checks all completed control manifests in
this integration. `integration_validation.json` records the current verification
scope and environment. `integration_files.sha256.json` covers the current source,
documentation, and results, excluding itself and ignored local build files.

The imported `notebook_execution.json`, `test_execution.json`, `environment.json`,
`kernel_equivalence.json`, `input_files.sha256.json`, `input_archive.json`,
`release_files.sha256.json`, and `original_release_validation.json` are historical
release evidence. Their paths may describe removed notebooks or files changed by
integration. `notebook_migration.json` records removed notebook hashes; recorded
MIME displays and figures are retained under `results/legacy_reference/`.
