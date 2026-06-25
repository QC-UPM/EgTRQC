# Dense Delay Analysis Report

## Objective

This analysis evaluates how the pure-delay feedback parameter modifies the behavior of the reduced dense-state OCTA model.
The goal is to inspect the delay dependence of the modular source, effective curvature response, and state-space diagnostics while preserving audited causal semantics.

## Configuration

- Delays analysed: 0, 1, 2, 5
- Total time: 0.5
- Time step: 0.05
- Seed: 7
- Model family: reduced dense OCTA model on graph balls.

## Generated Artifacts

- source_norm_vs_time.html
- curvature_norm_vs_time.html
- purity_vs_time.html
- entropy_vs_time.html
- trace_distance_vs_time.html
- summary_metrics.html
- dense_delay_timeseries.csv
- dense_delay_summary.csv
- manifest.json

## Summary Table

| delay_steps | final_trace_distance_to_reference | max_trace_distance_to_reference | final_purity | final_entropy | final_source_norm | final_curvature_norm |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.984375 | 0.992157 | 0.015625 | 4.158883 | 22.966304 | 46.870009 |
| 1 | 0.984375 | 0.992157 | 0.015625 | 4.158883 | 22.966304 | 46.870009 |
| 2 | 0.984375 | 0.992157 | 0.015625 | 4.158883 | 22.966304 | 46.870009 |
| 5 | 0.984375 | 0.992157 | 0.015625 | 4.158883 | 22.966304 | 46.870009 |

## Result Assessment

The smallest final trace distance to the reference state is obtained at `delay_steps = 0` with value `0.984375`.
The largest final trace distance is obtained at `delay_steps = 1` with value `0.984375`.
In this run the delay sweep is effectively degenerate at the summary-metric level: the final trace-distance and final-entropy spans across delays are below numerical relevance, so the present reduced dense model is not yet expressing a visible delay dependence for these settings.
The purity and entropy plots are especially useful for distinguishing genuine dynamical separation from changes dominated by local mixing.

## Interpretation Notes

- If a delay value lowers both the final trace distance and the peak trace distance, it is a stronger candidate for improved alignment with the reference dynamics.
- If entropy rises rapidly while the trace distance contracts, the apparent agreement may be partially explained by mixing rather than by detailed dynamical agreement.
- Source-norm and curvature-norm trajectories indicate whether the delay acts mainly on the forcing term or on the downstream geometric response.

## Limitations

This report is generated from the reduced dense-state OCTA model and is intended as a controlled PoC rather than a full high-fidelity production workflow.
The dense model is designed to preserve the core delay semantics while remaining computationally controlled and reproducible.

## Next Step

The next recommended step is to enrich the reduced model and expand the sweep protocol while preserving the same delay-buffer and audit contracts.