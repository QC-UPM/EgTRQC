# Benchmark protocols and interpretation

## Historical reference protocol

The extracted `egtrqc.notebook_kernel` preserves the supplied numerical kernels
and historical scheduler. `egtrqc.reference_workflow.run_reference_suite` executes
that protocol with either `smoke` (M=2, 0.04 s) or `reference` (M=10,20,40,
three folds, 2 s) parameters. The original delay indices are 0,1,2,5 at dt=0.02 s.
Recorded displays are exported as JSON and figures as PDF under
`results/legacy_reference/`. Smoke runs check execution and never replace paper data.

## Physical-step controlled protocol

`StudyConfig` defines an integer physical time grid. At each sample t_k, the
controller computes J_k from the current state and observes
J_eff,k = J_max(k-d,0). The buffer prehistory is J_0. The models evolve with this
input held fixed during the step; the buffer then advances once with J_k.
The terminal sample records sources but performs no state or buffer advance.
Frozen controls keep only the classical feedback input fixed; quantum state
updates, dephasing, and geometry dynamics continue.

The full register and matter-only closure use matched physical parameters and
initial matter states. Their main discrepancy is half the trace norm of the
matter density-matrix difference. The closure defect uses a shadow step from
the same matter input and applied source; it is a separate statistic.

For five delay steps at dt=0.02 s, the first changed input is at 0.12 s and its
first possible output effect is at 0.14 s. Thus the historical early interval
[0,0.10] s cannot demonstrate an evolving-feedback advantage under this rule.
The common active interval is [0.14,0.40] s, extended to 2 s in the long study.
The short-horizon refinement uses an active start of 0.014 s.
Physical delay stays fixed when dt is refined; delay_steps changes accordingly.

## Model levels and limits

The full experiment retains six matter qubits and an explicit finite geometry
register, with pointer-classical or coherent geometry. Reduced graph, product,
and dense engines test topology sensitivity within each approximation. Their
reference is their own zero-delay trajectory; they are not substitutes for the
full-register versus nonlinear-closure comparison.

The PDFs describe policy- and window-dependent responses, not a universal benefit
of delay. Three deterministic folds do not establish a continuum limit, hardware
validation, or an independent statistical sample for every cached execution.
Register support constraints and proposal-degree normalization must be checked
before interpreting register-size changes.

## Reproducibility and provenance

Configuration hashes identify result directories. JSON summaries contain audits;
NPZ files contain raw arrays; report generation reads those arrays and indices.
`--recompute` replaces cached numerical results explicitly. Validation checks
identities, sample counts, delay indices, buffer advance counts, trace, positivity,
and geometry curvature constraints. Checksums verify bytes, not scientific validity.

`provenance/release_files.sha256.json` is original ZIP evidence and includes files
removed or edited during integration. Use `python scripts/verify_files.py` with the
integration manifest instead. Original environment and execution records describe
the supplied release; they are not claims about this integration's test execution.
