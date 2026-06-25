Architecture
============

The architecture is intentionally centered on the delay contract.
Every other part of the repository is downstream of that decision.

Causal Contract
---------------

EgTRQC separates the feedback problem into four steps:

1. compute the instantaneous source from the current state,
2. observe the effective delayed source from the history,
3. advance the physical state using the effective source,
4. advance the delay buffer exactly once with the instantaneous source.

This keeps the implementation equivalent to the mathematical pure-delay rule.

Simulation Loop
---------------

.. graphviz::

   digraph loop {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f7fbf2"];
       a [label="Current State"];
       b [label="Compute Source J(t_n)"];
       c [label="Read J_eff(t_n)"];
       d [label="Compute Curvature / Response"];
       e [label="Advance Physical State"];
       f [label="Store Step Record"];
       g [label="Advance Delay Buffer"];

       a -> b -> c -> d -> e -> f -> g;
   }

Key Components
--------------

``DelayBuffer``
   Stores the physical source history and exposes observation-only and mutation-only operations.

``DelayAwareSimulator``
   Owns the time loop, seeds the RNG, collects records, and enforces the single-advance-per-step protocol.

``diagnostics``
   Reconstructs expected behavior from the physical log and checks exact compliance.

``analysis``
   Converts simulation outputs into tables, plots, manifests, and reports.

Data Flow
---------

.. graphviz::

   digraph dataflow {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#fff8e8"];
       sim [label="DelayAwareSimulator"];
       result [label="SimulationResult"];
       json [label="JSON Artifacts"];
       csv [label="CSV Tables"];
       plotly [label="Plotly HTML"];
       report [label="Markdown Report"];
       audits [label="Audit Reports"];

       sim -> result;
       result -> json;
       result -> csv;
       result -> plotly;
       result -> report;
       result -> audits;
   }

Extension Rules
---------------

The safest way to extend the PoC is:

- keep ``DelayBuffer`` unchanged unless the mathematical contract changes,
- keep observation and mutation separate,
- add model fidelity by implementing the same model interface,
- keep artifact generation outside the physical evolution loop.
