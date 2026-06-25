Overview
========

EgTRQC provides a clean simulation core for experiments where delayed feedback is part of the physical model rather than an implementation detail.

Design Goals
------------

The PoC is built around these goals:

- exact and auditable pure-delay semantics,
- reproducibility through deterministic seeds and serialized artifacts,
- model modularity across increasing levels of physical detail,
- clean separation between physics, diagnostics, and reporting.

Project Scope
-------------

EgTRQC is not a giant workflow framework.
It is a focused simulation and analysis core that exposes a small number of stable entry points and a well-typed internal architecture.

At its current stage, the PoC includes:

- a linear reference model for fast delay sweeps,
- graph-based OCTA-inspired reduced dynamics,
- product-state and dense-state quantum-inspired models,
- a delay-aware simulator shared across all model families,
- audits that verify causal correctness of the feedback history.

High-Level Picture
------------------

.. graphviz::

   digraph overview {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#eef4ff"];
       config [label="SimulationConfig"];
       model [label="Model Family"];
       delay [label="DelayBuffer"];
       source [label="Instantaneous Source"];
       effective [label="Effective Source"];
       state [label="State Advance"];
       records [label="Step Records"];
       audits [label="Audits"];
       artifacts [label="Artifacts"];

       config -> model;
       config -> delay;
       model -> source;
       source -> delay;
       delay -> effective;
       effective -> state;
       model -> state;
       state -> records;
       records -> audits;
       records -> artifacts;
       state -> delay [label="advance once"];
   }

Where to Start
--------------

If you are new to the repository:

1. Read :doc:`architecture` for the causal contract.
2. Read :doc:`models` for the available modeling layers.
3. Use :doc:`workflows` and :doc:`cli` to run the PoC.
4. Use :doc:`api` when you need module-level implementation detail.
