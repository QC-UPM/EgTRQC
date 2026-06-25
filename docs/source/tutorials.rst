Tutorials and Examples
======================

This page provides short guided paths through the PoC.
Each example is designed to exercise a different layer of the repository without requiring ad hoc scripting.

Tutorial Map
------------

.. graphviz::

   digraph tutorials {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#eef7ff"];
       t1 [label="1. Minimal Causal Sweep"];
       t2 [label="2. Dense Delay Analysis"];
       t3 [label="3. Extend with a New Model"];

       t1 -> t2 -> t3;
   }

Tutorial 1: Minimal Causal Sweep
--------------------------------

Goal
^^^^

Run the smallest end-to-end experiment and inspect the audit-friendly outputs.

Command
^^^^^^^

.. code-block:: bash

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli run-sweep        --output-dir artifacts/reference_sweep_vpy        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

What it generates
^^^^^^^^^^^^^^^^^

- one JSON file per delay value,
- a ``manifest.json`` file,
- embedded audit summaries for delay-definition and single-advance checks.

What to inspect
^^^^^^^^^^^^^^^

Open the manifest first.
It is the quickest way to confirm that:

- all requested delay cases ran,
- the expected artifacts were written,
- the audits passed for each run.

Why this example matters
^^^^^^^^^^^^^^^^^^^^^^^^

This is the cleanest path for validating the causal semantics of the PoC before moving into richer models.

Tutorial 2: Dense Delay Analysis
--------------------------------

Goal
^^^^

Run the richest currently supported built-in analysis pipeline and generate tables, interactive graphics, and a report.

Command
^^^^^^^

.. code-block:: bash

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli analyze-dense-delay        --output-dir artifacts/dense_delay_analysis        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

What it generates
^^^^^^^^^^^^^^^^^

- ``dense_delay_timeseries.csv``
- ``dense_delay_summary.csv``
- ``manifest.json``
- Plotly HTML files for source, curvature, purity, entropy, and trace distance
- ``REPORT.md``

Suggested reading order
^^^^^^^^^^^^^^^^^^^^^^^

1. Read ``REPORT.md`` for the executive summary.
2. Open ``dense_delay_summary.csv`` for compact numeric comparison.
3. Open the Plotly HTML files to understand the temporal structure.
4. Use the manifest when you need run metadata and audit state.

Why this example matters
^^^^^^^^^^^^^^^^^^^^^^^^

This example shows the PoC operating as a full artifact-producing analysis layer rather than just a simulator.

Tutorial 3: Add a New Model Family
----------------------------------

Goal
^^^^

Extend the PoC while preserving the delay contract.

Recommended steps
^^^^^^^^^^^^^^^^^

1. Create a new module under ``src/egtrqc/``.
2. Add a typed config dataclass for the model.
3. Implement these methods:

   - ``initial_state()``
   - ``source(state)``
   - ``curvature(effective_source)``
   - ``advance(state, effective_source, dt, rng)``

4. Add tests under ``tests/test_<new_model>.py``.
5. Export the model in ``src/egtrqc/__init__.py`` if it should be public.
6. Update the Sphinx docs if the model adds a new concept.

Minimal skeleton
^^^^^^^^^^^^^^^^

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np
   from numpy.random import Generator


   @dataclass(frozen=True, slots=True)
   class NewModelConfig:
       gain: float = 1.0


   class NewModel:
       def __init__(self, config: NewModelConfig) -> None:
           self._config = config

       def initial_state(self) -> np.ndarray:
           return np.array([1.0], dtype=np.float64)

       def source(self, state: np.ndarray) -> np.ndarray:
           return self._config.gain * np.asarray(state, dtype=np.float64)

       def curvature(self, effective_source: np.ndarray) -> np.ndarray:
           return np.asarray(effective_source, dtype=np.float64)

       def advance(
           self,
           state: np.ndarray,
           effective_source: np.ndarray,
           dt: float,
           rng: Generator,
       ) -> np.ndarray:
           del rng
           return np.asarray(state, dtype=np.float64) + dt * np.asarray(effective_source, dtype=np.float64)

Validation checklist
^^^^^^^^^^^^^^^^^^^^

After adding the model:

- run the targeted model tests,
- run the full suite,
- verify that the simulator can execute the model deterministically,
- confirm that the delay audits still pass for a short delta-kernel case.

Related pages
-------------

- :doc:`architecture`
- :doc:`theory`
- :doc:`developer_guide`
- :doc:`api`
