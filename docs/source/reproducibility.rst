Reproducibility
===============

EgTRQC is built as a reproducible PoC rather than as an exploratory notebook environment.

Reproducibility Strategy
------------------------

The project uses:

- explicit configuration objects,
- deterministic seeds,
- typed simulation results,
- machine-readable manifests,
- executable audits,
- checked test coverage.

Artifact Strategy
-----------------

.. graphviz::

   digraph reproducibility {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#eef5f7"];
       cfg [label="Config + Seed"];
       run [label="Simulation Run"];
       records [label="Step Records"];
       audits [label="Audits"];
       manifest [label="Manifest"];
       report [label="Report Bundle"];

       cfg -> run -> records;
       records -> audits;
       records -> report;
       audits -> manifest;
       report -> manifest;
   }

Core Commands
-------------

For an end-to-end rebuild of the PoC, run:

.. code-block:: bash

   bash scripts/build_all.sh

The script executes the test suite, rebuilds the reference sweep, rebuilds the dense analysis artifacts, and regenerates the Sphinx site.

You can also run each stage separately:

.. code-block:: bash

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli run-sweep --output-dir artifacts/reference_sweep_vpy
   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli analyze-dense-delay --output-dir artifacts/dense_delay_analysis
   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m pytest -q

What the Audits Check
---------------------

``audit_delay_definition``
   Verifies that the effective source matches the mathematical pure-delay rule.

``audit_single_advance_per_step``
   Verifies that the feedback history advanced exactly once per physical time step.

Why This Matters
----------------

A delay-aware PoC is only useful if the implemented history semantics are exact.
The audit layer turns that requirement into an executable condition instead of a documentation claim.
