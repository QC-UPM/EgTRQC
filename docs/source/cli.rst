CLI Reference
=============

The command-line interface is intentionally compact.
It exposes only the workflows that are already reproducible and well-audited.

Commands
--------

``run-sweep``
   Run the deterministic linear reference sweep over one or more delay values.

``analyze-dense-delay``
   Run the dense reduced OCTA-inspired delay analysis and emit tables, plots, and a report.

Examples
--------

Reference sweep:

.. code-block:: bash

   python -m egtrqc.cli run-sweep        --output-dir results/reference_sweep        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

Dense analysis:

.. code-block:: bash

   python -m egtrqc.cli analyze-dense-delay        --output-dir results/dense_delay_analysis        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

Design Notes
------------

The CLI is intentionally not a kitchen sink.
It exists to expose stable experiment surfaces that already satisfy the PoC's causal and reproducibility requirements.


Release-Style Build
-------------------

If you want the standard verification and artifact build in one pass, run:

.. code-block:: bash

   bash scripts/build_all.sh

Integrated Studies
------------------

``python -m egtrqc.reproduce`` dispatches manifest-driven full-model studies,
reduced topology runs, report generation, and stored-result validation::

   python -m egtrqc.reproduce --study validate
   python -m egtrqc.reproduce --study report
   python -m egtrqc.reproduce --study source_controls
   python -m egtrqc.reproduce --study all --recompute

``--root`` selects the repository containing configs and results. Omit
``--recompute`` to reuse completed matched-pair summaries. See
:doc:`integrated_release` for scientific protocol distinctions.
