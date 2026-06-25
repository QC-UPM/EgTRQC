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

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli run-sweep        --output-dir artifacts/reference_sweep_vpy        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

Dense analysis:

.. code-block:: bash

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli analyze-dense-delay        --output-dir artifacts/dense_delay_analysis        --delays 0 1 2 5        --total-time 0.5        --dt 0.05        --seed 7

Design Notes
------------

The CLI is intentionally not a kitchen sink.
It exists to expose stable experiment surfaces that already satisfy the PoC's causal and reproducibility requirements.


Release-Style Build
-------------------

If you want the standard verification and artifact build in one pass, run:

.. code-block:: bash

   bash scripts/build_all.sh
