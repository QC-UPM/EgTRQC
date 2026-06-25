Workflows
=========

The PoC currently exposes two primary workflows.

Workflow Map
------------

.. graphviz::

   digraph workflows {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#f4eefc"];
       cli [label="CLI Entry Point"];
       sweep [label="run-sweep"];
       dense [label="analyze-dense-delay"];
       json [label="JSON + Manifest"];
       analysis [label="CSV + Plotly + Report"];

       cli -> sweep -> json;
       cli -> dense -> analysis;
   }

Reference Sweep
---------------

The reference sweep is the smallest end-to-end workflow.
It runs the deterministic linear model for a set of delay values and writes one JSON artifact per case together with a manifest containing audit results.

Dense Delay Analysis
--------------------

The dense delay analysis runs the dense reduced OCTA-inspired model for a set of delays and generates:

- a per-time-step CSV file,
- a summary CSV file,
- a manifest,
- Plotly HTML figures,
- an English Markdown report.

Typical Session
---------------

.. code-block:: bash

   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli run-sweep --output-dir artifacts/reference_sweep_vpy
   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m egtrqc.cli analyze-dense-delay --output-dir artifacts/dense_delay_analysis
   PYTHONPATH=src /home/jordieres/soft/vpy/bin/python -m pytest -q

Generated Artifacts
-------------------

The workflows are intentionally artifact-driven.
They are meant to be easy to inspect programmatically and easy to archive for later comparison.
