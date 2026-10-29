API Reference
=============

The API reference is generated directly from the codebase.

Core Modules
------------

.. autosummary::
   :toctree: generated

   egtrqc.analysis
   egtrqc.cli
   egtrqc.config
   egtrqc.delay
   egtrqc.diagnostics
   egtrqc.ensemble
   egtrqc.geometry
   egtrqc.io
   egtrqc.model
   egtrqc.octa_dense_model
   egtrqc.octa_model
   egtrqc.octa_product_model
   egtrqc.quantum
   egtrqc.simulator

Integrated Benchmark Modules
----------------------------

.. autosummary::
   :toctree: generated

   egtrqc.controlled_studies
   egtrqc.study_controller
   egtrqc.study_reporting
   egtrqc.reproduce
   egtrqc.reference_workflow
   egtrqc.topology_studies

The compatibility kernel ``egtrqc.notebook_kernel`` retains the extracted
scientific definitions for numerical equivalence. New integrations should use
``controlled_studies`` rather than its historical execution and plotting helpers.
