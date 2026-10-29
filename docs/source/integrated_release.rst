Integrated Benchmark
====================

The integrated release extends the reduced-model core with a full six-qubit
finite geometry register and a matched nonlinear matter-only closure. The
supplied trajectories contain 732 matched full-model pairs and 60 reduced runs.
Repeated configurations across studies are execution counts, not sample sizes.

Execution
---------

Install the package and documentation dependencies::

   python -m pip install -e '.[dev,docs]'
   python -m egtrqc.reproduce --study validate
   python -m egtrqc.reproduce --study report
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Run a selected manifest with ``--study active_refinement``. Use ``--study all``
to run or reuse every study; add ``--recompute`` to replace cached trajectories.
The controller reads ``configs/<study>.json`` and writes parameter-hashed folders
under ``results/<study>/``. Reporting reads those folders and produces tables
and figures. Full recomputation may require substantial time and memory.

Controller and View Boundaries
------------------------------

``StudyConfig`` and ``FullRegister`` define the full experiment. The matched-pair
controller ``run_pair`` enforces physical-step delay scheduling using two
``DelayBuffer`` instances. ``study_controller.run_manifest`` coordinates batches.
``study_reporting`` consumes completed results and provides numerical validation,
aggregation, and rendering. ``reproduce`` dispatches the requested operations.

The inherited kernel retains its original class hierarchy and legacy plotting
helpers to preserve numerical equivalence. This is an explicit compatibility
boundary. New physics belongs in models; new execution belongs in controllers;
new presentation belongs in reporting. Adding empty wrapper classes does not
improve this separation.

Protocols and Scientific Scope
------------------------------

The effective source at sample k is J[max(k-d, 0)]. The controller observes
history, evolves with a fixed input, and advances history exactly once. The
terminal sample does not advance the state or history.

The root ``PROTOCOLS.md`` and ``DATA_DICTIONARY.md`` contain the full protocol
and storage definitions. Under the controlled rule, the five-step delayed and
frozen trajectories coincide through 0.12 s at dt=0.02 s, with a first possible
changed output at 0.14 s. The early [0,0.10] s window is therefore not an
active-feedback comparison. Results depend on the policy and interval.

Notebook Migration
------------------

No executable notebook is required or distributed. Recorded display dictionaries
and PDF figures are in ``results/legacy_reference``. The package exposes the
extracted calculations through ``reference_workflow`` and ``reproduce``.
``provenance/notebook_migration.json`` records source hashes. Historical release
hashes and environment records remain evidence of the supplied ZIP; the current
integration checksum manifest is separate.

Citation Maintenance
--------------------

The README retains the existing archive DOI and records the different identifier
in the supplied main paper. The release maintainer will update the README and
editable manuscript after the next Zenodo deposit. No editable manuscript source
was supplied, and the PDFs are not modified.
