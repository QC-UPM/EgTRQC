Developer Guide
===============

This guide explains how to extend EgTRQC without breaking the causal contract of the PoC.
The most important rule is simple: new physics is allowed, but hidden mutations of the feedback history are not.

Development Priorities
----------------------

When extending the repository, prioritize in this order:

1. preserve the delay semantics,
2. preserve deterministic reproducibility,
3. preserve typed, inspectable outputs,
4. add model fidelity only through explicit interfaces.

Core Contract
-------------

The PoC depends on three invariants:

- the delay prehistory is initialized with :math:`J(t_0)`,
- the effective source can be observed without mutating history,
- the delay history advances exactly once per physical step.

Anything that violates those rules is not a valid extension of the current PoC.

What a Model Must Provide
-------------------------

A model used by ``DelayAwareSimulator`` is expected to provide these methods:

``initial_state()``
   Return the initial matter state.

``source(state)``
   Return the instantaneous source :math:`J(t_n)` for the supplied state.

``curvature(effective_source)``
   Return the geometric or diagnostic response built from the effective source.

``advance(state, effective_source, dt, rng)``
   Return the next physical state.

In practice, this means a new model should be written so that the simulator can remain unchanged.

Recommended Extension Pattern
-----------------------------

The safe pattern is:

1. create a new typed config dataclass for the model,
2. implement the four core methods,
3. keep all model-local cached operators inside the model instance,
4. leave delay handling to ``DelayBuffer`` and ``DelayAwareSimulator``,
5. write focused tests for the new model,
6. reuse the existing analysis and audit layers whenever possible.

What Not to Do
--------------

Do not:

- append to the delay history from a model,
- make a diagnostic helper call ``advance()`` on the buffer,
- mix artifact generation into the model update logic,
- hide seeded randomness in global state,
- bypass ``SimulationConfig`` with ad hoc parameters scattered through scripts.

Those shortcuts tend to destroy reproducibility first and causal correctness immediately after.

Adding a New Model Family
-------------------------

A typical new model integration should touch these layers:

``src/egtrqc/<new_model>.py``
   The model implementation and its typed configuration.

``src/egtrqc/__init__.py``
   Export the public symbols if the model is part of the supported surface.

``tests/test_<new_model>.py``
   Add direct tests for initialization, source shape, curvature shape, and advance behavior.

Optionally:

``src/egtrqc/analysis.py``
   Extend analysis workflows if the new model needs a dedicated reporting path.

Adding a New Workflow
---------------------

If you add a new CLI workflow:

1. make it configuration-driven,
2. ensure it emits machine-readable artifacts,
3. ensure it emits a human-readable summary or report,
4. keep the physical simulation separate from reporting,
5. reuse ``audit_delay_definition`` and ``audit_single_advance_per_step`` whenever relevant.

A good workflow should be reproducible from one command and easy to archive.

Configuration Guidelines
------------------------

Use dataclasses in ``config.py`` or model-local config classes when:

- a parameter affects physics,
- a parameter affects reproducibility,
- a parameter affects artifact structure,
- the parameter needs validation.

Avoid unvalidated free-form dictionaries for core simulation logic.

Testing Checklist
-----------------

Before considering an extension complete, check:

- the model builds from typed config objects,
- the simulator can run it for at least one short deterministic case,
- the result passes the delay-definition audit when using the delta kernel,
- the result passes the single-advance audit,
- any analysis workflow writes its expected artifacts,
- the full test suite still passes.

Documentation Checklist
-----------------------

If a change introduces a new modeling concept or workflow, update:

- ``README.md`` for top-level discoverability,
- the relevant Sphinx guide pages,
- ``api.rst`` if a new module should appear in the API reference,
- theory documentation if the mathematical contract changes.

Decision Boundary
-----------------

If a proposed change requires altering the behavior of ``DelayBuffer`` or the ordering in ``DelayAwareSimulator``, treat it as an architectural change, not a local refactor.
That kind of modification should be justified explicitly because it changes the meaning of all delay-sensitive results.

Practical Rule of Thumb
-----------------------

If you can implement a richer model while leaving ``DelayBuffer``, ``DelayAwareSimulator``, and the audit functions untouched, you are probably extending the PoC in the intended direction.
If you need to weaken those components, stop and re-evaluate the design first.
