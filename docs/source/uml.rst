UML Design Reference
====================

These UML views describe the implemented package and its scientific compatibility
boundary. Diagrams use Graphviz, rendered as SVG by Sphinx; install the Graphviz
``dot`` executable before building. Function controllers are labeled explicitly.

MVC Package Dependencies
------------------------

Dependencies are conceptual UML package dependencies. Rendering does not invoke the study controller.

.. graphviz::
   :caption: MVC Package Dependencies

   digraph packages {
    rankdir=LR;
    node [shape=tab];
    cli [label="«controller»\ncli / reproduce"];
    controller [label="«controller»\nsimulator / study_controller\ncontrolled_studies.run_pair"];
    models [label="«model»\nmodel / octa_model\nocta_product_model / octa_dense_model\ncontrolled_studies.FullRegister"];
    delay [label="«model»\nconfig / delay / geometry"];
    view [label="«view»\nanalysis / study_reporting"];
    artifacts [shape=folder,label="results / configs / provenance"];
    cli -> controller [style=dashed,label="dispatches"];
    cli -> view [style=dashed,label="renders / validates"];
    controller -> models [style=dashed,label="evolves"];
    controller -> delay [style=dashed,label="observes / advances"];
    controller -> artifacts [label="reads / writes"];
    view -> artifacts [label="reads trajectories; writes reports"];
   }

Reduced Simulation Classes
--------------------------

The shared-behavior node describes the common callable contract; it is not an implemented base class or Protocol. Diamonds denote ownership; dashed hollow arrows denote behavioral conformance.

.. graphviz::
   :caption: Reduced Simulation Classes

   digraph reduced {
    rankdir=TB; node [shape=record];
    config [label="{SimulationConfig|dt; total_time; delay; reproducibility|num_steps}"];
    delaycfg [label="{DelayConfig|delay_steps; kernel|}"];
    simulator [label="{DelayAwareSimulator|model; config|run(): SimulationResult}"];
    buffer [label="{DelayBuffer|initial_value; advance_count|effective(); advance(); snapshot()}"];
    result [label="{SimulationResult|records; physical_log; total_buffer_advances|to_dict()}"];
    record [label="{StepRecord|time; state; source; effective_source|}"];
    model [label="{«shared behavior» Model|state and parameters|initial_state(); source(); curvature(); advance()}"];
    linear [label="{LinearBackreactionModel||}"];
    graphmodel [label="{GraphBackreactionModel||}"];
    product [label="{ProductStateOCTAModel||}"];
    dense [label="{DenseReducedOCTAModel||}"];
    config -> delaycfg [dir=both,arrowtail=diamond,arrowhead=none,label="1"];
    simulator -> config [label="1"];
    simulator -> buffer [dir=both,arrowtail=diamond,arrowhead=none,label="1 per run"];
    simulator -> model [label="1"];
    simulator -> result [style=dashed,label="creates"];
    result -> record [dir=both,arrowtail=diamond,arrowhead=none,label="0..*"];
    {linear graphmodel product dense} -> model [style=dashed,arrowhead=empty,label="provides behavior"];
   }

Full Register Inheritance and Control
-------------------------------------

Hollow solid arrows indicate actual Python inheritance. The controller is a function, not an invented runtime class. FrozenStepInput adapts the inherited baseline input interface.

.. graphviz::
   :caption: Full Register Inheritance and Control

   digraph full {
    rankdir=TB; node [shape=record];
    base [label="{notebook_kernel.GTRQCSimulator|rho; geometry; matter operators|step_strang(); matter_marginal()}"];
    octa [label="{OCTAReviewerExtensionsGTRQCSimulator|K_cfg; A_cfg; rate_mode|geometry and monitor extensions}"];
    full [label="{controlled_studies.FullRegister|step_source; cached factors|prepare_fast_factors()}"];
    nonlinear [label="{SemiclassicalNonlinearGKSLSimulator|rho; modular source|step()}"];
    cfg [label="{StudyConfig|dt; duration; delay; M; fold; mode|identifier; delay_steps; steps}"];
    pair [label="{«controller function» run_pair|two models; two delay buffers|observe; evolve; audit; persist}"];
    frozen [label="{FrozenStepInput|value|effective(); push(); reset()}"];
    full -> octa [arrowhead=empty];octa -> base [arrowhead=empty];
    pair -> cfg [style=dashed,label="reads"];
    pair -> full [style=dashed,label="constructs"];
    pair -> nonlinear [style=dashed,label="constructs"];
    pair -> frozen [style=dashed,label="injects held source"];
   }

Physical Step Interaction
-------------------------

This ordered UML interaction view shows the actual matched-pair controller, including the shadow closure-defect step and terminal sample. Each model has its own source history.

.. graphviz::
   :caption: Physical Step Interaction

   digraph sequence {
    rankdir=TB;
    node [shape=box];
    source [label="Controller → models: evaluate J_k from current states"];
    observe [label="Controller → buffers: effective(J_k, d)\nreturns J_max(k-d,0); no mutation"];
    record [label="Controller: record states and applied sources"];
    terminal [shape=diamond,label="terminal sample?"];
    stop [label="Return arrays / audits; no advance"];
    evolve [label="Controller → full register: step_strang(dt) with held source"];
    shadow [label="Controller → baseline: shadow step from register marginal\nmeasure closure defect"];
    baseline [label="Controller → baseline: restore own state; step(dt) with own source"];
    advance [label="Controller → each buffer: advance(J_k) exactly once"];
    source -> observe -> record -> terminal;
    terminal -> stop [label="yes"];
    terminal -> evolve [label="no"];
    evolve -> shadow -> baseline -> advance;
    advance -> source [style=dashed,label="next physical sample"];
   }

Study Execution Activity
------------------------

The cache identifier hashes every configuration field. Cached summaries are reused by execution; validate explicitly checks the associated trajectories and audits.

.. graphviz::
   :caption: Study Execution Activity

   digraph activity {
    node [shape=box];
    start [shape=circle,label=""];
    config [label="Read manifest; validate StudyConfig; compute identifier"];
    cache [shape=diamond,label="summary exists and\nrecompute is false?"];
    reuse [label="Read cached summary"];
    run [label="Construct matched models; execute physical steps"];
    audit [label="Check exact source indices; collect numerical diagnostics"];
    save [label="Write trajectory.npz and summary.json"];
    next [shape=diamond,label="more configurations?"];
    index [label="Write ordered study index.json"];
    end [shape=doublecircle,label=""];
    start -> config -> cache;cache -> reuse [label="yes"];cache -> run [label="no"];
    run -> audit -> save;save -> next;reuse -> next;next -> config [label="yes"];
    next -> index [label="no"];index -> end;
   }

Artifact and Validation Relationships
-------------------------------------

Numerical validation and checksum verification answer different questions. Original ZIP manifests remain archived provenance; the integration manifest describes the current distribution.

.. graphviz::
   :caption: Artifact and Validation Relationships

   digraph artifacts {
    rankdir=LR;node [shape=record];
    cfg [label="{configs/study.json|ordered StudyConfig values}"];
    index [label="{results/study/index.json|ordered matched-pair summaries}"];
    summary [label="{results/study/id/summary.json|config; run_id; source and numerical audits}"];
    npz [label="{results/study/id/trajectory.npz|time; trace_distance; source histories; geometry; optional states}"];
    report [label="{«view» study_reporting|tables; figures; paired contrasts}"];
    validation [label="{provenance/release_validation.json|counts; diagnostic extrema}"];
    checksum [label="{provenance/integration_files.sha256.json|relative file paths and SHA-256 digests}"];
    cfg -> index [label="1 study : ordered runs"];
    index -> summary [label="run_id"];
    summary -> npz [label="same id directory"];
    {index npz} -> report [style=dashed,label="reads"];
    {cfg index npz} -> validation [style=dashed,label="audits"];
    checksum -> npz [style=dashed,label="byte integrity"];
   }

