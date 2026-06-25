Model Families
==============

EgTRQC includes several model families so that the same delay-aware infrastructure can be exercised at different levels of physical richness.

Family Map
----------

.. graphviz::

   digraph families {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#eefaf4"];
       root [label="Shared Delay-Aware Simulator"];
       linear [label="LinearBackreactionModel"];
       graph_model [label="GraphBackreactionModel"];
       product [label="ProductStateOCTAModel"];
       dense [label="DenseReducedOCTAModel"];

       root -> linear;
       root -> graph_model;
       root -> product;
       root -> dense;
   }

LinearBackreactionModel
-----------------------

This is the smallest and fastest causal reference model.
It is useful when we want to verify delay semantics, audit behavior, and baseline reproducibility with minimal physical overhead.

GraphBackreactionModel
----------------------

This model introduces an OCTA-inspired graph structure.
It uses node-wise source construction, graph balls, and a response operator that is closer to the geometry-aware reduced setting.

ProductStateOCTAModel
---------------------

This model keeps a factored matter-state representation.
It is a useful intermediate layer when we want more structure than the linear model but lower cost than a dense-state simulation.

DenseReducedOCTAModel
---------------------

This model keeps a full dense state for small controlled systems.
It is the richest currently implemented PoC model and is the basis for the dense delay analysis pipeline.

Choosing a Model
----------------

Use the linear model when:

- you are validating causal semantics,
- you need fast sweeps,
- you are debugging the simulator or audit layer.

Use the graph model when:

- you need OCTA-inspired geometry dependence,
- you want node-level source and curvature structure.

Use the product or dense models when:

- you need quantum-inspired state dynamics,
- you want to compare reduced versus richer matter-state treatments,
- you want analysis outputs beyond scalar trajectories.
