Theory and Equations
====================

This page collects the minimal mathematical layer behind the PoC.
The goal is not to provide a full derivation of a production OCTA theory, but to make the implemented contract explicit and easy to audit.

Discrete-Time Delay Rule
------------------------

Let :math:`J(t_n)` denote the instantaneous source evaluated from the current physical state at time :math:`t_n = n\,\Delta t`.
For a pure discrete delay of :math:`d` time steps, the implemented effective source is

.. math::

   J_{\mathrm{eff}}(t_n) =
   \begin{cases}
   J(t_n), & d = 0, \\
   J(t_{n-d}), & n \ge d, \\
   J(t_0), & n < d.
   \end{cases}

The early-time branch is important.
The PoC does not use zero-filled prehistory.
Instead, the prehistory is initialized with the physical initial source value :math:`J(t_0)`.

Delay-Buffer Update Protocol
----------------------------

The simulator enforces the following order at each physical step:

1. evaluate :math:`J(t_n)` from the current state,
2. read :math:`J_{\mathrm{eff}}(t_n)` from the delay buffer,
3. compute the curvature or response proxy,
4. advance the physical state,
5. append :math:`J(t_n)` to the physical history exactly once.

In symbolic form, if :math:`x_n` denotes the physical state, the loop is

.. math::

   J_n = J(x_n)

.. math::

   J_n^{\mathrm{eff}} = \mathcal{D}[J_0, J_1, \ldots, J_n; d]

.. math::

   x_{n+1} = \Phi(x_n, J_n^{\mathrm{eff}}, \Delta t)

followed by the history update

.. math::

   \mathcal{H}_{n+1} = \mathcal{H}_n \cup \{J_n\}

This is the reason diagnostics can be evaluated safely after the fact: they consume stored records and do not enter the physical history operator.

Reference Linear Model
----------------------

In the smallest reference model, the state is a real vector :math:`x_n \in \mathbb{R}^m` and the evolution is explicit Euler:

.. math::

   x_{n+1} = x_n + \Delta t\,(A x_n + B J_n^{\mathrm{eff}}) + \eta_n

where :math:`A` is a damping matrix, :math:`B` is the feedback coupling, and :math:`\eta_n` is an optional seeded perturbation term.
The instantaneous source is

.. math::

   J_n = C x_n + b

and the curvature proxy is

.. math::

   K_n = M J_n^{\mathrm{eff}}

This model is intentionally simple and is used as the fastest causal reference layer.

Geometry Response Operator
--------------------------

The OCTA-inspired graph models introduce a discrete geometry graph and a response operator of the form

.. math::

   G = \frac{1}{h^2}L + m^2 I

where :math:`L` is the graph Laplacian, :math:`h` is the mesh scale, and :math:`m` is a mass-like parameter.
Given an effective source, the reduced response solve is

.. math::

   G k_{\mathrm{eq}} = -\frac{1}{\alpha} J_{\mathrm{eff}}

so that

.. math::

   k_{\mathrm{eq}} = -\frac{1}{\alpha} G^{-1} J_{\mathrm{eff}}

The reported curvature vector is then constructed around the graph reference curvature :math:`k_{\mathrm{ref}}`:

.. math::

   K = k_{\mathrm{ref}} + k_{\mathrm{eq}}

If :math:`m \approx 0`, the PoC can enforce a zero-mean projection so that the massless response remains well-posed in the reduced setting.

Modular-Source Construction
---------------------------

The OCTA-inspired product-state and dense-state models use graph balls to define local reduced subsystems.
For each ball :math:`B_i`, a reference reduced state :math:`\rho_{B_i}^{\mathrm{ref}}` defines a modular Hamiltonian

.. math::

   H_{B_i}^{\mathrm{mod}} = -\log\!\left((1-\varepsilon)\rho_{B_i}^{\mathrm{ref}} + \varepsilon \frac{I}{d_{B_i}}\right)

where :math:`\varepsilon > 0` is a regularization parameter.
Given a current state :math:`\rho`, the modular source component on ball :math:`B_i` is the centered expectation

.. math::

   J_i(\rho) = \operatorname{Tr}(\rho_{B_i} H_{B_i}^{\mathrm{mod}}) - \operatorname{Tr}(\rho_{B_i}^{\mathrm{ref}} H_{B_i}^{\mathrm{mod}})

This construction gives the product-state and dense-state models a local-information structure closer to the OCTA-inspired setting than the linear model.

Product-State Approximation
---------------------------

In the product-state model, the dynamical variables are local excitation probabilities.
A product density matrix is reconstructed from those marginals, reduced on each ball, and inserted into the modular-source rule above.
The state update remains low-dimensional while the source construction carries a reduced quantum-statistical structure.

Dense-State Reduced Model
-------------------------

In the dense reduced model, the matter state is a full density matrix :math:`\rho_n` on a small qubit system.
The model uses two ingredients:

- a curvature-dependent effective Hamiltonian,
- curvature-dependent dephasing rates.

At a high level, one step is a completely positive trace-preserving map of the form

.. math::

   \rho_{n+1} = \mathcal{E}_{\Delta t, K_n}(\rho_n)

where :math:`\mathcal{E}` combines unitary rotation and local dephasing channels.
This is the richest currently implemented PoC layer and the basis for the dense delay analysis workflow.

Audit Equations
---------------

The two main audits correspond directly to executable mathematical checks.

Pure-delay audit:

.. math::

   \max_n \left\lVert J_{\mathrm{eff}}^{\mathrm{logged}}(t_n) - J_{\mathrm{eff}}^{\mathrm{expected}}(t_n) \right\rVert_\infty \le 10^{-12}

Single-advance audit:

.. math::

   N_{\mathrm{advances}} = N_{\mathrm{physical\ steps}}

These checks are part of the design of the PoC, not an optional afterthought.

Interpretation Boundary
-----------------------

The PoC is mathematically explicit about the delay semantics and the reduced response structure.
What it does not claim is that these reduced equations already constitute a full, final physical theory of the broader OCTA setting.
Instead, the current theory layer should be read as:

- exact for the implemented causal semantics,
- reduced but faithful for the current model families,
- extensible without changing the delay contract.
