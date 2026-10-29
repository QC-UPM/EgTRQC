"""Topology sensitivities of the existing reduced graph, product and dense models."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from .geometry import GeometryGraph,build_octa_graph
from .octa_model import GraphBackreactionModel,GraphBackreactionModelConfig
from .octa_product_model import ProductStateOCTAModel,ProductStateOCTAModelConfig
from .octa_dense_model import DenseReducedOCTAModel,DenseReducedOCTAModelConfig
from .config import SimulationConfig,DelayConfig,ReproducibilityConfig
from .simulator import DelayAwareSimulator
from .diagnostics import audit_delay_definition,audit_single_advance_per_step
from .quantum import product_density_matrix
from .controlled_studies import distance


def topology(name):
    """Construct one of the five six-node benchmark topologies.

    Args:
        name: One of octahedral, ring, chain, star, or complete.

    Returns:
        GeometryGraph with the requested adjacency and inherited positions.
    """
    base=build_octa_graph()
    if name=='octahedral':return base
    a=np.zeros((6,6))
    if name in ('ring','chain'):
        edges=[(i,i+1) for i in range(5)]
        if name=='ring':edges.append((5,0))
    elif name=='star':edges=[(0,i) for i in range(1,6)]
    elif name=='complete':edges=[(i,j) for i in range(6) for j in range(i+1,6)]
    else:raise ValueError(name)
    for i,j in edges:a[i,j]=a[j,i]=1
    return GeometryGraph(a,np.zeros(6),base.node_positions)


def run_topology_study(outdir):
    """Run all reduced families over five topologies and four delays.

    Args:
        outdir: Destination for raw NPZ trajectories and JSON summaries.

    Returns:
        Sixty reduced-model summaries, each with delay and advance audits.
    """
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    rows=[];parameters={}
    for family in ('graph','product','dense'):
      for graph_name in ('octahedral','ring','chain','star','complete'):
        graph=topology(graph_name)
        if family=='graph':cls,mc=GraphBackreactionModel,GraphBackreactionModelConfig()
        elif family=='product':cls,mc=ProductStateOCTAModel,ProductStateOCTAModelConfig()
        else:cls,mc=DenseReducedOCTAModel,DenseReducedOCTAModelConfig()
        parameters[family]=asdict(mc)
        reference=None
        for tau in (0.,.02,.04,.1):
            config=SimulationConfig(total_time=.4,dt=.01,delay=DelayConfig(delay_steps=int(round(tau/.01))),reproducibility=ReproducibilityConfig(seed=7))
            result=DelayAwareSimulator(cls(graph,mc),config).run()
            states=np.array([r.state for r in result.physical_log]);t=np.array([r.time for r in result.physical_log])
            sources=np.array([r.source for r in result.physical_log]);eff=np.array([r.effective_source for r in result.physical_log])
            if reference is None:reference=states
            if family=='graph':metric=np.linalg.norm(states-reference,axis=1)/np.sqrt(6)
            elif family=='product':metric=np.array([distance(product_density_matrix(x),product_density_matrix(y)) for x,y in zip(states,reference)])
            else:metric=np.array([distance(x,y) for x,y in zip(states,reference)])
            da=audit_delay_definition(result);sa=audit_single_advance_per_step(result)
            assert da.passed and sa.passed
            row=dict(model_level='reduced_'+family,topology=graph_name,delay=tau,dt=.01,duration=.4,
              statistic='root_mean_square_state_amplitude_difference' if family=='graph' else 'within_model_trace_distance_to_zero_delay',
              peak=float(metric.max()),area=float(np.trapezoid(metric,t)),
              max_source_change=float(np.linalg.norm(eff-eff[0],axis=1).max()),delay_audit=asdict(da),advance_audit=asdict(sa))
            rows.append(row)
            np.savez_compressed(outdir/f'{family}_{graph_name}_delay{int(round(tau/.01))}.npz',time=t,states=states,source=sources,effective_source=eff,discrepancy=metric)
    (outdir/'index.json').write_text(json.dumps(rows,indent=2))
    (outdir/'parameters.json').write_text(json.dumps(parameters,indent=2))
    return rows
