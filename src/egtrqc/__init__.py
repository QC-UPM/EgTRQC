"""EgTRQC reference package."""

from egtrqc.analysis import DenseAnalysisArtifacts, DenseAnalysisConfig, run_dense_delay_analysis
from egtrqc.config import DelayConfig, KernelConfig, ReproducibilityConfig, SimulationConfig
from egtrqc.delay import DelayBuffer
from egtrqc.diagnostics import (
    DelayAuditReport,
    SingleAdvanceAuditReport,
    audit_delay_definition,
    audit_single_advance_per_step,
)
from egtrqc.ensemble import (
    GeometryEnsemble,
    build_config_graph,
    build_geometry_ensemble,
    diff_vertex_and_sign,
    gibbs_distribution,
    make_geometry_ensemble,
    metropolis_rate_matrix,
    slice_free_energy,
)
from egtrqc.geometry import GeometryGraph, build_balls, build_octa_graph
from egtrqc.model import LinearBackreactionModel, LinearBackreactionModelConfig
from egtrqc.octa_dense_model import DenseReducedOCTAModel, DenseReducedOCTAModelConfig
from egtrqc.octa_model import GraphBackreactionModel, GraphBackreactionModelConfig, GeometryRegisterConfig
from egtrqc.octa_product_model import ProductStateOCTAModel, ProductStateOCTAModelConfig
from egtrqc.quantum import ensure_density, matrix_log_psd, partial_trace_qubits, product_density_matrix
from egtrqc.simulator import DelayAwareSimulator, SimulationResult, StepRecord

__all__ = [
    'DelayAwareSimulator',
    'DelayAuditReport',
    'DelayBuffer',
    'DelayConfig',
    'DenseAnalysisArtifacts',
    'DenseAnalysisConfig',
    'DenseReducedOCTAModel',
    'DenseReducedOCTAModelConfig',
    'GeometryEnsemble',
    'GeometryGraph',
    'GeometryRegisterConfig',
    'GraphBackreactionModel',
    'GraphBackreactionModelConfig',
    'KernelConfig',
    'LinearBackreactionModel',
    'LinearBackreactionModelConfig',
    'ProductStateOCTAModel',
    'ProductStateOCTAModelConfig',
    'ReproducibilityConfig',
    'SimulationConfig',
    'SimulationResult',
    'SingleAdvanceAuditReport',
    'StepRecord',
    'audit_delay_definition',
    'audit_single_advance_per_step',
    'build_balls',
    'build_config_graph',
    'build_geometry_ensemble',
    'build_octa_graph',
    'diff_vertex_and_sign',
    'ensure_density',
    'gibbs_distribution',
    'make_geometry_ensemble',
    'matrix_log_psd',
    'metropolis_rate_matrix',
    'partial_trace_qubits',
    'product_density_matrix',
    'run_dense_delay_analysis',
    'slice_free_energy',
]
