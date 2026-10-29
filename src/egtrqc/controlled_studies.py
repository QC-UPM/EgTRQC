"""Matched source-control experiments using the archived full-register kernels.

The orchestration evaluates the source at the start of each physical step and
uses egtrqc.delay.DelayBuffer. Numerical quantum factors are inherited from
notebook_kernel; vectorized factors are checked against their scalar versions.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib, json, math, platform, time, types
import numpy as np
from . import notebook_kernel as nk
from .delay import DelayBuffer

MODES = ('fixed_projector_signed', 'fixed_projector_unsigned',
         'instantaneous_projector_signed', 'instantaneous_projector_unsigned')

@dataclass(frozen=True)
class StudyConfig:
    """Parameters for a matched finite-register and nonlinear-closure experiment.

    Time quantities are in seconds. The physical delay and duration must be
    integer multiples of ``dt``. All fields participate in the cache identifier.

    Attributes:
        fold: Deterministic ensemble fold.
        mode: Monitor and geometry-rate policy.
        M: Number of geometry configurations.
        geometry_quantum: Whether off-diagonal geometry blocks are retained.
        dt: Physical time step in seconds.
        duration: Total simulated time in seconds.
        delay: Physical feedback delay in seconds.
        frozen: Hold the source at its initial value throughout evolution.
        initial_population: Excited population in each initial matter qubit.
        geometry_initial: Geometry initialization policy.
        rate_scale: Common dynamical rate multiplier.
        alpha_area: Area-response coupling.
        gamma0: Matter dephasing rate.
        gamma_curv: Curvature-dependent dephasing coefficient.
        beta_geom: Inverse geometry temperature.
        gamma_geom: Geometry transition rate.
        pointer_dephase: Geometry pointer dephasing rate.
        trace_tol: Maximum tolerated trace error.
        source_regularization: Regularizer for the modular source.
        ensemble_seed: Seed defining the nested geometry ensemble.
        normalized_proposals: Normalize rates by maximum proposal degree.
        decoupled: Disable matter and geometry feedback couplings.
        store_states: Include density matrices in trajectory files."""
    fold: int = 0
    mode: str = 'instantaneous_projector_signed'
    M: int = 10
    geometry_quantum: bool = False
    dt: float = .02
    duration: float = .4
    delay: float = .1
    frozen: bool = False
    initial_population: float = .5
    geometry_initial: str = 'delta'
    rate_scale: float = 1.
    alpha_area: float = 1.
    gamma0: float = .3
    gamma_curv: float = 1.2
    beta_geom: float = 8.
    gamma_geom: float = 40.
    pointer_dephase: float = 2.
    trace_tol: float = 5e-11
    source_regularization: float = .002
    ensemble_seed: int = 77
    normalized_proposals: bool = False
    decoupled: bool = False
    store_states: bool = False

    def __post_init__(self):
        if self.mode not in MODES: raise ValueError('Unknown mode')
        if self.M < 2 or self.M > 180: raise ValueError('M must be between 2 and 180')
        if self.dt <= 0 or self.duration <= 0 or self.delay < 0: raise ValueError('Invalid time grid')
        for x in (self.duration, self.delay):
            if not np.isclose(x/self.dt, round(x/self.dt), atol=1e-10, rtol=0):
                raise ValueError('Duration and physical delay must be integer multiples of dt')
        if not 0 <= self.initial_population <= 1: raise ValueError('Population outside [0,1]')
        if self.rate_scale <= 0 or self.alpha_area <= 0: raise ValueError('Scales must be positive')
        if self.geometry_initial not in ('delta', 'uniform', 'gibbs', 'coherent_gibbs'):
            raise ValueError('Unknown geometry initialization')
        if self.geometry_initial == 'coherent_gibbs' and not self.geometry_quantum:
            raise ValueError('Coherent initialization requires quantum geometry')

    @property
    def delay_steps(self): return int(round(self.delay/self.dt))
    @property
    def steps(self): return int(round(self.duration/self.dt))
    @property
    def identifier(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:20]


def distance(rho, sigma):
    """Compute trace distance between two Hermitian density matrices.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Half the trace norm of their Hermitian difference."""
    delta = rho-sigma
    return float(.5*np.abs(np.linalg.eigvalsh(.5*(delta+delta.conj().T))).sum())


def population_state(population, n=6):
    """Build a pure product density matrix with a shared qubit population.

    Args:
        population: Excited-state probability per qubit.
        n: Number of matter qubits.

    Returns:
        Complex density matrix of dimension 2**n.
    """
    ket=np.array([math.sqrt(1-population),math.sqrt(population)],complex)
    psi=nk.tensor_kets([ket]*n)
    return np.outer(psi,psi.conj())


class FrozenStepInput:
    """Read-only input for a single inherited baseline step."""
    def __init__(self, value): self.value=np.asarray(value,float).copy()
    def push(self, value): pass
    def effective(self, **kwargs): return self.value.copy()
    def reset(self): pass


class FullRegister(nk.OCTAReviewerExtensionsGTRQCSimulator):
    """Archived full joint model with externally scheduled source and exact fast factors."""
    def _compute_rates_from_current_J(self):
        j=self.step_source
        F=np.array([nk.slice_F(self._k(K),j,self.G,self.alpha_area) for K in self.K_cfg])
        R=nk.metropolis_rates(F,self.A_cfg,self.beta_geom,self.gamma_geom_base)
        return R,R.sum(axis=0),nk.gibbs_pi(F,self.beta_geom)

    def prepare_fast_factors(self):
        self._unitaries={}
        self._dephasing={}
        self._geometry_unitaries={}
        self._effects={}
        for _,(v,sgn) in self.edge_info.items():
            if (v,sgn) not in self._effects:
                _,sf=self._select_F_and_sqrtF(v,sgn)
                if not np.allclose(sf,np.diag(np.diag(sf)),atol=1e-13,rtol=0):
                    raise ValueError('Fast diagonal monitor kernel is not applicable')
                self._effects[v,sgn]=np.diag(sf).copy()
        self.max_kraus_error=0.
        self.min_nojump_eigenvalue=1.
        self.maximum_jump_substeps=0

    def _apply_matter_unitary(self,dt):
        if dt not in self._unitaries:
            self._unitaries[dt]=np.stack([nk.expm_hermitian(H,dt) for H in self._H_list_edge])
        U=self._unitaries[dt]; Ud=U.conj().transpose(0,2,1)
        if self.geometry_quantum:
            self.rho[0]=(U[:,None]@self.rho[0])@Ud[None,:]
        else: self.rho[0]=(U@self.rho[0])@Ud

    def _apply_geometry_unitary(self,dt):
        if not self.geometry_quantum or self.geom_mix_rate==0:return
        if dt not in self._geometry_unitaries:
            self._geometry_unitaries[dt]=nk.expm_hermitian(self.HG,dt)
        U=self._geometry_unitaries[dt]
        self.rho[0]=np.einsum('aA,ABij,bB->abij',U,self.rho[0],U.conj(),optimize=True)

    def _apply_conditional_dephasing_exact(self,dt):
        if dt not in self._dephasing:
            shape=(self.M,self.M,self.d,self.d) if self.geometry_quantum else (self.M,self.d,self.d)
            fac=np.ones(shape,float)
            for i in range(self.N):
                z=np.diag(self.Z_ops[i]).real; zz=z[:,None]*z[None,:]
                g=np.maximum(self.gamma_deph[i],0.)
                if self.geometry_quantum:
                    a=np.sqrt(g)[:,None];b=np.sqrt(g)[None,:]
                    lp=np.exp(-.5*(a-b)**2*dt);lm=np.exp(-.5*(a+b)**2*dt)
                    fac*=.5*(lp+lm)[:,:,None,None]+.5*(lp-lm)[:,:,None,None]*zz
                else:
                    p=.5*(1-np.exp(-2*g*dt))
                    fac*=1-p[:,None,None]+p[:,None,None]*zz
            self._dephasing[dt]=fac
        self.rho[0]*=self._dephasing[dt]

    def _apply_geometry_pointer_dephase(self,dt):
        if not self.geometry_quantum:return
        f=np.full((self.M,self.M),np.exp(-self.geom_pointer_dephase*dt));np.fill_diagonal(f,1.)
        self.rho[0]*=f[:,:,None,None]

    def _apply_geometry_jump_kraus(self,dt):
        R=self._compute_rates_from_current_J()[0] if self.backreaction=='instantaneous' else self.R0
        S=np.zeros((self.M,self.d));edges=[]
        for b,neighbours in enumerate(self.neigh_by_source):
            for a,v,sgn in neighbours:
                r=float(R[a,b])
                if r>0:
                    sf=self._effects[v,sgn]
                    S[b]+=r*np.abs(sf)**2
                    edges.append((a,b,r,sf[:,None]*sf.conj()[None,:]))
        lam=float(S.max())
        if lam<=0:return
        remaining=float(dt); substeps=0
        while remaining>0:
            ds=min(remaining,self.jump_substep_safety/lam)
            q=1-ds*S
            if q.min() < -1e-12:raise FloatingPointError('Negative no-jump effect')
            k=np.sqrt(np.maximum(q,0))
            self.max_kraus_error=max(self.max_kraus_error,float(np.abs(k*k+ds*S-1).max()))
            self.min_nojump_eigenvalue=min(self.min_nojump_eigenvalue,float(q.min()))
            old=self.rho[0]
            if self.geometry_quantum:
                out=old*k[:,None,:,None]*k[None,:,None,:]
                for a,b,r,f in edges:out[a,a]+=ds*r*f*old[b,b]
            else:
                out=old*k[:,:,None]*k[:,None,:]
                for a,b,r,f in edges:out[a]+=ds*r*f*old[b]
            self.rho[0]=out
            remaining-=ds;substeps+=1
            if substeps>100000:raise RuntimeError('Excessive jump substeps')
        self.maximum_jump_substeps=max(self.maximum_jump_substeps,substeps)


def install_source_cache(model):
    """Cache embedded modular-source operators on a model.

    Args:
        model: Model whose source evaluator is replaced by an equivalent cached evaluator.
    """
    operators=np.stack([nk.embed_operator(K,keep=B,N=model.N) for K,B in zip(model.KB_list,model.balls)])
    offsets=np.einsum('ij,vji->v',model.rho_ref_pure,operators).real
    diag=np.diagonal(operators,axis1=-2,axis2=-1)
    if np.max(np.abs(operators-np.array([np.diag(x) for x in diag])))<1e-13:
        def source(self,rho):return (diag@np.diag(rho)).real-offsets
    else:
        def source(self,rho):return np.einsum('ij,vji->v',rho,operators).real-offsets
    model.modular_sources_J=types.MethodType(source,model)


def install_baseline_hamiltonian_cache(model):
    """Cache curvature-independent factors of the baseline Hamiltonian.

    Args:
        model: Matter-only model updated in place.
    """
    edges=np.stack([.5*(model.X_ops[i]@model.X_ops[j]+model.Y_ops[i]@model.Y_ops[j]) for i,j in model._edges])
    xs=np.sum(model.X_ops,axis=0); zs=np.stack(model.Z_ops)
    j0=np.array([model.exchange_J[i,j] for i,j in model._edges])
    inds=np.array(model._edges)
    def H(self,K):
        w=self.omega_0*(1+self.alpha_curv_freq*(K-self.K_ref))
        dk=K/self._A_vertex-self._Kdens_ref
        integ=.5*(dk[inds[:,0]]+dk[inds[:,1]])*self._edge_len
        fac=np.clip(1+self.edge_curv_strength*integ,*self.edge_curv_clip)
        return nk.hermitize(.5*np.einsum('v,vij->ij',w,zs)+.5*self.transverse_hx*xs+np.einsum('e,eij->ij',j0*fac,edges))
    model._H_from_K_edge_integrated=types.MethodType(H,model)


def make_models(c:StudyConfig,fast=True):
    """Construct matched joint-register and matter-only models.

    Args:
        c: Shared physical and numerical parameters.
        fast: Use factors verified against the archived scalar kernels.

    Returns:
        Finite-register model and nonlinear matter-only comparator."""
    surface=nk.geometry_builder('octa',seed=100+17*c.fold)
    nodes,graph,_=nk.surface_to_nodes_graph(surface,name_prefix='o',compute_frames=False,frame_seed=0)
    s=c.rate_scale
    common=dict(nodes=nodes,graph=graph,surface=surface,ball_radius_hops=1,
        alpha_area=c.alpha_area,mass_m=.7,mesh_h=1.,omega_0=2*np.pi*40*s,
        alpha_curv_freq=0. if c.decoupled else .2,gamma0=c.gamma0*s,
        gamma_curv=0. if c.decoupled else c.gamma_curv*s,exchange_J_strength=2.5*s,
        transverse_hx=.25*s,eps_modular=c.source_regularization,eps_ref_rank=1e-6,
        reference_state='zeros',initial_state='plus_product',
        initial_rho_override=population_state(c.initial_population),
        edge_curv_strength=0. if c.decoupled else .15,collective_jump_base=0.,
        collective_jump_kappa=2.,delay_steps=c.delay_steps,memory_kernel='delta')
    klass=FullRegister if fast else nk.OCTAReviewerExtensionsGTRQCSimulator
    fr=klass(**common,regge_a=.8,beta_geom=c.beta_geom,gamma_geom_base=c.gamma_geom*s,
        levels_per_node=7,deltaK_scale=.12,max_configs=c.M,
        seed_geometry_ensemble=c.ensemble_seed+13*c.fold,rate_mode=c.mode,
        geometry_quantum=c.geometry_quantum,geometry_initial='diagonal_gibbs',
        geom_mix_rate=s if c.geometry_quantum else 0.,
        geom_pointer_dephase=c.pointer_dephase*s if c.geometry_quantum else 0.,
        monitor_eta=0. if c.decoupled else .9,enforce_gauss_bonnet=True,
        enforce_zero_mean_if_massless=True,causal_apply_every=0,
        trace_mode='cptp',trace_tol=c.trace_tol,trace_hygiene_tol=1e-6,memory_guard_gib=2.)
    if c.normalized_proposals:
        scale=max(1.,float(fr.A_cfg.sum(axis=0).max()))
        fr.gamma_geom_base/=scale
        fr.R0/=scale;fr.out0/=scale
    if c.geometry_initial=='delta':nk.force_geometry_delta(fr)
    elif c.geometry_initial=='uniform':nk.force_geometry_uniform(fr)
    else:
        weights=fr.pi0
        fr.rho[:]=0
        if c.geometry_initial=='coherent_gibbs':
            fr.rho[0]=np.sqrt(weights[:,None]*weights[None,:])[:,:,None,None]*fr.rho0
        elif c.geometry_quantum:
            for a in range(fr.M):fr.rho[0,a,a]=weights[a]*fr.rho0
        else:fr.rho[0]=weights[:,None,None]*fr.rho0
    nl=nk.SemiclassicalNonlinearGKSLSimulator(**common,feedback_mode='modular')
    install_source_cache(fr);install_source_cache(nl)
    install_baseline_hamiltonian_cache(nl)
    if fast:fr.prepare_fast_factors()
    return fr,nl


def integrate(t,y,left,right):
    """Integrate sampled data on a closed comparison interval.

    Args:
        t: Sample times in seconds.
        y: Metric values at each sample.
        left: Left interval endpoint.
        right: Right interval endpoint.

    Returns:
        Trapezoidal integral, or NaN when fewer than two samples are available.
    """
    mask=(t>=left-1e-12)&(t<=right+1e-12)
    if mask.sum()<2:return float('nan')
    return float(np.trapezoid(y[mask],t[mask]))


def run_pair(c:StudyConfig,output_dir:Path|None=None,overwrite=False):
    """Run a matched pair using the physical-step source controller.

    Args:
        c: Validated experiment configuration.
        output_dir: Parent directory for parameter-hashed result folders.
        overwrite: Recompute even when a cached summary exists.

    Returns:
        Configuration, trajectory metrics, source audits, and numerical diagnostics.

    Raises:
        AssertionError: Recorded sources violate the pure-delay index rule."""
    dest=Path(output_dir)/c.identifier if output_dir else None
    if dest and (dest/'summary.json').exists() and not overwrite:
        return json.loads((dest/'summary.json').read_text())
    started=time.perf_counter();fr,nl=make_models(c)
    jfr0=fr.modular_sources_J(fr.matter_marginal());jnl0=nl.modular_sources_J(nl.rho)
    bf=DelayBuffer(initial_value=jfr0,max_history=max(c.delay_steps+2,8))
    bn=DelayBuffer(initial_value=jnl0,max_history=max(c.delay_steps+2,8))
    times=np.arange(c.steps+1)*c.dt
    rf=[];rn=[];jf=[];jn=[];ef=[];en=[];trace=[];defects=[];pop=[]
    for k,t in enumerate(times):
        fm=fr.matter_marginal().copy();nm=nl.rho.copy()
        f=fr.modular_sources_J(fm);n=nl.modular_sources_J(nm)
        fu=jfr0.copy() if c.frozen else bf.effective(f,delay_steps=c.delay_steps,kernel_kind='delta',dt=c.dt,tau=None)
        nu=jnl0.copy() if c.frozen else bn.effective(n,delay_steps=c.delay_steps,kernel_kind='delta',dt=c.dt,tau=None)
        rf.append(fm);rn.append(nm);jf.append(f);jn.append(n);ef.append(fu);en.append(nu)
        trace.append(fr.total_trace());pop.append(fr.geom_prob())
        if k==c.steps:break
        fr.step_source=fu.copy()
        fr.step_strang(c.dt)
        # A one-step closure defect on the same matter input and same classical source.
        nl.rho=fm.copy();nl._J_buffer=FrozenStepInput(fu)
        shadow,_,_=nl.step(c.dt)
        defects.append(distance(fr.matter_marginal(),shadow))
        nl.rho=nm;nl._J_buffer=FrozenStepInput(nu)
        nl.step(c.dt)
        bf.advance(f);bn.advance(n)
    rf=np.stack(rf);rn=np.stack(rn);jf=np.array(jf);jn=np.array(jn);ef=np.array(ef);en=np.array(en)
    D=np.array([distance(a,b) for a,b in zip(rf,rn)])
    source_activity_fr=np.linalg.norm(ef-ef[0],axis=1)
    source_activity_nl=np.linalg.norm(en-en[0],axis=1)
    def first_active(a):
        ix=np.flatnonzero(a[:-1]>1e-10*max(1.,np.linalg.norm(ef[0])))
        return float(times[ix[0]]) if len(ix) else None
    # The common interval begins after the first update using J(t_1) for max delay .10 s.
    active_start=.014 if c.duration<=.1000000001 else .14
    summary=dict(config=asdict(c),run_id=c.identifier,model_level='full_six_qubit_joint_register',
        source_schedule='step_start_J0_prehistory',duration=c.duration,
        peak_early=float(D[times<=.1+1e-12].max()),area_early=integrate(times,D,0,.1),
        peak_full=float(D.max()),area_full=integrate(times,D,0,c.duration),
        active_start=active_start,area_active=integrate(times,D,active_start,c.duration),
        mean_active=integrate(times,D,active_start,c.duration)/(c.duration-active_start) if c.duration>active_start else None,
        first_active_source_fr=first_active(source_activity_fr),first_active_source_nl=first_active(source_activity_nl),
        max_source_change_fr=float(source_activity_fr.max()),max_source_change_nl=float(source_activity_nl.max()),
        max_trace_error=float(np.abs(np.array(trace)-1).max()),
        min_eigenvalue_fr=float(min(np.linalg.eigvalsh(x).min().real for x in rf)),
        min_eigenvalue_nl=float(min(np.linalg.eigvalsh(x).min().real for x in rn)),
        max_closure_defect=float(max(defects)),mean_closure_defect=float(np.mean(defects)),
        max_kraus_error=fr.max_kraus_error,min_nojump_eigenvalue=fr.min_nojump_eigenvalue,
        maximum_jump_substeps=fr.maximum_jump_substeps,
        geometry_channels=int((fr.A_cfg>0).sum()),max_proposal_degree=float(fr.A_cfg.sum(axis=0).max()),
        connected_component_count=int(component_count(fr.A_cfg)),
        fr_buffer_advances=bf.advance_count,nl_buffer_advances=bn.advance_count,
        wall_seconds=time.perf_counter()-started,python=platform.python_version(),numpy=np.__version__)
    # Direct index verification, independent of DelayBuffer implementation.
    expected=np.maximum(np.arange(c.steps+1)-c.delay_steps,0)
    if c.frozen:expected[:]=0
    summary['source_index_error_fr']=float(np.abs(ef-jf[expected]).max())
    summary['source_index_error_nl']=float(np.abs(en-jn[expected]).max())
    if max(summary['source_index_error_fr'],summary['source_index_error_nl'])>1e-12:
        raise AssertionError('Source history does not satisfy pure-delay rule')
    arrays=dict(time=times,trace_distance=D,source_fr=jf,source_nl=jn,effective_source_fr=ef,
        effective_source_nl=en,geometry_population=np.array(pop),closure_defect=np.array(defects),
        geometry_configurations=fr.K_cfg,configuration_codes=fr.codes,adjacency=fr.A_cfg)
    if c.store_states:arrays.update(rho_fr=rf,rho_nl=rn)
    if dest:
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'trajectory.npz',**arrays)
        (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def component_count(A):
    """Count connected components of a configuration adjacency matrix.

    Args:
        A: Configuration graph adjacency matrix.

    Returns:
        Number of graph components, including isolated configurations.
    """
    seen=set();count=0
    for i in range(len(A)):
        if i in seen:continue
        count+=1;todo=[i];seen.add(i)
        while todo:
            j=todo.pop()
            for k in np.flatnonzero(A[j]):
                if int(k) not in seen:seen.add(int(k));todo.append(int(k))
    return count


def run_many(configurations,output_dir,progress=True):
    """Execute or reuse an ordered sequence of matched pairs.

    Args:
        configurations: Iterable of StudyConfig objects.
        output_dir: Parent directory for result folders.
        progress: Print a completion line for each pair.

    Returns:
        Ordered per-pair summaries.
    """
    out=[]
    for i,c in enumerate(configurations):
        row=run_pair(c,Path(output_dir));out.append(row)
        if progress:print(f'{i+1}/{len(configurations)} {c.identifier} {c.mode} M={c.M} delay={c.delay:g} frozen={c.frozen}',flush=True)
    return out
