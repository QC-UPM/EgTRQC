"""Regression checks for physical-step scheduling and full-register controls."""
import types
from dataclasses import replace
import numpy as np
import pytest
from egtrqc.controlled_studies import (StudyConfig,FullRegister,FrozenStepInput,
    make_models,run_pair,distance,MODES)

@pytest.mark.parametrize('quantum',[False,True])
@pytest.mark.parametrize('mode',MODES)
def test_vectorized_factors_equal_archived_factors(quantum,mode):
    c=StudyConfig(M=5,mode=mode,geometry_quantum=quantum,duration=.04)
    fast,_=make_models(c);old,_=make_models(c,fast=False)
    j=fast.modular_sources_J(fast.matter_marginal())
    fast.step_source=j;old.step_source=j
    old._compute_rates_from_current_J=types.MethodType(FullRegister._compute_rates_from_current_J,old)
    for _ in range(2):fast.step_strang(c.dt);old.step_strang(c.dt)
    assert np.max(np.abs(fast.rho-old.rho))<2e-11

@pytest.mark.parametrize('q',[False,True])
def test_frozen_identity_and_activation(tmp_path,q):
    c=StudyConfig(M=5,duration=.18,geometry_quantum=q,store_states=True)
    a=run_pair(c,tmp_path);b=run_pair(replace(c,frozen=True),tmp_path)
    x=np.load(tmp_path/a['run_id']/'trajectory.npz');y=np.load(tmp_path/b['run_id']/'trajectory.npz')
    hold=x['time']<=.12+1e-12
    assert np.array_equal(x['rho_fr'][hold],y['rho_fr'][hold])
    assert np.array_equal(x['rho_nl'][hold],y['rho_nl'][hold])
    assert a['first_active_source_fr']==pytest.approx(.12)
    assert a['first_active_source_nl']==pytest.approx(.12)
    assert a['source_index_error_fr']==0.
    assert a['source_index_error_nl']==0.
    assert a['fr_buffer_advances']==9
    assert a['nl_buffer_advances']==9

@pytest.mark.parametrize('dt',[.02,.01,.005,.002])
def test_physical_delay_not_changed_by_step_refinement(dt):
    c=StudyConfig(dt=dt)
    assert c.delay_steps*dt==pytest.approx(.1)


def test_nested_ensembles():
    small,_=make_models(StudyConfig(M=10));large,_=make_models(StudyConfig(M=20))
    assert np.array_equal(small.codes,large.codes[:10])
    assert np.allclose(small.K_cfg,large.K_cfg[:10],atol=0,rtol=0)


def test_homogeneous_source_cannot_shift_constrained_curvature():
    fr,nl=make_models(StudyConfig())
    k=fr.K_cfg-fr.K_ref
    assert np.max(np.abs(k.sum(axis=1)))<1e-13
    j=np.ones(6)*2.
    baseline_shift=nl.solve_G(-j)
    assert np.linalg.norm(baseline_shift)==pytest.approx(2*np.sqrt(6)/.49)
    e0=np.array([.5*fr.alpha_area*x@(fr.G@x) for x in k])
    ej=e0+k@j
    assert np.max(np.abs(ej-e0))<1e-12


def test_fixed_rate_is_a_delay_negative_control(tmp_path):
    c=StudyConfig(M=5,mode='fixed_projector_signed',duration=.16,store_states=True)
    a=run_pair(replace(c,delay=0.),tmp_path);b=run_pair(c,tmp_path)
    x=np.load(tmp_path/a['run_id']/'trajectory.npz');y=np.load(tmp_path/b['run_id']/'trajectory.npz')
    assert np.array_equal(x['rho_fr'],y['rho_fr'])

@pytest.mark.parametrize('q',[False,True])
def test_decoupled_null_model(tmp_path,q):
    c=StudyConfig(M=5,decoupled=True,geometry_quantum=q,duration=.16)
    r=run_pair(c,tmp_path)
    assert r['peak_full']<5e-12
    assert r['max_kraus_error']<5e-12


def test_invalid_grids_rejected():
    with pytest.raises(ValueError):StudyConfig(dt=.03)
    with pytest.raises(ValueError):StudyConfig(initial_population=1.1)
