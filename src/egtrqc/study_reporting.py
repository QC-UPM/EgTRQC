"""Tables, figures and numerical checks for the integrated benchmark release."""
from __future__ import annotations
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .controlled_studies import StudyConfig, distance

from .study_controller import STUDIES


def load_study(root,study):
    """Load completed study summaries into a table.

    Args:
        root: Repository root.
        study: Study index name.

    Returns:
        DataFrame combining configuration fields and scalar diagnostics.
    """
    root=Path(root);p=root/f'results/{study}/index.json'
    rows=json.loads(p.read_text())
    return pd.DataFrame([{**r['config'],**{k:v for k,v in r.items() if k!='config'}} for r in rows])


def summarize(root):
    """Regenerate aggregate tables from completed study indices.

    Args:
        root: Repository root containing the supplied studies.

    Returns:
        Directory containing run tables, paired contrasts, and fold summaries.
    """
    root=Path(root);tables=root/'results/tables';tables.mkdir(exist_ok=True,parents=True)
    for study in STUDIES:
        df=load_study(root,study)
        df.to_csv(tables/f'{study}_runs.csv',index=False)
        if study in ('source_controls','active_short_lag','long_window'):
            b=df[~df.frozen]
            b.groupby(['mode','geometry_quantum','delay'])[['area_early','area_active','area_full','peak_early','peak_full','max_source_change_fr','max_source_change_nl']].agg(['mean','std']).to_csv(tables/f'{study}_summary.csv')
            pivot=b.pivot(index=['fold','mode','geometry_quantum'],columns='delay',values='area_active')
            pivot.subtract(pivot[0.],axis=0).to_csv(tables/f'{study}_paired_contrasts.csv')
        elif study=='register_scaling':
            df.groupby(['mode','normalized_proposals','delay','M'])[['area_active','peak_early','mean_closure_defect']].agg(['mean','std']).to_csv(tables/f'{study}_summary.csv')
        elif study=='population_refinement':
            population_refinement_summary(root).to_csv(tables/'population_refinement_summary.csv',index=False)
        elif study=='active_refinement':
            refinement_summary(root).to_csv(tables/'active_refinement_summary.csv',index=False)
        else:
            labels=json.loads((root/'configs/full_robustness_labels.json').read_text());df['condition']=df.run_id.map(labels)
            rows=[]
            for name,g in df.groupby('condition'):
                z=g[g.delay==0.].set_index('fold');d=g[g.delay==.1].set_index('fold');delta=d.area_active-z.area_active
                rows.append(dict(condition=name,area0_mean=z.area_active.mean(),area5_mean=d.area_active.mean(),delta_mean=delta.mean(),delta_std=delta.std(),peak_min=g.peak_early.min(),peak_max=g.peak_early.max(),trace_error_max=g.max_trace_error.max()))
            pd.DataFrame(rows).to_csv(tables/f'{study}_summary.csv',index=False)
    reduced=pd.DataFrame(json.loads((root/'results/reduced_topology/index.json').read_text()))
    reduced.to_csv(tables/'reduced_topology_summary.csv',index=False)
    return tables


def frozen_checks(root,study='source_controls'):
    """Compare frozen and delayed trajectories during the common hold interval.

    Args:
        root: Repository root.
        study: Study containing frozen and active controls.

    Returns:
        Per-fold and per-mode source, curve, and optional state comparisons.
    """
    root=Path(root);df=load_study(root,study);checks=[]
    for keys,g in df.groupby(['fold','mode','geometry_quantum']):
        frozen=g[g.frozen].iloc[0];delayed=g[(~g.frozen)&(g.delay==g.delay.max())].iloc[0]
        a=np.load(root/f'results/{study}/{frozen.run_id}/trajectory.npz')
        b=np.load(root/f'results/{study}/{delayed.run_id}/trajectory.npz')
        hold=b['time']<=delayed['delay']+delayed['dt']+1e-12
        source_hold=b['time']<delayed['delay']+delayed['dt']-1e-12
        row=dict(fold=int(keys[0]),mode=keys[1],geometry_quantum=bool(keys[2]),
                 maximum_trace_distance_curve_difference_during_hold=float(np.abs(a['trace_distance'][hold]-b['trace_distance'][hold]).max()),
                 maximum_applied_source_difference_during_hold=float(max(np.abs(a['effective_source_fr'][source_hold]-b['effective_source_fr'][source_hold]).max(),np.abs(a['effective_source_nl'][source_hold]-b['effective_source_nl'][source_hold]).max())))
        if 'rho_fr' in a.files and 'rho_fr' in b.files:
            row['max_state_distance_fr_during_hold']=max(distance(x,y) for x,y in zip(a['rho_fr'][hold],b['rho_fr'][hold]))
            row['max_state_distance_nl_during_hold']=max(distance(x,y) for x,y in zip(a['rho_nl'][hold],b['rho_nl'][hold]))
            row['max_state_distance_fr_after_hold']=max(distance(x,y) for x,y in zip(a['rho_fr'][~hold],b['rho_fr'][~hold]))
            row['max_state_distance_nl_after_hold']=max(distance(x,y) for x,y in zip(a['rho_nl'][~hold],b['rho_nl'][~hold]))
        checks.append(row)
    (root/f'results/{study}/frozen_checks.json').write_text(json.dumps(checks,indent=2))
    return checks


def curvature_support_checks(root):
    """Measure the finite-register curvature support obstruction.

    Args:
        root: Repository root containing register-scaling results.

    Returns:
        Per-configuration curvature support diagnostics.
    """
    root=Path(root);df=load_study(root,'register_scaling');rows=[]
    for r in df.itertuples():
        a=np.load(root/f'results/register_scaling/{r.run_id}/trajectory.npz')
        J=a['effective_source_nl'];lower=np.abs(J.sum(axis=1))/(r.alpha_area*.7**2*np.sqrt(6))
        rows.append(dict(run_id=r.run_id,fold=r.fold,mode=r.mode,M=r.M,delay=r.delay,
            normalized_proposals=r.normalized_proposals,
            curvature_common_mode_lower_bound_initial=float(lower[0]),
            curvature_common_mode_lower_bound_minimum=float(lower.min()),
            curvature_common_mode_lower_bound_maximum=float(lower.max())))
    out=pd.DataFrame(rows);out.to_csv(root/'results/tables/curvature_support.csv',index=False)
    return out


def figures(root):
    """Render benchmark figures from completed results.

    Args:
        root: Repository root containing study indices and trajectories.

    Returns:
        Directory containing the rendered figures.
    """
    root=Path(root);out=root/'results/figures';out.mkdir(exist_ok=True,parents=True)
    df=load_study(root,'source_controls');sel=df[(df.fold==0)&(df['mode']=='instantaneous_projector_signed')&(~df.geometry_quantum)]
    frozen=sel[sel.frozen].iloc[0];delayed=sel[(~sel.frozen)&(sel.delay==.1)].iloc[0]
    a=np.load(root/f'results/source_controls/{frozen.run_id}/trajectory.npz')
    b=np.load(root/f'results/source_controls/{delayed.run_id}/trajectory.npz');t=b['time']
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for key,label in [('rho_fr','Finite-register matter'),('rho_nl','Matter-only baseline')]:
        d=np.array([distance(x,y) for x,y in zip(a[key],b[key])]);ax.semilogy(t,np.maximum(d,1e-16),marker='.',label=label)
    ax.axvline(.12,linestyle='--',label='First non-initial source used')
    ax.set(xlabel='Time (s)',ylabel='Distance from frozen-source trajectory',title='Explicit frozen-source control')
    ax.legend();fig.tight_layout();fig.savefig(out/'frozen_source_control.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for key,label in [('effective_source_fr','Finite register'),('effective_source_nl','Baseline')]:
        j=b[key];ax.plot(t,np.linalg.norm(j-j[0],axis=1),marker='.',label=label)
    ax.axvline(.12,linestyle='--');ax.set(xlabel='Time (s)',ylabel='Applied-source change (Euclidean norm)',title='Delayed input activation');ax.legend();fig.tight_layout();fig.savefig(out/'source_activation.png',dpi=180);plt.close(fig)
    for study in ('source_controls','active_short_lag','long_window'):
        data=load_study(root,study);data=data[(~data.frozen)&(~data.geometry_quantum)]
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for mode,g in data.groupby('mode'):
            p=g.pivot(index='fold',columns='delay',values='area_active');d=p.subtract(p[0.],axis=0)
            ax.errorbar(d.columns.to_numpy()*1000,d.mean().values,yerr=d.std().values,marker='o',capsize=3,label=mode.replace('_projector','').replace('_',' '))
        ax.axhline(0.,linestyle=':');ax.set(xlabel='Physical feedback delay (ms)',ylabel='Paired active-window area difference (s)',title=study.replace('_',' ').capitalize())
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{study}_contrasts.png',dpi=180);plt.close(fig)
    data=load_study(root,'register_scaling');data=data[(~data.normalized_proposals)&(data.delay==0.)]
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for mode,g in data.groupby('mode'):
        p=g.pivot(index='fold',columns='M',values='area_active')
        ax.errorbar(p.columns,p.mean(),yerr=p.std(),marker='o',capsize=3,label=mode.replace('_projector','').replace('_',' '))
    ax.set(xlabel='Register size M',ylabel='Active-window integrated discrepancy (s)',title='Nested register ensembles; fixed base jump rate');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'nested_register_scaling.png',dpi=180);plt.close(fig)
    data=load_study(root,'long_window'); data=data[(~data.frozen)&data['mode'].str.startswith('instantaneous')]
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for mode in ('instantaneous_projector_signed','instantaneous_projector_unsigned'):
        for delay in (0.,.1):
            group=data[(data['mode']==mode)&(data.delay==delay)]
            curves=[]
            for row in group.itertuples():
                z=np.load(root/f'results/long_window/{row.run_id}/trajectory.npz')
                curves.append(z['trace_distance']);t=z['time']
            ax.plot(t,np.mean(curves,axis=0),linestyle='-' if delay==0 else '--',
                label=f"{mode.replace('instantaneous_projector_','')}; delay={delay:.2f} s")
    ax.set(xlabel='Time (s)',ylabel='Mean trace distance over three folds',title='Computed full-horizon controls')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'full_horizon_trace_distance.png',dpi=180);plt.close(fig)
    pop=population_refinement_summary(root);fig,ax=plt.subplots(figsize=(7.2,4.5))
    for population,g in pop.groupby('population'):
        ax.errorbar(g.dt,g.delta_mean,yerr=g.delta_std,marker='o',capsize=3,label=f'Initial population {population}')
    ax.axhline(0.,linestyle=':');ax.set(xlabel='Time step (s)',ylabel='Paired active-window area difference (s)',title='Population dependence at fixed physical delay')
    ax.legend();fig.tight_layout();fig.savefig(out/'population_refinement.png',dpi=180);plt.close(fig)
    return out


def refinement_summary(root):
    """Summarize paired active contrasts at fixed physical short delay.

    Args:
        root: Repository root.

    Returns:
        DataFrame of fold contrasts grouped by policy and time step.
    """
    base=load_study(root,'active_short_lag')
    base=base[(~base.geometry_quantum)&(~base.frozen)&base['mode'].str.startswith('instantaneous')]
    data=pd.concat([base,load_study(root,'active_refinement')],ignore_index=True)
    rows=[]
    for (mode,dt),group in data.groupby(['mode','dt']):
        p=group.pivot(index='fold',columns='delay',values='area_active')
        contrast=p[.01]-p[0.]
        rows.append(dict(mode=mode,dt=dt,delta_mean=contrast.mean(),delta_std=contrast.std(),
            delta_min=contrast.min(),delta_max=contrast.max(),area0_mean=p[0.].mean(),
            peak_mean=group[group.delay==0.].peak_early.mean()))
    return pd.DataFrame(rows)


def validate_release(root):
    """Validate manifest identities, trajectories, and numerical audits.

    Args:
        root: Repository root containing the completed results.

    Returns:
        Execution counts and extrema of the numerical diagnostics.

    Raises:
        AssertionError: A count, identifier, source audit, or numerical bound fails."""
    root=Path(root);counts={};maxima={};minimum={};allrows=[]
    for study in STUDIES:
        configs=json.loads((root/f'configs/{study}.json').read_text())
        rows=json.loads((root/f'results/{study}/index.json').read_text())
        assert len(configs)==len(rows),(study,'manifest count')
        for config,row in zip(configs,rows):
            c=StudyConfig(**config)
            assert c.identifier==row['run_id']
            assert c.identifier==StudyConfig(**row['config']).identifier
            assert row['fr_buffer_advances']==c.steps==row['nl_buffer_advances']
            assert max(row['source_index_error_fr'],row['source_index_error_nl'])<=1e-12
            assert row['max_trace_error']<=c.trace_tol
            assert min(row['min_eigenvalue_fr'],row['min_eigenvalue_nl'])>=-1e-10
            path=root/f'results/{study}/{c.identifier}/trajectory.npz'
            with np.load(path,allow_pickle=False) as z:
                assert len(z['time'])==c.steps+1
                assert len(z['closure_defect'])==c.steps
                assert np.isfinite(z['trace_distance']).all()
                assert z['trace_distance'].min()>=-1e-12
                assert z['trace_distance'].max()<=1+1e-10
                assert np.max(np.abs(z['geometry_configurations'].sum(axis=1)-4*np.pi))<1e-10
        counts[study]=len(rows);allrows.extend(rows)
    for key in ['max_trace_error','max_kraus_error','source_index_error_fr','source_index_error_nl']:
        maxima[key]=max(r[key] for r in allrows)
    for key in ['min_eigenvalue_fr','min_eigenvalue_nl','min_nojump_eigenvalue']:
        minimum[key]=min(r[key] for r in allrows)
    reduced=json.loads((root/'results/reduced_topology/index.json').read_text())
    assert len(reduced)==60
    assert all(r['delay_audit']['passed'] and r['advance_audit']['passed'] for r in reduced)
    out=dict(status='passed',full_matched_pairs=sum(counts.values()),
        full_model_trajectories=2*sum(counts.values()),reduced_model_trajectories=len(reduced),
        studies=counts,numerical_maxima=maxima,numerical_minima=minimum,
        note='Counts include repeated configurations across studies; they are execution counts, not independent statistical samples.')
    (root/'provenance/release_validation.json').write_text(json.dumps(out,indent=2))
    return out


def population_refinement_summary(root):
    """Summarize population-dependent contrasts across time refinements.

    Args:
        root: Repository root.

    Returns:
        DataFrame of population and time-step contrasts with fold ordering counts.
    """
    base=load_study(root,'full_robustness')
    base=base[base.initial_population.isin([.2,.8])]
    data=pd.concat([base,load_study(root,'population_refinement')],ignore_index=True)
    rows=[]
    for (population,dt),group in data.groupby(['initial_population','dt']):
        p=group.pivot(index='fold',columns='delay',values='area_active')
        contrast=p[.1]-p[0.]
        rows.append(dict(population=population,dt=dt,mean0=p[0.].mean(),mean5=p[.1].mean(),
            delta_mean=contrast.mean(),delta_std=contrast.std(),delta_min=contrast.min(),
            delta_max=contrast.max(),decreasing_folds=int((np.diff(p.values,axis=1)<0).all(axis=1).sum()),
            max_source_fr=group.max_source_change_fr.max(),max_source_nl=group.max_source_change_nl.max()))
    return pd.DataFrame(rows)
