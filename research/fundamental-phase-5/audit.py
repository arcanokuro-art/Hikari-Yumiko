"""Decision traces and matched interventions for the Phase 4 controller."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / 'fundamental-phase-4'))
import benchmark as p4
from environment import ACTIONS, Memory, initial, legal, planner, profile, terminal_values

SCENARIOS = (*p4.SCENARIOS, 'joint-new')
POLICIES = ('net','no-margin','absolute','broad','miope-veto','p2','residual',
            'myopic','two-step','oracle','fixed','deterministic','stop')
EDGES = (-.15, -.075, -.035, 0., .02, .05, .10, .20)


def broad_profiles(rng, count):
    profiles = []
    for i in range(count):
        p = profile(rng, 'train')
        if i % 2:
            p = replace(p, fidelity=tuple(rng.uniform(.5,.99,2)),
                        erasure=tuple(rng.uniform(.03,.65,2)),
                        costs=tuple(map(int,rng.integers(1,6,3))))
        profiles.append(p)
    return profiles


def fit_broad(seed, steps):
    start = time.perf_counter()
    train = p4.target_data(broad_profiles(np.random.default_rng(seed+10000),1200))
    valid_profiles = broad_profiles(np.random.default_rng(seed+20000),200)
    valid = p4.target_data(valid_profiles)
    rng = np.random.default_rng(seed)
    model = p4.NetPolicy(rng)
    loss = model.fit((train[0],train[1],train[3]),rng,steps,(valid[0],valid[1],valid[3]))
    scores = {m: statistics.mean(p4.policy_value(p,lambda s,p:model.decide(s,p,m))
                                for p in valid_profiles) for m in p4.MARGINS}
    margin = max(scores,key=scores.__getitem__)
    return (model, margin), dict(mse=loss, margin=margin, margin_utilities=scores,
        seconds=time.perf_counter()-start, parameters=sum(x.size for x in model.params),
        macs_per_call=sum(x.size for x in model.params[::2]), train_states=len(train[0]),
        validation_states=len(valid[0]))


def cases_for(seed, scenario, budget, count):
    if scenario != 'joint-new':
        return p4.cases_for(seed,scenario,budget,count)
    rng = np.random.default_rng(seed+90000)
    cases=[]
    for _ in range(count):
        p = replace(profile(rng,'id'), fidelity=tuple(rng.uniform(.67,.91,2)),
                    erasure=tuple(rng.uniform(.28,.71,2)),
                    costs=(int(rng.integers(1,3)),int(rng.integers(1,4)),int(rng.integers(2,6))),
                    budget=budget)
        cases.append((p,*p4.prior.world(p,rng.random(),rng.random(2))))
    return cases


def get_policy(name, models, seed):
    if name == 'broad':
        model, margin = models[name]
        return lambda s,p:model.decide(s,p,margin)
    if name == 'miope-veto':
        def choose(s,p):
            a = models['net'][0].decide(s,p,models['net'][1])
            return int(np.argmax(planner(p)(s,1))) if a<3 and any(legal(s,p)[3:]) else a
        return choose
    if name == 'p2':
        return p4.prior.policy(name,models,seed)
    return p4.get_policy(name,models,seed)


def oracle_gain(q, stop, legal_ops):
    return [float(q[i]-stop) if legal_ops[i-3] else None for i in range(3,6)]


def score_decisions(cases, model, margin, scenario, budget, trace):
    tally = Counter()
    bins = [dict(n=0,predicted_sum=0.,exact_sum=0.,positive=0) for _ in range(len(EDGES)+1)]
    errors = []
    for episode,(p,label,reports) in enumerate(cases):
        s, memory, evaluate = initial(p), Memory(reports), planner(p)
        for step in range(4):
            mask = legal(s,p)[3:]
            q = evaluate(s,3)
            stop = max(q[:3])
            exact = oracle_gain(q,stop,mask)
            active = [i for i,item in enumerate(mask) if item]
            prediction = model.forward(np.asarray(p4.features(s,p)))[1] if active else None
            a = model.decide(s,p,margin)
            optimal = [i for i,v in enumerate(q) if max(q)-v<=p4.EPS]
            tally['decisions'] += 1
            if active:
                best = max(active,key=prediction.__getitem__)
                pred, truth = float(prediction[best]), exact[best]
                errors.append((pred-truth)**2)
                cell=bins[int(np.searchsorted(EDGES,pred,side='right'))]
                cell['n']+=1
                cell['predicted_sum']+=pred
                cell['exact_sum']+=truth
                cell['positive']+=int(truth>p4.EPS)
                tally['sign_error']+=int((pred>0)!=(truth>p4.EPS))
                tally['scored_states']+=1
            if a not in optimal:
                if a < 3:
                    category = 'margin' if pred>0 else 'nonpositive'
                    tally['premature_'+category]+=1
                    tally['missed_one_step']+=int(max(evaluate(s,1))>stop+p4.EPS)
                else:
                    category = 'harmful' if q[a]<stop-p4.EPS else 'ranking'
                tally['errors']+=1
                if trace is not None:
                    record=dict(scenario=scenario,budget=budget,episode=episode,step=step,
                        state=asdict(s),profile=asdict(p),decision=ACTIONS[a],
                        optimal=[ACTIONS[i] for i in optimal],stop=stop,exact_gain=exact,
                        estimated_gain=[float(prediction[i]) if mask[i] else None for i in range(3)],
                        margin=margin,regret=float(max(q)-q[a]),category=category,
                        realized_label=label,private_reports=reports)
                    trace.write(json.dumps(record,separators=(',',':'))+'\n')
            if a<3:
                break
            s=memory.acquire(s,a,p)
    n=sum(b['n'] for b in bins)
    total=sum(abs(b['predicted_sum']-b['exact_sum']) for b in bins)
    return dict(counts=tally, bins=bins, edges=EDGES, marginal_rmse=statistics.mean(errors)**.5 if n else None,
                weighted_bin_gap=total/n if n else None)


def alias_probe():
    p=replace(profile(np.random.default_rng(4),'id'),prior=(.46,.46,.08),
              fidelity=(.65,.50),erasure=(.04,.04),costs=(1,1,1),budget=3)
    s=initial(p)
    rows=[]
    for f in (.50,.98):
        latent=replace(p,fidelity=(f,p.fidelity[1]))
        q=planner(latent)(s,3)
        rows.append(dict(actual_local_fidelity=f,optimal=[ACTIONS[i] for i,v in enumerate(q) if max(q)-v<=p4.EPS],
                         best_stop=max(q[:3]),best_acquire=max(q[3:]),optimal_utility=max(q)))
    return dict(public_profile=asdict(p),public_features=p4.features(s,p),worlds=rows)


def run(seed,steps,count,trace):
    models,fit=p4.fit_models(seed,steps)
    models['broad'],fit['broad']=fit_broad(seed,steps)
    train_rng,val_rng=(np.random.default_rng(seed+n) for n in (10000,20000))
    train=[profile(train_rng,'train') for _ in range(1200)]
    val=[profile(val_rng,'train') for _ in range(200)]
    start=time.perf_counter()
    rng=np.random.default_rng(seed)
    models['p2']=p4.prior.Controller(rng)
    fit['p2']={'validation_mse':models['p2'].fit(p4.prior.absolute_examples(train,3,False),rng,
                                               steps,p4.prior.absolute_examples(val,3,False)),
               'seconds':time.perf_counter()-start,'parameters':sum(x.size for x in models['p2'].params)}
    outcomes,diagnostic,cf={},{},{}
    for i,scenario in enumerate(SCENARIOS):
        outcomes[scenario],diagnostic[scenario]={},{}
        for budget in p4.BUDGETS:
            cases=cases_for(seed,scenario,budget,count)
            rows={}
            offset=seed%len(POLICIES)
            for name in POLICIES[offset:]+POLICIES[:offset]:
                chooser=lambda:get_policy(name,models,seed+60000+i)
                metric=p4.prior.previous.evaluate(cases,chooser())
                metric.update(p4.selector_timing(cases,chooser()))
                metric['decisions']=p4.diagnose(cases,chooser())
                rows[name]=metric
                if budget==5 and scenario in ('id','ood','joint-new'):
                    cf.setdefault(scenario,{})[name]=p4.prior.previous.counterfactual(cases[:128],chooser())
            outcomes[scenario][budget]=rows
            diagnostic[scenario][budget]={name:score_decisions(cases,*models[name],scenario,budget,
                                            trace if name=='net' else None) for name in ('net','broad')}
    return dict(seed=seed,steps=steps,episodes=count,test=outcomes,calibration=diagnostic,
                counterfactual=cf,fit=fit,alias=alias_probe(),machine=platform.machine(),
                python=platform.python_version(),numpy=np.__version__,
                rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,energy_joules=None)


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--seed',type=int,required=True)
    args.add_argument('--steps',type=int,default=1800)
    args.add_argument('--count',type=int,default=1024)
    args.add_argument('--trace',type=Path,required=True)
    opts=args.parse_args()
    if opts.seed<0 or opts.steps<1 or opts.count<17:
        args.error('nonnegative seed, positive steps and at least 17 episodes required')
    with gzip.open(opts.trace,'wt',encoding='utf-8') as stream:
        print(json.dumps(run(opts.seed,opts.steps,opts.count,stream),allow_nan=False,default=np.generic.item,separators=(',',':')))
