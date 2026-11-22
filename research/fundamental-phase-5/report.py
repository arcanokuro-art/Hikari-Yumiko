"""Produce full metric tables, paired intervals, and calibration plots."""
import csv
import gzip
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
KEYS = ('accuracy','utility','cost_units','accesses','abstention','median_us','p95_us',
        'selector_mean_us_per_episode','selector_mean_us_per_call')
CONTROLS = ('net','no-margin','p2','residual','myopic','two-step','oracle','fixed',
            'deterministic','stop','absolute')
COLORS = {'net':'#1f77b4','broad':'#00704d','miope-veto':'#a64b00','myopic':'#994888',
          'residual':'#555555','two-step':'#c72642','oracle':'#111111'}


def load():
    runs=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted((ROOT/'results').glob('seed-*.json.gz'))]
    if {r['seed'] for r in runs} != {7,19,43,71,101}:
        raise ValueError('Expected five independent seeds')
    return sorted(runs,key=lambda r:r['seed'])


def ci(xs):
    m,s=mean(xs),stdev(xs)
    e=2.776445105*s/len(xs)**.5
    return m,s,m-e,m+e


def export(runs):
    with (ROOT/'metricas.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['escenario','presupuesto','politica','metrica','media','desviacion','ic95_inf','ic95_sup'])
        for scenario, budgets in runs[0]['test'].items():
            for budget, model in budgets.items():
                for name in sorted(model):
                    for k in KEYS:
                        w.writerow((scenario,budget,name,k,*ci([r['test'][scenario][budget][name][k] for r in runs])))
    with (ROOT/'contrastes.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['escenario','presupuesto','politica','control','metrica','media','desviacion','ic95_inf','ic95_sup'])
        for scenario, budgets in runs[0]['test'].items():
            for budget in budgets:
                for name in ('broad','miope-veto'):
                    for control in CONTROLS:
                        for k in ('accuracy','utility','cost_units','median_us','selector_mean_us_per_episode'):
                            w.writerow((scenario,budget,name,control,k,*ci([
                                r['test'][scenario][budget][name][k]-r['test'][scenario][budget][control][k]
                                for r in runs])))


def draw(runs):
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'hana-phase5','font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for col,scenario in enumerate(('id','ood','joint-new')):
        for name,color in COLORS.items():
            a,b,c,d=([mean(r['test'][scenario][k][name][metric] for r in runs) for k in ('3','5','7')]
                      for metric in ('cost_units','accuracy','median_us','utility'))
            axes[0,col].plot(a,[100*x for x in b],'-o',ms=3,color=color,label=name)
            axes[1,col].plot(c,d,'-o',ms=3,color=color)
        axes[0,col].set(title=scenario,xlabel='Unidades simuladas',ylabel='Precisión (%)')
        axes[1,col].set(xlabel='Latencia mediana de CPU (µs)',ylabel='Utilidad neta')
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=7)
    save(fig,'rendimiento')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,s in zip(axes,('id','ood','joint-new')):
        for name in ('net','broad'):
            bins=[r['calibration'][s]['5'][name]['bins'] for r in runs]
            xy=[]
            for group in zip(*bins):
                n=sum(g['n'] for g in group)
                if n:
                    xy.append(tuple(sum(g[k] for g in group)/n for k in ('predicted_sum','exact_sum')))
            ax.plot(*zip(*xy),'-o',color=COLORS[name],label=name)
        lo,hi=ax.get_xlim()
        ax.plot([lo,hi],[lo,hi],'--',color='gray',label='calibración ideal')
        ax.set(title=s,xlabel='Ganancia predicha',ylabel='Ganancia exacta')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    save(fig,'calibracion')


def save(fig,name):
    path=ROOT/f'{name}.svg'
    fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(f'/tmp/hana5-{name}.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    records=load()
    export(records)
    draw(records)
