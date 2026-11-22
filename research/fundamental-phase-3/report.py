"""Build all aggregate tables and matched budget plots from raw seed files."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEEDS = (7, 19, 43, 71, 101)
BUDGETS = ("3", "5", "7")
SCENARIOS = ("id", "ood", "noise", "incomplete", "cost-shift", "unhelpful")
POLICIES = ("residual", "p2", "myopic", "fixed", "deterministic", "random", "oracle",
            "always-residual", "no-cost-residual", "two-step")
METRICS = ("accuracy", "utility", "cost_units", "accesses", "abstention", "median_us", "p95_us")


def interval(values):
    m = mean(values)
    s = stdev(values)
    half = 2.776445 * s / len(values) ** .5
    return m, s, m-half, m+half


def stats(values):
    m, s, _, _ = interval(values)
    return f"{m:.4f} ± {s:.4f}"


def tables(runs):
    lines = ["# Resultados completos de la Fase 3", "",
             "Media ± desviación estándar de cinco semillas. Acierto y abstención son proporciones.",
             "La matriz por semilla, el pormenor de decisiones y los conteos exactos están en los JSON.", ""]
    for scenario in SCENARIOS:
        for budget in BUDGETS:
            lines += [f"## {scenario} · presupuesto {budget}", "",
                      "| Política | Acierto | Utilidad | Costo | Accesos | Abstención | Mediana µs | p95 µs |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for p in POLICIES:
                group = [r["test"][scenario][budget][p] for r in runs]
                lines.append("| " + " | ".join([p,*[stats([x[k] for x in group]) for k in METRICS]]) + " |")
            lines.append("")
    lines += ["## Diferencias emparejadas del residual", "",
              "Intervalo t exploratorio del 95 % por semilla, cuatro grados de libertad.",
              "Las diferencias son residual menos comparador. Comparación primaria predefinida: utilidad ID a presupuesto 5 frente a miope.", "",
              "| Escenario | Presupuesto | Comparador | Diferencia de utilidad e IC | Diferencia de acierto e IC |",
              "|---|---:|---|---|---|"]
    for s in SCENARIOS:
        for b in BUDGETS:
            for control in ("p2","myopic","two-step"):
                row=[]
                for k in ("utility","accuracy"):
                    m, _, lo, hi = interval([r["test"][s][b]["residual"][k]
                                              - r["test"][s][b][control][k] for r in runs])
                    row.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                lines.append(f"| {s} | {b} | {control} | {' | '.join(row)} |")
    lines += ["","## Entrenamiento y memoria","",
              "| Modelo | Parámetros | MAC por llamada | Tiempo de ajuste (s) | MSE de validación |",
              "|---|---:|---:|---:|---:|"]
    for p in ("p2","residual","no-cost-residual"):
        group=[r["fit"][p] for r in runs]
        lines.append(f"| {p} | {group[0]['parameters']} | {group[0]['macs_per_network_call']} | "
                     f"{stats([g['seconds'] for g in group])} | {stats([g['validation_mse'] for g in group])} |")
    lines += ["", "Pico RSS (MiB), proceso completo: " + stats([r["rss_mib"] for r in runs]) + ". Energía no medida."]
    return "\n".join(lines)+"\n"


def curves(runs, png=None):
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"hana-phase3","font.size":9})
    fig, axes = plt.subplots(2,3,figsize=(12,6.6),layout="constrained")
    selected = ("residual","p2","myopic","fixed","two-step","oracle")
    for ax, s in zip(axes.flat,SCENARIOS):
        for p in selected:
            xx = [mean(r["test"][s][b][p]["cost_units"] for r in runs) for b in BUDGETS]
            yy = [100*mean(r["test"][s][b][p]["accuracy"] for r in runs) for b in BUDGETS]
            ax.plot(xx,yy,marker="o",ms=3,label=p)
        ax.set(title=s,xlabel="Costo realizado (unidades)",ylabel="Acierto (%)")
        ax.grid(alpha=.2)
    handles, labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=6,frameon=False)
    path=ROOT/"curvas.svg"
    fig.savefig(path,metadata={"Date":None})
    path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())+"\n")
    if png:
        fig.savefig(png,dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--png")
    args=ap.parse_args()
    runs=[json.loads((ROOT/"results"/f"seed-{s}.json").read_text()) for s in SEEDS]
    (ROOT/"resultados.md").write_text(tables(runs))
    curves(runs,args.png)
