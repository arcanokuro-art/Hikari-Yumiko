"""Regenerate aggregate figures and tables from the per-seed JSON."""
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEEDS = (7, 19, 43, 71, 101)
RUNS = [json.loads((ROOT / "results" / f"seed-{s}.json").read_text()) for s in SEEDS]
POLICIES = ("neural", "myopic", "deterministic", "random", "fixed", "oracle",
            "no-cost", "one-step", "no-memory")
LABELS = ("Neuronal", "Miope", "Determinista", "Aleatorio", "Fijo", "Oráculo",
          "Sin costos explícitos", "Un paso supervisado", "Sin memoria")
SPLITS = ("id", "ood", "unhelpful")


def series(split, policy, metric):
    return [r["test"][split][policy][metric] for r in RUNS]


def spread(xs, factor=1, digits=3):
    return f"{mean(xs)*factor:.{digits}f} ± {stdev(xs)*factor:.{digits}f}"


def tables():
    out = ["# Resultados reproducibles", "", "Medias ± desviación estándar muestral de cinco semillas. Los JSON conservan todas las métricas por semilla y escenario.", ""]
    for split in SPLITS:
        out.extend([f"## {split}", "", "| Política | Acierto (%) | Utilidad | Unidades | Accesos | Mediana (µs) | p95 (µs) |", "|---|---:|---:|---:|---:|---:|---:|"])
        for policy, label in zip(POLICIES, LABELS):
            cells = [spread(series(split, policy, key), 100 if key == "accuracy" else 1)
                     for key in ("accuracy", "utility", "cost_units", "accesses", "median_us", "p95_us")]
            out.append("| " + " | ".join([label, *cells]) + " |")
        out.append("")
    comparison = [a-b for a,b in zip(series("id","neural","utility"), series("id","myopic","utility"))]
    center = mean(comparison)
    margin = 2.776 * stdev(comparison) / len(comparison)**.5
    out.extend(["## Diagnósticos", "",
        f"- Diferencia emparejada de utilidad neuronal menos miope en ID: {center:+.4f}, IC t del 95 % [{center-margin:+.4f}, {center+margin:+.4f}] con 4 grados de libertad.",
        "- Diferencias por semilla: " + ", ".join(f"{s}: {d:+.4f}" for s,d in zip(SEEDS, comparison)) + ".",
        f"- Parámetros del selector: {RUNS[0]['learned_parameters']}. MAC densas por decisión: {RUNS[0]['neural_macs_per_decision']}, más activaciones, posterior y manejo de estado.",
        "- RSS máximo del proceso en MiB: " + spread([r["peak_rss_kib_linux"] for r in RUNS],1/1024) + ". Incluye entrenamiento y todas las políticas.",
        "- Tiempo de ajuste neuronal en segundos, incluida generación supervisada: " + spread([r["training_s"]["neural"] for r in RUNS]) + ".",
        "- Medianas aisladas de acceso en µs: " + ", ".join(f"{a} {spread([r['operator_median_us'][a] for r in RUNS])}" for a in ("local","index","verify")) + ". Las unidades de presupuesto son tarifas simuladas.",
        "- Energía: no medida. Hardware: " + RUNS[0]["machine"] + ". NumPy: " + RUNS[0]["numpy"] + ".", "",
        "| Política | Acierto en ambos extremos del contrafactual (%) | Cambio de acción (%) | Lecturas locales / índice / verificación en ID |", "|---|---:|---:|---:|"])
    for policy, label in zip(POLICIES, LABELS):
        row = [r["counterfactual"][policy] for r in RUNS] if policy != "random" else []
        agreement = spread([r["oracle_pair_agreement"] for r in row],100) if row else "—"
        flips = spread([r["action_change_rate"] for r in row],100) if row else "—"
        calls = " / ".join(f"{mean(r['test']['id'][policy]['action_counts'][a] for r in RUNS):.1f}"
                           for a in ("local","index","verify"))
        out.append(f"| {label} | {agreement} | {flips} | {calls} |")
    (ROOT/"resultados.md").write_text("\n".join(out)+"\n")


def chart():
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"hana-phase-2","font.size":10, "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(9,3.8), constrained_layout=True)
    for i, (p,label,color) in enumerate(zip(POLICIES[:6], LABELS[:6],
                                           ("#006b69","#32475b","#8b5a31","#8a6a9e","#687078","#b24c60"))):
        vals = [series(s,p,"utility") for s in SPLITS]
        axes[0].errorbar([0+i*.07,1+i*.07,2+i*.07], [mean(v) for v in vals],
                         yerr=[stdev(v) for v in vals],fmt="o",color=color,capsize=2,label=label)
        axes[1].scatter(mean(series("id",p,"cost_units")),mean(series("id",p,"accuracy"))*100,
                        color=color,s=42,label=label)
    axes[0].set(xticks=(.175,1.175,2.175),xticklabels=("Conocida","Fuera de distribución","No informativa"),
                ylabel="Utilidad neta")
    axes[1].set(xlabel="Unidades simuladas por episodio",ylabel="Acierto en distribución conocida (%)")
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    for ax in axes:
        ax.grid(alpha=.2)
    svg = ROOT / "rendimiento.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig("/tmp/hana-phase2-chart.png",dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    tables()
    chart()
