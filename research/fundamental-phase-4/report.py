"""Regenerate quantitative tables and the three Phase 4 figures."""
import csv
import gzip
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy", "utility", "cost_units", "accesses", "memory_payload_reads",
           "abstention", "median_us", "p95_us", "selector_mean_us_per_episode",
           "selector_mean_us_per_call")
CONTROLS = ("residual", "myopic", "two-step", "oracle", "no-margin", "absolute")
PLOTS = ("net", "residual", "myopic", "two-step", "oracle", "fixed")
FILES = ("precision-costo", "utilidad-presupuesto", "latencia-calidad")


def load_runs():
    return [json.loads(gzip.decompress(p.read_bytes())) for p in sorted((ROOT / "results").glob("*.json.gz"))]


def interval(values):
    middle = mean(values)
    sd = stdev(values)
    error = 2.776445105 * sd / len(values) ** .5
    return middle, sd, middle - error, middle + error


def matrix(runs):
    rows = []
    intervals = []
    for scenario, budgets in runs[0]["test"].items():
        for budget, policies in budgets.items():
            measured = {p: {k: [d["test"][scenario][budget][p][k] for d in runs]
                            for k in METRICS} for p in policies}
            for p, metric in sorted(measured.items()):
                rows.extend((scenario, budget, p, k, *interval(v)) for k, v in metric.items())
            for p in CONTROLS:
                for k in METRICS:
                    differences = [a - b for a, b in zip(measured["net"][k], measured[p][k])]
                    intervals.append((scenario, budget, p, k, *interval(differences)))
    return rows, intervals


def csv_file(name, rows):
    with (ROOT / name).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("scenario", "budget", "policy_or_control", "metric", "mean", "sd", "ci95_low", "ci95_high"))
        writer.writerows(rows)


def produce_figures(runs):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "hana-phase4", "font.size": 10})
    specs = (("cost_units", "accuracy", "Costo simulado", "Acierto (%)"),
             ("budget", "utility", "Presupuesto simulado", "Utilidad neta"),
             ("median_us", "accuracy", "Mediana de latencia (µs)", "Acierto (%)"))
    for name, (x, y, xlab, ylab) in zip(FILES, specs):
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8), layout="constrained")
        for ax, scenario in zip(axes.flat, runs[0]["test"]):
            for p in PLOTS:
                points = [(int(b) if x == "budget" else mean(r["test"][scenario][b][p][x] for r in runs),
                           mean(r["test"][scenario][b][p][y] for r in runs)) for b in ("3", "5", "7")]
                ax.plot([a for a, _ in points], [b * (100 if y == "accuracy" else 1) for _, b in points],
                        marker="o", markersize=3, label=p)
            ax.set(title=scenario, xlabel=xlab, ylabel=ylab)
            ax.grid(alpha=.2)
        axes.flat[-1].axis("off")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=6, frameon=False)
        path = ROOT / f"{name}.svg"
        fig.savefig(path, metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        fig.savefig(f"/tmp/hana4-{name}.png", dpi=170)
        plt.close(fig)


def main():
    runs = load_runs()
    if len(runs) != 5:
        raise ValueError("Expected the five seed files")
    result, differences = matrix(runs)
    csv_file("rendimiento.csv", result)
    csv_file("diferencias.csv", differences)
    produce_figures(runs)


if __name__ == "__main__":
    main()
