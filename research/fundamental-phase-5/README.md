# HANA Fundamental Research Fase 5

Hikari Yumiko · Zaskar Startup

- [Protocolo e hipótesis falsables](protocolo.md)
- [Método, auditoría y conclusiones](informe.md)
- [Resultados individuales](results/) y [reproducciones de Fase 4](reproduction/)
- [Métricas completas](metricas.csv), [contrastes emparejados](contrastes.csv), [rendimiento](rendimiento.svg) y [calibración](calibracion.svg)

## Reproducción

En Linux con Python 3.12, NumPy fijado por Fase 2 y un hilo de BLAS:

```bash
cd research/fundamental-phase-5
python -m pip install -r ../fundamental-phase-2/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest -v
mkdir -p /tmp/hana5 traces
for seed in 7 19 43 71 101; do
  python ../fundamental-phase-4/benchmark.py --seed "$seed" > "/tmp/hana5/seed-$seed.json"
  python reproduce.py "/tmp/hana5/seed-$seed.json"
  (set -o pipefail; python audit.py --seed "$seed" --trace "traces/seed-$seed.jsonl.gz" | gzip -n > "results/seed-$seed.json.gz") || exit 1
  python uncertainty.py --seed "$seed" > "results/posterior-$seed.json"
done
python -m pip install matplotlib==3.10.8
python report.py
```

El flujo de GitHub Actions genera y publica los resultados y las trazas por
semilla. Las trazas también se pueden regenerar sin GitHub. Cada línea
contiene una decisión subóptima de Net con perfil público, estado, valores
por acción, margen, arrepentimiento, decisión tomada, óptimos, etiqueta real
e informes privados registrados después del episodio. Ese último material
solo se escribe en la auditoría y nunca entra en la red. Los contadores
omitidos en los JSON representan cero. La carpeta de trazas se genera
localmente y se distribuye como artefacto de CI.
