# HANA Fundamental Research — Fase 4

Hikari Yumiko · Zaskar. Política de adquisición por ganancia neta frente a la parada.

- [Informe científico, antecedentes y metodología](informe.md)
- [Auditoría de la Fase 3 reproducida](auditoria.md)
- [Datos crudos de las cinco semillas](results/) y [reproducciones completas de Fase 3](reproduction/)
- [Tabla completa de rendimiento](rendimiento.csv) y [diferencias emparejadas](diferencias.csv)
- [Precisión frente a costo](precision-costo.svg), [utilidad frente a presupuesto](utilidad-presupuesto.svg) y [latencia frente a calidad](latencia-calidad.svg)

## Cómo reproducirlo

En Linux x86-64 con Python 3.12 y un hilo de BLAS por proceso:

```sh
cd research/fundamental-phase-4
python -m pip install -r ../fundamental-phase-2/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest -v
mkdir -p /tmp/hana-phase4-repro
for seed in 7 19 43 71 101; do
  python ../fundamental-phase-3/experiment.py --seed "$seed" > "/tmp/hana-phase4-repro/seed-$seed.json"
  python reproduce.py "/tmp/hana-phase4-repro/seed-$seed.json"
  python benchmark.py --seed "$seed" | gzip -n > "results/seed-$seed.json.gz"
done
python -m pip install matplotlib==3.10.8
python report.py
```

El comando de experimentos reescribe datos locales. Comparar latencias de la
misma máquina y sesión, con carga controlada. Los JSON comprimidos contienen
los valores sin redondeo de las tablas. Los resultados conservados proceden
de cinco procesos concurrentes de un hilo en un anfitrión x86-64 con nueve
CPU visibles. GitHub Actions repite las pruebas y experimentos en CPU y
publica un artefacto por semilla.
