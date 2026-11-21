# HANA-0.3 Experimental

El controlador estima dos probabilidades condicionales a partir del token
y del estado actualizado. La primera, p2, decide si ejecutar el segundo
ciclo. La segunda, p3, decide si ejecutar el tercero después del segundo.
En inferencia se usa un umbral de 0,5 y solo se actualizan los ejemplos
que continúan. Durante el entrenamiento se calculan los tres candidatos
para usar decisiones discretas con gradientes straight-through. El costo
diferenciable de ciclos adicionales es p2 + p2 × p3, promediado sobre
todos los tokens. Con peso 0,035 se suma al error de predicción, y con
peso cero se mide el efecto de quitar la penalización.

## Protocolo

Se alternan progresiones modulares en un vocabulario de 7 símbolos y
recuperación asociativa con 8 o 40 distractores. Se evalúan nuevas
muestras de esas tareas y dos distancias de recuperación reservadas, 52
y 80, con secuencias de 77 y 105 tokens. El experimento largo de 105
tokens sitúa todas las claves antes de la ventana de 64 entradas.
El control de 128 entradas conserva esas claves sin aumentar los pesos
entrenables. Los distractores son constantes. Esto mide retención al
aumentar la distancia y deja pendiente la interferencia de distractores
diversos.

Se ejecutan 240 pasos por modelo y semilla con el mismo flujo de datos
y un hilo de CPU. La evaluación usa 96 ejemplos por tarea. Se registra
el promedio de ciclos sobre toda la secuencia y sobre las posiciones
evaluadas, además de histogramas de estas últimas. La latencia mide
una secuencia completa con lote unitario. El pico RSS incluye el
entrenamiento y el entorno Python/PyTorch.

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
for seed in 7 19 43; do
  for model in penalized unpenalized memory-128 adaptive fixed-1 znc gru; do
    python -m hikari.hana03_experiment --model "$model" --seed "$seed"
  done
done
```

`adaptive` y `fixed-1` reutilizan HANA-0.2. `znc` y `gru` son
las referencias recurrentes con presupuestos similares. Los experimentos
entrenan desde pesos aleatorios. GitHub Actions publica métricas por
semilla como artefactos.
