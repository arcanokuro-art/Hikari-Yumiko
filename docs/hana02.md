# HANA-0.2 Experimental

HANA-0.2 conserva el núcleo GRUCell, la memoria de trabajo con compuerta y el
registro asociativo de HANA-0. El modelo aprende desde pesos aleatorios. La ruta
de inferencia para un ejemplo evita seleccionar y reinsertar filas al continuar
un ciclo. La proyección de salida se aplica de una vez a la secuencia.

## Protocolo en CPU

Cada proceso entrena un modelo desde cero con un hilo, AdamW, 180 pasos,
lotes de 12 y una de tres semillas (7, 19 y 43). Los lotes se generan
separadamente de la inicialización y son iguales entre modelos para una
misma semilla. Se alternan estas tareas dentro de un mismo entrenamiento:

| Tarea | Tokens | Posiciones evaluadas |
|---|---:|---|
| Progresión modular | 12 | Predicción después de observar dos tokens |
| Recuperación corta | 13 | Cuatro claves consultadas en orden aleatorio |
| Recuperación larga | 49 | Dieciséis claves consultadas en orden aleatorio |

Cada clave es única dentro del ejemplo. Los valores se asignan al azar y se
presentan antes de su clave para que el registro pueda guardar el estado
correspondiente al procesarla. Los 64 espacios permiten conservar todas
las asociaciones de estas tres tareas. La evaluación usa 96 secuencias
generadas con semillas separadas para cada tarea.

El control sin memoria conserva la misma entrada del núcleo con un vector
cero para igualar los parámetros. Los controles fijos usan uno, dos o tres
ciclos y omiten la compuerta de continuación. ZNC-0 y la GRU usan anchuras
que aproximan el presupuesto de HANA. El Nano original también se prueba
con 64 espacios para igualar la capacidad. La compuerta de HANA se entrena
mediante una estimación de gradiente straight-through y el error de predicción.
El benchmark registra histogramas de ciclos en las posiciones puntuadas.
No se proporciona una etiqueta de dificultad ni una cantidad objetivo
de ciclos.

La latencia es la mediana de 15 pasadas completas de un ejemplo después de
cuatro calentamientos. El pico RSS abarca todo el proceso, incluido PyTorch,
entrenamiento y evaluación. La medición pareada de rutas reutiliza los mismos
pesos, entradas y decisiones, alternando el orden de medición.

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python -m hikari.hana02_experiment --profile
for seed in 7 19 43; do
  for model in nano adaptive no-memory fixed-1 fixed-2 fixed-3 znc gru; do
    python -m hikari.hana02_experiment --model "$model" --seed "$seed"
  done
done
```

GitHub Actions guarda un JSON por modelo y semilla, además del perfil de
operaciones. Para la siguiente ronda conviene probar distancias más allá de
la capacidad del registro y separar los costos de la memoria de trabajo, la
memoria asociativa y el enrutamiento. El objetivo de Android con 4 GB
requiere una medición de inferencia en el dispositivo.
