# HANA-0 Nano

Este experimento de Hikari Yumiko predice tokens con pesos inicializados al azar.
El núcleo GRUCell conserva un estado compacto. Una compuerta mantiene una
memoria de trabajo y un registro externo al núcleo almacena estados asociados
a los últimos tokens. El registro vive en RAM, tiene capacidad fija y se
transfiere con el estado entre bloques de una secuencia.

El controlador evalúa cada ejemplo por separado y permite entre uno y tres
ciclos por token. En inferencia, los ejemplos detenidos omiten las actualizaciones
restantes del núcleo. Durante el entrenamiento se evalúa un candidato adicional
para estimar el gradiente del controlador en cada decisión. El decodificador
combina el estado recurrente y la memoria de trabajo.

## Reproducir en Linux

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python -m hikari.hana_experiment --model hana --save hana0.pt
python -m hikari.hana_experiment --model znc
python -m hikari.hana_experiment --model gru
```

Cada comando usa un proceso nuevo y un hilo de CPU. Los tres modelos reciben
los mismos lotes sintéticos de progresiones cíclicas, usan AdamW y tienen un
número similar de parámetros. La evaluación excluye el primer token, cuyo
sucesor es ambiguo sin conocer el paso de la progresión. Usa muestras nuevas
de la misma distribución, que pueden coincidir con secuencias vistas.
La latencia es el promedio de 30 secuencias de 12 tokens en lotes de tamaño uno
después de cinco pasadas de calentamiento. El pico RSS incluye PyTorch,
entrenamiento y evaluación. El recuento de ciclos indica refinamientos del
núcleo y no equivale a medir operaciones ni energía.

La prueba inicial mide aprendizaje sintético en CPU Linux. Android y el
límite de 4 GB son objetivos pendientes de validación. El siguiente experimento
debe comparar varias semillas y longitudes, probar recuperación a mayor
distancia y medir variantes con ciclos fijos o sin memoria para aislar el
aporte de cada mecanismo.
