# HANA-0.5 Experimental

## Profundidad oculta

La consulta siempre tiene un nodo inicial, tres relaciones distintas y
un atributo objetivo. La respuesta es el primer nodo encontrado cuyo
atributo coincide con el objetivo. Nunca se entregan al modelo la
profundidad, el nodo objetivo, la fase del grafo ni una señal de fin
insertada en la secuencia. Al llegar a tres ciclos se devuelve el
resultado del último ciclo.

Cada episodio ofrece tres consultas sobre el mismo grafo, con el mismo
inicio y orden de relaciones. Los tres objetivos se seleccionan como
atributos distintos en orden aleatorio. Solo cambia cuál de esos tres
atributos solicita la consulta. Todas las fases tienen el mismo
histograma de atributos y las aristas y nodos se barajan. Los atributos
tienen múltiples coincidencias en el grafo, incluso en cada fase.
La diferencia entre las consultas se determina al recorrer las aristas.

Los conjuntos reservados incluyen una etiqueta de banco ausente del
entrenamiento, un orden de composición ausente y grafos de 48 nodos y
144 aristas frente a 32 y 96. Los contrafactuales intercambian dos
destinos del mismo banco y fase. Ambos tienen el mismo atributo
objetivo. Esto cambia la respuesta mientras conserva la consulta, la
profundidad, el orden de filas y las frecuencias marginales. Un oráculo
independiente verifica el primer resultado válido.

## Comparación de políticas

Una lectura exacta de la tabla recupera el siguiente nodo. Una GRUCell
actualiza el estado y un decodificador entrenado reconstruye la identidad
del nodo actual. El controlador neuronal recibe el estado y los dos
atributos por separado. Aprende la condición de parada a partir de
comprobaciones de igualdad realizadas durante el entrenamiento. La
regla determinista aplica directamente la misma comprobación al nodo
recuperado. Para ambas, solo los ejemplos activos ejecutan el ciclo
siguiente.

Las políticas neuronal, determinista, aleatoria y de uno, dos y tres
ciclos fijos comparten exactamente los mismos pesos entrenados por
semilla, además de memoria e interfaz. El control fijo responde en su
último ciclo y no puede detenerse antes. El aleatorio toma un presupuesto
uniforme de uno a tres ciclos antes de leer. La ablación sin memoria
compara tres ciclos fijos con el mismo número de parámetros, conserva
resúmenes neuronales de todos los hechos y atributos, y entrena
directamente contra el objetivo al carecer de nodos recuperados.

Este diseño prueba una política neuronal para un criterio explícito y
verificable en el estado recuperado. La búsqueda usa una tabla
simbólica y la señal de entrenamiento de parada deriva de una igualdad
de atributos. La generalización de etiqueta mide el acceso a un banco
nuevo con la misma interfaz. El experimento no evalúa la adquisición
de nuevos significados ni razonamiento general.

## Reproducción

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
for seed in 7 19 43; do
  python -m hikari.hana05_experiment --seed "$seed"
  python -m hikari.hana05_experiment --seed "$seed" --no-memory
done
```

Se usan 400 pasos, lotes de 24 y un hilo de CPU. Los flujos aleatorios
de entrenamiento, evaluación y política aleatoria son independientes.
Cada conjunto evalúa 192 ejemplos por profundidad y semilla. Los JSON
registran precisión, histograma de ciclos, detenciones tempranas,
exactas y tardías, certificados de coincidencia y latencia unitaria.
El contrafactual registra además la precisión conjunta del par y los
cambios de respuesta. La latencia es la mediana de 11 consultas
unitarias tras tres calentamientos e
incluye la construcción de la tabla. El pico RSS incluye Python,
PyTorch, entrenamiento y evaluación. Los tiempos y el RSS de un
modelo compartido se registran una vez, no como mediciones independientes
por política.
