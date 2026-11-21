# HANA-0.4 Experimental

## Tarea y controles

Cada episodio usa un grafo nuevo. Tres relaciones representan permutaciones
independientes sobre un subconjunto aleatorio de 10 de 24 identificadores.
Se baraja el orden de todas las aristas y se ejecuta una composición de
una a tres relaciones distintas desde un nodo inicial uniforme. Un salto
mide recuperación asociativa y dos o tres saltos requieren encadenarla.
El destino correcto solo se entrega a la función de pérdida. Todas las
entidades activas aparecen como destino una vez por relación, de modo
que su frecuencia de aparición no identifica la respuesta. Usar bancos
independientes y relaciones distintas evita sesgos de retorno al aplicar
repetidamente una misma permutación. El número de relaciones se desprende
de la consulta completa, visible por igual para todos los modelos.

La evaluación separa grafos nuevos con la distribución de entrenamiento,
una etiqueta de relación ausente del entrenamiento (renombra un banco
sin cambiar el tamaño), órdenes de relaciones reservados y grafos largos
de 18 nodos y 54 aristas. El control de relación evalúa el acceso a un
banco con una etiqueta nueva, no la adquisición de una regla semántica.
Se verifica el objetivo con un solucionador independiente y mediante un
contrafactual que cambia el resultado al intercambiar dos destinos sin
cambiar la consulta ni el histograma de entidades.

## Modelo

El estado inicial combina las aristas comprimidas, el nodo inicial y
la consulta completa. Cada ciclo lee un banco explícitamente indexado
por la relación de ese paso. Una GRUCell combina el puntero actual con
lo recuperado, y una compuerta entrenable actualiza el puntero. El
decodificador aprende a generar la respuesta a partir del estado y del
puntero. La tabla indexada ofrece un sesgo estructural: cada ciclo
dispone de una recuperación exacta por relación sobre un puntero
distribucional. HANA-0.3, ZNC-0 y GRU reciben las mismas aristas y consultas
como secuencia de tokens y carecen de ese formato de acceso. Sus
resultados miden la diferencia entre sistemas completos, incluida esta
estructura.

Las variantes fijas comparten núcleo, memoria y codificadores. Si
ejecutan más ciclos que relaciones, conservan el puntero en los ciclos
de relleno y siguen actualizando el estado. La ablación sin memoria
conserva la consulta completa y el resumen comprimido de hechos. Para
entrenar el controlador se evalúan los tres candidatos, se calienta el
decodificador y se usa una distribución objetivo proporcional a
exp(-(error de salida + 0,15 × ciclos adicionales) / 0,08). Este
objetivo está separado del gradiente del decodificador. Quitar el costo
conserva el mismo procedimiento. No se usan las profundidades como
etiquetas objetivo del controlador. La longitud de la consulta y el
índice del ciclo son entradas explícitas, junto al estado neuronal.
En inferencia se ejecutan solo los ciclos seleccionados.

## Reproducción en CPU

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
for seed in 7 19 43; do
  for model in adaptive no-cost no-memory fixed-1 fixed-2 fixed-3 hana03 znc gru; do
    python -m hikari.hana04_experiment --model "$model" --seed "$seed"
  done
done
```

Cada ejecución entrena desde cero durante 240 pasos con lotes de 16,
un hilo de CPU y secuencias de datos reproducibles. Se evalúan 192
consultas por combinación de profundidad y conjunto. La latencia
incluye la construcción de la tabla para una consulta y se mide como
la mediana de siete pasadas tras tres calentamientos. Los ciclos de
razonamiento se registran para HANA-0.4. En las referencias
se registran actualizaciones recurrentes de la secuencia completa,
que representan otra unidad de cómputo. El pico de RAM incluye
PyTorch, entrenamiento y evaluación. GitHub Actions publica los JSON
por modelo y semilla.
