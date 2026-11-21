# HANA-0.6 Experimental

## Evidencia parcial y desconocido

Se reutilizan los grafos de HANA-0.5. Todas las consultas tienen tres
relaciones y un atributo objetivo. El objetivo se elige uniformemente
entre los tres atributos del recorrido limpio y el cuarto atributo,
ausente en el recorrido. Se oculta alrededor del 15 % de las aristas
de cada grafo antes de construir la memoria. Un salto sin arista visible
corta la verificación. La respuesta de referencia es el primer nodo
alcanzable cuyo atributo limpio coincide, o **desconocido**. Por tanto,
una respuesta desconocida puede deberse a ausencia del atributo o a
una arista interrumpida. Los datos de inferencia excluyen atributos
limpios, etiquetas objetivo, profundidades y tasas de corrupción.

Cada nodo tiene cinco testigos, con una probabilidad de 15 % de omisión
independiente. El ruido cambia un atributo por uno de los otros tres
con igual probabilidad. Pueden coexistir registros contradictorios.
Los modelos reciben recuentos por atributo, las aristas visibles, el
inicio y la consulta. Los recuentos para el objetivo y el competidor
más votado se derivan de esas mismas observaciones. Los episodios OOD
incluyen 55 % de corrupción, dos testigos correlacionados, una etiqueta
de relación reservada, un orden reservado y grafos más largos.

## Controladores comparados

El modelo recurrente estima una distribución sobre nodos recuperados
y desconocido. La cabeza de detención aprende a comparar la utilidad
de cada salida usando pérdidas supervisadas disponibles en entrenamiento,
con un costo de 0,15 por paso y una penalización para una respuesta
todavía inaccesible. El gradiente de la cabeza no se propaga al predictor.
Durante inferencia, el controlador puede usar la confianza y probabilidad
de desconocido de la distribución, además del estado recurrente. La
ablación sin confianza conserva dimensiones, entrena de nuevo y pone
esas dos entradas a cero. Una arista rota impide nuevas lecturas
adaptativas, pero una salida puede usar evidencia de nodos anteriores.

El comparador determinista exige tres votos favorables entre los cinco
testigos y devuelve desconocido si no encuentra una coincidencia. El
probabilístico usa un modelo de ruido simétrico independiente con tasa
nominal de 20 %, combina probabilidades de primera coincidencia y elige
entre los nodos recuperados y la hipótesis desconocida. No modela las
dependencias entre etiquetas del recorrido ni entre testigos. Ambos
usan la misma tabla de aristas y votos que el modelo neuronal. Las reglas
tienen cero pesos entrenables. Los modelos neuronales conservan el mismo
presupuesto. Los controles de ciclos fijos comparten pesos con el
adaptativo, y la referencia recurrente se entrena desde cero solo para el
tercer ciclo, con la misma memoria. La ablación sin memoria usa resúmenes
de hechos y votos y predice sobre todos los identificadores.

## Reproducción, calibración y métricas

```sh
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
for seed in 7 19 43; do
  for variant in shared no-confidence no-memory recurrent; do
    python -m hikari.hana06_experiment --seed "$seed" --variant "$variant"
  done
done
```

Cada modelo se inicializa al azar y se entrena durante 650 pasos de
32 ejemplos, alternando ruido de 10 % y 30 %. La validación usa un flujo
independiente con 384 consultas en cada nivel. En validación se eligen
los umbrales de salida por precisión menos 0,02 por ciclo adicional
y la temperatura de confianza por verosimilitud. Los resultados de
prueba usan otro flujo y los parámetros de validación congelados.
Los tres niveles de ruido y el estrés correlacionado se comparan sobre
los mismos grafos y consultas. Los contrafactuales cambian el destino
de la arista decisiva, preservando la consulta, los votos y la máscara
de visibilidad. Se reportan aciertos conjuntos para los pares cuya
respuesta realmente cambia.

Cada escenario incluye 512 ejemplos por semilla. Se distinguen
profundidad 0 (desconocido) y profundidades 1 a 3. Se mide precisión
selectiva, cobertura, precisión y sensibilidad de desconocido, ECE de
10 intervalos y Brier de confianza en la respuesta elegida. La latencia
es la mediana de 15 consultas unitarias y abarca la preparación de la
memoria. RSS es el pico del proceso, incluido entrenamiento, PyTorch y
evaluación. GitHub Actions publica los JSON por semilla. Este protocolo
no evalúa razonamiento general ni aplica un criterio de certeza lógica
cuando las etiquetas son aleatorias: evalúa recuperación del objetivo
latente bajo el ruido explícito del generador.
