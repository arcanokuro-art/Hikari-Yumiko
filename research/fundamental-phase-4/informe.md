# Informe científico · HANA Fundamental Research Fase 4

## Pregunta y criterio

Investigar si un selector pequeño puede amortizar la planificación y elegir
adquisiciones frente a una salida inmediata con menor costo total. El
contraste de interés es utilidad neta en ID y OOD con presupuesto 5 frente a
miope, residual y plan exacto de dos pasos. Se examinan además costo y
latencia por pares, sin un margen de no inferioridad preespecificado. Un
intervalo compatible con cero no demuestra equivalencia. Los intervalos de
la matriz son exploratorios y carecen de ajuste por multiplicidad.

## Estado, objetivo y parada

El entorno conserva las tres fases previas. El estado público s contiene
dos informes adquiridos, estado del índice y presupuesto restante. El
perfil público p contiene la creencia inicial, fidelidades, omisiones y
tarifas. La etiqueta y los informes realizables están fuera del selector.
Cada adquisición se hace como máximo una vez. El índice habilita
verificación y no cambia la distribución posterior. La lista legal exige
presupuesto para índice y verificación, y bloquea verificación antes del
índice. Local y verificación obtienen documentos simulados en RAM. Todas
las políticas usan el mismo posterior bayesiano y la misma memoria.

Sea R(y,a) igual a 1 si la salida coincide, 0,25 por abstenerse ante una
etiqueta binaria y 0 en otro caso. Las acciones terminales son
A_T = {responder 0, responder 1, abstenerse}. Se definen:

```text
T(s,p)   = max_{a ∈ A_T} E[R(Y,a) | s,p]
V_0(s,p) = T(s,p)
Q_h(s,a,p) = −0,035 · c(a,p) + Σ_o P(o | s,a,p) · V_{h−1}(T_a(s,o),p)
V_h(s,p) = max(T(s,p), max_{a ∈ adquisiciones legales} Q_h(s,a,p))
G_a(s,p) = Q_3(s,a,p) − T(s,p)
```

La política ejecuta la adquisición de mayor G estimado si supera el margen
m y es legal. En cualquier otro caso, emite la salida terminal de mayor
utilidad exacta. A falta de operaciones legales, omite la red. La red no
ejecuta Q₁ ni Q₃ en inferencia. Puede predecir valores negativos y comparte
el cálculo de creencia entre la entrada y la salida terminal. Frente al
residual anterior, que estima Q₃−Q₁ y trunca valores negativos, esta
parametrización se centra directamente en la decisión de continuar.

## Entrenamiento y controles

Dos redes 20→32→3 con tanh, 771 parámetros y 736 MAC densas por llamada.
Una estima G y otra estima Q₃ absoluto y resta T al decidir. La ablación
sin margen reutiliza pesos de G y fija m=0. Se usa error cuadrático en las
acciones legales y el menor MSE de validación cada 100 pasos. Los 20 rasgos
son posterior (3), fidelidades y omisiones (4), tarifas y presupuesto (4),
informes adquiridos codificados (8) e índice (1). Las reglas de desempate
toman la primera acción de máximo valor y prefieren parar en el umbral.

Cinco semillas 7, 19, 43, 71 y 101. Para cada semilla se generan 1 200
perfiles de entrenamiento con semilla +10 000 y 200 de validación con
semilla +20 000. Se enumeran transiciones posibles y se excluyen estados
sin adquisición legal al entrenar las dos redes nuevas. El objetivo se
obtiene marginalizando todas las observaciones posibles con Q₃ exacto.
Se inicializa desde cero. Adam heredado usa 1 800 pasos, lotes de 192,
tasa 0,004, betas 0,9/0,999 y épsilon 1e−8. Validación elige m entre
0, 0,005, 0,01, 0,02, 0,035, 0,05 y 0,075 por utilidad esperada de la
trayectoria completa, integrada sobre sus resultados posibles. Márgenes del
selector net por semilla: 0,005 / 0,020 / 0,020 / 0,020 / 0,005. La selección
de pesos y margen utiliza validación y excluye los escenarios reservados.

Controles: residual de Fase 3 con la misma arquitectura, planificadores
exactos uno, dos o tres con horizonte reiniciado en cada decisión,
orden fijo local/índice/verificación, determinista con umbral de salida 0,78,
parada inmediata, una lectura local forzada y aleatorio uniforme entre
parar y adquisiciones legales. Cada episodio es compartido por todos los
controles. El planificador exacto de dos pasos puede seguir adquiriendo en
decisiones sucesivas, mientras que la lectura local forzada tiene límite
real de una adquisición.

## Escenarios, contrafactuales y métricas

Se cruzan los presupuestos 3, 5 y 7 con siete escenarios de 1 024 casos
por celda y semilla. Los seis escenarios anteriores usan semilla +50 000,
y el caso reservado two-required usa semilla +70 000. Los informes del mismo
mundo se comparten en todos los presupuestos de un escenario. Los cambios
aislados conservan el perfil base y las variables aleatorias de ID. Se
mantienen RNG separados para entrenamiento, validación y aleatoria
(semilla +60 000 más índice de escenario). Se rota el orden de las políticas
según la semilla. El informe complementa precisión con utilidad realizada,
abstención, accesos totales, lecturas de documento (local + verificación),
precio abstraído, mediana y p95 de latencia. Los primeros 16 episodios se
excluyen de los cronómetros. En una segunda repetición del mismo episodio se
cronometra únicamente el selector, sin contaminar la medida de episodio.
RSS es un pico de proceso Linux e incluye todos los modelos y entrenamiento.
Los MAC omiten posterior, máscara, tanh, Python y memoria.

- ID conserva el generador histórico y perfiles de prueba nuevos.
- OOD conjunto invierte la fiabilidad típica, sube omisiones y precios.
- Ruido fija fidelidades en 0,53/0,55.
- Incompleta fija omisiones en 0,10/0,95.
- Cambio de tarifas suma dos unidades a local y una a verificación.
- Sin señal fija ambas fidelidades en 0,5 y hace óptimo detenerse.
- Two-required usa prior (0,46, 0,46, 0,08), fidelidades (0,50, 0,99),
  omisiones (0,10, 0,03) y costos (2,1,1). Local no aporta información.
  Índice y verificación permiten una mejora estricta sobre horizonte 0 o 1.

En ID a presupuesto 5, el planificador clasifica en promedio 537,6 casos
de parada inicial óptima, 136,6 resolubles con un paso y 349,8 con mejora
estricta al ampliar el horizonte. La dificultad es una etiqueta ex post,
fuera de las entradas. En el contrafactual de tarifa cambian solo las
unidades locales en los primeros 128 casos de ID y OOD a presupuesto 5.
Se mide acuerdo en ambos extremos cuando cambia la acción oracle.
Las pruebas unitarias comprueban máscara, presupuesto, casos de parada y
dos adquisiciones, ganancia con signo, auditoría y el efecto nulo de
cambiar un informe privado antes de adquirirlo.

Los intervalos t del 95 % usan cinco unidades independientes de réplica
(semillas), cuatro grados de libertad y diferencias emparejadas cuando
se comparan políticas. El costo simulado se mide en unidades del generador
y nunca se convierte a tiempo o energía. Las curvas conectan tres puntos
operativos y no identifican un efecto a costo igual.

## Antecedentes y alcance

- Russell y Wefald (1991), *Principles of metareasoning*, derivan el valor
  del cómputo desde su efecto en decisiones externas. [DOI](https://doi.org/10.1016/0004-3702(91)90015-C)
- Hay, Russell, Tolpin y Shimony (2012), *Selecting Computations*,
  estudian elegir cómputos o parar bajo incertidumbre. Usan simulación para
  informar decisiones. Aquí adquirir cambia la evidencia del episodio.
  [Actas UAI](https://proceedings.mlr.press/r10/hay12a.html)
- Howard (1966) formaliza valor de información desde efectos esperados en
  decisiones; Kaelbling, Littman y Cassandra (1998) describen planeación
  bajo observaciones parciales. HANA usa una creencia conocida en un
  dominio finito. [Howard](https://doi.org/10.1109/TSSC.1966.300074) · [Kaelbling et al.](https://doi.org/10.1016/S0004-3702(98)00023-X)
- Ross, Gordon y Bagnell (2011) tratan el desajuste entre entrenamiento e
  interacción en imitación. Aquí se enumeran estados alcanzables bajo
  perfiles de entrenamiento y se ajusta un oráculo supervisado. No se
  aplica DAgger ni aprendizaje por refuerzo en línea.
  [AISTATS](https://proceedings.mlr.press/v15/ross11a.html)
- Graves (2016) aprende cuántos pasos ejecutar en una recurrencia, y
  Bengio, Léonard y Courville (2013) estudian estimación de gradientes
  para unidades estocásticas con cómputo condicional. Aquí la decisión
  dura compara adquisiciones heterogéneas y detenerse, con un objetivo
  supervisado sobre valores. [ACT](https://arxiv.org/abs/1603.08983) · [Gradientes](https://arxiv.org/abs/1308.3432)

## Resultados y decisión

Con presupuesto 5 en ID, net obtuvo 51,66 % y utilidad 0,5441. Frente al
miope (47,91 %, 0,5347), la diferencia de utilidad +0,00946 tuvo IC
[−0,00147; +0,02039]. Costó 1,171 unidades frente a 0,465. Frente al residual
(53,48 %, 0,5467, 1,499 unidades), la diferencia de utilidad −0,00258 tuvo
IC [−0,01080; +0,00564]. Ahorró 0,328 unidades [0,087; 0,569] y la mediana
de episodio bajó de 48,91 a 13,09 µs en esta corrida. Frente a dos pasos
(53,50 %, 0,5543, 1,295 unidades, 62,90 µs) la pérdida de utilidad fue
0,01021 [0,00541; 0,01501]. El ahorro de unidades frente a este control
es inconcluyente. En esta CPU el costo propio del selector no anuló el
ahorro de tiempo frente al residual o a dos pasos. El cronómetro aislado
del selector también favoreció net (18,10 µs/episodio en promedio frente a
69,91 del residual y 95,00 del exacto de dos pasos). El promedio aislado y
la mediana total describen resúmenes distintos y no se restan.

En OOD a presupuesto 5, la utilidad bajó de 0,5432 compartida por residual,
miope y dos pasos a 0,5323. Diferencia −0,01095, IC [−0,02104; −0,00086].
El costo menor acompañó una caída de 49,71 % a 45,43 % y 283,8 paradas
prematuras de media. El contrafactual ID mostró acuerdo bilateral del
43,8 % frente a 65,5 % residual y 88,6 % exacto de dos pasos. En OOD, net
obtuvo 29,4 % frente a 100 % de ambos controles. La memoria incompleta
también penalizó net. Con fuentes sin señal, net eligió parar siempre.
En two-required net, residual y exacto de dos pasos lograron 89,45 % y
utilidad 0,8245 con dos unidades, frente a 46,21 % y 0,4621 del miope.

Quitar el margen aumentó el gasto ID en 0,305 unidades de media y elevó
operaciones perjudiciales de 112,6 a 187,8, pero redujo paradas prematuras
de 137,4 a 83,8. Su diferencia de utilidad es inconcluyente. Fuera de
distribución sigue existiendo un déficit con margen cero, consistente con
una combinación de error de aproximación y exceso de cautela. La ablación
de Q absoluto tiene una utilidad ID de 0,5496 y su diferencia frente a
net incluye cero. Los resultados no atribuyen una ventaja de calidad al
centrado respecto de estimar Q absoluto con los mismos parámetros.

El ajuste net toma 0,279 s de media, más 0,486 s para etiquetas compartidas
y 0,041 s para calibración. El pico RSS total es 47,63 ± 0,47 MiB y no
permite atribuir memoria a una política. Las tarifas son simuladas, la
latencia proviene de CPU x86-64 y las cinco mediciones se ejecutaron en
procesos concurrentes de un hilo. No se midió energía, almacenamiento
físico, latencia móvil ni robustez cuando los metadatos de fiabilidad son
incorrectos. Las comparaciones temporales son específicas del ejecutor
y del planificador exacto de referencia sin reutilización entre decisiones.

**Decisión.** Conservar el mecanismo como referencia de latencia y modificar
la hipótesis de generalización. No sustituir los controles por el selector
actual. Para revisión posterior, la evidencia favorece estudiar un retorno
a una política analítica cuando la fiabilidad o la omisión sale del soporte
de entrenamiento, o estimar una cota de incertidumbre explícita y controlar
el porcentaje de casos revertidos. Esas ideas se proponen para revisión y
quedan fuera de esta fase. El experimento no demuestra originalidad de
la parada, la decisión bayesiana ni la aproximación neuronal de valores.
