# Metodología de HANA Fundamental Research Fase 3

## Base y reproducción

La rama parte del commit 560c0dc11f73609d061d839cbad97a59231d774c del PR #9.
El PR #8 aporta las hipótesis, la bibliografía y el protocolo. El código de
Fase 2 sigue siendo la autoridad para verosimilitud, presupuesto, recompensa,
memoria, políticas antiguas y entrenamiento de referencia. Antes de construir
el residual se ejecutaron su suite y las cinco semillas canónicas con 1 800
pasos y 1 024 episodios. La carpeta reproduction contiene los JSON reejecutados.
verify_repro.py exige igualdad exacta de los campos deterministas de dichos
JSON, incluidos acierto, utilidad, abstenciones, acciones y contrafactuales.
El MSE permite error relativo 1e−9 o absoluto 1e−12 entre equipos.
En la reproducción local coincidió exactamente.
Se excluyen etiquetas de plataforma y mediciones de tiempo y RSS, porque se
miden de nuevo en cada ambiente. Los resultados deterministas reprodujeron
las cinco semillas.

## Pregunta, hipótesis y formulación

¿Puede aprenderse solo la ventaja del horizonte adicional, conservando el
cálculo miope exacto para la recompensa inmediata y los casos simples? La
hipótesis es que esto reduce adquisiciones dominadas frente a Fase 2 y
mejora utilidad frente a la regla miope, sin depender de que la fuente de
verificación conserve su fiabilidad o costo habitual.

Un perfil público p especifica prior, fidelidades, omisiones, tarifas y
presupuesto. El estado s contiene informes adquiridos, disponibilidad del
índice y presupuesto restante. La etiqueta y los informes privados permanecen
en el entorno. La misma memoria, actualización posterior y utilidad terminal
se aplican a todas las políticas. Cada consulta tiene tres adquisiciones
posibles: lectura local, índice y verificación. El índice habilita la
verificación sin modificar el posterior. No se repite una adquisición.
La recompensa terminal es 1 por acierto, 0 por error binario y 0,25 por
abstenerse en una etiqueta binaria. El costo resta 0,035 por unidad.

Para adquisición a, la recurrencia de valores es

$$
Q_h(s,a)=-0.035c_a+\sum_o P(o\mid s,a,p)V_{h-1}(T(s,a,o)),
\qquad V_h(s)=\max_{a\text{ legal}}Q_h(s,a).
$$

Las salidas terminales conservan su utilidad posterior esperada en todos
los horizontes. $Q_1$ compara cada adquisición seguida de salida inmediata.
El objetivo supervisado de la red es
$\Delta_a(s,p)=Q_3(s,a)-Q_1(s,a)$ para las tres adquisiciones legales.
La ampliación de horizonte conserva la opción de parar, por lo que el
residual es no negativo. El selector usa $Q_1(s,a)+\max(0,f_\theta(s,p)_a)$,
con valores terminales exactos y máscara de presupuesto. Solo se llama al
predictor si el índice es factible o si, con el índice activo, caben las dos
fuentes pendientes. Fuera de esos casos, un paso agota las oportunidades de
adquirir evidencia y el residual real es cero. La red opera exclusivamente
sobre 20 rasgos públicos, incluidos posterior, estado de adquisición,
presupuesto y tarifas. Los valores de entrenamiento integran resultados
posibles bajo el modelo, sin mostrar el informe realizado ni la respuesta.
La red es 20→32→3 con tanh, 771 parámetros y 736 MAC densas por llamada.
Fase 2 se reentrena con su red 20→32→6, 870 parámetros y 832 MAC.

## Configuración y controles

Semillas 7, 19, 43, 71 y 101. Por semilla se generan 1 200 perfiles de
entrenamiento y 200 de validación con RNG semilla + 10 000 y + 20 000.
En cada perfil se enumeran estados alcanzables y acciones legales. Se
emplean 1 800 pasos del optimizador Adam heredado, lote 192, tasa 0,004,
betas 0,9 y 0,999 y épsilon 1e−8. Se escoge el menor MSE en validación
cada 100 pasos. Se compara con Fase 2, regla miope, regla fija,
determinista, aleatoria, planificación exacta de dos y tres pasos.
La ablación sin puerta usa el mismo estimador en toda decisión con
alguna adquisición legal. La ablación sin costos explícitos readapta
la red sin sus cuatro rasgos de costos y presupuesto. Conserva la base
miope, la máscara y el objetivo con costos. No equivale a ocultar el
presupuesto al sistema entero.

Se cruzan presupuestos 3, 5 y 7 con seis escenarios, 1 024 episodios por
celda y semilla. ID usa el generador de perfiles anterior. OOD combina
fiabilidad invertida, más omisiones y tarifas elevadas. Ruido reduce
fidelidades a 0,53 y 0,55. Memoria incompleta fija omisiones en 0,10 y
0,95. Cambio de tarifas suma 2 a lectura local y 1 a verificación.
Fuentes no informativas fijan fidelidad 0,5. Las intervenciones aisladas
comparten perfil base y números aleatorios con ID. Para cada escenario
se conservan los mismos mundos a través de presupuestos y políticas.
Se usa RNG semilla + 50 000 para la prueba y + 60 000 más índice de
escenario para la política aleatoria. La auditoría nativa usa perfiles,
presupuestos y flujos idénticos a la Fase 2.

## Auditoría, métricas e inferencia

Cada estado visitado se compara con el planificador exacto de horizonte
tres. El arrepentimiento por decisión es la mejor utilidad esperada
menos la de la acción elegida. Adquisición innecesaria significa que su
valor planificado está por debajo del de parar, con tolerancia 1e−9.
Una adquisición subóptima puede ser útil y aun así perder contra otra
acción. Parada prematura significa que la mejor adquisición supera la
mejor salida terminal. Estas definiciones no dependen de si el episodio
acabó acertado. Los conteos son condicionales a los estados visitados
por cada política y no equivalen a una comparación causal por sí solos.
El diagnóstico contrafactual cambia solo la tarifa local y mide acuerdo
en ambos extremos con el oráculo cuando éste cambia de acción. El test
de información oculta interviene sobre informes sin adquirir y verifica
que rasgos y decisión se mantengan. La intervención de tarifas también
preserva los informes realizados.

Se mide acierto estricto, utilidad neta, costo simulado, abstención,
accesos y conteos de acciones. La latencia en µs excluye los primeros
16 episodios y mide inferencia y manejo de estado. La auditoría se
ejecuta fuera de la medición. RAM se reporta como pico RSS de proceso
incluyendo entrenamiento y todas las políticas. MAC excluye Bayes,
activaciones y enrutamiento. Las tarifas no son mediciones de energía.
Los intervalos t del 95 % usan diferencias emparejadas por semilla
con cuatro grados de libertad y son exploratorios. La comparación
primaria se fijó para utilidad ID con presupuesto 5 frente a miope.
Las curvas conectan puntos operativos de presupuestos crecientes,
no interpolan efecto causal a igual gasto.

## Incidencia registrada y límites

El primer intento de descubrimiento de pruebas importó experiment.py
de Fase 2 por precedencia de ruta. Cambiar el orden de la ruta de
módulos dejó la carpeta experimental propia en primer lugar. Las
pruebas se repitieron. La ejecución principal carga el benchmark de
Fase 2 por ruta explícita y los módulos de cálculo no cambiaron.

El generador es pequeño y su modelo de observación es conocido. Un
horizonte exacto de dos o tres pasos es viable y constituye un control
fuerte. El método no valida un selector en lenguaje natural ni inferencia
en dispositivos finales. Los cambios OOD conjuntos no aíslan causas,
por eso se añaden perturbaciones separadas. Ningún resultado justifica
originalidad de la planificación residual ni ahorro energético.
