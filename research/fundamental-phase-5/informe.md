# Auditoría de decisiones y generalización de HANA

## Continuidad experimental

El PR #8 formuló selección de operaciones, retención y estado multiescala.
El PR #9 implementó selección bayesiana con memoria y tarifas simuladas.
El PR #10 aprendió una corrección sobre valores de un paso. El PR #11
estimó directamente la ventaja de continuar y calibró un margen de parada.
La Fase 5 deriva de 74ee27879aaf59d576532e0c99e765e98d96ecfc. Se revisaron
las formulaciones, las interfaces, las pruebas y las matrices de resultados.
Las cinco evaluaciones completas de Fase 4 se volvieron a ejecutar antes
de evaluar las intervenciones. El verificador conserva igualdad de enteros
y usa tolerancia relativa 1e−9 y absoluta 1e−12 para valores flotantes.
Solo excluye metadatos de plataforma, latencia, tiempos y RSS.

## Formulación

Sea s el historial adquirido y saldo, p el perfil público, T(s,p) la
máxima utilidad terminal esperada y c(a,p) la tarifa de adquisición.
La política de referencia evalúa

    V₀(s,p) = T(s,p)
    Qₕ(s,a,p) = −0,035 c(a,p) + Σₒ P(o|s,a,p) Vₕ₋₁(sₐₒ,p)
    Vₕ(s,p) = max(T(s,p), maxₐ Qₕ(s,a,p))
    G(s,a,p) = Q₃(s,a,p) − T(s,p)

La recompensa vale uno por acertar, 0,25 por abstenerse sobre etiqueta
binaria y cero por error binario. Si el mejor de los valores aprendidos
ĝ de adquisiciones legales es mayor que el margen m, Net ejecuta esa
acción. En otro caso toma la mejor salida terminal. La primera acción
de máximo valor resuelve empates. Este mecanismo y su optimizador se
reutilizan intactos.

Una decisión es incorrecta si su desventaja frente al mejor valor exacto
excede 1e−9. Una parada incorrecta es atribuible localmente al margen
cuando 0 < max ĝ ≤ m. Cuando max ĝ ≤ 0 persiste al quitar el margen en
ese estado. Esta atribución local se complementa con una evaluación de
margen cero, que puede cambiar los estados visitados. En caso de adquirir,
se indica acción perjudicial si Q₃ < T−1e−9 y error de selección en otro
caso. Registrar una lectura beneficiosa según Q₃ supone continuación
óptima. No equivale al beneficio que obtuvo el controlador en ese episodio.

La incertidumbre auditada corresponde al valor de la información. Para
la adquisición legal preferida por la red, se compara ĝ con su ganancia
exacta G. Se reporta RMSE, error de signo y diferencia absoluta entre
medias predichas y exactas por intervalo, ponderada por número de estados.
Los límites son −0,15, −0,075, −0,035, 0, 0,02, 0,05, 0,10 y 0,20; hay
intervalos exteriores. La fracción exacta positiva se guarda por bin.
La red no emite probabilidades de incertidumbre y estas medidas no son
calibración probabilística de la etiqueta ni cobertura de intervalos.
El diagnóstico describe los estados visitados por cada política.
Como medición probabilística separada, en presupuesto 5 se puntúa el
posterior compartido antes de leer y al detenerse con cada red. Brier
multiclase es Σᵧ (pᵧ−1{Y=y})². ECE pondera las diferencias absolutas
entre confianza máxima y frecuencia de acierto del argmáx posterior,
con diez bins de ancho 0,1. Se usa argmáx de etiqueta para este
diagnóstico, porque la salida de máxima recompensa puede diferir.

## Intervenciones y condiciones emparejadas

Se conservaron Net, Net sin margen, Q absoluto, Fase 2, residual de
Fase 3, parada, regla fija, determinista, miope y los exactos de dos y
tres pasos. La política miope de veto consulta Q₁ únicamente cuando Net se
dispone a detenerse y hay una adquisición legal. Mantiene la acción
aprendida cuando Net quiere adquirir. Sirve como intervención diagnóstica
sobre la parada y obliga a medir el tiempo del planificador adicional.

El control de cobertura reemplaza la mitad de los perfiles de
entrenamiento y validación por perfiles generados con fidelidad uniforme
[0,50, 0,99], omisión [0,03, 0,65] y tarifas enteras 1–5. Conserva 1 200
perfiles para entrenamiento, 200 para validación, inicialización por
semilla, la red 20→32→3, 771 parámetros y 736 MAC por llamada. Cada
perfil se expande en estados alcanzables, por lo que la cantidad de
estados puede diferir. El efecto estimado corresponde al cambio conjunto
de cobertura de entrenamiento y validación. Los checkpoints se escogen
por MSE de validación con Adam de 1 800 pasos, lotes de 192 y tasa 0,004.
El margen se escoge por utilidad exacta de validación entre los siete
candidatos de Fase 4. No hay pesos preentrenados.

Las semillas son 7, 19, 43, 71 y 101. Cada escenario tiene 1 024
episodios por presupuesto 3, 5 y 7, con mundos emparejados entre políticas
y presupuestos. Se mantienen ID, OOD conjunto, ruido, memoria incompleta,
cambio de tarifas, ausencia de señal y dos adquisiciones indispensables.
El escenario nuevo combina fidelidades entre 0,67 y 0,91, omisiones entre
0,28 y 0,71 y tarifas locales 1–2, de índice 1–3 y verificación 2–5.
Usa un flujo reservado con semilla +90 000. Parte de dicho escenario
intersecta la distribución ampliada. Se mide generalización a una
distribución distinta, no extrapolación fuera de todo el soporte ampliado.
En contrafactuales de tarifa, solo se altera el costo local y se compara
el acuerdo bilateral con el oráculo para los pares donde este cambia.

La prueba de identificabilidad presenta el mismo vector público en dos
mundos. La fidelidad declarada de la fuente local se mantiene en 0,65.
La fidelidad real oculta vale 0,50 o 0,98. El oráculo privilegiado puede
usar la fidelidad real. Con fidelidad 0,50 conviene responder sin leer
(utilidad 0,4600, mejor adquisición 0,4250). Con 0,98 conviene leer
(valor 0,848936 frente a 0,4600). Los conjuntos de acciones óptimas son
disjuntos. En el benchmark normal p es conocido y veraz, y el planificador
comparte esa entrada. La colisión demuestra falta de identificabilidad
solo en el contrafactual con metadatos imperfectos.

## Métricas de hardware y análisis estadístico

La latencia de episodio y p95 incluyen acceso simulado, estado y control.
Se mide aparte el tiempo del selector, en una segunda pasada. Los
cronómetros excluyen los primeros 16 episodios. RSS es el máximo del
proceso e incluye entrenamiento, instrumentación y todas las políticas.
Los precios son unidades simuladas. La latencia se midió en CPU x86-64
durante cinco procesos concurrentes de un hilo. No hay medición de energía,
almacenamiento físico ni hardware móvil. Los IC t del 95 % se calculan
sobre diferencias emparejadas por semilla, con cuatro grados de libertad.
Los contrastes secundarios son exploratorios y no se ajustan por
comparaciones múltiples. Un IC que cruza cero no prueba equivalencia.

## Incidencias y límites

El primer verificador incluyó un tiempo compartido de etiquetado. Se
excluyó explícitamente de la paridad determinista mediante el patrón de
tiempos. El primer test de identificabilidad supuso abstención como salida
en ausencia de señal. Se corrigió para comprobar las dos respuestas
binarias empatadas, como exige la recompensa fijada. La primera ejecución
nueva produjo trazas y falló al serializar contadores derivados de
booleanos de NumPy. Se convirtieron booleanos y escalares NumPy a tipos nativos, se validó
una ejecución breve con JSON y se repitió la matriz completa.

El entorno es sintético, con dos fuentes y horizonte finito. La
intervención de cobertura prueba sensibilidad a los datos con la misma
capacidad, sin identificar cuánto error restante corresponde a
optimización, capacidad o elección de pérdida. El oráculo se usa como
diagnóstico y no implica acceso a evidencia futura realizada.

## Resultados y refutaciones

La auditoría incluye 18 708 decisiones incorrectas del controlador histórico
en las cinco semillas, con 3 549, 4 090, 3 604, 4 103 y 3 362 para las semillas
7, 19, 43, 71 y 101 respectivamente. Cada error tiene una entrada JSONL.
Los siguientes valores corresponden a presupuesto 5 y promedian semillas.

| Escenario | Política | Precisión | Utilidad | Costo | Mediana CPU µs |
|---|---|---:|---:|---:|---:|
| ID | Net | 51,66 % | 0,54411 | 1,17129 | 13,41 |
| ID | Cobertura amplia | 49,77 % | 0,53889 | 0,88672 | 12,38 |
| ID | Veto miope | 53,22 % | 0,55122 | 1,32266 | 35,94 |
| ID | Exacto dos pasos | 53,50 % | 0,55432 | 1,29531 | 59,12 |
| OOD histórico | Net | 45,43 % | 0,53226 | 0,06738 | 8,94 |
| OOD histórico | Cobertura amplia | 46,27 % | 0,53579 | 0,19805 | 8,96 |
| OOD histórico | Veto miope | 49,73 % | 0,54319 | 0,80410 | 28,73 |
| OOD histórico | Miope | 49,71 % | 0,54321 | 0,79805 | 23,07 |
| OOD nuevo | Net | 47,40 % | 0,53869 | 0,31328 | 10,44 |
| OOD nuevo | Cobertura amplia | 48,55 % | 0,54519 | 0,41914 | 9,94 |
| OOD nuevo | Veto miope | 49,86 % | 0,54856 | 0,59766 | 31,96 |
| OOD nuevo | Exacto dos pasos | 49,77 % | 0,55135 | 0,49297 | 53,02 |

H1 queda refutada en OOD histórico. Net registró 1 419 paradas prematuras,
1 279 (90,1 %) con ganancia predicha no positiva y 140 con valor positivo
por debajo del margen. Quitar el margen dejó 255,8 de las 283,8 paradas
originales por cada 1 024 episodios. En ID la proporción de paradas
atribuidas al margen fue mayor (301 de 687) y sigue siendo minoritaria.

H2 recibe apoyo localizado. La cobertura amplia subió la utilidad OOD
histórica en 0,00353 [0,00070, 0,00637] y la de OOD nuevo en 0,00650
[0,00432, 0,00868]. En ID la redujo en 0,00523 [0,00107, 0,00938] en
magnitud. En OOD histórico la brecha de calibración de ganancia bajó
de 0,04712 a 0,00884 y el RMSE de 0,06035 a 0,03768. En ID el RMSE
subió de 0,04404 a 0,04983. Los cambios de datos mejoran una parte del
fallo y presentan un intercambio entre distribuciones. No establecen
que más capacidad o una pérdida diferente sean innecesarias.

H3 se confirma para el par construido con fiabilidad oculta. Las acciones
óptimas en ambos mundos son disjuntas. En el entorno histórico las
fidelidades son públicas, de modo que este resultado no explica por sí
solo sus errores. El diseño actual puede calcular el valor exacto con
los rasgos públicos y todavía aprender una aproximación insuficiente.

H4 recibe apoyo. El cálculo de un paso ya detectaba una mejora positiva
en las 1 419 paradas prematuras OOD históricas. El veto eliminó esas
paradas en la evaluación observada y mejoró la utilidad 0,01093
[0,00015, 0,02172] frente a Net. Su selector consumió 32,31 µs/episodio
en promedio frente a 8,46 de Net. La regla miope alcanzó utilidad
prácticamente igual en ese escenario con mediana 23,07 µs, menor que
el veto (28,73 µs). Mantener la evaluación de ambos controladores para
casi todas las paradas tiene un costo visible. El veto también redujo
las paradas en memoria incompleta, aunque conservó adquisiciones
perjudiciales cuando Net ya había decidido continuar.

En el contrafactual de tarifa sobre OOD histórico, el acuerdo bilateral
fue 29,4 % para Net, 56,8 % con cobertura amplia y 100 % con veto o
exacto de dos pasos. Con dos adquisiciones indispensables, Net, cobertura
amplia, veto y exacto de dos pasos acertaron 89,45 % con utilidad 0,82453.
La regla miope se detuvo con 46,21 % y utilidad 0,46211. Con fuentes sin
señal, Net se detuvo en todos los episodios. Cobertura amplia hizo algunas
lecturas, con 0,00645 unidades por episodio y utilidad indistinguible
en la comparación exploratoria.

El entrenamiento de cobertura, incluido etiquetado y calibración, tomó
0,913 ± 0,116 s. Pico RSS total 49,14 ± 0,85 MiB. Los JSON y los CSV
conservan medidas por semilla, desviaciones, intervalos y diagnósticos.

En OOD histórico, el Brier del posterior pasa de 0,61101 antes de leer
a 0,60637 con Net y 0,59951 con cobertura. El ECE de etiqueta permanece
cerca de 0,031 en ambos controles. En la tarea que requiere dos
adquisiciones, el Brier baja de 0,57135 a 0,17816 con ambas redes.
La calibración de clase y la calibración de ganancia responden a
preguntas distintas. Un posterior razonable puede coexistir con una
política que decide mal cuándo obtener evidencia.

## Decisión científica

**B. Rediseñar el mecanismo de decisión.** El fallo dominante es la
subestimación del valor de información al cambiar la distribución.
La cobertura corrige parte del error a costa de utilidad ID. El veto
muestra que una comprobación analítica detecta el fallo, y también mide
su sobrecosto. Conviene conservar los módulos como referencias y diseñar
una separación explícita entre detección de error de valor, confiabilidad
del perfil y decisión de consultar un planificador. Debe compararse con
la regla miope sola y con el exacto de dos pasos. Si la fiabilidad no es
observable, el rediseño necesitará modelar esa incertidumbre o adquirir
evidencia que distinga los mundos. La Fase 6 requiere autorización.
