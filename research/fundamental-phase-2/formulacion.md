# Formulación y controles experimentales

## Revisión de las hipótesis de la Fase 1

Se revisaron los cinco documentos y su bibliografía. La hipótesis A se
elige primero porque puede falsarse variando solo el selector, con memoria
y núcleo congelados y un problema pequeño con solución de referencia.
Su supuesto más importante es que existan acciones informativas de
distinto valor y que el costo de decidir no anule el ahorro. Un segundo
supuesto es que la evidencia no adquirida quede fuera del estado inicial,
algo que HANA-0.6 no aislaba con su resumen de aristas y votos.

En B, la utilidad de retener episodios exige una distribución futura
representativa y el efecto de añadir recuerdos no es aditivo. Un
experimento tendría que separar reaparición de tareas, revisión de
hechos e interferencia, usando replay uniforme con los mismos bytes.
En C, la contracción vale para cada actualización lenta ejecutada,
pero entre eventos la dinámica lenta es identidad. La reducción de
tráfico requiere omitir trabajo físicamente, no solo reducir un
contador. La memoria dispersa necesita un límite explícito y control
de costo de índice. Ambas quedan como hipótesis aún sin probar.

## Entorno y aprendizaje

La respuesta latente es $Y\in\{0,1,?\}$. El perfil público $p$
contiene el prior $P(Y)$, tasas de fiabilidad y omisión de las
fuentes, tarifas positivas y presupuesto. El estado $s=(r_L,r_V,i,b)$
contiene informes adquiridos, indicador de índice habilitado y
presupuesto restante. El mapa de estado no contiene identificador de
episodio, tipo de escenario ni informes sin leer. A igualdad de
perfil, solo la acción solicitada puede revelar su informe. No se
permite volver a leer una fuente, gastar más del presupuesto ni
verificar antes de consultar el índice.

| Operación | Efecto |
|---|---|
| Lectura local | Adquiere un informe ruidoso o una omisión |
| Consulta del índice | Activa el acceso a la verificación; deja intacto el posterior |
| Verificación | Adquiere otro informe o una omisión |
| Responder 0, responder 1 o abstenerse | Termina y aplica la utilidad correspondiente |

Las dos fuentes son independientes condicionadas a $Y,p$. Para
etiquetas binarias hay canal simétrico y omisión independiente;
si $Y=?$, los votos no omitidos son equiprobables. El estado
compartido usa $P(Y\mid r_L,r_V,p)\propto P(Y\mid p)
\prod_{k\,\text{leído}}P(r_k\mid Y,p)$. Todas las políticas
usan ese mismo decodificador de utilidad terminal. Un acierto vale
1, una respuesta binaria errónea vale 0 y abstenerse sobre una clase
binaria vale 0,25. La precisión cuenta la abstención como acierto
únicamente para la etiqueta desconocida.

Para acciones de lectura, con horizonte máximo tres:

$$
Q_h(s,a)=-0.035\,c(a,p)+\sum_{o}P(o\mid s,a,p)\, V_{h-1}(T(s,a,o)),
\qquad V_h(s)=\max_{a\in A(s)}Q_h(s,a).
$$

Las acciones terminales usan utilidad posterior esperada; en horizonte
cero solo quedan esas acciones. El planificador exhaustivo integra
las observaciones posibles, sin leer informes reales por adelantado.
El control miope usa la misma ecuación con horizonte uno. La red
$20\to32\to6$ minimiza el error cuadrático sobre objetivos exactos
de cada acción factible, con máscara de acciones ilegales. Al decidir
usa los valores terminales analíticos y las tres estimaciones
neuronales para adquisiciones. Sus 870 parámetros y 832 MAC densas por
decisión no incluyen Bayes, activaciones, máscara ni manejo de memoria.
El modelo aprende a imitar valores de planificación con datos
calculados. No explora en línea ni aprende por refuerzo del premio
realizado. El costo lineal y presupuesto son simulados y se reportan
por separado de los tiempos de CPU.

## Controles, particiones y amenazas a la validez

La regla determinista usa un umbral de utilidad terminal de 0,78 y
prioridad verificación/local/índice. El fijo ejecuta local/índice/
verificación si es factible y responde al final. El aleatorio elige de
manera uniforme entre parar y una adquisición disponible. El oráculo
es un control de utilidad esperada, no una cota de acierto en cada
muestra. Las ablaciones ocultan las cuatro entradas de tarifa y
presupuesto de la red, reducen sus objetivos al horizonte uno, o
eliminan todas las lecturas. La ablación de tarifas mantiene la máscara
de factibilidad y el objetivo de entrenamiento con tarifas.

Se separan perfiles de entrenamiento, validación y prueba por flujos
aleatorios. Validación elige el checkpoint por error, no por resultados
de prueba. En OOD se invierte la fiabilidad típica de las fuentes,
suben omisiones y aparecen tarifas mayores. Son cambios simultáneos;
no se puede atribuir el efecto a uno de ellos. El tercer conjunto
asigna fiabilidad 0,5 a ambos canales. En contrafactuales se conserva
perfil, prior y mundo, y se cambia solo la tarifa local. Se informa
acuerdo en ambos extremos para los pares donde el planificador cambia.
Las pruebas unitarias comprueban invariancia al cambiar un informe
no adquirido, simetría de etiquetas con prior simétrico, máscara de
presupuesto y ganancia indirecta del índice. Quedan pendientes
controles con topologías de grafos y corrupción de metadatos de
fiabilidad. No hay texto, identificadores serializados ni datos
preentrenados. El estado y las fuentes son deliberadamente pequeños.

Se mide latencia de episodio, de operación aislada, conteo de accesos,
RSS total y unidades simuladas. La latencia incluye el selector
y la gestión de estado pero no acceso físico a almacenamiento o red.
La asignación de presupuesto de la Fase 1 no se considera cumplida
en energía: no se instaló un medidor y no se estimaron julios.
El intervalo t de diferencias por semilla es exploratorio.
No deben contarse las consultas como réplicas independientes de entrenamiento.

## Antecedentes y originalidad

- Russell y Wefald (1991) formulan el valor del cómputo bajo
  restricciones, antecedente directo del objetivo. [Principles of metareasoning](https://doi.org/10.1016/0004-3702(91)90015-C).
- Kaelbling, Littman y Cassandra (1998) formalizan planeación con
  observaciones parciales y estados de creencia. Nuestro entorno aplica
  esa estructura con horizonte tres. [Planning and acting in partially observable stochastic domains](https://doi.org/10.1016/S0004-3702(98)00023-X).
- Sutton y Barto (2018) describen ecuaciones de Bellman y aprendizaje por
  refuerzo. Aquí se calculan etiquetas de valores con el simulador,
  sin actualizar una política durante la interacción de evaluación.
  Ross, Gordon y Bagnell (2011) estudian imitación bajo distribución
  inducida por la política. La enumeración de estados legales del
  benchmark evita depender solo de trayectorias del experto y
  simplifica ese problema, sin resolverlo en dominios abiertos.
  [Sutton y Barto](https://mitpress.ublish.com/book/reinforcement-learning-an-introduction-2) · [Ross et al.](https://arxiv.org/abs/1011.0686).
- Graves (2016) y Banino, Balaguer y Blundell (2021) estudian
  cómputo adaptativo y parada. Aquí se elige también entre operaciones
  heterogéneas y un índice habilita una acción futura.
  [ACT](https://arxiv.org/abs/1603.08983) · [PonderNet](https://arxiv.org/abs/2107.05407).
- Jiang et al. (2023) y Jeong et al. (2024) desencadenan recuperación
  adaptativa en generación o eligen estrategias de recuperación.
  Esta prueba separa informes adquiridos, costos simulados y control
  de horizonte en un entorno finito, sin generador de lenguaje.
  [FLARE](https://arxiv.org/abs/2305.06983) · [Adaptive-RAG](https://arxiv.org/abs/2403.14403).

La implementación muestra cómo aislar selección, acceso y costo en
HANA. El principio de valor marginal, la recursión de Bellman, el
direccionamiento y la parada aprendida tienen antecedentes. No se
demostró originalidad de esos mecanismos ni ventaja general del
selector entrenado.
