# Hipótesis arquitectónicas falsables

## A. Elegir operaciones por beneficio marginal y costo medido

**Pregunta.** ¿Conviene aprender qué operación ejecutar cuando una
regla fija ya sabe cuándo abstenerse? Se plantea un menú pequeño de
operadores tipados: actualización interna, lectura local por clave,
búsqueda en un índice, comprobación independiente, respuesta o abstención.
Cada lectura produce un informe parcial nuevo. Un registro de evidencia
mantiene procedencia y fiabilidad. Las lecturas dependen del estado
observado y pueden equivocarse o aportar información redundante.

**Modelo.** Sea $s_t=(h_t,E_t,b_t,d)$ el estado compacto, evidencias
adquiridas, presupuesto restante y perfil de dispositivo. Para una
acción $a$, $c_d(s_t,a)$ es el vector de costos, por ejemplo tiempo
y energía, y $b_{t+1}=b_t-c_d(s_t,a)$. Se filtran operaciones que exceden
el presupuesto con un límite conservador estimado y se mide cualquier
violación en ejecución. La utilidad terminal es $u(y,\hat y)$, con
costos de error y abstención fijados previamente. Se estima

$$
Q(s,a)=\mathbb E[V(s')\mid s,a]-\lambda_d^\top c_d(s,a),
\qquad
V(s)=\max\bigl\{U_{\text{responder}}(s),U_{\text{abstener}}(s),
\max_{a\in A_{\text{factibles}}}Q(s,a)\bigr\}.
$$

$U_{\text{responder}}$ y $U_{\text{abstener}}$ integran incertidumbre
bajo una estimación calibrada. Para un primer prototipo futuro se
limita el horizonte a tres acciones. En simulación se puede bifurcar
el mismo prefijo observado y ejecutar cada acción factible usando
estados de mundo emparejados para medir el retorno posterior. Se ajusta
el estimador sobre ese retorno, sin mostrarle el mundo latente ni el
resultado de la acción candidata antes de elegir. No se propone
estimar contrafactuales causales de registros observacionales sin
exploración ni cobertura de acciones.

**Antecedentes y candidato propio.** Metarrazonamiento y POMDP definen
el problema, ACT y PonderNet ilustran parada, y FLARE, Adaptive-RAG y
ReAct ilustran recuperación condicionada. La diferencia candidata a
probar es el estimador compacto entrenado con comparaciones de
operaciones emparejadas y costo real por dispositivo, con las mismas
interfaces de memoria y acciones para todos los controles.

**Predicción y refutación.** En tareas con costos y fiabilidades
cruzados, la política cambia de operador cuando se invierte cuál
fuente es fiable, aun con idéntico nivel nominal de dificultad. Gana
utilidad al mismo presupuesto frente a una regla miope que usa el mismo
estimador de fiabilidad. Si el ahorro desaparece al incluir el costo
del selector, si gana solo por acceso anticipado a evidencia, o si no
mejora el frente frente al planificador miope en escenarios reservados,
la hipótesis de ventaja práctica pierde apoyo. Riesgos adicionales:
deriva térmica del perfil, explotación de pistas y sensibilidad a la
penalización de abstención.

## B. Retener episodios por utilidad futura y costo de memoria

**Pregunta.** ¿Un búfer pequeño puede preservar hechos y habilidades
bajo cambios de distribución con mejor relación retención/costo que
replay uniforme, FIFO o sorpresa? Se separa un núcleo estable, un
adaptador plástico de bajo rango y un conjunto episódico $M$.

**Modelo.** Para un episodio observado $e$, se estima el beneficio de
retenerlo y su costo amortizado:

$$
v(e)=\mathbb E_{q\sim\widehat D_{\rm futuro}}
[\ell(q;M\setminus\{e\})-\ell(q;M\cup\{e\})]
-\lambda_m\,\mathrm{bytes}(e)-\lambda_r\,\mathbb E[\mathrm{lecturas}(e)].
$$

$widehat D_{\rm futuro}$ se aproxima por una distribución histórica
o por un generador de entrenamiento, jamás por la evaluación reservada.
La admisión y expulsión aproximan $\max_M \sum_{e\in M}v(e)$ bajo un
límite duro de bytes. La suma ignora interacciones entre recuerdos y
se compara con una ablación que controla redundancia. El adaptador
$\phi$ actualiza con datos nuevos y un minibatch de replay:

$$
\phi_{t+1}=\phi_t-\eta\nabla_\phi
[\ell_{\rm nuevo}+\alpha\ell_{\rm replay}+\beta\|\phi-\phi_{\rm ref}\|_2^2].
$$

**Antecedentes y candidato propio.** NTM y DNC preceden al manejo de
memoria, EWC y GEM al control de interferencia, Titans y TTT a estados
plásticos durante inferencia. La candidata es la admisión conjunta
por utilidad contrafactual amortizada y el adaptador con presupuesto
medible, no la existencia de replay ni de memoria plástica.

**Predicción y refutación.** Con los mismos bytes totales, número de
actualizaciones y tiempo CPU, mejora la retención al reaparecer
habilidades antiguas sin degradar el aprendizaje de revisiones nuevas.
Medir olvido, latencia de escritura y transferencia hacia atrás.
Refutar si FIFO, reservorio o replay uniforme igualan el frente, si
la utilidad predicha no se corresponde con el efecto de eliminar un
episodio, o si sobrerretener hechos viejos empeora las revisiones.
Riesgos: sesgo a consultas frecuentes, deriva, consolidar ruido y
costo de estimar utilidad por episodio.

## C. Estado multiescala con correcciones episódicas dispersas

**Pregunta.** ¿Se puede conservar evidencia de largo alcance con un
estado rápido de tamaño fijo y un estado lento actualizado solo en
eventos útiles? Se investiga el núcleo de representación con un
selector externo común, para evitar confundirlo con la hipótesis A.

**Modelo.** Para entrada $x_t$ y evento binario $g_t$, usar

$$
f_t=\sigma(W_f[f_{t-1},x_t]),\quad
m_t=(1-g_t)m_{t-1}+g_t(A(x_t)m_{t-1}+B(x_t)f_t),\quad
z_t=D[f_t,m_t,r_t].
$$

$f_t$ y $m_t$ tienen dimensiones fijas. $A(x_t)$ es diagonal
contractiva, por ejemplo $\mathrm{diag}(\exp(-\mathrm{softplus}(a(x_t))))$.
Se calcula $r_t$ por acceso a un registro disperso de eventos
cuando se supera un umbral sobre error de compresión estimado.
El entrenamiento conjunto penaliza $\sum_t g_t$ y lecturas/escrituras.
En inferencia los eventos duros deben omitir efectivamente las
operaciones sobre $m_t$; una multiplicación densa por cero no cuenta.

**Antecedentes y candidato propio.** RWKV, Mamba y Skip RNN preceden
a estado recurrente eficiente, selectividad y omisión de actualizaciones.
NTM/DNC y Titans preceden al uso de memoria externa o plástica. Se
probaría el aporte específico de ligar las correcciones al error de
compresión y amortizar tráfico entre dos velocidades de estado.

**Predicción y refutación.** A igual calidad de recuperación y
composición reservada, reduce tráfico o energía frente a una
recurrencia de dos escalas de actualización fija y configuraciones
pequeñas de Mamba/RWKV. Separa el efecto del disparador frente a igual
frecuencia aleatoria. Si el registro aumenta de manera no acotada,
si los eventos raros no se recuperan, o si se reduce el número de
actualizaciones pero no el costo real, la variante no cumple su objetivo.
El costo de indexar y el bajo paralelismo de operaciones esparsas pueden
invertir la ventaja en CPU o servidores.
