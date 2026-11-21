# Protocolo experimental propuesto

## Diseño previo a implementación

Probar primero A con un núcleo congelado y una interfaz idéntica de memoria
y operadores. Este aislamiento permite atribuir diferencias a la decisión
de qué ejecutar. Las hipótesis B y C son alternativas independientes y se
evalúan después de revisión. La siguiente matriz fija qué factor se manipula.

| Serie | Tratamiento | Controles y diagnósticos |
|---|---|---|
| A. Decisión | Cambiar por separado fiabilidad, redundancia, costo, disponibilidad y presupuesto de cada operador | Rutina fija, aleatoria emparejada por costo, umbral de confianza, ACT/PonderNet reducido a continuar o parar, regla de valor miope y oráculo exhaustivo hasta horizonte tres |
| B. Plasticidad | Corriente con tareas A→B→C→A, correcciones de hechos y retorno de patrones raros | Adaptador congelado, adaptador sin replay, FIFO, reservorio y sorpresa; EWC y replay uniforme con memoria comparable |
| C. Estado | Series de recuperación y composición con distractores, distancias y combinaciones no vistas | GRU/ZNC-0, recurrente multiescala fija, eventos aleatorios a igual frecuencia; Mamba y RWKV pequeños inicializados desde cero si son ejecutables en CPU |

El oráculo de A integra las distribuciones conocidas del simulador
condicionadas al mismo prefijo y horizonte, sin observar hechos latentes.
Un oráculo que sí ve la verdad solo sirve como cota optimista y se rotula
por separado. Los controles no entrenables registran cero parámetros y sus
tablas, índices y costos de inicialización se contabilizan aparte.

### Tareas y ausencia de información adelantada

- **A:** un generador de grafos ofrece varias fuentes de evidencia con
  fiabilidad observada solo mediante metadatos públicos o informes ya
  adquiridos. Incluye evidencia repetida, verificación que resuelve
  contradicciones, accesos fallidos y consultas irresolubles. Consultas de
  forma y longitud común. Las aristas o informes todavía no recuperados
  se mantienen en un almacén opaco, fuera del resumen, normalización y
  estado inicial. Cargar o indexar el almacén solo está permitido detrás
  de la interfaz y se incluye en el costo cuando corresponda.
- **B:** hechos temporales con fecha y episodios etiquetados por tarea.
  Separar recordar la versión vigente, preservar historia y reconocer
  una revisión. El evaluador recorre todas las tareas después de cada
  bloque; sus consultas no alimentan la memoria.
- **C:** recuperación con distancias controladas, listas con distractores
  y composición de reglas. Probar longitud 2× y 4×, combinaciones
  disjuntas y cambios de topología que no se reduzcan a renombrar bancos.
- **Contrafactuales:** clonar un prefijo y variar fiabilidad, costo o
  contenido de una fuente conservando histograma, consulta y orden.
  Verificar con oráculo independiente cuándo cambia la operación
  preferida o el objetivo. Medir tanto fidelidad en pares que cambian
  como estabilidad en pares cuyo objetivo no cambia. Hacer pruebas
  negativas con memorias intercambiadas entre episodios, un modelo que
  solo recibe la consulta y un detector de fuga que inspecciona la
  serialización, los metadatos, la máscara y la longitud.

### Particiones, presupuesto y reproducibilidad

Entrenamiento, validación y prueba usan semillas de generador y conjuntos
disjuntos de grafos y plantillas. Separar las semillas de inicialización
de las de datos, política y generación de ruido. Congelar validación
antes de ajustar hiperparámetros y reservar un test OOD adicional sin
reutilizarlo para diseñar el mecanismo. Exigir cinco semillas emparejadas
(7, 19, 43, 71 y 101) para la rama principal y fijar los pasos de
entrenamiento y criterio de parada antes de abrir test. Propuesta de
partida: modelos neuronales hasta 100 mil parámetros, cinco configuraciones
por familia en validación y 1 024 consultas por estrato y semilla.
Permitir un ensayo de instrumentación con datos de entrenamiento para
comprobar viabilidad y estimar dispersión. Si cinco semillas no ofrecen
precisión suficiente en los intervalos, registrar resultado inconcluso
y presupuestar una réplica independiente.

Igualar entradas, operadores, memoria disponible, generador, optimizador,
datos y búsqueda de hiperparámetros cuando aíslen el factor de interés.
Ajustar presupuestos de parámetros al ±5 % entre redes donde sea viable
y documentar diferencias. Igualar bytes de memoria **incluyendo índices,
metadatos y estado del adaptador**, y número de pasos de optimización.
Reportar también un frente a presupuesto de entrenamiento equivalente
en tiempo/operaciones. Reutilizar pesos entre políticas al aislar
inferencia y entrenar cada núcleo desde cero en comparaciones de
arquitectura. Para separar eficiencia del sistema de eficiencia
arquitectónica, reportar por separado implementaciones en un runtime
común y versiones con kernels optimizados. Publicar commit, dependencias
fijadas, configuración de equipo, comandos y artefactos con resultados
por consulta, semilla y ambiente.

### Mediciones

| Dimensión | Registro previsto |
|---|---|
| Calidad y abstención | Precisión global y por profundidad/incertidumbre, cobertura y riesgo selectivo, utilidad con costeo explícito, ECE/Brier y abstenciones erróneas separadas de aciertos «desconocido» |
| Generalización | Exactitud y utilidad por conjunto reservado, resultado contrafactual por par y caídas frente a distribución conocida |
| Aprendizaje continuo | Matriz tarea × etapa, peor tarea, olvido medio $\max_{u<T}a_{i,u}-a_{i,T}$ y transferencia hacia atrás respecto a la primera prueba de cada tarea |
| Cómputo | Operadores ejecutados, lecturas/escrituras, bytes de estado e índice, MAC/FLOPs convencionados y estimaciones de bytes movidos; separar estimaciones de contadores de hardware |
| Tiempo y RAM | Medianas y p95 de latencia extremo a extremo, inicio en frío y caliente, lote 1 y lotes 8/32, rendimiento bajo concurrencia, tiempo de entrenamiento, RSS máximo del proceso y memoria residente del modelo e índice |
| Energía | Julios por consulta y por consulta correcta, reportando potencia o energía bruta y, si es medible, incremental sobre reposo; sincronizar con ventana de inferencia y declarar qué dominios cubre el sensor |

Primero Linux CPU de un hilo y lote 1, con afinidad y configuración
térmica registradas. Medir preparaciones de tabla e índice, cache fría
y caliente, memoria en procesos separados y coste de la política.
Intercalar bloques por política y repetir al menos diez bloques por
configuración para latencia y energía. Instrumentar con contador CPU
si el equipo lo expone o medidor externo calibrado. Identificar si
la lectura cubre paquete, DRAM o sistema completo, descontar reposo
solo con un intervalo comparable. Sin sensor declarar NE, no inferir
julios de latencia ni potencia de diseño. De ser aprobado y viable,
repetir en ARM local, luego medir servidor de CPU y eventualmente
acelerador. Documentar cuantización, fallback de operadores, colas,
RAM de índices, frecuencia y estrangulamiento térmico. Una ventaja
en x86 de lote unitario no establece ventaja en Android o servidor.

### Análisis y decisiones

Registrar por semilla medias y dispersión, más diferencias emparejadas
con intervalos al 95 % que respeten agrupamiento por semilla y escenario.
No tratar consultas individuales como réplicas independientes del
entrenamiento. Usar comparación primaria A contra la regla miope en
prueba reservada, a presupuestos seleccionados en validación. Conservar
los demás contrastes como secundarios salvo registro previo.

Para revisión se propone como umbral de avance de A una reducción
de al menos 10 % en energía por consulta al mismo riesgo selectivo
y cobertura comparable (margen de 1 punto porcentual para ambos),
frente al control miope; informar el intervalo de diferencias y no
usar un punto estimado aislado como prueba. Si no se puede medir energía,
la latencia extremo a extremo sirve como objetivo exploratorio, con
p95 reportado. Para B, proponer al menos 2 puntos menos de olvido
a igual capacidad, sin más de 1 punto de deterioro en tareas nuevas.
Para C, proponer al menos 10 % de mejora en latencia o energía en
CPU a igual presupuesto y una pérdida de precisión inferior a
1 punto en longitud y composición reservadas. Umbrales y sensor
requieren aprobación antes de experimentar, y no se reinterpretan
después de observar test.

Revisar la frontera calidad-costo y los casos que invierten la política.
Si el selector agrega costo sin mejora o la ventaja depende del acceso
a datos adelantados, priorizar el control sencillo y documentar el fallo.
Si el efecto es incierto, conservar la hipótesis como inconclusa.
Esta fase no implementa ninguna de las alternativas.
