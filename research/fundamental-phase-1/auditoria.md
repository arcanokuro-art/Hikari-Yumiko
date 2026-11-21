# Auditoría de HANA-0 Nano a HANA-0.6

Se inspeccionaron los modelos, generadores, ejecutores de experimentos,
pruebas y workflows del historial acumulado hasta HANA-0.6. Las cifras
provienen de los informes de los PR. Se verificaron las conclusiones de
GitHub Actions en sus revisiones de cabecera. Esta fase documental no
reentrenó los modelos ni reprodujo independientemente las mediciones.

| Versión | Mecanismo inspeccionado | Resultado registrado y alcance |
|---|---|---|
| Nano | GRUCell, compuerta de trabajo, memoria reciente por igualdad y refinamiento hasta tres ciclos | 97,48 % en progresiones, frente a 99,96 % de ZNC-0 y 99,79 % de GRU. HANA fue más lento. Una semilla, 12 tokens. [PR #2](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/2) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44883469343) |
| 0.2 | Ablaciones de ciclos y memoria, proyección conjunta y ruta sin indexación por lote de un ejemplo | En recuperación larga de 49 tokens, 99,48 % con 6,25 ms frente a 9,14 ms de Nano. Un ciclo fijo obtuvo 99,89 % con 3,77 ms. Las progresiones quedaron cerca del azar en este entrenamiento mixto. [PR #3](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/3) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44889998252) |
| 0.3 | Dos probabilidades condicionales, costo esperado de ciclos, registro de 64 o 128 entradas | A 105 tokens, penalizar redujo 2,98 a 1,15 ciclos. Con 64 entradas, 5,82 % de precisión penalizada. Con 128, 100 % adaptable y 99,35 % fijo de un ciclo, con 10,16 y 7,67 ms respectivamente. La capacidad explicó la recuperación en este generador. [PR #4](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/4) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44893730442) |
| 0.4 | Tabla relacional exacta, puntero distribuido, GRUCell y costo de salida | 100 % y 1,05 / 2,02 / 3,00 ciclos en grafos frescos. Los controles de uno y dos ciclos fallaron a tres saltos. La longitud de la consulta informaba el número de relaciones y el índice del ciclo entraba en la política. Frente a referencias seriales cambió también la interfaz de acceso. [PR #5](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/5) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44913912686) |
| 0.5 | Tres relaciones para todas las consultas, lectura exacta, parada al encontrar un atributo objetivo | Ambas políticas lograron 100 % y usaron los ciclos correctos en grafos frescos y en 1 728 pares contrafactuales. La igualdad directa fue más rápida a las tres profundidades. La política neuronal se detuvo tarde en seis consultas con etiqueta de banco reservada. [PR #6](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/6) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44915706847) |
| 0.6 | Evidencia parcial y votos, predicción de desconocido, cabeza de parada y calibración en validación | A 30 % de ruido, 88,09 ± 1,37 % adaptable, 88,61 ± 1,81 % probabilística y 91,54 ± 0,63 % recurrente fija entrenada para el tercer ciclo. Latencia adaptable 0,733 ms frente a 0,361 ms probabilística. En 862 contrafactuales con cambio de objetivo, acierto conjunto 77,80 % y 69,31 % respectivamente. [PR #7](https://github.com/arcanokuro-art/Hikari-Yumiko/pull/7) · [CI](https://github.com/arcanokuro-art/Hikari-Yumiko/actions/runs/44916934029) |

## Hallazgos de la inspección

- **La memoria cambió de naturaleza.** Nano a 0.3 almacenan estados recientes
  con capacidad fija. De 0.4 en adelante se construye una tabla de hechos
  estructurada por episodio. La capacidad, direccionamiento, costo de
  preparación, exactitud y permanencia de ambos diseños difieren. Una
  ablation sin tabla mide la contribución de la interfaz completa.
- **Los costos son heterogéneos.** El contador de ciclos no mide el costo de
  preparar la tabla, mover datos, ejecutar compuertas o programar un lote
  divergente. El pico RSS publicado involucra también entrenamiento y
  PyTorch. No hay mediciones históricas de energía ni de memoria residente
  exclusiva de inferencia. Los hitos usan tareas distintas, por lo que sus
  latencias y precisiones no constituyen una serie de mejora controlada.
- **0.6 hace parada, no selección de operaciones.** En
  [hana06.py](../../hikari/hana06.py) la consulta ya fija el banco siguiente.
  El costo entrenado compara pérdidas supervisadas futuras de esa misma
  trayectoria. La inferencia recibe las observaciones, pero el codificador
  inicial resume globalmente las aristas y votos visibles. No puede
  interpretarse como adquisición de datos que el modelo aún no ha visto.
  En un experimento de valor de información, las observaciones no adquiridas
  deben quedar fuera incluso del resumen inicial.
- **La incertidumbre no sigue una curva simple.** Con los mismos grafos,
  de 10 % a 55 % de ruido la ruta adaptable pasa de 2,04 a 1,92 ciclos.
  En 55 % de ruido y tres saltos obtiene 35,73 %, frente a 54,26 % de tres
  ciclos fijos. La abstención es una predicción del criterio del generador,
  que usa etiquetas limpias ocultas. No certifica verdad lógica a partir
  de votos contradictorios.
- **La generalización es acotada.** Cambiar un identificador de banco en
  0.4/0.5 evalúa acceso con etiqueta nueva en una interfaz estable. Cambiar
  el orden de relaciones, el tamaño del grafo o los niveles de ruido son
  pruebas distintas. Ninguna por sí sola prueba transferencia de reglas,
  razonamiento abierto o escalamiento a servidores.
- **Leer, escribir y aprender son distintos.** Escribir en una memoria
  episódica o cambiar un estado recurrente no implica retener habilidades
  nuevas tras secuencias de tareas. Los experimentos anteriores no aislaron
  retención, interferencia, revisiones de hechos ni transferencia tras
  actualizaciones en producción.
