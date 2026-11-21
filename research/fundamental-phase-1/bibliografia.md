# Revisión bibliográfica y mapa de antecedentes

Revisión narrativa dirigida por mecanismos. Se consultaron publicaciones
primarias, actas y prepublicaciones mediante título o identificador, y se
buscaron antecedentes de control de cómputo, valor de información, recuperación
activa, adaptación y medición energética. No es una revisión sistemática
exhaustiva ni una búsqueda de patentes. Las referencias identifican métodos
y condiciones, no transfieren sus resultados a HANA.

| Trabajo verificable | Mecanismo y relación con HANA |
|---|---|
| Peng et al. (2023), *RWKV: Reinventing RNNs for the Transformer Era*. [EMNLP / arXiv](https://arxiv.org/abs/2305.13048) | Mezcla temporal recurrente derivada de una forma de atención lineal. Su estado de inferencia no crece con la longitud para un modelo fijo. Referencia para comparar estado compacto, no memoria direccionable ni política de deliberación. |
| Gu y Dao (2023, prepublicación; 2024, COLM), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv](https://arxiv.org/abs/2312.00752) | Dinámica de espacio de estados selectiva dependiente de la entrada y cálculo optimizado para hardware. Una complejidad lineal en secuencia no demuestra una ventaja CPU para operadores pequeños ni recuperación exacta de hechos arbitrarios. |
| Behrouz, Zhong y Mirrokni (2024, primera versión en arXiv; identificador de 2025), *Titans: Learning to Memorize at Test Time*. [arXiv](https://arxiv.org/abs/2501.00663) | Memoria neuronal de largo plazo actualizada durante inferencia, con señales de sorpresa, olvido y variantes que combinan atención. Antecedente claro de actualización a tiempo de prueba, no equivalente al registro exacto episódico de HANA. |
| Sun et al. (2024), *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. [arXiv](https://arxiv.org/abs/2407.04620) | Una red pequeña, actualizada con objetivo autosupervisado durante inferencia, cumple el papel de estado recurrente. Control conceptual para separar memoria paramétrica rápida de escrituras tradicionales. |
| Graves (2016), *Adaptive Computation Time for Recurrent Neural Networks*. [arXiv](https://arxiv.org/abs/1603.08983) | Aprende una parada diferenciable entre entradas y salidas con costo de cómputo. Precede a la idea de dar más ciclos a un token. La dinámica y la pérdida de HANA no son una implementación literal de ACT. |
| Dehghani et al. (2018, arXiv; 2019, ICLR), *Universal Transformers*. [arXiv](https://arxiv.org/abs/1807.03819) | Reutiliza un bloque recurrentemente en profundidad y estudia cómputo adaptativo por posición. Precedente de profundidad variable, aunque use atención propia de Transformer. |
| Banino, Balaguer y Blundell (2021), *PonderNet: Learning to Ponder*. [arXiv](https://arxiv.org/abs/2107.05407) | Define una distribución de parada mediante probabilidades condicionales y regularización. Es próximo a HANA-0.3, que usa una penalización explícita y enrutamiento duro con estimación aproximada de gradientes. |
| Campos et al. (2018, ICLR), *Skip RNN*. [arXiv](https://arxiv.org/abs/1708.06834) | Aprende a omitir actualizaciones de estado. Comparador para diseños recurrentes que procesan solo eventos seleccionados. |
| Graves, Wayne y Danihelka (2014), *Neural Turing Machines*. [arXiv](https://arxiv.org/abs/1410.5401) | Controlador neuronal con lecturas y escrituras diferenciables en memoria externa. Antecedente de memoria direccionable y aprendizaje algorítmico. |
| Graves et al. (2016), *Hybrid computing using a neural network with dynamic external memory*. [Nature, DOI](https://doi.org/10.1038/nature20101) | Differentiable Neural Computer. Amplía direccionamiento con asignación, uso de memoria y enlaces temporales. Es más flexible que la tabla exacta con claves provistas por el generador de HANA, y ofrece un control conceptual para memoria y acceso relacional. |
| Russell y Wefald (1991), *Principles of metareasoning*. [DOI](https://doi.org/10.1016/0004-3702(91)90015-C) | Selección de operaciones por valor de cómputo esperado y costo. Antecedente directo de la pregunta prioritaria, incluso si el estimador de HANA fuese neuronal. |
| Kaelbling, Littman y Cassandra (1998), *Planning and acting in partially observable stochastic domains*. [DOI](https://doi.org/10.1016/S0004-3702(98)00023-X) | Un estado de creencia integra historia y observación. Las acciones tienen valor por sus consecuencias y por la información que revelan. Base para un oráculo pequeño y un control miope que comparten observaciones. |
| Andreas et al. (2016), *Neural Module Networks*. [arXiv](https://arxiv.org/abs/1511.02799) | Composición de módulos especializados para responder consultas. Precede a la idea de un menú de operaciones tipadas. |
| Yao et al. (2023, ICLR), *ReAct: Synergizing Reasoning and Acting in Language Models*. [arXiv](https://arxiv.org/abs/2210.03629) | Intercala razonamiento y acciones para acceder a información externa. No se adopta su modelo ni su interfaz, pero es un antecedente de control de operaciones y evidencia. |
| Jiang et al. (2023), *Active Retrieval Augmented Generation*. [arXiv](https://arxiv.org/abs/2305.06983) | FLARE dispara recuperación en función de la generación anticipada y su confianza. Precedente de solicitar evidencia selectivamente durante inferencia. |
| Jeong et al. (2024), *Adaptive-RAG*. [arXiv](https://arxiv.org/abs/2403.14403) | Elige estrategias de recuperación según complejidad de consulta. Referencia de selección entre estrategias, distinta de una decisión secuencial con valor marginal y costo medido por equipo. |
| Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*. [PNAS, DOI](https://doi.org/10.1073/pnas.1611835114) | EWC penaliza cambios en pesos importantes para tareas anteriores. Control conceptual para olvido e interferencia. |
| Lopez-Paz y Ranzato (2017), *Gradient Episodic Memory for Continual Learning*. [arXiv](https://arxiv.org/abs/1706.08840) | Una memoria episódica informa restricciones sobre las actualizaciones. Antecedente de emplear un búfer para preservar desempeño en tareas pasadas. |
| Lake y Baroni (2018), *Generalization without systematicity*. [PMLR](https://proceedings.mlr.press/v80/lake18a.html) | Controla combinaciones reservadas para distinguir extrapolación composicional de simple recombinación cercana al entrenamiento. |
| Guo et al. (2017), *On Calibration of Modern Neural Networks*. [arXiv](https://arxiv.org/abs/1706.04599) | Temperatura y evaluación de calibración. Útil para HANA-0.6, con el matiz de que ahí se calibra la probabilidad de la predicción elegida y no toda la distribución. |
| Dao et al. (2022), *FlashAttention*. [arXiv](https://arxiv.org/abs/2205.14135) | Separa complejidad aritmética de transferencia de datos. Ilustra por qué el costo de IO y el kernel pueden dominar la comparación incluso para operaciones matemáticamente equivalentes. |
| Henderson et al. (2020), *Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning*. [arXiv](https://arxiv.org/abs/2002.05651) | Reporte de energía y configuración experimental. Una estimación de FLOPs, potencia nominal o latencia no sustituye energía medida. |

## Clasificación de las propuestas

- **Conocidos:** recurrencia, espacio de estados selectivo, profundidad
  adaptativa, direccionamiento, actualización de memoria, recuperación
  activa, calibración, selección por utilidad y costo, y composición modular.
- **Instanciaciones propias por probar:** una política que estime la
  ganancia contrafactual de accesos heterogéneos, condicionada por un
  perfil medido de costos y un presupuesto; una puntuación de retención
  amortizada por bytes y lecturas futuras; una interfaz multiescala con
  corrección dispersa de pérdida de información. Cada una tiene vecinos
  científicos cercanos. Una implementación distinta o una combinación
  de elementos conocidos no demuestra por sí misma novedad.
- **Sin demostrar:** que el controlador sea superior a un planificador
  miope bien calibrado, que el almacenamiento preserve conocimiento en
  el tiempo, que el estado comprimido generalice, o que alguna opción
  ahorre julios en un dispositivo real. Para una afirmación de novedad,
  contrastar el mecanismo exacto y su objetivo con trabajos vecinos
  adicionales antes de divulgarla.
