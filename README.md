# Hikari Yumiko

**Proyecto independiente de investigación de inteligencia artificial de ZASKAR.** Hikari se desarrollará y entrenará desde parámetros aleatorios, sin pesos preentrenados de terceros. Se podrá ejecutar desde ZASKIA en una fase futura mediante un adaptador, sin depender de ZASKIA.

## Objetivo inicial
Investigar una arquitectura neuronal no Transformer para modelado secuencial, lenguaje y razonamiento. El primer núcleo `ZNC-0` usa una transición recurrente con actualización controlada, y será evaluado frente a líneas base. **No se afirma novedad científica:** la recurrencia y las compuertas son técnicas existentes; la investigación buscará comprobar mejoras medibles antes de introducir hipótesis más ambiciosas.

## Primer experimento
- Predicción del siguiente token a partir de secuencias sintéticas.
- Entrenamiento reproducible en CPU mediante PyTorch.
- Prueba de que el error disminuye y la salida conserva su forma.
- Control de versiones independiente y CI en GitHub Actions.

## Ejecución

```bash
python -m pip install -r requirements.txt
python -m hikari.train --steps 250
python -m unittest discover -s tests -v
```

Los checkpoints generados quedan en `checkpoints/` (ignorados por Git). El código todavía es un experimento de aprendizaje, **no un chatbot listo ni una IA de razonamiento general**.

## Fases
1. ZNC-0: recurrencia entrenable y mediciones.
2. Memoria externa, recuperación y tareas de dependencia larga.
3. Comparativas con RNN/GRU y Transformer de presupuesto equivalente.
4. Vocabulario, datos lingüísticos auditados, evaluación de generalización.
5. Empaquetado local e integración opcional con ZASKIA.

Se documentarán los experimentos, seeds, datasets, costos computacionales y resultados negativos. No se modificarán versiones estables sin validación.
