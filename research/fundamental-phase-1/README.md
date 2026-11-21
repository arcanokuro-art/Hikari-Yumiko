# HANA Fundamental Research — Fase 1

**Proyecto:** Hikari Yumiko, desarrollado por Zaskar.
**Corte de revisión:** 21 de noviembre de 2026.
**Base de código:** HANA-0.6, commit `4759569b115551f3268553153a361433eab872c9`.

## Resumen para decisión

La pregunta prioritaria es si un núcleo compacto puede **elegir qué operación
ejecutar y a qué memoria acceder** bajo un presupuesto, con mayor utilidad
por joule que políticas sencillas que tienen las mismas opciones. HANA-0.6
probó cuándo detener una secuencia de lecturas predefinida. El nuevo problema
introduce acciones alternativas con efectos y costos diferentes. Controlar el
cómputo, tener memoria externa y estimar el valor de la información cuentan
con antecedentes directos. Una contribución podría surgir de una formulación
concreta que mantenga ventaja tras aislar el selector, la representación, la
memoria y el hardware.

| Prioridad propuesta | Pregunta | Condición para continuar |
|---|---|---|
| 1. Selección de operaciones | ¿Predice HANA si conviene verificar, recuperar, computar, responder o abstenerse? | Mejora frente a una regla miope con el mismo menú y costo medido |
| 2. Retención bajo presupuesto | ¿Qué experiencias vale la pena conservar cuando cambia el entorno? | Reduce olvido frente a repetición y expulsión simples con los mismos bytes y actualizaciones |
| 3. Estado de dos escalas | ¿Puede una recurrencia multiescala y una memoria dispersa conservar detalles con menor tráfico? | Mantiene calidad en longitudes y combinaciones reservadas y reduce latencia o energía en CPU |

**Evidencia existente.** HANA-0.4 y HANA-0.5 mostraron asignación de ciclos
en grafos sintéticos con una interfaz de memoria estructurada. En HANA-0.5,
una igualdad exacta superó al controlador aprendido en latencia y en un
conjunto reservado. En HANA-0.6, una regla probabilística fue aproximadamente
dos veces más rápida y similar en precisión con 30 % de ruido. Aumentar el
ruido indujo al controlador a detenerse antes en promedio. Estos resultados
motivan un experimento de decisión, pero no prueban eficiencia energética,
aprendizaje continuo, escalabilidad ni originalidad arquitectónica.

Los documentos son una agenda para revisión. Las hipótesis, márgenes y
presupuestos son propuestas previas a cualquier implementación o entrenamiento
de esta fase.

- [Auditoría de HANA-0 Nano a HANA-0.6](auditoria.md)
- [Revisión bibliográfica y mapa de antecedentes](bibliografia.md)
- [Formulación matemática de las tres hipótesis](hipotesis.md)
- [Protocolo experimental y criterios de decisión](protocolo.md)
