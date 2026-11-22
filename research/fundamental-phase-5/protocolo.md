# Protocolo de diagnóstico

Se conservan la memoria, las acciones, la recompensa y el posterior de las fases anteriores. Los contrastes se fijan antes de ejecutar las nuevas políticas.

| Hipótesis | Predicción | Experimento y control | Criterio de refutación |
|---|---|---|---|
| H1 El margen explica la mayoría de las paradas prematuras | En OOD legado, más de la mitad de las paradas incorrectas tienen ganancia predicha positiva | Clasificar cada parada con pesos fijos por signo de la predicción y comprobar el efecto de margen cero en los mismos episodios | La proporción es menor o igual a 0,5, o quitar el margen no elimina la mayoría de las paradas |
| H2 La cobertura de entrenamiento es un factor modificable | Ampliar la cobertura de fidelidades, omisiones y precios recupera utilidad OOD con la misma capacidad | Comparar ajuste histórico y cobertura amplia con el mismo número de perfiles, red, optimizador y márgenes. IC emparejado de utilidad OOD a presupuesto 5 | Sin intervalo positivo, no se confirma. Una diferencia no positiva en todo el IC refuta la predicción de mejora |
| H3 La fiabilidad no observada introduce ambigüedad estructural | Dos mundos con idéntica entrada pública pueden exigir decisiones opuestas | Cambiar únicamente la fiabilidad real en un par del mismo estado público y resolver con un oráculo informado | Si las acciones óptimas se intersectan, el par no demuestra una colisión |
| H4 La red aprende mal el signo del valor adquirido | Corregir solo las paradas que una regla miope contradice recupera parte de la utilidad OOD | Control de diagnóstico que consulta Q₁ solo al parar. Comparar con pesos intactos y exacto de dos pasos | Si no mejora la utilidad OOD o la pérdida resulta explicada por margen, el diagnóstico de una señal de valor aprovechable pierde apoyo |
