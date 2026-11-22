# Auditoría de Fase 3

Se revisaron los documentos de Fase 1 (PR #8), la formulación, los
ejecutores y resultados de Fase 2 (PR #9) y la implementación, los informes,
las pruebas y los datos de Fase 3 (PR #10). La base inmediata es
`ca08937a3f24cbe56d127091a2a7dae70874764a`. Fase 1 estableció el problema
de selección de operaciones. Fase 2 fijó entorno, núcleo de creencia y
referencias. Fase 3 añadió una corrección neuronal al plan miope.

Antes de modificar el mecanismo se ejecutaron las seis pruebas de Fase 3
y sus cinco corridas completas (1 800 pasos, 1 024 episodios por celda).
Coincidieron acciones, precisión, utilidad, costo, abstención, máscaras de
auditoría y contrafactuales. Se exige igualdad de valores deterministas salvo
MSE y arrepentimiento medio (tolerancia relativa 1e−9, absoluta 1e−12).
Latencias, tiempo de ajuste, RSS y metadatos de entorno se registran aparte.
Los archivos `reproduction/seed-*.json.gz` conservan los registros nuevos.

Para un estado visitado se compara el valor exacto Q₃ de la acción elegida
con el valor T de detenerse. Útil significa Q₃−T > 1e−9, perjudicial significa
Q₃−T < −1e−9, redundante significa empate dentro de esa tolerancia.
Una acción se marca además como costosa cuando su tarifa simulada es ≥3.
La marca «costosa» puede coexistir con cualquiera de las tres categorías.
Una adquisición puede ser útil aunque exista una alternativa de mayor
utilidad. La referencia Q₃ supone continuación óptima. La clasificación
describe cada estado visitado y el costo del modelo, así que no estima el
efecto causal de cambiar de política en un estado inicial común.

Medias de cinco semillas, 1 024 episodios por semilla, presupuesto 5.
La Fase 3 reprodujo el residual y la Fase 4 lo reentrenó con la misma
configuración para aplicar el clasificador ampliado sobre los mismos casos.

| Escenario y política | Operación | Útiles | Redundantes | Perjudiciales | Costosas | Unidades |
|---|---|---:|---:|---:|---:|---:|
| ID residual | Local | 270,8 | 0 | 13,2 | 42,4 | 450,2 |
| ID residual | Índice | 237,0 | 0 | 129,2 | 37,0 | 559,2 |
| ID residual | Verificación | 320,4 | 0 | 1,0 | 49,8 | 526,0 |
| ID net | Local | 143,6 | 0 | 38,6 | 0,6 | 218,0 |
| ID net | Índice | 248,8 | 0 | 59,2 | 31,4 | 466,6 |
| ID net | Verificación | 287,8 | 0 | 14,8 | 53,8 | 514,8 |
| OOD residual | Local | 315,0 | 0 | 0 | 143,2 | 817,2 |
| OOD net | Local | 31,2 | 0 | 3,0 | 0,6 | 69,0 |

En ID, el residual heredado eligió índices perjudiciales en promedio 129,2
veces y se detuvo antes de tiempo 28,8 veces. Net redujo esos índices a 59,2,
pero elevó las paradas prematuras a 137,4 y desplazó parte del error hacia
lecturas locales y verificaciones. En OOD conjunto, el residual y el miope
acertaron la política exacta en todas las decisiones visitadas en presupuesto
5. Net se detuvo prematuramente 283,8 veces en promedio. En memoria incompleta
net registró 106,8 operaciones perjudiciales frente a 71,2 del residual. Por
tanto, menos gasto no implica menos errores de selección. Los JSON conservan
los otros presupuestos, políticas, costos y contadores.
