# Auditoría de RF25 para comparación con campo — 2026-09-13

Base: `main`, `b59256942596b7ce80941bc2f63254024ed4f3d3`. Sin commit.
La producción científica congelada no se modifica. Los resultados generados
se encuentran en `outputs/evaluation/field_validation/` y no se versionan.

## Decisión y demostración

La implementación anterior de `field_rf25_local.py` **no acredita equivalencia
con el valor publicado por la producción canónica**. Se retiró `LocalPredictor`
del workflow. Permanecen las funciones pequeñas de componentes para probar
las propiedades algebraicas y el auditor de procedencia.

Sean `w_pi` las áreas positivas de intersección calculadas por
`build_overlap_edges`, `P` los padres representados con MODIS finito y soporte
usable >=0.90, y `A_pi = w_pi / sum_i(w_pi)` para `p` en `P`. La producción
resuelve, sobre todos los píxeles activos (incluyendo soporte neutral interno):

```
G = A A^T
G lambda = m - A x0
x = x0 + A^T lambda
x+ = max(x, 0)
accept iff max(abs(A x+ - m)) <= 0.01 mm
```

`x0` tampoco es puramente puntual: usa Kc neutral, medias de Kc y escalas de
todos los padres elegibles que comparten cada celda fina. Dos padres están
conectados si comparten cualquier celda de 20 m con área positiva, incluso si
esa celda no es publicable. Las entradas correspondientes de `G` son positivas.

**A: cierre local exacto, condicional.** En un dominio finito el recorrido
termina. Una componente completa de padres elegibles, más sus vecinos
inelegibles para la máscara de publicación, contiene todas sus ecuaciones.
Después de permutar índices, `G` es diagonal por bloques. Resolver ese bloque
reproduce la solución matemática del bloque global, siempre que la clasificación
de padres y todos sus soportes sean idénticos. No basta un halo fijo. La prueba
sintética compara el bloque cerrado con la solución global y muestra un error
con un recorte que elimina un vecino acoplado. Esto es equivalencia algebraica,
no garantía de identidad bit a bit de factorizaciones diferentes.

**B: no hay garantía de ahorro local.** Se construyó la geometría canónica sobre
el mosaico archivado de 2022-03-30, sin descargar ni reconciliar productos.
El diagnóstico reproducible está en `audit_support_graph.py` y sus resultados
en `support_graph_audit.json` dentro del directorio de evaluación.

| Caso geométrico/de soporte | Padres elegibles | Componente que comparten ST01–ST03/ST05 | Tiles de esa componente, mínimo |
|---|---:|---:|---:|
| Todos los padres representados potencialmente elegibles | 19.090 | 19.090 | 261/261 |
| Usable >=90% archivado, MODIS exterior supuesto disponible | 9.935 | 9.523 | 175/261 |
| Usable >=90% archivado y MODIS conocido en raster nativo de cuenca | 5.251 | 5.229 | 124/261 |

El último caso ya abarca 2.816.334 celdas finas; el segundo, 5.124.307. Los
tiles de padres inelegibles de frontera aún pueden aumentar el coste. El MODIS
persistido está enmascarado a la cuenca: no permite reconstruir exactamente la
elegibilidad del halo exterior. Los dos últimos casos acotan conectividad del
estado archivado, **no son una reproducción del RF25 actual**. No se midieron
las 14 fechas ni se afirma una frecuencia estadística de componentes grandes.
Sí se demuestra que la hipótesis de una pequeña vecindad local no es válida
en este soporte: puede requerir gran parte o todos los tiles canónicos.

Además, la propuesta local construía geometría, adyacencia y arrays para todo
el dominio antes del cierre. Incluso una componente pequeña no eliminaba ese
coste global.

**C: vía exacta más simple.** Leer el píxel que contiene la estación en
`ET_mm_period` del raster final conserva exactamente el valor `float32`, la
grilla y la máscara publicada. Si falta el producto o su atribución, ejecutar
`scripts/produce_rf25_rasters.py` con el modelo/AOA congelados. El código final
no ofrece una fórmula puntual o un factor MODIS independiente que sustituya
esta reconciliación global.

## Auditoría de semántica

| Elemento | Evidencia en la propuesta local y diferencia relevante |
|---|---|
| 25 predictores | `_download_raw_tile` reutilizaba `_build_raw_tile` y `build_rf25_production_stack`, con los mismos tiles de 4.000 m y buffer de 1.000 m. Selección y orden en `RF25_MODEL_FEATURES`; descarga `float32`, scoring `float64`, persistencia raw `float32`, como la producción canónica. No se detectó una fórmula alternativa en esta etapa. |
| Ópticos (16) | Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, EVI, SAVI, NDWI, NDMI, RedEdge1/2/3, NIR_Broad y NDRE (nombres de modelo con `_mean`). Misma colección S2, Cloud Score+ 0.50, mosaicos diarios, medoid y cálculo posterior de índices. No S1 ni CHIRPS. |
| Meteorología (5) | Tair media/máxima, VPD medio, radiación solar media diaria y viento medio. Mismas transformaciones horarias ERA5-Land y agregación del período MODIS; sin muestreo meteorológico alternativo por estación. |
| Armónicos (4) | Seno/coseno de frecuencias 1 y 2 con día del año inicial y denominador 365.25, iguales a producción. |
| RF final | Cargaba `rf25_virtual10_ge90.joblib` y llamaba `validate_rf25_model`: 300 árboles, max_features=0.33, min_samples_leaf=3, max_depth=None, bootstrap, seed=42, n_jobs=-1. Sin entrenamiento ni OOF de estaciones. |
| AOA | Mismo `rf25_weighted_aoa.joblib` y `score_local_rf25`: stack completo, DI ponderado con parámetros finales, AOA dura y Kc finito >=0. LPD diagnóstico. La carga del artefacto actual no prueba que un raster anterior lo usara. |
| MODIS | Misma `_download_native_modis`, período y QA de `build_modis_period_context`, transform nativo con fase preservada y CRS sinusoidal esférico explícito. La propuesta añadía una exclusión si el padre que contiene el **punto** era NaN. La publicación canónica usa todos los padres **representados que intersectan la celda fina**; no contiene ese veto puntual adicional. En campo se registra MODIS disponible como una puerta de comparación independiente. |
| Reconciliación | Mismo solver, pero sobre índices compactados. La elegibilidad de descubrimiento se duplicaba usando `np.sum` en lugar de las acumulaciones `np.bincount` del solver; cerca del umbral 0.90 no se demostró una clasificación numéricamente idéntica. Un cierre incompleto no sirve. |
| Aceptación global | Un bloque local exitoso no certifica que el resto de bloques pase la condición global. Se añadió un contraejemplo: el bloque de estación pasa y otro padre con media Kc no positiva hace fallar la producción completa. La propuesta podía etiquetar disponible una estación sin que existiera producto global aceptable. |
| Publicación | Requiere usable original, todos los padres representados incidentes elegibles y máscara de cuenca rasterizada con all_touched=False. Se aplica después del solver y del piso no negativo. La propuesta no convertía ET final a `float32`, por lo que ni siquiera pretendía devolver el número exactamente almacenado. |
| Soporte y grilla | La propuesta compartía `_support_tiles(root, 4000)`, origen UTM, malla de 20 m y halo por anillos completos. La función de geometría almacenaba todo el dominio. No hay equivalencia automática con otros tamaños de tile. El muestreo actual usa la grilla del producto, sin interpolación y sin un filtro vectorial extra `inside_basin` que pudiera contradecir la máscara del píxel. |
| Caché local | La identidad incluía algunos archivos pero omitía dependencias como `ee_download.py`; una tabla por fecha tampoco incluía las coordenadas/selección de estaciones en su clave. Son fallos adicionales de garantía, aunque no prueban que un resultado particular estuviera alterado. |

## Procedencia de 2022-03-30

Ruta exacta del raster:

```
C:\Users\User\Desktop\cristian\GEE\26-et-downscaling-fundacion\outputs\current\rasters\2022-03-30\ET_rf25_cs050_ge90_weighted_aoa_exact_overlap_support90_tol001_v1_2022-03-30_20m.tif
```

Creación: `2026-09-12T19:28:02.563995+00:00`; modificación:
`2026-09-12T19:28:06.146735+00:00`. Tamaño: 43.595.673 bytes.

| Archivo | SHA-256 |
|---|---|
| Raster anterior | `08e8473492654526b4fa2e037bf4fb423c39852e8d3a439dec1b91d2df3f5532` |
| `outputs/current/models/rf25_virtual10_ge90.joblib` | `f9f6ef05649cbf07ab4d92d38cb41269517cfb27db027dfba40920834ea1bed4` |
| `outputs/current/models/rf25_weighted_aoa.joblib` | `db3dba544514742604674428703e7d72efb37d01590c0d17005424eec78588b1` |
| `outputs/current/rasters/2022-03-30/production_metadata_rf25_cs050_ge90_weighted_aoa_exact_overlap_support90_tol001_v1.json` | `c41c3e952cea0db6eefcfed9f0132185982c4fd552f9136ee27d5f468974973c` |
| `outputs/current/models/rf25_model_metadata.json` | `dcc384a0f653be5d6f980e65af99e36f0c918d310fc5f0cd9ef18df3b57b1c1d` |

La metadata describe 261 tiles, 9.521 padres elegibles, 5.131.340 celdas activas,
2.669.878 píxeles publicados, cero negativos y error máximo
`1.9184653865522705e-13 mm`. Son declaraciones del producto archivado, no prueba
de identidad con el modelo actual. La firma guardada es
`6b8244038a461a3c64550c676ae3951575842c4b5103a43f7e65524fa82f17c2`.

En tres recargas del mismo archivo, las firmas canónicas fueron distintas,
mientras el hash de los campos numéricos de todos los árboles fue idéntico:
`2fdc7fd7758eca931a2c91a47a7344eb6e79b7b6fd35383e37082d859253ce68`.
Los valores completos de cada experimento, rutas, fechas, bandas y metadata
se conservan en `raster_2022_03_30_provenance_audit.json`.

`rf25_model_signature` usa `joblib.hash(model)`. El hash de un objeto serializado
no está actuando aquí como una identidad científica estable. No se demostró la
causa binaria exacta de la inestabilidad; los bytes de padding de estructuras
son una hipótesis, no un resultado de esta auditoría. `model_source` está
hardcoded a `fitted_in_current_run` en `download_rf25_basin`, aunque el entry
point carga joblib. El manifiesto esperado
`outputs/current/logs/rf25_run_provenance.json` no existe. El tile manifest
enumera tiles/rutas; no vincula modelo y AOA por SHA-256 en tiempo de ejecución.

Conclusión: **atribución no demostrada**. No se usa el raster para métricas RF25
actuales ni como evidencia de reproducción.

Corrección mínima propuesta para una futura modificación canónica: registrar
al comienzo y al final de producción SHA-256 de modelo, AOA, configuración/código
y, al completar, raster y metadata en un manifiesto de ejecución; declarar
correctamente `loaded_artifact` frente a `fitted_in_current_run`. Para una firma
científica independiente de serialización, hashear campos de árboles por nombre,
forma, dtype/endian y valores, junto con parámetros/orden de features y arrays
AOA; probar estabilidad entre recargas y sensibilidad a un cambio numérico.
No se implementa esa modificación canónica. El wrapper de campo registra solo
su propia ejecución futura en el espacio aislado, sin certificar retroactivamente.

## Resultados disponibles y attrition

Se reutilizaron los CSV de ERA5 y satélite ya adquiridos en la ejecución anterior,
registrando sus hashes. No se repitió la adquisición. Los cuatro escenarios y
sus coeficientes permanecen según `METHODOLOGY.md`.

MODIS frente al proxy de campo, `available_sample` (mm por período):

| Escenario | n | R2 | RMSE | MAE | BIAS | KGE |
|---|---:|---:|---:|---:|---:|---:|
| fixed_kc_main | 35 | -0.268866 | 9.219647 | 7.252176 | 1.620980 | 0.244340 |
| historical_all_stations | 55 | -0.917036 | 11.940908 | 9.911224 | 5.889461 | 0.161438 |
| fao_sensitivity | 55 | -0.864528 | 11.715155 | 9.657949 | 6.223599 | 0.221156 |
| ndvi20_all | 48 | -2.205505 | 13.635035 | 11.251280 | 9.571901 | 0.118339 |

La muestra MODIS común a los cuatro escenarios contiene 28 filas de ST01–ST03;
la común a las tres sensibilidades contiene 48. Las seis métricas completas
para cada muestra y por estación están en `field_validation_metrics.csv` y
`field_validation_by_station.csv`. Los hashes de claves son idénticos dentro
de cada comparación/familia. Las métricas RF25 y MODIS–RF25 tienen `n=0` y los
otros cinco indicadores indefinidos; no se infieren errores a partir de n=0.

| Escenario / estación | Candidatos | >=5 días | Proxy | MODIS | Stack RF25 confirmado | AOA | Padres elegibles | Reconciliación | Par final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed / ST01 | 16 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| fixed / ST02 | 16 | 11 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| fixed / ST03 | 16 | 14 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| fixed / ALL | 48 | 35 | 35 | 35 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ST01 | 16 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ST02 | 16 | 11 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ST03 | 16 | 14 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ST04 | 16 | 12 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ST05 | 16 | 13 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| historical y fao / ALL | 80 | 60 | 55 | 55 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ST01 | 16 | 10 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ST02 | 16 | 11 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ST03 | 16 | 14 | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ST04 | 16 | 12 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ST05 | 16 | 13 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| ndvi20 / ALL | 80 | 60 | 48 | 48 | 0 | 0 | 0 | 0 | 0 |

Los ceros desde stack son **ausencia de evidencia verificable**, no fallos
científicos observados. La tabla CSV conserva `n_unknown_at_stage` separado de
`n_failed_known_at_stage`; todos los casos que llegan al stack son desconocidos.
`field_period_pairs.csv` conserva diagnósticos nulos y el motivo del producto.

Hay 14 fechas con al menos un período de campo válido: 2022-03-14, 03-22, 03-30,
04-07, 04-15, 04-23, 05-01, 05-09, 05-17, 05-25, 06-02, 06-10, 06-18 y 06-26.
Trece carecen de raster final; 03-30 tiene procedencia no demostrada. Se registran
en `rf25_field_product_inventory.csv`. No se ejecutó nueva producción de cuenca
en esta auditoría; se dejó implementada la vía canónica opcional por fecha.

## Comparación con git history

Fuentes inspeccionadas: `46b7d30:docs/EXPERIMENT_HISTORY.md`, su revisión actual,
`46b7d30^:docs/decisions/23_field_reporting_and_final_closure.md`,
`46b7d30^:scripts/evaluate_v5_field_proxy.py`,
`857adcc:scripts/validate_field_downscaling.py` y reglas de QC restauradas.
No se recuperaron tablas históricas de pares RF25 en los archivos versionados
o outputs locales; las cifras RF siguientes están preservadas en la narrativa
versionada. No se completan métricas que allí no están publicadas.

| Resultado histórico | n | R2 | RMSE | MAE | BIAS | KGE |
|---|---:|---:|---:|---:|---:|---:|
| RF Virtual10 fixed Kc, dominio propio | 11 | no reportado | 12.505 | no reportado | no reportado | no reportado |
| RF Virtual10 fixed Kc, matched Stable5 | 9 | -0.101 | 9.675 | no reportado | no reportado | no reportado |
| RF Virtual10 retrospectivo, protocolo 30 m, AOA compartida con Ridge | 24 | -1.007 | 11.898 | 9.248 | 4.522 | 0.264 |

El filtro responsable del cambio histórico de dominio propio a matched es la
intersección adicional con disponibilidad del producto Stable5: elimina dos
filas del subconjunto fijo (11 -> 9). El código histórico separa >=5 días de
8/8; no se aplica aquí 8/8 como filtro principal. La comparación retrospectiva
de 24 filas además exige disponibilidad conjunta RF/Ridge bajo AOA y conserva
el protocolo espacial histórico de 30 m. Ninguna es una muestra RF25 final
global de 20 m acreditada por los productos actuales.

La diferencia actual `n=0` frente a esas cifras tiene una causa identificable:
falta de producto final atribuible, antes de conocer el stack/AOA/soporte. No
se adjudica a peor AOA ni a distinta reconciliación sin observar esos diagnósticos.
Tampoco se compara RMSE MODIS de 35 filas con RMSE RF de 11 como si fueran las
mismas observaciones. El flujo de `857adcc` usa RF espacial OOF, predictores
ópticos/S1 y siete meteorológicos, y normalización multiplicativa dentro de un
solo footprint; no es el RF25 final con AOA ponderada y solver global.

Sin las tablas originales y productos finales actuales no puede calcularse un
delta numérico fila por fila ni aislar cuánto aporta cada cambio histórico de
NDVI, modelo, AOA y soporte. Esta limitación se mantiene explícita, sin forzar
coincidencias. Las reglas diarias restauradas (cm -> mm, ETr/ETo diario,
>=5 días y expansión por duración real) sí tienen pruebas específicas.

## Comprobaciones y límites

La configuración pytest existente se revisó: `pyproject.toml` no tenía sección
pytest y no existía otra configuración aplicable. Solo se añadió
`[tool.pytest.ini_options] testpaths = ["tests"]` para que la colección normal
no explore `outputs/`.

La validación usa el entorno ya instalado `et-fundacion`; el alias WindowsApps
de Python no contenía las dependencias. La ejecución de tests necesita acceso
a los temporales de pytest fuera del sandbox. No se alteraron dependencias.

Comprobaciones finales: `python -m pytest -q`: **80 passed, 8 warnings**
(20 pruebas de campo incluidas); `python scripts/run_pipeline.py preflight`:
PASS; `git diff --check`: PASS. Los warnings son deprecaciones Affine/Rasterio
y el aviso de coexistencia de runtimes OpenMP. No se detectó regresión RF25.
La validación offline se ejecutó con éxito, reutilizando los inputs previos.

Los hashes previos de 35 artefactos protegidos de la sesión anterior se
verificaron antes de trabajar. Se amplió el registro a 1.685 archivos (tiles,
rasters, metadata, modelos, población/OOF, selección y código/datos canónicos),
guardado en `protected_before_audit.json`; la verificación final se guarda en
`protected_after_audit.json`: **1.685/1.685 idénticos**, también 35/35 de la sesión
anterior. `HEAD` y rama siguen en `b592569` y `main`. No hay ajuste de ciencia
ni entrenamiento de RF25.

Quedan pendientes productos canónicos atribuibles para las 14 fechas y una
comparación histórica por filas cuando se disponga de sus tablas originales.
Los valores de campo siguen siendo proxies de Kc; `ndvi20_all` comparte
Sentinel-2 con RF25 y es sensibilidad, no validación independiente.
