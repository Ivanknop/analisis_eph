# analisis_eph

TFI de la Especialización en Inteligencia de Datos (UNLP). Propuesta en definición.

## Pregunta

¿Qué tan probable es que un hogar **caiga en mayor informalidad laboral, o no logre salir de ella**, entre una
observación y la siguiente (a un trimestre y a un año), según su **quintil de ingreso** y sus **condiciones
habitacionales**?

Dicho de otro modo: ¿quien vive en condiciones habitacionales deficitarias tiene más dificultad para salir de
la precariedad laboral, y más riesgo de caer en ella, que un hogar del mismo quintil sin ese déficit?

El objetivo es un sistema de alerta temprana a nivel hogar, basado en el panel de la Encuesta Permanente de
Hogares (EPH) continua.

### Eje del análisis

- **Unidad**: el hogar comparado consigo mismo entre `t` y `t+h` (`h=1` y `h=4`, los que permite la rotación
  2-2-2 del panel).
- **Resultado a predecir**, en dos direcciones:
  - **Permanencia**: hogares con activos y sin empleo formal en `t` que siguen igual en `t+h` (no salen).
  - **Caída**: hogares con algún empleo formal en `t` que lo pierden en `t+h` (pasan a informalidad,
    independiente o sin ocupados).
- **Estratificadores principales**: quintil/decil de ingreso per cápita (`DECCFR`) y el ancla habitacional
  (`ancla_con_deficit`: hacinamiento, materiales, condiciones sanitarias). La vivienda varía poco entre
  visitas, por eso funciona como ancla y predictor, no como resultado.
- **Ingreso**: entra como posición en `t` (quintil/decil) y como eje complementario de la etiqueta; la
  definición final de la población en riesgo está abierta (ver D14 y D15 en `docs/decisiones_metodologicas.md`).

### Qué es contexto, no eje

El mandato presidencial (y, eventualmente, la alineación provincia–Nación) es **una variable de contexto más**:
se usa para describir el período y se evaluará al final como feature candidata (con y sin ella), no organiza
el análisis. Con 6 mandatos en la serie, cualquier comparación entre ellos es descriptiva. Las secciones por
mandato de los notebooks 02 y 05 quedan como descripción de contexto (ver D15).

## Fuente de datos

EPH continua (INDEC), microdatos trimestrales de las bases Hogar e Individual, 2003T3–2025T4: 86 trimestres
publicados en 31 aglomerados urbanos.

Cortes de la serie:
- El INDEC no publicó 2007T3 ni 2015T3–2016T1.
- El identificador de vivienda (`CODUSU`) cambia de formato entre la base histórica (hasta 2015) y la regular
  (desde 2016), por lo que ningún par de panel cruza ese corte.
- El tratamiento de ingresos no declarados cambia en 2016.

Documentación: diseño de registros de las bases (INDEC): [link al codebook]. El detalle de cada decisión
metodológica está en `docs/decisiones_metodologicas.md`.

Insumo auxiliar: índice FACPCE para el empalme del IPC (FACPCE, Resolución JG 539/18),
en `data/indice-FACPCE.*` (URL de la fuente: pendiente).

## Cómo obtener los microdatos

Los microdatos no se versionan en este repositorio: se descargan del INDEC.

- 2016 en adelante: `https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/`
- 2003T3–2015T2: solo disponibles vía Wayback Machine (el INDEC no los mantiene en su sitio actual).

`src/eph_client.py` (`EphClient`) descarga y cachea ambos casos en `data/eph_cache/`, con alcance nacional
(no filtra por aglomerado). `data/meta/manifiesto_insumos.csv` registra URL, fecha de descarga y hash SHA-256
de cada archivo usado, para verificar que se trabaja con los mismos insumos.

## Requisitos

- Python 3.12
- `unar`, para extraer los trimestres 2014–2015 que el INDEC publica en `.rar`
  (Debian/Ubuntu: `apt-get install unar`; macOS: `brew install unar`)
- Dependencias de Python: `pip install -r requirements.txt`
- `nbstripout` (recomendado, no obligatorio), para que las notebooks se commiteen sin
  salidas: `pip install nbstripout && nbstripout --install --attributes .gitattributes`
  una vez por clon. El filtro solo actúa al pasar por `git add`/commit -- el archivo
  local conserva las salidas que hayas corrido vos.

Tests: `pytest tests/`

## Pipeline

Las notebooks se corren en orden: cada una lee la capa que escribió la anterior.

| Notebook | Lee | Escribe |
|---|---|---|
| `01_ingesta_armonizacion` | `data/eph_cache/` | `data/01_armonizado/`, `inventario_variables.csv`, `flags_trimestre.csv` |
| `01b_capa_tipada_nucleo` | `data/01_armonizado/` | `data/02_nucleo/`, `formato_decimal_monetario.csv`, `p21_series_trimestral.csv` |
| `02_panel_hogares` | `data/02_nucleo/` | `data/03_panel/pares_h1/`, `data/03_panel/pares_h4/`, `panel_resumen_pares.csv` |
| `03_exploracion_descriptiva` | `data/02_nucleo/`, `data/03_panel/` | `quintiles_*.csv`, `ancla_habitacional_estabilidad.csv`, `condiciones_vida_*.csv` (transiciones laborales por quintil × ancla) |
| `04_trayectorias` | `data/03_panel/` | `data/04_trayectorias/`, `trayectorias_*.csv` |
| `05_composicion_quintiles` | `data/02_nucleo/` | `composicion_quintiles_*.csv` (contexto descriptivo, transversal) |
| `06_poblacion_riesgo` | `data/02_nucleo/`, `data/03_panel/`, `data/04_trayectorias/` | `riesgo_*.csv` (población en riesgo y salida, por quintil/decil × ancla) |
| `07_cascada_targets` | `data/02_nucleo/`, `data/01_armonizado/`, `data/03_panel/` | `data/07_targets/pares_targets/`, `cascada_trimestral.csv`, `vivienda_frecuencias*.csv`, `cascada_folds.csv` (cascada de casos y targets Y1/Y2, D24) |

Dónde está cada parte de la pregunta:

| Pieza | Notebook | Salida principal |
|---|---|---|
| Ancla habitacional (definición y estabilidad entre visitas) | 03, Parte E | `ancla_habitacional_estabilidad.csv` |
| Caída desde empleo formal y salida de la informalidad, por quintil × ancla | 03, Parte E | `condiciones_vida_transiciones.csv`, `condiciones_vida_resumen_deficit.csv` |
| Población en riesgo y tasa de salida (3 definiciones candidatas) | 06 | `riesgo_salida_trimestre.csv`, `riesgo_salida_ancla_bootstrap.csv` |
| Brecha por ancla dentro de cada decil (no solo composición) | 06, Sección 3 | `riesgo_salida_ancla_por_decil.csv`, `riesgo_salida_ancla_estandarizada.csv` |
| Salida sostenida (trayectorias de 4 apariciones) | 04 + 06, Sección 2c | `riesgo_salida_estricta.csv` |
| Contexto por trimestre y mandato | 02 (resumen), 05 | `panel_resumen_final.csv`, `composicion_quintiles_*.csv` |

Los CSV se escriben en `data/meta/`.

- **Qué se versiona:** los Parquet de `data/0N_*/` se regeneran y no se versionan. Los CSV de `data/meta/` son
  chicos, documentan resultados y sí se versionan.
- **Parámetros:** cada notebook acepta `TRIMESTRES` (subconjunto opcional) y `REHACER` (por defecto `False`:
  reprocesa solo lo que falta o está desactualizado). La primera corrida completa tarda unos 10 minutos,
  sobre todo por el parseo de las bases históricas.
- **Registro:** cada corrida queda en `data/meta/corridas.csv`, y el avance por trimestre en
  `data/meta/resumen_0N_*.csv`.

`notebooks/00_explorar.ipynb` es de solo lectura: arma vistas DuckDB sobre las capas ya escritas para
explorar sin volver a correr el pipeline.
