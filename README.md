# analisis_eph

TFI de la Especialización en Inteligencia de Datos (UNLP). Propuesta en definición.

## Pregunta

¿Qué hogares tienen mayor riesgo de empeorar su situación de ingresos o laboral entre una observación y la
siguiente, a un trimestre y a un año? ¿Cambian los factores que lo predicen según el horizonte y según el
contexto político (fase del mandato presidencial, alineación provincia–Nación)?

El objetivo es un sistema de alerta temprana basado en el panel de hogares de la Encuesta Permanente de
Hogares (EPH) continua.

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

## Estado

Pipeline de datos y panel de hogares construidos. En curso: definición de la variable a predecir y
revisión bibliográfica.