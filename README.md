# analisis_eph

TFI de la Especialización en Inteligencia de Datos (UNLP). Propuesta aprobada por los
directores.

## Pregunta

¿Qué tan probable es que un hogar se aleje de la línea de pobreza/indigencia, o no logre
acercarse a ella, entre una observación y la siguiente (a un trimestre y a un año)?

El objetivo es anticipar la trayectoria de cada hogar respecto de la línea de pobreza, a
partir del panel rotativo de la Encuesta Permanente de Hogares (EPH) continua.

### Eje del análisis

- **Objeto central**: la distancia de cada hogar a la línea de pobreza/indigencia
  (`ratio_linea_pobreza`/`ratio_linea_indigencia`, CBA/CBT por adulto equivalente) y su variación entre `t` y `t+h`
  (`h=1` y `h=4`, los horizontes que permite la rotación 2-2-2 del panel) — para **todos**
  los hogares, no solo los pobres. `pobre`/`indigente` (ratio < 1) son derivados, no el eje.
- **Predictores candidatos**, no eje: composición del hogar, situación laboral de los
  ocupados (formal/informal/desocupado), ingresos por fuente, vivienda y eventos del
  hogar.
- **Subanálisis**: salida de la pobreza entre los pobres, `t`→`t+4`.

## Fuente de datos

EPH continua (INDEC), microdatos trimestrales de las bases Hogar e Individual, 2003T3–2025T4.


### Línea de pobreza

- **2016T2–2025T4**: línea oficial regional (CBA/CBT por adulto equivalente, 6 regiones).
- **2003T3–2015T2**: retropolación con el procedimiento de dos anclas de Tornarolli (2018,
  CEDLAS DT 226) — diciembre 2006 (Metodología INDEC N°22) y abril 2016 (canasta oficial),
  índice encadenado IPC-GBA INDEC + IPC Provincias CIFRA-CTA.

## Cómo obtener los microdatos

Los microdatos no se versionan en este repositorio: se descargan del INDEC.

- 2016 en adelante: `https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/`
- 2003T3–2015T2: solo disponibles vía Wayback Machine (el INDEC no los mantiene en su sitio
  actual).

`src/eph_client.py` (`EphClient`) descarga y cachea ambos casos en `data/eph_cache/`

## Requisitos

- Python 3.12
- `unar`, para extraer los trimestres 2014–2015 que el INDEC publica en `.rar`
  (Debian/Ubuntu: `apt-get install unar`; macOS: `brew install unar`)
- Dependencias de Python: `pip install -r requirements.txt`

Tests: `pytest tests/`

## Pipeline

Las notebooks se corren en orden: cada una lee la capa que escribió la anterior (ver
`scripts/regenerar.sh` para el orden exacto).

| Notebook | Lee | Escribe |
|---|---|---|
| `01_ingesta_armonizacion` | `data/eph_cache/` | `data/01_armonizado/`, `inventario_variables.csv`, `flags_trimestre.csv` |
| `01b_capa_tipada_nucleo` | `data/01_armonizado/` | `data/02_nucleo/`, `formato_decimal_monetario.csv`, `p21_series_trimestral.csv` |
| `02_panel_hogares` | `data/02_nucleo/` | `data/03_panel/pares_h1/`, `data/03_panel/pares_h4/`, `panel_resumen_pares.csv`, `panel_resumen_final.csv` |
| `03_exploracion_descriptiva` | `data/02_nucleo/`, `data/03_panel/` | CSV descriptivos (quintiles, NBI) |
| `04_trayectorias` | `data/03_panel/` | `data/04_trayectorias/`, `trayectorias_*.csv` — diagnóstico de factibilidad del panel dinámico |
| `05_distancia_linea_pobreza` | `data/02_nucleo/`, `data/01_armonizado/` | `data/02b_distancia_linea_pobreza/hogar/`  `data/canastas_indec/*_crudo.csv` |
| `05b_canastas_retropoladas` | `data/insumos_lineas/`, `data/canastas_indec/` | canasta histórica según Tornarolli, 2003T1–2016T1 |
| `05c_distancia_historica` | `data/02_nucleo/`, `data/01_armonizado/` | `data/02b_distancia_linea_pobreza/hogar/` (era `"historica"`, 2003T3–2015T2, mismas particiones que `05`), `linea_pobreza_trimestral.csv`, `frontera_2015_2016_vs_cedlas.csv` |
| `05d_no_respuesta_historica` | `data/01_armonizado/`, `data/02_nucleo/`, `data/02b_distancia_linea_pobreza/hogar/`, `data/03_panel/` | `imputacion_*.csv` — diagnóstico de la imputación de ingresos de INDEC en la era histórica (`IDIMPH`/`IDIMPP`), no escribe capas |
| `06_trayectorias_distancia` | `data/02b_distancia_linea_pobreza/hogar/`, `data/03_panel/` | `data/05_trayectoria_distancia/{pares_h1,pares_h4}/` (solo era `"regional_2016"`) |
| `06c_trayectorias_distancia_historica` | `data/02b_distancia_linea_pobreza/hogar/`, `data/03_panel/`, `data/01_armonizado/` | `data/05_trayectoria_distancia/{pares_h1,pares_h4}/` (era `"historica"`, 2003T3–2015T2, mismas particiones que `06`; agrega `par_valido_c`, imputación y flags de corte) |
| `06b_diagnosticos_trayectoria` | `data/05_trayectoria_distancia/` | CSV descriptivos (tendencia, regresión a la media, conteos train/holdout) |
| `06d_diagnosticos_dos_universos` | `data/05_trayectoria_distancia/`, `data/02b_distancia_linea_pobreza/hogar/` | `dos_universos_*.csv` — 2003–2025 contra 2016–2025: conteos, líneas base, dispersión por era, sensibilidades |

`notebooks/00_explorar.ipynb` es de solo lectura: arma vistas DuckDB sobre las capas ya
escritas para explorar sin volver a correr el pipeline.