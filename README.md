# analisis_eph

TFI de la Especialización en Inteligencia de Datos (UNLP). Propuesta en definición, no
validada todavía por los directores.

## Pregunta

¿Qué tan probable es que un hogar se aleje de la línea de pobreza/indigencia, o no logre
acercarse a ella, entre una observación y la siguiente (a un trimestre y a un año)?

El objetivo es anticipar la trayectoria de cada hogar respecto de la línea de pobreza, a
partir del panel rotativo de la Encuesta Permanente de Hogares (EPH) continua — no solo
clasificar quién es pobre hoy, sino cómo se mueve.

### Eje del análisis

- **Objeto central**: la distancia de cada hogar a la línea de pobreza/indigencia
  (`ratio_lp`/`ratio_li`, CBA/CBT por adulto equivalente) y su variación entre `t` y `t+h`
  (`h=1` y `h=4`, los horizontes que permite la rotación 2-2-2 del panel) — para **todos**
  los hogares, no solo los pobres. `pobre`/`indigente` (ratio < 1) son derivados, no el eje.
- **Predictores candidatos**, no eje: informalidad (intensificación del share de activos
  informales/desocupados del hogar) y déficit habitacional (ancla NBI: hacinamiento,
  materiales, saneamiento).
- **Subanálisis**, no objetivo principal: salida de la pobreza entre los pobres, `t`→`t+4`.

### Qué es contexto, no eje

El mandato presidencial es una variable de contexto: se usa para describir el período y se
evaluará como feature candidata (con y sin ella), no organiza el análisis. Con 6 mandatos en
la serie, cualquier comparación entre ellos es descriptiva.

## Fuente de datos

EPH continua (INDEC), microdatos trimestrales de las bases Hogar e Individual, 2003T3–2025T4.

Cortes conocidos de la serie:
- El INDEC no publicó 2007T3 ni 2015T3–2016T1.
- Período intervenido INDEC 2007–2015: series de precios/ingresos de ese tramo bajo reserva
  metodológica.
- 2020T2: operativo telefónico (pandemia), no comparable sin salvedad con el resto de la serie.
- 2006T3: INDEC incorpora 3 aglomerados nuevos (29→32) — quiebre de cobertura geográfica.
- El identificador de vivienda (`CODUSU`) cambia de formato entre la base histórica
  (hasta 2015) y la regular (desde 2016) — ningún par del panel cruza ese corte.
- El tratamiento de ingresos no declarados cambia en 2016 (ver `.claude/decisiones.md`).

### Línea de pobreza

- **2016T2–2025T4**: línea oficial regional (CBA/CBT por adulto equivalente, 6 regiones).
- **2003T3–2015T2**: retropolación con el procedimiento de dos anclas de Tornarolli (2018,
  CEDLAS DT 226) — diciembre 2006 (Metodología INDEC N°22) y abril 2016 (canasta oficial),
  índice encadenado IPC-GBA INDEC + IPC Provincias CIFRA-CTA. Aprobada por los directores;
  validada contra CEDLAS (Anexo B) y UCA-EDSA. Detalle completo en `.claude/decisiones.md`.
- **Medición B** (`src/posicion_relativa.py`, 2003T3–2025T4): posición relativa de
  `ITF/unidades_ae` sin CBA/CBT, para entrenar/diagnosticar en toda la serie — sus niveles
  **no** son comparables a tasas oficiales de pobreza/indigencia.

## Cómo obtener los microdatos

Los microdatos no se versionan en este repositorio: se descargan del INDEC.

- 2016 en adelante: `https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/`
- 2003T3–2015T2: solo disponibles vía Wayback Machine (el INDEC no los mantiene en su sitio
  actual).

`src/eph_client.py` (`EphClient`) descarga y cachea ambos casos en `data/eph_cache/`, con
alcance nacional (no filtra por aglomerado). `data/meta/manifiesto_insumos.csv` registra URL,
fecha de descarga y hash SHA-256 de cada archivo usado, para verificar que se trabaja con los
mismos insumos.

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

Las notebooks se corren en orden: cada una lee la capa que escribió la anterior (ver
`scripts/regenerar.sh` para el orden exacto).

| Notebook | Lee | Escribe |
|---|---|---|
| `01_ingesta_armonizacion` | `data/eph_cache/` | `data/01_armonizado/`, `inventario_variables.csv`, `flags_trimestre.csv` |
| `01b_capa_tipada_nucleo` | `data/01_armonizado/` | `data/02_nucleo/`, `formato_decimal_monetario.csv`, `p21_series_trimestral.csv` |
| `02_panel_hogares` | `data/02_nucleo/` | `data/03_panel/pares_h1/`, `data/03_panel/pares_h4/`, `panel_resumen_pares.csv` |
| `03_exploracion_descriptiva` | `data/02_nucleo/`, `data/03_panel/` | CSV descriptivos (quintiles, NBI) — insumo para decidir la etiqueta, no la etiqueta en sí |
| `04_trayectorias` | `data/03_panel/` | `data/04_trayectorias/`, `trayectorias_*.csv` — diagnóstico de factibilidad del panel dinámico |
| `05_distancia_linea_pobreza` | `data/02_nucleo/`, `data/01_armonizado/` | `data/02b_distancia_lp/hogar/` (era `"regional_2016"`, 2016T2–2025T4), `data/canastas_indec/*_crudo.csv` |
| `05b_canastas_retropoladas` | `data/insumos_lineas/`, `data/canastas_indec/` | canasta retropolada Tornarolli, 2003T1–2016T1 (en memoria/`data/meta/canasta_retropolada_*.csv`, no versionado) |
| `05c_distancia_historica` | `data/02_nucleo/`, `data/01_armonizado/` | `data/02b_distancia_lp/hogar/` (era `"historica"`, 2003T3–2015T2, mismas particiones que `05`) |
| `06_trayectorias_distancia` | `data/02b_distancia_lp/hogar/`, `data/03_panel/` | `data/05_trayectoria_distancia/{pares_h1,pares_h4}/` (solo era `"regional_2016"`) |
| `06b_diagnosticos_trayectoria` | `data/05_trayectoria_distancia/` | CSV descriptivos (tendencia, regresión a la media, conteos train/holdout) |
| `07_posicion_relativa_sin_lp` | `data/02_nucleo/` | `data/02c_posicion_relativa/hogar/` (Medición B, 2003T3–2025T4) |
| `08_trayectoria_relativa` | `data/02c_posicion_relativa/`, `data/03_panel/` | `data/06_trayectoria_relativa/{pares_h1,pares_h4}/` |
| `08b_diagnosticos_relativa` | `data/05_trayectoria_distancia/`, `data/06_trayectoria_relativa/` | CSV comparativos Medición A vs. B |

`notebooks/00_explorar.ipynb` es de solo lectura: arma vistas DuckDB sobre las capas ya
escritas para explorar sin volver a correr el pipeline.

## Qué se versiona

- Los Parquet de las capas numeradas (`data/0N_*/`, `data/0N[a-z]_*/`) y la mayoría de los
  CSV descriptivos de `data/meta/` **no** se versionan — se regeneran corriendo
  `scripts/regenerar.sh`. Sí se versionan: los logs/trazabilidad del pipeline
  (`corridas.csv`, `manifiesto_insumos.csv`, `resumen_0N_*.csv`, `flags_trimestre.csv`,
  `inventario_variables.csv`, `formato_decimal_monetario.csv`, `p21_series_trimestral.csv`,
  `linea_pobreza_trimestral.csv`) y los insumos externos chicos que no se pueden reconstruir
  corriendo código del repo (`data/canastas_indec/`, `data/insumos_lineas/`,
  `data/gini_indec_publicado.csv`).
- **Parámetros:** cada notebook de la capa 01/01b acepta `TRIMESTRES` (subconjunto opcional)
  y `REHACER` (por defecto `False`: reprocesa solo lo que falta o está desactualizado). La
  primera corrida completa tarda unos 10 minutos, sobre todo por el parseo de las bases
  históricas.
- **Registro:** cada corrida queda en `data/meta/corridas.csv`, y el avance por trimestre en
  `data/meta/resumen_0N_*.csv`.
