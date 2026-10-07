#!/usr/bin/env bash
# Regenera las capas Parquet y los CSV descriptivos de data/ corriendo el
# pipeline de notebooks en orden de dependencia (ver README.md "Qué se
# versiona"). No modifica ningún notebook: ejecuta con nbconvert --inplace,
# los .ipynb versionados quedan con sus salidas actualizadas.
#
# Requiere antes de correr:
#   - data/eph_cache/ con los microdatos ya descargados (EphClient, aparte
#     de este script) -- 5 trimestres .rar (2014T1/T3/T4, 2015T1/T2) además
#     necesitan `unar` instalado para extraerse.
#   - El resto de los insumos versionados de data/ (canastas_indec/, los CSV
#     de data/meta/ listados como "D" en el README).
#
# Orden topológico (no es el orden numérico de los notebooks): 04 corre
# antes que 03/05/06/07 porque 06 y 07 leen
# data/meta/trayectorias_completitud_cohorte.csv, que escribe 04; 01 tiene
# que haber corrido antes que 06 porque 06 lee
# data/meta/inventario_variables.csv, que escribe 01. 00_explorar.ipynb no
# entra: es de solo lectura.
#
# Tiempo estimado con todo ya cacheado/descargado: ~1 hora (86 trimestres
# completos). Ver duraciones reales de corridas anteriores en
# data/meta/corridas.csv.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

NOTEBOOKS=(
    01_ingesta_armonizacion
    01b_capa_tipada_nucleo
    02_panel_hogares
    04_trayectorias
    03_exploracion_descriptiva
    05_composicion_quintiles
    06_poblacion_riesgo
    07_cascada_targets
    08_distancia_linea_pobreza
    08b_retropolacion_pobreza
    09_trayectorias_distancia
    09b_diagnosticos_trayectoria
)

for nb in "${NOTEBOOKS[@]}"; do
    ruta="notebooks/${nb}.ipynb"
    echo "=== ${nb} ==="
    inicio=$(date +%s)
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "$ruta"
    fin=$(date +%s)
    echo "--- ${nb}: $((fin - inicio))s ---"
done

echo "Pipeline completo. Capas Parquet y CSV de data/meta/ regenerados."
