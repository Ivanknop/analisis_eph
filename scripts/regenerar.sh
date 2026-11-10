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
# Orden numérico de los notebooks, salvo 06c junto a 06 (escriben la misma
# capa, una era cada uno, antes de los diagnósticos 06b/06d). 07_features va al
# final: lee 05 completa (las dos eras). 00_explorar.ipynb no entra: es de solo
# lectura.
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
    03_exploracion_descriptiva
    04_trayectorias
    05_distancia_linea_pobreza
    05b_canastas_retropoladas
    05c_distancia_historica
    05d_no_respuesta_historica
    06_trayectorias_distancia
    06c_trayectorias_distancia_historica
    06b_diagnosticos_trayectoria
    06d_diagnosticos_dos_universos
    07_features
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
