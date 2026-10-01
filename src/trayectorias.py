"""Diagnóstico de trayectorias de hogares en 4 apariciones superpuestas
(t, t+1, t+4, t+5) sobre el panel vinculado del notebook 02 -- no construye
ninguna etiqueta de deterioro (ver notebooks/04_trayectorias.ipynb y D12 en
docs/decisiones_metodologicas.md).
"""
# Todas las funciones son puras (reciben DataFrames/dicts, no tocan `data/`)
# para poder testearlas con fixtures sintéticas chicas
# (`tests/test_trayectorias.py`), mismo criterio que `panel.py`.
from __future__ import annotations

import pandas as pd

from panel import sumar_trimestres

# La rotación 2-2-2 es propiedad de la VIVIENDA (INDEC vuelve a visitarla
# según su cronograma, viva ahí quien viva); la identidad de HOGAR (D9) es
# otra cosa -- si es la misma familia. `inferir_visita` usa la primera,
# `encadenar_trayectoria` y el resto usan la segunda.
CLAVE_VIVIENDA = ["CODUSU", "AGLOMERADO"]
CLAVE_HOGAR = ["CODUSU", "NRO_HOGAR", "AGLOMERADO"]

# Offsets relativos a `t` evaluados por `inferir_visita` (D12): los 4
# pares de visitas posibles están separados por 1, 4 o 5 trimestres entre
# sí, salvo el par (visita 2, visita 3), separado por 3 -- de ahí el ±3,
# necesario para que las 4 hipótesis prediccionen sobre el mismo conjunto
# de offsets (si se lo excluye, las hipótesis 2 y 3 quedan con menos
# chequeos que la 1 y la 4, y el puntaje deja de ser comparable entre sí).
OFFSETS_VISITA = (-5, -4, -3, -1, 1, 3, 4, 5)

# Desplazamiento de `t0` (entrada a la cohorte) respecto de `t`, por
# hipótesis de visita.
_T0_RELATIVO_A_T = {1: 0, 2: -1, 3: -4, 4: -5}
# Offsets del calendario de 4 visitas (t0, t0+1, t0+4, t0+5) respecto de t0.
_CALENDARIO_RELATIVO_A_T0 = (0, 1, 4, 5)


def _calendario_relativo_a_t(hipotesis: int) -> set[int]:
    t0_relativo = _T0_RELATIVO_A_T[hipotesis]
    return {t0_relativo + offset for offset in _CALENDARIO_RELATIVO_A_T0}


def claves_vivienda(hogar: pd.DataFrame) -> pd.DataFrame:
    """`CODUSU+AGLOMERADO` únicos de una partición de `02_nucleo/hogar` (unidad de rotación)."""
    return hogar[CLAVE_VIVIENDA].drop_duplicates().reset_index(drop=True)


def claves_hogar(hogar: pd.DataFrame) -> pd.DataFrame:
    """`CODUSU+NRO_HOGAR+AGLOMERADO` únicos de una partición de `02_nucleo/hogar` (unidad de identidad, D9)."""
    return hogar[CLAVE_HOGAR].drop_duplicates().reset_index(drop=True)


def inferir_visita(presencia_vivienda: dict[int, pd.DataFrame | None]) -> pd.DataFrame:
    """Infiere la visita (1-4, D12) de cada vivienda presente en el offset `0` de
    `presencia_vivienda`, por selección de hipótesis sobre `OFFSETS_VISITA`. `None` en un
    offset significa "no determinable" (gap de publicación o cruce de era, D7) -- lo decide
    quien llama, no esta función. Devuelve CODUSU, AGLOMERADO, visita_inferida (Int8, NA si hay
    empate), n_matches, n_mismatches, n_determinables (Int8) y ambigua (boolean)."""
    universo = presencia_vivienda[0][CLAVE_VIVIENDA].drop_duplicates().reset_index(drop=True)

    presente_por_offset: dict[int, pd.Series | None] = {}
    for offset in OFFSETS_VISITA:
        tabla = presencia_vivienda.get(offset)
        if tabla is None:
            presente_por_offset[offset] = None
            continue
        claves = tabla[CLAVE_VIVIENDA].drop_duplicates().assign(_presente=True)
        marca = universo.merge(claves, on=CLAVE_VIVIENDA, how="left")
        presente_por_offset[offset] = marca["_presente"].fillna(False).astype(bool)

    n_determinables = sum(serie is not None for serie in presente_por_offset.values())

    matches, mismatches = {}, {}
    for hipotesis in (1, 2, 3, 4):
        calendario = _calendario_relativo_a_t(hipotesis)
        m = pd.Series(0, index=universo.index)
        mm = pd.Series(0, index=universo.index)
        for offset in OFFSETS_VISITA:
            presente = presente_por_offset[offset]
            if presente is None:
                continue
            coincide = presente if offset in calendario else ~presente
            m += coincide
            mm += ~coincide
        matches[hipotesis] = m
        mismatches[hipotesis] = mm

    matches_df = pd.DataFrame(matches)
    mismatches_df = pd.DataFrame(mismatches)
    scores_df = matches_df - mismatches_df

    mejor_score = scores_df.max(axis=1)
    ambigua = scores_df.eq(mejor_score, axis=0).sum(axis=1) > 1
    ganadora = scores_df.idxmax(axis=1)

    # n_matches/n_mismatches de la hipótesis ganadora -- en empate todas las
    # hipótesis con el mejor score comparten estos valores por construcción
    # (mismo score y mismo n_determinables => misma resta), así que tomar la
    # de `idxmax` (la de menor número entre las empatadas) no pierde información.
    posiciones = ganadora.map({h: i for i, h in enumerate(matches_df.columns)}).to_numpy()
    filas = range(len(universo))
    n_matches = pd.array(matches_df.to_numpy()[filas, posiciones], dtype="Int8")
    n_mismatches = pd.array(mismatches_df.to_numpy()[filas, posiciones], dtype="Int8")

    resultado = universo.copy()
    resultado["visita_inferida"] = ganadora.where(~ambigua).astype("Int8")
    resultado["n_matches"] = n_matches
    resultado["n_mismatches"] = n_mismatches
    resultado["n_determinables"] = n_determinables
    resultado["ambigua"] = ambigua
    return resultado


def clasificar_patron_apariciones(claves_hogar_t0: pd.DataFrame, claves_hogar_t0_1: pd.DataFrame | None,
                                   claves_hogar_t0_4: pd.DataFrame | None,
                                   claves_hogar_t0_5: pd.DataFrame | None) -> pd.DataFrame:
    """Patrón de presencia cruda (p. ej. `"1111"`, `"1100"`) de cada hogar de `claves_hogar_t0`
    en `t0, t0+1, t0+4, t0+5`, por clave exacta `CODUSU+NRO_HOGAR+AGLOMERADO`. Un trimestre `None`
    se marca `"?"` en esa posición (no determinable, no se confunde con ausencia)."""
    resultado = claves_hogar_t0[CLAVE_HOGAR].drop_duplicates().reset_index(drop=True)
    resultado["patron"] = "1"
    for tabla in (claves_hogar_t0_1, claves_hogar_t0_4, claves_hogar_t0_5):
        if tabla is None:
            resultado["patron"] += "?"
            continue
        claves = tabla[CLAVE_HOGAR].drop_duplicates().assign(_presente=True)
        presente = resultado.merge(claves, on=CLAVE_HOGAR, how="left")["_presente"].fillna(False)
        resultado["patron"] += presente.map({True: "1", False: "0"})
    return resultado


def encadenar_trayectoria(pares_h1_t: pd.DataFrame, pares_h4_t: pd.DataFrame,
                           pares_h1_t4: pd.DataFrame) -> pd.DataFrame:
    """Encadena los 3 eslabones t->t+1, t->t+4 y t+4->t+5 por `CODUSU+NRO_HOGAR+AGLOMERADO`
    (clave invariante por construcción dentro de cada eslabón, D9/`panel.vincular_hogares`):
    intersección exacta de los 3, sin exigir `identidad_confirmada` en ninguno."""
    def _eslabon(pares: pd.DataFrame, columna: str) -> pd.DataFrame:
        return pares.rename(columns={"NRO_HOGAR_t": "NRO_HOGAR", "identidad_confirmada": columna})[
            ["CODUSU", "NRO_HOGAR", "AGLOMERADO", columna]
        ]

    resultado = _eslabon(pares_h1_t, "identidad_confirmada_1").merge(
        _eslabon(pares_h4_t, "identidad_confirmada_h4"), on=CLAVE_HOGAR, how="inner")
    resultado = resultado.merge(_eslabon(pares_h1_t4, "identidad_confirmada_2"), on=CLAVE_HOGAR, how="inner")
    resultado["identidad_confirmada_trayectoria"] = (
        resultado["identidad_confirmada_1"] & resultado["identidad_confirmada_h4"] & resultado["identidad_confirmada_2"]
    )
    return resultado


def verificar_contra_h4_t1(trayectoria: pd.DataFrame, pares_h4_t1: pd.DataFrame) -> pd.DataFrame:
    """Agrega a `trayectoria` el eslabón redundante t+1->t+5 (`pares_h4` con origen `t+1`,
    mismo horizonte real que la trayectoria completa) como verificación cruzada -- no se usa
    para construir la trayectoria, solo para detectar inconsistencias."""
    verificacion = pares_h4_t1.rename(
        columns={"NRO_HOGAR_t": "NRO_HOGAR", "identidad_confirmada": "identidad_confirmada_verificacion"}
    )[["CODUSU", "NRO_HOGAR", "AGLOMERADO", "identidad_confirmada_verificacion"]].assign(presente_en_verificacion=True)
    resultado = trayectoria.merge(verificacion, on=CLAVE_HOGAR, how="left")
    resultado["presente_en_verificacion"] = resultado["presente_en_verificacion"].fillna(False)
    return resultado


def cohorte_completable(anio_t0: int, trimestre_t0: int, trimestres_publicados: set[tuple[int, int]],
                         es_historico) -> dict:
    """Evalúa si la cohorte que entra en `(anio_t0, trimestre_t0)` puede en teoría completar las
    4 visitas: existencia de `t0, t0+1, t0+4, t0+5` en `trimestres_publicados`, y que el eslabón
    `t0->t0+4` no cruce la frontera histórico/regular (D7). `es_historico` se recibe como
    parámetro (`armonizacion.es_historico`) para no acoplar este módulo a `armonizacion`."""
    trimestres = {nombre: sumar_trimestres(anio_t0, trimestre_t0, offset)
                  for nombre, offset in (("t0", 0), ("t0+1", 1), ("t0+4", 4), ("t0+5", 5))}
    faltantes = [nombre for nombre, t in trimestres.items() if t not in trimestres_publicados]
    if faltantes:
        return {"completable": False, "motivo": f"trimestre no publicado: {', '.join(faltantes)}"}
    if es_historico(*trimestres["t0"]) != es_historico(*trimestres["t0+4"]):
        return {"completable": False, "motivo": "eslabón h=4 cruza la frontera histórico/regular (D7)"}
    return {"completable": True, "motivo": None}


def encadenar_dos_eslabones(pares_1: pd.DataFrame, pares_2: pd.DataFrame) -> pd.DataFrame:
    """Intersección por `CODUSU+NRO_HOGAR+AGLOMERADO` de 2 eslabones consecutivos de pares
    (no 3, a diferencia de `encadenar_trayectoria`) -- insumo de los diseños de panel
    dinámico (a)/(b) de D13, que no requieren la trayectoria completa de 4 visitas."""
    def _eslabon(pares: pd.DataFrame, columna: str) -> pd.DataFrame:
        return pares.rename(columns={"NRO_HOGAR_t": "NRO_HOGAR", "identidad_confirmada": columna})[
            ["CODUSU", "NRO_HOGAR", "AGLOMERADO", columna]
        ]

    resultado = _eslabon(pares_1, "identidad_confirmada_1").merge(
        _eslabon(pares_2, "identidad_confirmada_2"), on=CLAVE_HOGAR, how="inner")
    resultado["identidad_confirmada_ambos"] = resultado["identidad_confirmada_1"] & resultado["identidad_confirmada_2"]
    return resultado


def contar_n_diseno(tabla: pd.DataFrame, columna_identidad: str, anio_t_col: str = "ANO4_t",
                     trimestre_t_col: str = "TRIMESTRE_t") -> pd.DataFrame:
    """Cuenta hogares `vinculada` (todas las filas de `tabla`) e `identidad_confirmada`
    (`columna_identidad`, True), por trimestre de origen -- mismo conteo sirve para
    cualquiera de los 3 diseños candidatos, pasándole la tabla y columna que corresponda."""
    base = tabla[[anio_t_col, trimestre_t_col]].copy()
    base["vinculada"] = 1
    base["identidad_confirmada"] = tabla[columna_identidad].fillna(False).astype(bool)
    return base.groupby([anio_t_col, trimestre_t_col])[["vinculada", "identidad_confirmada"]].sum().reset_index()


def clasificar_perdida_por_patron(patron: pd.Series) -> pd.Series:
    """Clasifica cada patrón de 4 caracteres (salida de `clasificar_patron_apariciones`)
    en un estado de selectividad: `"completa"` (`"1111"`), `"hueco_con_reaparicion"` (un
    `"1"` después de un `"0"`), `"se_pierde_en_t0_1"`/`"_t0_4"`/`"_t0_5"` (última posición
    con `"1"` en un patrón sin huecos antes), o `"no_determinable"` (patrón con `"?"`)."""
    etiquetas_corte = {0: "se_pierde_en_t0_1", 1: "se_pierde_en_t0_4", 2: "se_pierde_en_t0_5"}

    def _estado(p: str) -> str:
        if "?" in p:
            return "no_determinable"
        if p == "1111":
            return "completa"
        ultimo_uno = max(i for i, c in enumerate(p) if c == "1")
        if "0" in p[:ultimo_uno]:
            return "hueco_con_reaparicion"
        return etiquetas_corte[ultimo_uno]

    return patron.map(_estado)
