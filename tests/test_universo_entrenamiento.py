"""Tests de `universo_entrenamiento` y del completado de `armonizar_pares`.
Los de datos reales comparan `data/06_features/` contra 05 + regla C; se saltan
si las capas no están."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import constantes as cte
import cortes_trayectoria as ct
import universo_entrenamiento as ue
from features import CLAVE_PAR, armonizar_pares

REPO = Path(__file__).resolve().parents[1]
TRAYECTORIA = REPO / "data" / "05_trayectoria_distancia"
FEATURES = REPO / "data" / "06_features"

REQUIERE_CAPAS = pytest.mark.skipif(
    not (ue.particiones(TRAYECTORIA, 1) and ue.particiones(FEATURES, 1)),
    reason="requiere data/05_trayectoria_distancia/ y data/06_features/ reales")

# Regla C sobre la 05: la era regular no trae `par_valido_c`, ahí vale `par_valido`.
REGLA_C_05 = "CASE WHEN era_t = 'regional_2016' THEN {base} ELSE {c} END"


def _glob(raiz: Path, h: int) -> str:
    return str(raiz / f"pares_h{h}" / "ANO4=*" / "TRIMESTRE=*" / "data.parquet")


def _consulta(sql: str) -> pd.DataFrame:
    return duckdb.connect().execute(sql).df()


def _pares_sinteticos() -> pd.DataFrame:
    """Un par histórico completo y uno regular con la validez y los cortes en NULL,
    como quedan al leer la 05 entera con `union_by_name`."""
    pares = pd.DataFrame({
        "CODUSU": ["H1", "R1"], "NRO_HOGAR_t": 1, "NRO_HOGAR_th": 1,
        "ANO4_t": [2010, 2020], "TRIMESTRE_t": [1, 1], "ANO4_th": [2010, 2020], "TRIMESTRE_th": [2, 2],
        "era_t": ["historica", "regional_2016"], "era": ["historica", None],
        "PONDIH_t": pd.array([pd.NA, 80], dtype="Float64"), "peso_hogar_t": pd.array([40, pd.NA], dtype="Float64"),
        "gap_previo": pd.array([pd.NA, pd.NA], dtype="Int8"),
        "par_valido": pd.array([True, True], dtype="boolean"),
        "par_valido_log": pd.array([True, False], dtype="boolean"),
        "par_valido_c": pd.array([False, pd.NA], dtype="boolean"),
        "par_valido_log_c": pd.array([False, pd.NA], dtype="boolean"),
    })
    for flag in ct.FLAGS_CORTE:
        pares[flag] = pd.array([False, pd.NA], dtype="boolean")
    return pares


def test_armonizar_pares_completa_columnas_nulas_de_la_regular():
    resultado = armonizar_pares(_pares_sinteticos(), 1, [])
    regular = resultado.iloc[1]
    assert regular["era"] == "regional_2016"
    assert regular["par_valido_c"] == True and regular["par_valido_log_c"] == False  # noqa: E712
    assert regular["par_toca_2020t2"] == True  # noqa: E712
    for flag in ct.FLAGS_CORTE:
        assert not pd.isna(regular[flag])


def test_armonizar_pares_no_toca_la_historica_ni_valores_regulares_presentes():
    pares = _pares_sinteticos()
    pares.loc[1, "par_valido_c"] = False
    resultado = armonizar_pares(pares, 1, [])
    assert resultado.loc[0, "par_valido_c"] == False  # noqa: E712
    assert resultado.loc[1, "par_valido_c"] == False  # noqa: E712
    assert resultado.loc[1, "par_valido_log_c"] == False  # noqa: E712


def test_normalizar_validez_pasa_na_a_false_y_conserva_identidad():
    pares = pd.DataFrame({"identidad_confirmada": pd.array([pd.NA, True], dtype="boolean")})
    for columna in ue.COLUMNAS_VALIDEZ:
        pares[columna] = pd.array([pd.NA, True], dtype="boolean")
    resultado = ue.normalizar_validez(pares)
    for columna in ue.COLUMNAS_VALIDEZ:
        assert resultado[columna].dtype == bool and resultado[columna].tolist() == [False, True]
    assert resultado["identidad_confirmada"].isna().sum() == 1


def test_columna_validez_cubre_objetivos_y_criterios():
    assert set(cte.COLUMNA_VALIDEZ) == {(o, c) for o in cte.OBJETIVOS for c in cte.CRITERIOS_IMPUTACION}


def test_en_universo_y_sin_cortes():
    pares = pd.DataFrame({"ANO4_t": [2005, 2016, 2024], "par_periodo_intervenido": [True, False, False],
                          **{f: [False, False, f == "par_toca_2020t2"] for f in ct.FLAGS_PUNTUALES}})
    assert ue.en_universo(pares, "universo_2003_2025").tolist() == [True, True, True]
    assert ue.en_universo(pares, "universo_2016_2025").tolist() == [False, True, True]
    assert ue.sin_cortes(pares, "con_marcados").tolist() == [True, True, True]
    assert ue.sin_cortes(pares, "sin_cortes_puntuales").tolist() == [True, True, False]
    assert ue.sin_cortes(pares, "sin_cortes_ni_intervenido").tolist() == [False, True, False]
    with pytest.raises(ValueError):
        ue.en_universo(pares, "a_2003_2025")


def test_conteos_validos_sintetico():
    pares = pd.DataFrame({"ANO4_t": [2010, 2022, 2023], "TRIMESTRE_t": [1, 4, 2], "ANO4_th": [2010, 2023, 2023],
                          "TRIMESTRE_th": [2, 1, 3], "h": 1, "era": ["historica", "regional_2016", "regional_2016"],
                          "par_valido_simetrica": [True, True, True], "par_valido_c": [False, True, True],
                          "par_valido_log": [True, True, False], "par_valido_log_c": [False, True, False]})
    conteos = ue.conteos_validos(pares).set_index(["universo", "era", "particion", "objetivo", "criterio_imputacion"])
    n = conteos["n_pares_validos"]
    assert n[("universo_2003_2025", "historica", "train", "simetrica", "imputacion_conserva")] == 1
    assert ("universo_2003_2025", "historica", "train", "simetrica", "imputacion_excluye") not in n.index
    assert n[("universo_2016_2025", "regional_2016", "embargo", "simetrica", "imputacion_excluye")] == 1
    assert n[("universo_2016_2025", "regional_2016", "holdout", "simetrica", "imputacion_excluye")] == 1
    assert ("universo_2016_2025", "historica", "train", "simetrica", "imputacion_conserva") not in n.index


@REQUIERE_CAPAS
@pytest.mark.parametrize("h", [1, 4])
def test_06_features_sin_validez_c_nula_en_pares_validos(h):
    nulos = _consulta(f"""
        SELECT count(*) FILTER (WHERE par_valido AND par_valido_c IS NULL) AS c,
               count(*) FILTER (WHERE par_valido_log AND par_valido_log_c IS NULL) AS log_c,
               count(*) FILTER (WHERE era = 'regional_2016' AND par_valido_c IS DISTINCT FROM par_valido) AS reg_c,
               count(*) FILTER (WHERE era = 'regional_2016' AND par_valido_log_c IS DISTINCT FROM par_valido_log) AS reg_log
        FROM read_parquet('{_glob(FEATURES, h)}', union_by_name=true)""")
    assert nulos.iloc[0].tolist() == [0, 0, 0, 0]


@REQUIERE_CAPAS
@pytest.mark.parametrize("h", [1, 4])
def test_conteos_por_era_05_con_regla_c_igual_a_06_features(h):
    regla_c = REGLA_C_05.format(base="par_valido", c="par_valido_c")
    regla_log_c = REGLA_C_05.format(base="par_valido_log", c="par_valido_log_c")
    desde_05 = _consulta(f"""
        SELECT era_t AS era, count(*) AS n, count(*) FILTER (WHERE {regla_c}) AS n_c,
               count(*) FILTER (WHERE {regla_log_c}) AS n_log_c
        FROM read_parquet('{_glob(TRAYECTORIA, h)}', union_by_name=true) GROUP BY 1 ORDER BY 1""")
    desde_06 = _consulta(f"""
        SELECT era, count(*) AS n, count(*) FILTER (WHERE par_valido_c) AS n_c,
               count(*) FILTER (WHERE par_valido_log_c) AS n_log_c
        FROM read_parquet('{_glob(FEATURES, h)}', union_by_name=true) GROUP BY 1 ORDER BY 1""")
    pd.testing.assert_frame_equal(desde_05, desde_06)


@REQUIERE_CAPAS
@pytest.mark.parametrize("h", [1, 4])
def test_par_a_par_05_con_regla_c_igual_a_06_features(h):
    union = " AND ".join(f"a.{c} IS NOT DISTINCT FROM b.{c}" for c in CLAVE_PAR)
    # Las marcas de corte de la era regular no están en la 05: se comparan solo las históricas.
    flags = ", ".join(f"count(*) FILTER (WHERE a.era_t = 'historica' AND a.{f} IS DISTINCT FROM b.{f})"
                      for f in ct.FLAGS_CORTE)
    diferencias = _consulta(f"""
        SELECT count(*) FILTER (WHERE a.CODUSU IS NULL OR b.CODUSU IS NULL),
               count(*) FILTER (WHERE b.par_valido_c IS DISTINCT FROM
                                {REGLA_C_05.format(base="a.par_valido", c="a.par_valido_c")}),
               count(*) FILTER (WHERE b.par_valido_log_c IS DISTINCT FROM
                                {REGLA_C_05.format(base="a.par_valido_log", c="a.par_valido_log_c")}),
               count(*) FILTER (WHERE b.era IS DISTINCT FROM a.era_t), {flags}
        FROM read_parquet('{_glob(TRAYECTORIA, h)}', union_by_name=true) a
        FULL JOIN read_parquet('{_glob(FEATURES, h)}', union_by_name=true) b ON {union}""")
    assert diferencias.iloc[0].tolist() == [0] * diferencias.shape[1]


@REQUIERE_CAPAS
@pytest.mark.parametrize("h", [1, 4])
def test_particion_cubre_todos_los_validos_y_embargo_es_regular(h):
    universo = ue.leer_universo(FEATURES, h)
    conteos = ue.conteos_validos(universo)
    for (nombre, objetivo, criterio), g in conteos.groupby(["universo", "objetivo", "criterio_imputacion"]):
        columna = cte.COLUMNA_VALIDEZ[(objetivo, criterio)]
        assert g["n_pares_validos"].sum() == int(universo.loc[ue.en_universo(universo, nombre), columna].sum())
    fuera_de_train = conteos[conteos["particion"] != "train"]
    assert (fuera_de_train["era"] == "regional_2016").all()
    excluye = conteos[conteos["criterio_imputacion"] == "imputacion_excluye"].set_index(
        ["universo", "era", "particion", "objetivo"])["n_pares_validos"]
    conserva = conteos[conteos["criterio_imputacion"] == "imputacion_conserva"].set_index(
        ["universo", "era", "particion", "objetivo"])["n_pares_validos"]
    regulares = excluye.index.get_level_values("era") == "regional_2016"
    assert (excluye[regulares] == conserva.loc[excluye.index[regulares]]).all()


@REQUIERE_CAPAS
def test_unir_objetivo_trae_la_variacion_de_la_05():
    universo = ue.leer_universo(FEATURES, 1)
    unido = ue.unir_objetivo(universo, TRAYECTORIA, 1)
    assert len(unido) == len(universo)
    validos = unido[unido["par_valido_c"]]
    assert validos["variacion_simetrica"].notna().all()
    assert np.isfinite(validos.loc[validos["par_valido_log_c"], "variacion_log"].astype(float)).all()
