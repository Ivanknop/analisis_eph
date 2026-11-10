import numpy as np
import pandas as pd
import pytest

from cortes_trayectoria import (
    marcar_cortes_par, pct_itf_cero_declarado, peso_hogar_era, peso_normalizado_trimestre, trimestres_itf0_elevado,
)


def _pares(origen_destino, gap=None):
    filas = [{"ANO4_t": a, "TRIMESTRE_t": t, "ANO4_th": b, "TRIMESTRE_th": u} for (a, t), (b, u) in origen_destino]
    df = pd.DataFrame(filas)
    df["gap_previo"] = pd.array(gap if gap is not None else [pd.NA] * len(df), dtype="Int8")
    return df


def test_periodo_intervenido_bordes():
    marcado = marcar_cortes_par(_pares([((2006, 1), (2006, 2)), ((2006, 4), (2007, 4)), ((2015, 4), (2016, 1)),
                                        ((2016, 2), (2016, 3))]), [])
    assert marcado["par_periodo_intervenido"].tolist() == [False, True, True, False]


def test_ruptura_codusu_par_y_previa():
    marcado = marcar_cortes_par(_pares([((2013, 1), (2014, 1)), ((2013, 4), (2014, 4)), ((2014, 2), (2014, 3)),
                                        ((2014, 2), (2014, 3))], gap=[pd.NA, pd.NA, 3, 0]), [])
    assert marcado["par_cruza_ruptura_codusu"].tolist() == [True, False, False, False]
    assert marcado["previa_cruza_ruptura_codusu"].tolist() == [False, False, True, False]


def test_hueco_2007t3():
    marcado = marcar_cortes_par(_pares([((2007, 1), (2008, 1)), ((2007, 4), (2008, 1)), ((2008, 3), (2008, 4)),
                                        ((2007, 1), (2007, 2))]), [])
    assert marcado["par_atraviesa_2007t3"].tolist() == [True, False, False, False]
    assert marcado["previa_no_observable_2007t3"].tolist() == [False, True, True, False]


def test_itf0_elevado_y_2020t2():
    pares = _pares([((2004, 1), (2004, 2)), ((2005, 1), (2005, 2)), ((2020, 1), (2020, 2)), ((2020, 2), (2021, 2))])
    marcado = marcar_cortes_par(pares, trimestres_itf0_elevado({(2004, 2): 4.5, (2005, 1): 0.9, (2003, 3): np.nan}))
    assert marcado["par_itf0_elevado"].tolist() == [True, False, False, False]
    assert marcado["par_toca_2020t2"].tolist() == [False, False, True, True]
    assert len(marcado) == len(pares)


def test_trimestres_itf0_umbral_estricto():
    assert trimestres_itf0_elevado({(2004, 1): 2.1, (2004, 3): 2.0, (2003, 3): 2.85}) == [(2003, 3), (2004, 1)]


def test_pct_itf_cero_excluye_solo_no_respuesta_confirmada():
    regular = pd.DataFrame({"ITF": [0.0, 0.0, 500.0, 800.0],
                            "ingreso_no_declarado": pd.array([True, False, False, False], dtype="boolean")})
    assert pct_itf_cero_declarado(regular) == pytest.approx(100 / 3)


def test_pct_itf_cero_era_historica_con_flag_na():
    historica = pd.DataFrame({"ITF": [0.0, 300.0, 400.0, 500.0],
                              "ingreso_no_declarado": pd.array([pd.NA] * 4, dtype="boolean")})
    assert pct_itf_cero_declarado(historica) == pytest.approx(25.0)


def test_peso_hogar_era():
    pares = pd.DataFrame({"era_t": ["regional_2016", "historica"],
                          "PONDIH_t": pd.array([150, pd.NA], dtype="Int32"),
                          "peso_hogar_t": pd.array([pd.NA, 80], dtype="Int32")})
    assert peso_hogar_era(pares, "t").tolist() == [150, 80]
    pares.loc[1, "PONDIH_t"] = 5
    with pytest.raises(ValueError):
        peso_hogar_era(pares, "t")


def test_peso_hogar_era_sin_columna_historica():
    pares = pd.DataFrame({"era_t": ["regional_2016"], "PONDIH_t": pd.array([150], dtype="Int32")})
    assert peso_hogar_era(pares, "t").tolist() == [150]


def test_peso_normalizado_suma_uno_por_trimestre():
    pares = pd.DataFrame({"ANO4_t": [2005, 2005, 2018], "TRIMESTRE_t": [1, 1, 1], "w": [1.0, 3.0, 500.0]})
    normalizado = peso_normalizado_trimestre(pares, "w")
    assert normalizado.tolist() == pytest.approx([0.25, 0.75, 1.0])
