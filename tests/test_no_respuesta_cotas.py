import numpy as np
import pandas as pd

from no_respuesta_cotas import (
    clasificar_cota, cota_historica, cota_ingreso_hogar, ingreso_declarado_persona, pobre_segun_cota,
    recuperacion_pares,
)


def _personas(*filas):
    base = {"CODUSU": "A", "NRO_HOGAR": 1, "P21": 0, "TOT_P12": 0, "T_VI": 0, "P47T": 0,
            "PP08D1": 0, "PP08F1": 0, "PP08F2": 0, "PP06C": 0, "PP06D": 0,
            "PP08J1": np.nan, "PP08J2": np.nan, "PP08J3": np.nan, "V2_M": 0, "V5_M": 0}
    return pd.DataFrame([{**base, **f} for f in filas])


def test_cota_suma_lo_declarado_y_cuenta_menos9_como_cero():
    ind = _personas({"P21": 1000, "T_VI": -9, "V2_M": 300, "V5_M": -9, "P47T": -9},
                    {"P21": -9, "PP08D1": 500, "TOT_P12": 200, "P47T": -9})
    personas = ingreso_declarado_persona(ind)
    assert personas["cota_persona"].tolist() == [1300.0, 700.0]
    assert personas["algun_menos9"].all() and personas["p47t_menos9"].all()
    assert not personas["p47t_positivo_con_menos9"].any()


def test_cota_respaldo_cuenta_propia_y_aguinaldo():
    ind = _personas({"P21": -9, "PP06C": 800, "PP08J1": 100})
    assert ingreso_declarado_persona(ind)["cota_persona"].iloc[0] == 900.0


def test_cota_sin_menos9_reproduce_p47t():
    ind = _personas({"P21": 1000, "PP08J1": 50, "TOT_P12": 10, "T_VI": 40, "P47T": 1100})
    personas = ingreso_declarado_persona(ind)
    assert personas["cota_persona"].iloc[0] == 1100.0 and not personas["algun_menos9"].iloc[0]


def test_p47t_parcial_detectado():
    ind = _personas({"P21": -9, "T_VI": 100, "P47T": 100})
    assert ingreso_declarado_persona(ind)["p47t_positivo_con_menos9"].iloc[0]


def test_cota_hogar_agrega_por_hogar():
    ind = _personas({"P21": 100}, {"P21": -9, "T_VI": 50}, {"NRO_HOGAR": 2, "P21": 7})
    hogar = cota_ingreso_hogar(ind).set_index("NRO_HOGAR")
    assert hogar.loc[1, "cota_ingreso"] == 150.0 and hogar.loc[1, "n_personas_menos9"] == 1
    assert hogar.loc[2, "cota_ingreso"] == 7.0


def test_cota_historica_recorta_y_propaga_na():
    resultado = cota_historica(pd.Series([1000.0, 500.0, 300.0]), pd.Series([400.0, 600.0, np.nan]))
    assert resultado.iloc[0] == 600.0 and resultado.iloc[1] == 0.0 and pd.isna(resultado.iloc[2])


def test_clasificar_cota():
    clase = clasificar_cota(pd.Series([0.5, 1.0, 0.3, 0.0, np.nan]), pd.Series([False, True, True, True, True]))
    assert clase.tolist() == ["declarado", "seguro_no_pobre", "indeterminado", "sin_ingreso_declarado", "sin_cota"]


def test_pobre_segun_cota():
    pobre = pobre_segun_cota(pd.Series(["declarado", "seguro_no_pobre", "indeterminado"]),
                             pd.Series([True, pd.NA, pd.NA], dtype="boolean"))
    assert pobre.iloc[0] == True and pobre.iloc[1] == False and pd.isna(pobre.iloc[2])  # noqa: E712


def test_recuperacion_pares_por_objetivo_y_via():
    pares = pd.DataFrame({
        "identidad_confirmada": [True, True, True, True, False, True],
        "valido_actual": [False, False, False, True, False, False],
        "clase_t": ["seguro_no_pobre", "declarado", "declarado", "declarado", "seguro_no_pobre", "indeterminado"],
        "clase_th": ["declarado", "seguro_no_pobre", "indeterminado", "declarado", "declarado", "declarado"],
        "pobre_t": pd.array([pd.NA, True, False, False, pd.NA, pd.NA], dtype="boolean"),
        "pobre_th": pd.array([True, pd.NA, pd.NA, False, False, True], dtype="boolean"),
    })
    rec = recuperacion_pares(pares)
    assert rec["recupera_entrada"].tolist() == [True, False, False, False, False, False]
    assert rec["recupera_permanencia"].tolist() == [False, True, False, False, False, False]
    assert rec["via"].tolist() == ["t", "th", "", "", "", ""]
    assert rec["pobre_th_cota"].iloc[1] == False  # noqa: E712
