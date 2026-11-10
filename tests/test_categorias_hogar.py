import numpy as np
import pandas as pd
import pytest

from categorias_hogar import (
    calcular_estrategias, calificacion_ocupacion, composicion_hogar, fuentes_ingreso_hogar, fuentes_por_trimestre,
    grupo_ingreso_en_trimestre, hay_menor_no_asiste, laboral_hogar, precariedad_hogar, vivienda_hogar,
)


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "CH03", "CH04", "CH06", "ESTADO", "CAT_OCUP", "PP07H", "NIVEL_ED", "CH10",
                "PONDERA", "PP04D_COD"]
    df = pd.DataFrame(filas, columns=columnas[:len(filas[0])])
    for c in columnas:
        if c not in df:
            df[c] = 0
    return df


def _por_hogar(df, codusu):
    return df.set_index("CODUSU").loc[codusu]


def test_estrategias_combinan_preguntas():
    hogar = pd.DataFrame({"V13": [1, 2, 2], "V14": [2, 2, 2], "V15": [2, 1, 2], "V16": [2, 2, 2], "V17": [2, 2, 2]})
    resultado = calcular_estrategias(hogar)
    assert resultado["recurre_desahorro"].tolist() == [True, False, False]
    assert resultado["recurre_endeudamiento_formal"].tolist() == [False, True, False]


def test_grupo_ingreso_alineado_a_claves():
    hogar = pd.DataFrame({"CODUSU": ["a", "b"], "NRO_HOGAR": [1, 1], "DECCFR": [1, 12]})
    claves = pd.DataFrame({"CODUSU": ["b", "z", "a"], "NRO_HOGAR": [1, 1, 1]}, index=[10, 11, 12])
    grupo = grupo_ingreso_en_trimestre(hogar, claves)
    assert grupo.index.tolist() == [10, 11, 12]
    assert grupo.iloc[0] == "ingreso_no_declarado" and pd.isna(grupo.iloc[1]) and grupo.iloc[2] == "Q1"


def test_composicion_cuenta_menor_de_un_anio_y_dependientes():
    individual = _individual([
        ("a", 1, 1, 2, 40, 1, 3, 1, 4), ("a", 1, 3, 1, -1, 3, 0, 0, 1), ("a", 1, 3, 1, 70, 3, 0, 0, 2),
        ("b", 1, 1, 1, 30, 2, 0, 0, 6),
    ])
    resultado = composicion_hogar(individual)
    a = _por_hogar(resultado, "a")
    assert a["n_miembros"] == 3 and a["n_menores_14"] == 1 and a["n_mayores_65"] == 1
    assert a["share_dependientes"] == pytest.approx(2 / 3)
    assert a["edad_jefe"] == 40 and a["sexo_jefe"] == 2 and a["nivel_ed_jefe"] == 4
    assert a["nivel_ed_jefe_bucket"] == "secundario completo o más" and not a["jefe_ambiguo"]
    assert _por_hogar(resultado, "b")["condicion_actividad_jefe"] == "desocupado"


def test_composicion_jefe_ambiguo_anula_features_del_jefe():
    individual = _individual([("a", 1, 1, 1, 40, 1, 3, 1, 4), ("a", 1, 1, 2, 38, 1, 3, 1, 4),
                              ("b", 1, 3, 1, 20, 1, 3, 1, 4)])
    resultado = composicion_hogar(individual)
    for codusu in ("a", "b"):
        fila = _por_hogar(resultado, codusu)
        assert fila["jefe_ambiguo"]
        assert pd.isna(fila["edad_jefe"]) and pd.isna(fila["sexo_jefe"]) and pd.isna(fila["nivel_ed_jefe"])


def test_menor_no_asiste_false_sin_menores_y_na_sin_dato():
    individual = _individual([
        ("a", 1, 1, 1, 40, 1, 3, 1, 4, 2), ("a", 1, 3, 1, 12, 3, 0, 0, 1, 2),
        ("b", 1, 1, 1, 40, 1, 3, 1, 4, 2),
        ("c", 1, 1, 1, 40, 1, 3, 1, 4, 2), ("c", 1, 3, 1, 10, 3, 0, 0, 1, 9),
        ("d", 1, 1, 1, 40, 1, 3, 1, 4, 2), ("d", 1, 3, 1, 10, 3, 0, 0, 1, 1),
    ])
    resultado = hay_menor_no_asiste(individual)
    assert resultado.loc[("a", 1)] == True  # noqa: E712
    assert resultado.loc[("b", 1)] == False  # noqa: E712
    assert pd.isna(resultado.loc[("c", 1)])
    assert resultado.loc[("d", 1)] == False  # noqa: E712


def test_calificacion_completa_ceros_a_la_izquierda():
    codigos = pd.Series(["05001", 5002.0, "10334", 99999, None, 1.0, "abc"])
    resultado = calificacion_ocupacion(codigos)
    assert resultado.iloc[0] == 1 and resultado.iloc[1] == 2 and resultado.iloc[2] == 4
    assert resultado.iloc[3:].isna().tolist() == [True, True, False, True]
    assert resultado.iloc[5] == 1


def test_laboral_conteos_y_share():
    individual = _individual([("a", 1, 1, 1, 40, 1, 3, 1), ("a", 1, 3, 2, 38, 2, 0, 0), ("a", 1, 3, 1, 10, 3, 0, 0),
                              ("b", 1, 1, 1, 70, 3, 0, 0)])
    resultado = laboral_hogar(individual)
    a, b = _por_hogar(resultado, "a"), _por_hogar(resultado, "b")
    assert (a["n_ocupados"], a["n_desocupados"], a["n_activos"]) == (1, 1, 2)
    assert a["share_ocupados"] == pytest.approx(1 / 3) and a["situacion_laboral"] == "algun_asalariado_formal"
    assert b["sin_activos"] and b["situacion_laboral"] == "sin_ocupados"


def test_precariedad_cuatro_variantes():
    # formal, informal, cuenta propia profesional, cuenta propia no calificada, patrón, desocupado
    individual = pd.DataFrame({
        "CODUSU": ["a"] * 6, "NRO_HOGAR": [1] * 6,
        "ESTADO": [1, 1, 1, 1, 1, 2], "CAT_OCUP": [3, 3, 2, 2, 1, 0], "PP07H": [1, 2, 0, 0, 0, 0],
        "PP04D_COD": [10333, 10334, 10331, 10334, 5001, None],
    })
    fila = _por_hogar(precariedad_hogar(individual), "a")
    assert fila["share_precario_A"] == pytest.approx(2 / 6)
    assert fila["share_precario_B"] == pytest.approx(4 / 6)
    assert fila["share_precario_C"] == pytest.approx(5 / 6)
    assert fila["share_precario_D"] == pytest.approx(3 / 6)


def test_precariedad_sin_activos_es_na_y_cuenta_propia_sin_calificacion_sale_de_d():
    individual = pd.DataFrame({
        "CODUSU": ["a", "b", "b"], "NRO_HOGAR": [1, 1, 1], "ESTADO": [3, 1, 1], "CAT_OCUP": [0, 2, 3],
        "PP07H": [0, 0, 2], "PP04D_COD": [None, 99999, 10334],
    })
    resultado = precariedad_hogar(individual).set_index("CODUSU")
    assert resultado.loc["a", ["share_precario_A", "share_precario_D"]].isna().all()
    assert resultado.loc["b", "n_activos_clasificables_D"] == 1
    assert resultado.loc["b", "share_precario_D"] == 1.0 and resultado.loc["b", "share_precario_B"] == 1.0


def test_fuentes_por_trimestre_cambia_en_2023t4():
    assert "V2_M" in fuentes_por_trimestre(2023, 3)["jubilacion"]
    assert "V2_01_M" in fuentes_por_trimestre(2023, 4)["jubilacion"]
    assert fuentes_por_trimestre(2024, 1)["laboral"] == fuentes_por_trimestre(2010, 1)["laboral"]


def _individual_ingresos(**columnas):
    base = {"CODUSU": ["a", "a", "b"], "NRO_HOGAR": [1, 1, 1], "ESTADO": columnas.get("ESTADO", [1, 1, 1])}
    for c in ["P21", "TOT_P12", "PP08J1", "PP08J2", "PP08J3", "V2_M", "V21_M", "V5_M", "V11_M",
              "V3_M", "V4_M", "V8_M", "V9_M", "V10_M", "V12_M", "V18_M", "V19_AM", "T_VI"]:
        base[c] = columnas.get(c, [0, 0, 0])
    return pd.DataFrame(base)


def test_fuentes_shares_suman_uno_con_residuo():
    individual = _individual_ingresos(P21=[600, 0, 0], V2_M=[0, 200, 0], V5_M=[0, 100, 0], V3_M=[0, 50, 0],
                                      T_VI=[0, 400, 0])
    itf = pd.Series([1000.0, 0.0], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 1)], names=["CODUSU", "NRO_HOGAR"]))
    resultado = fuentes_ingreso_hogar(individual, fuentes_por_trimestre(2010, 1), itf).set_index("CODUSU")
    a = resultado.loc["a"]
    assert (a["share_laboral"], a["share_jubilacion"], a["share_transferencias"]) == pytest.approx((0.6, 0.2, 0.1))
    assert (a["share_otros_no_laborales"], a["share_residuo_no_laboral"]) == pytest.approx((0.05, 0.05))
    assert a[[c for c in resultado.columns if c.startswith("share_")]].sum() == pytest.approx(1.0)
    assert resultado.loc["b"].drop("NRO_HOGAR").isna().all()


def test_fuentes_codigo_negativo_anula_la_fuente():
    individual = _individual_ingresos(P21=[-9, 500, 0], T_VI=[0, 0, 0])
    itf = pd.Series([500.0], index=pd.MultiIndex.from_tuples([("a", 1)], names=["CODUSU", "NRO_HOGAR"]))
    resultado = fuentes_ingreso_hogar(individual, fuentes_por_trimestre(2010, 1), itf).set_index("CODUSU")
    assert pd.isna(resultado.loc["a", "share_laboral"])
    assert resultado.loc["a", "share_jubilacion"] == 0.0


def test_fuentes_desde_2023t4_suman_componentes():
    individual = pd.DataFrame({"CODUSU": ["a"], "NRO_HOGAR": [1], "ESTADO": [1], "P21": [0], "TOT_P12": [0], "PP08J1": [0],
                               "PP08J2": [0], "PP08J3": [0], "V2_01_M": [300], "V2_02_M": [100], "V22_03_M": [100],
                               "V5_02_M": [500], "T_VI": [1000]})
    itf = pd.Series([1000.0], index=pd.MultiIndex.from_tuples([("a", 1)], names=["CODUSU", "NRO_HOGAR"]))
    for c in ["V2_03_M", "V21_01_M", "V21_02_M", "V21_03_M", "V22_01_M", "V22_02_M", "V5_01_M", "V5_03_M",
              "V11_01_M", "V11_02_M", "V3_M", "V4_M", "V8_M", "V9_M", "V10_M", "V12_M", "V18_M", "V19_AM"]:
        individual[c] = 0
    fila = fuentes_ingreso_hogar(individual, fuentes_por_trimestre(2024, 2), itf).iloc[0]
    assert (fila["share_jubilacion"], fila["share_transferencias"], fila["share_residuo_no_laboral"]) == pytest.approx((0.5, 0.5, 0.0))


def test_vivienda_v1_y_v2():
    hogar = pd.DataFrame({
        "CODUSU": ["a", "b", "c"], "NRO_HOGAR": [1, 1, 1],
        "II1": [1, 3, 3], "II2": [1, 2, 2], "II5": [2, 2, 2], "II5_1": [0, 0, 0], "II7": [6, 3, 1], "II8": [3, 1, 1],
        "IV1": [1, 1, 1], "IV3": [1, 1, 1], "IV4": [1, 9, 6], "IV6": [1, 1, 1], "IV8": [1, 1, 1], "IV10": [1, 1, 1],
        "IV11": [0, 1, 1], "IV12_1": [2, 1, 2], "IV12_2": [2, 2, 2], "IV12_3": [2, 2, 2],
    }).set_index(["CODUSU", "NRO_HOGAR"])
    n = pd.Series([4, 2, 2], index=hogar.index)
    resultado = vivienda_hogar(hogar, n).set_index("CODUSU")
    assert resultado.loc["a", "hacinamiento_nbi"] and resultado.loc["a", "vivienda_precaria_v1"]
    assert resultado.loc["a", "tenencia_precaria"] and resultado.loc["a", "cocina_lena_kerosene"]
    assert not resultado.loc["b", "techo_precario"] and resultado.loc["b", "inquilino"]
    assert resultado.loc["b", "entorno_basural_o_inundable"] and not resultado.loc["b", "vivienda_precaria_v2"]
    assert not resultado.loc["c", "vivienda_precaria_v1"] and resultado.loc["c", "vivienda_precaria_v2"]


def _pp08j(estado, pp08j1):
    individual = _individual_ingresos(P21=[600, 0, 0], ESTADO=estado)
    individual["PP08J1"] = pp08j1
    itf = pd.Series([700.0, 0.0], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 1)], names=["CODUSU", "NRO_HOGAR"]))
    return fuentes_ingreso_hogar(individual, fuentes_por_trimestre(2018, 2), itf).set_index("CODUSU").loc["a"]


def test_fuentes_pp08j_vacio_en_no_ocupado_cuenta_como_cero():
    assert _pp08j([1, 3, 3], [100.0, np.nan, np.nan])["share_laboral"] == pytest.approx(1.0)


def test_fuentes_pp08j_vacio_en_ocupado_sigue_siendo_na():
    assert pd.isna(_pp08j([1, 1, 3], [100.0, np.nan, np.nan])["share_laboral"])
    assert pd.isna(_pp08j([1, 3, 3], [np.nan, np.nan, np.nan])["share_laboral"])


def test_fuentes_pp08j_codigo_negativo_en_no_ocupado_sigue_siendo_na():
    assert pd.isna(_pp08j([1, 3, 3], [100.0, -9.0, np.nan])["share_laboral"])
