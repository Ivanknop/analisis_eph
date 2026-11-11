import numpy as np
import pandas as pd

from no_respuesta_modelo import (
    a_categoricas, cobertura_medica, dimensiones_cno, grupos_vivienda, metricas_binarias, metricas_continuas,
    pesos_inversos, predecir_fuera_de_fold, rama_caes, resumen_pesos, variables_hogar,
)


def test_rama_caes_cambia_de_tabla_en_2012():
    codigos = pd.Series(["4500", "9500", "4000", "9700", "12", None])
    viejo = rama_caes(codigos, 2011, 4)
    nuevo = rama_caes(codigos, 2012, 1)
    assert viejo.iloc[0] == "construccion" and viejo.iloc[1] == "servicio_domestico"
    assert nuevo.iloc[2] == "construccion" and nuevo.iloc[3] == "servicio_domestico"
    assert pd.isna(viejo.iloc[4]) and pd.isna(viejo.iloc[5])


def test_dimensiones_cno_completa_ceros():
    dims = dimensiones_cno(pd.Series([1011.0, 30134.0, 99.0, "56203"]))
    assert dims.iloc[0].tolist() == [1, 0, 1, 1]
    assert dims.iloc[1].tolist() == [30, 1, 3, 4]
    assert dims.iloc[2].isna().all()
    assert dims.iloc[3].tolist() == [56, 2, 0, 3]


def test_cobertura_medica():
    resultado = cobertura_medica(pd.Series([1, 12, 2, 3, 4, 9, 23]))
    assert resultado.iloc[:5].tolist() == ["obra_social", "obra_social", "prepaga", "publica", "sin_cobertura"]
    assert pd.isna(resultado.iloc[5]) and resultado.iloc[6] == "prepaga"


def test_pesos_inversos_recorta():
    p = pd.Series([0.01, 0.5, 1.0] * 100)
    w = pesos_inversos(p, p_min=0.05, percentil_recorte=1.0)
    assert w.max() == 20.0 and w.min() == 1.0
    r = resumen_pesos(pd.Series([1.0, 1.0, 1.0]))
    assert r["n_efectivo_relativo"] == 1.0 and r["cv"] == 0.0


def test_grupos_vivienda_estable_y_por_vivienda():
    df = pd.DataFrame({"CODUSU": ["A", "A", "B", "C"], "AGLOMERADO": [1, 1, 1, 2]})
    g = grupos_vivienda(df, 3)
    assert g[0] == g[1] and (grupos_vivienda(df, 3) == g).all() and set(g) <= {0, 1, 2}


def test_metricas():
    m = metricas_binarias(pd.Series([0, 1, 0, 1]), pd.Series([0.1, 0.9, 0.2, 0.8]))
    assert m["auc"] == 1.0 and m["acierto_umbral_0_5"] == 1.0
    c = metricas_continuas(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 4.0]))
    assert np.isclose(c["mae"], 1 / 3) and np.isclose(c["sesgo"], 1 / 3)


def test_predecir_fuera_de_fold_no_usa_su_fold():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=2000), "c": pd.Categorical(rng.choice(["a", "b"], 2000))})
    y = pd.Series((X["x"] + rng.normal(scale=0.5, size=2000) > 0).astype(int))
    pred, imp = predecir_fuera_de_fold(X, y, np.arange(2000) % 4)
    assert pred.notna().all() and metricas_binarias(y, pred)["auc"] > 0.8
    assert imp.index[0] == "x" and np.isclose(imp["ganancia"].sum(), 1.0)


def _escribir(tmp_path, tipo, df):
    ruta = tmp_path / tipo / "ANO4=2019" / "TRIMESTRE=2"
    ruta.mkdir(parents=True)
    df.to_parquet(ruta / "data.parquet")


def test_variables_hogar_jefe_ambiguo_y_sin_ingresos(tmp_path):
    hogar = pd.DataFrame({"CODUSU": ["A", "B"], "NRO_HOGAR": [1, 1], "AGLOMERADO": [32, 32], "REGION": [1, 1],
                          "MAS_500": ["S", "S"], "IV1": [1, 2], "II1": [3, 2], "IX_TOT": [2, 2], "ITF": [100, 0]})
    individual = pd.DataFrame({"CODUSU": ["A", "A", "B", "B"], "NRO_HOGAR": [1] * 4, "CH03": [1, 2, 1, 1],
                               "CH06": [40, 38, 50, 20], "ESTADO": [1, 3, 1, 2], "CAT_OCUP": [3, 0, 2, 0],
                               "CH08": [1, 4, 4, 4], "PP04B_COD": [4000.0, np.nan, 4803.0, np.nan],
                               "PP04D_COD": [30134.0, np.nan, 1011.0, np.nan], "PP07H": [1, 0, 0, 0], "P21": [9, 0, 9, 0]})
    _escribir(tmp_path, "hogar", hogar)
    _escribir(tmp_path, "individual", individual)
    v = variables_hogar(2019, 2, tmp_path).set_index("CODUSU")
    assert v.loc["A", "CH06_jefe"] == 40 and pd.isna(v.loc["B", "CH06_jefe"])
    assert v.loc["A", "rama_jefe"] == "construccion" and v.loc["A", "cno_calificacion_jefe"] == 4
    assert v.loc["A", "n_asalariados_con_descuento"] == 1 and v.loc["A", "ambientes_por_persona"] == 1.5
    assert not {"ITF", "P21"} & set(v.columns)
    assert a_categoricas(v.reset_index())["rama_jefe"].dtype == "category"


def test_variables_hogar_armoniza_no_corresponde_y_menores(tmp_path):
    hogar = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [32], "REGION": [1], "MAS_500": ["S"],
                          "IX_TOT": [2], "IX_MEN10": [np.nan], "IX_MAYEQ10": [2]})
    individual = pd.DataFrame({"CODUSU": ["A", "A"], "NRO_HOGAR": [1, 1], "CH03": [1, 2], "ESTADO": [3, 1],
                               "PP04A": [0, 2], "PP07K": [0, 1], "PP3E_TOT": [0, 40], "PP04D_COD": ["0", "30134"],
                               "CH08": [1, 1]})
    _escribir(tmp_path, "hogar", hogar)
    _escribir(tmp_path, "individual", individual)
    v = variables_hogar(2019, 2, tmp_path).iloc[0]
    assert v["IX_MEN10"] == 0 and pd.isna(v["PP04A_jefe"]) and pd.isna(v["PP07K_jefe"]) and pd.isna(v["PP3E_TOT_jefe"])
