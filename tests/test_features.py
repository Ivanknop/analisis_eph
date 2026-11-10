from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import (
    CATALOGO_FEATURES, InsumosTrimestre, armonizar_pares, catalogo_como_tabla, construir_features_trimestre,
    construir_matriz_X, decil_ponderado, nombres_prohibidos, reporte_na,
)
from trayectoria_distancia import mediana_ponderada

ANIO, TRIMESTRE = 2018, 2
HOGARES = [f"v{i}" for i in range(6)]
FUENTES_VIEJAS = ["V2_M", "V21_M", "V5_M", "V11_M", "V3_M", "V4_M", "V8_M", "V9_M", "V10_M", "V12_M", "V18_M", "V19_AM"]


def _individual(semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    filas = []
    for i, codusu in enumerate(HOGARES):
        filas.append({"CODUSU": codusu, "NRO_HOGAR": 1, "COMPONENTE": 1, "CH03": 1, "CH04": 1 + i % 2, "CH06": 40 + i,
                      "ESTADO": 1, "CAT_OCUP": [3, 3, 2, 1, 3, 3][i], "PP07H": [1, 2, 0, 0, 2, 1][i],
                      "P21": int(rng.integers(100, 900)), "T_VI": 0, "NIVEL_ED": 4, "CH10": 2, "CAT_INAC": 0,
                      "PP04D_COD": 10333})
        filas.append({"CODUSU": codusu, "NRO_HOGAR": 1, "COMPONENTE": 2, "CH03": 3, "CH04": 2, "CH06": 10 + i,
                      "ESTADO": 3, "CAT_OCUP": 0, "PP07H": 0, "P21": 0, "T_VI": 0, "NIVEL_ED": 1,
                      "CH10": [1, 2, 1, 1, 1, 1][i], "CAT_INAC": 3, "PP04D_COD": 0})
    df = pd.DataFrame(filas)
    df["AGLOMERADO"] = 2
    df["PONDERA"] = 100
    for c in ["TOT_P12", "PP08J1", "PP08J2", "PP08J3", *FUENTES_VIEJAS]:
        df[c] = 0
    return df


def _hogar(individual: pd.DataFrame) -> pd.DataFrame:
    itf = individual.groupby("CODUSU")["P21"].sum().reindex(HOGARES).to_numpy()
    hogar = pd.DataFrame({"CODUSU": HOGARES, "NRO_HOGAR": 1, "AGLOMERADO": 2, "ITF": itf, "IPCF": itf / 2,
                          "PONDIH": 50, "DECCFR": [1, 3, 5, 7, 12, 10]})
    for c, v in {"II1": 2, "II2": 1, "II5": 2, "II5_1": 0, "II7": 1, "II8": 1, "IV1": 1, "IV3": 1, "IV4": 1, "IV6": 1,
                 "IV8": 1, "IV10": 1, "IV11": 1, "IV12_1": 2, "IV12_2": 2, "IV12_3": 2, "V2": 2, "V13": 2, "V14": 2,
                 "V15": 2, "V16": 2, "V17": 2}.items():
        hogar[c] = v
    return hogar


def _pares_05(origen, destino, h, codusus, identidad=True, valido_c=None, ratio_t=None, ratio_th=None):
    n = len(codusus)
    ratio_t = np.linspace(0.5, 2.0, n) if ratio_t is None else np.asarray(ratio_t, dtype=float)
    ratio_th = ratio_t * 1.1 if ratio_th is None else np.asarray(ratio_th, dtype=float)
    pares = pd.DataFrame({
        "CODUSU": codusus, "NRO_HOGAR_t": 1, "NRO_HOGAR_th": 1, "ANO4_t": origen[0], "TRIMESTRE_t": origen[1],
        "ANO4_th": destino[0], "TRIMESTRE_th": destino[1], "AGLOMERADO_t": 2, "AGLOMERADO_th": 2, "REGION_t": 1,
        "era_t": "regional_2016", "era_th": "regional_2016", "PONDIH_t": 50, "PONDIH_th": 50,
        "identidad_confirmada": pd.array([identidad] * n, dtype="boolean"),
        "ratio_linea_pobreza_t": ratio_t, "ratio_linea_pobreza_th": ratio_th,
        "log_ratio_linea_pobreza_t": np.log(ratio_t), "log_ratio_linea_pobreza_th": np.log(ratio_th),
        "unidades_adulto_equivalente_t": 2.0, "unidades_adulto_equivalente_th": 2.5,
        "ITF_t": 1000.0, "ITF_th": 1100.0, "gap_previo": pd.array([pd.NA] * n, dtype="Int8"),
        "variacion_simetrica_previa": pd.array([pd.NA] * n, dtype="Float64"),
    })
    pares["variacion_simetrica"] = (ratio_th - ratio_t) / ((ratio_th + ratio_t) / 2)
    pares["variacion_log"] = np.log(ratio_th) - np.log(ratio_t)
    for c in ("par_valido", "par_valido_simetrica", "par_valido_log"):
        pares[c] = pares["identidad_confirmada"]
    if valido_c is not None:
        pares["par_valido_c"] = pd.array(valido_c, dtype="boolean")
        pares["par_valido_log_c"] = pd.array(valido_c, dtype="boolean")
    return armonizar_pares(pares, h, [])


def _panel(origen, destino, h, codusus, n_altas=0):
    return pd.DataFrame({"CODUSU": codusus, "NRO_HOGAR_t": 1, "NRO_HOGAR_th": 1, "AGLOMERADO": 2, "ANO4_t": origen[0],
                         "TRIMESTRE_t": origen[1], "ANO4_th": destino[0], "TRIMESTRE_th": destino[1], "h": h,
                         "n_altas": n_altas, "n_bajas": 0,
                         "identidad_confirmada": pd.array([True] * len(codusus), dtype="boolean")})


@pytest.fixture
def insumos() -> InsumosTrimestre:
    individual = _individual(0)
    hogar = _hogar(individual)
    distancia = pd.DataFrame({"CODUSU": HOGARES, "NRO_HOGAR": 1, "AGLOMERADO": 2, "PONDIH": 50,
                              "ratio_linea_pobreza": np.linspace(0.5, 2.0, 6), "ratio_linea_indigencia": np.linspace(1, 4, 6),
                              "pobre": pd.array([True, True, False, False, False, False], dtype="boolean"),
                              "indigente": pd.array([True] + [False] * 5, dtype="boolean")})
    pares = _pares_05((ANIO, TRIMESTRE), (ANIO, 3), 1, HOGARES)
    # v0, v1: 2ª visita (par h=1 desde t-1, gap 0); v2: 3ª visita (h=1 desde t-4 y h=4 desde t-4, gap 3).
    pares.loc[pares["CODUSU"].isin(["v0", "v1"]), "gap_previo"] = 0
    pares.loc[pares["CODUSU"] == "v2", "gap_previo"] = 3
    # v9: 4ª visita en t (par h=1 desde t-1 y h=4 desde t-4), no origina pares en t.
    previo_h1_t1 = _pares_05((ANIO, 1), (ANIO, TRIMESTRE), 1, ["v0", "v1", "v9"], valido_c=[True, False, True],
                             ratio_t=[0.8, 0.9, 3.0])
    previo_h1_t4 = _pares_05((2017, 2), (2017, 3), 1, ["v2"], ratio_t=[1.0], ratio_th=[1.3])
    previo_h4_t4 = _pares_05((2017, 2), (ANIO, TRIMESTRE), 4, ["v2", "v9"], ratio_t=[1.0, 2.0])
    pares.loc[pares["CODUSU"] == "v0", "variacion_simetrica_previa"] = previo_h1_t1["variacion_simetrica"].iloc[0]
    pares.loc[pares["CODUSU"] == "v1", "variacion_simetrica_previa"] = previo_h1_t1["variacion_simetrica"].iloc[1]
    pares.loc[pares["CODUSU"] == "v2", "variacion_simetrica_previa"] = previo_h1_t4["variacion_simetrica"].iloc[0]
    contexto = pd.DataFrame({"AGLOMERADO": [2], "tasa_pobreza_aglo": [0.3], "tasa_indigencia_aglo": [0.05],
                             "tasa_informalidad_aglo": [0.4], "tasa_desocupacion_aglo": [0.08],
                             "gini_ipcf_aglo": [0.42], "n_hogares_ventana_aglo": [900]})
    return InsumosTrimestre(
        anio=ANIO, trimestre=TRIMESTRE, h=1, pares=pares, distancia=distancia, hogar=hogar, individual=individual,
        pares_previos=pd.concat([previo_h1_t1, previo_h1_t4, previo_h4_t4], ignore_index=True),
        panel_previos=pd.concat([_panel((ANIO, 1), (ANIO, TRIMESTRE), 1, ["v0", "v1", "v9"], n_altas=1),
                                 _panel((2017, 2), (ANIO, TRIMESTRE), 4, ["v2", "v9"])], ignore_index=True),
        individual_previo={(ANIO, 1): _individual(1), (2017, 2): _individual(2)},
        contexto_aglomerado=contexto,
    )


def _columnas_catalogo(features):
    return [f.nombre for f in CATALOGO_FEATURES if f.nombre in features]


def test_catalogo_consistente():
    nombres = [f.nombre for f in CATALOGO_FEATURES]
    assert len(nombres) == len(set(nombres))
    for f in CATALOGO_FEATURES:
        assert f.rezago_max <= 0 and f.rezago_min <= f.rezago_max, f.nombre
    tabla = catalogo_como_tabla()
    assert tabla.loc[tabla["nombre"] == "vivienda_precaria_v1_t", "en_x"].item()
    assert not tabla.loc[tabla["nombre"] == "vivienda_precaria_v2_t", "en_x"].item()
    en_x = dict(zip(tabla["nombre"], tabla["en_x"]))
    assert en_x["share_jubilacion_continua_t"] and en_x["motivo_previa_na_t"]
    assert not (en_x["motivo_previa_na_detalle_t"] or en_x["n_pares_terminan_en_t"])
    assert not (en_x["share_jubilacion_t"] or en_x["share_residuo_no_laboral_t"] or en_x["cuestionario_2023t4_t"])


def test_lista_blanca_sin_patrones_prohibidos():
    assert nombres_prohibidos([f.nombre for f in CATALOGO_FEATURES if f.en_x]) == []


@pytest.mark.parametrize("nombre", [
    "ratio_linea_pobreza_th", "variacion_simetrica", "variacion_log", "variacion_comun_trimestre_simetrica",
    "variacion_propia_aglomerado_log", "par_valido_c", "identidad_confirmada", "cambio_composicion", "n_altas",
    "aporte_ingreso", "mandato_presidencial_t", "regimen_ingreso", "par_toca_2020t2", "peso_hogar_era_t",
    "imputado_th", "direccion", "transicion", "diferencia_absoluta", "cambio_jefatura", "cuestionario_2023t4_t",
])
def test_patrones_atrapan_columnas_de_t_mas_h(nombre):
    assert nombres_prohibidos([nombre]) == [nombre]


def test_patrones_admiten_rezagadas():
    assert nombres_prohibidos(["variacion_comun_trimestre_simetrica_rezagada_t", "variacion_simetrica_previa_c",
                               "variacion_log_hasta_t_c"]) == []


def test_construir_matriz_x_rechaza_prohibidas():
    catalogo_malo = [*CATALOGO_FEATURES[:2], type(CATALOGO_FEATURES[0])("ratio_linea_pobreza_th", "x", "x", "x")]
    with pytest.raises(ValueError):
        construir_matriz_X(pd.DataFrame({"ratio_linea_pobreza_th": [1.0]}), catalogo_malo)


def test_decil_ponderado():
    valores = pd.Series([5.0, 1.0, 3.0, np.nan, 2.0])
    pesos = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0])
    decil = decil_ponderado(valores, pesos)
    assert decil.tolist()[:3] == [9, 2, 6] and decil.isna().tolist()[3:] == [True, True]


def test_features_una_fila_por_par_y_todo_el_catalogo(insumos):
    features = construir_features_trimestre(insumos)
    assert len(features) == len(insumos.pares)
    assert features["CODUSU"].tolist() == insumos.pares["CODUSU"].tolist()
    faltan = [f.nombre for f in CATALOGO_FEATURES if f.nombre not in features]
    assert faltan == []


def test_previa_c_y_ratio_previo(insumos):
    f = construir_features_trimestre(insumos).set_index("CODUSU")
    assert f.loc["v0", "variacion_simetrica_previa_c"] == pytest.approx(f.loc["v0", "variacion_simetrica_previa"])
    assert np.isnan(f.loc["v1", "variacion_simetrica_previa_c"]) and not np.isnan(f.loc["v1", "variacion_simetrica_previa"])
    assert f.loc["v0", "ratio_linea_pobreza_previo"] == pytest.approx(0.8)
    assert f.loc["v2", "ratio_linea_pobreza_previo"] == pytest.approx(1.3)
    assert np.isnan(f.loc["v3", "variacion_simetrica_previa_c"]) and np.isnan(f.loc["v3", "ratio_linea_pobreza_previo"])


def test_hasta_t_y_eventos_rezagados(insumos):
    f = construir_features_trimestre(insumos).set_index("CODUSU")
    assert f.loc["v2", "variacion_simetrica_hasta_t_c"] == pytest.approx(f.loc["v2", "variacion_simetrica_hasta_t"])
    assert np.isnan(f.loc["v1", "variacion_simetrica_hasta_t_c"])
    assert f.loc["v0", "cambio_ae_previo_t"] == pytest.approx(0.25)
    assert f.loc["v0", "tasa_altas_bajas_previa_t"] == pytest.approx(0.5)
    assert f.loc["v2", "tasa_altas_bajas_previa_t"] == 0.0
    assert f.loc[["v0", "v1", "v2"], "tiene_par_hasta_t"].all() and not f.loc["v3", "tiene_par_hasta_t"]
    assert np.isnan(f.loc["v3", "n_se_jubila_previo_t"])


def test_componente_comun_rezagado_usa_el_mismo_horizonte(insumos):
    f = construir_features_trimestre(insumos)
    validos = insumos.pares_previos[(insumos.pares_previos["h"] == 1) & (insumos.pares_previos["ANO4_th"] == ANIO)
                                    & (insumos.pares_previos["TRIMESTRE_th"] == TRIMESTRE)]
    validos = validos[validos["par_valido_c"] == True]  # noqa: E712
    esperado = mediana_ponderada(validos["variacion_simetrica"], validos["peso_hogar_era_t"])
    assert np.allclose(f["variacion_comun_trimestre_simetrica_rezagada_t"], esperado)


def test_contexto_y_calendario(insumos):
    f = construir_features_trimestre(insumos)
    assert (f["gini_ipcf_aglo_t"] == 0.42).all() and not f["cuestionario_2023t4_t"].any()
    assert not f["encuesta_telefonica_t"].any() and (f["era"] == "regional_2016").all()


def _perturbar_futuro(ins: InsumosTrimestre, semilla: int) -> InsumosTrimestre:
    rng = np.random.default_rng(semilla)
    pares = ins.pares.copy()
    columnas_futuro = [c for c in pares.columns if c.endswith("_th") and c not in ("NRO_HOGAR_th", "ANO4_th", "TRIMESTRE_th")]
    columnas_futuro += ["variacion_simetrica", "variacion_log"]
    for c in columnas_futuro:
        if pd.api.types.is_numeric_dtype(pares[c]) and not pd.api.types.is_bool_dtype(pares[c]):
            pares[c] = rng.uniform(0.01, 50, len(pares))
    for c in ("identidad_confirmada", "par_valido", "par_valido_simetrica", "par_valido_log", "par_valido_c",
              "par_valido_log_c"):
        pares[c] = pd.array(rng.integers(0, 2, len(pares)).astype(bool), dtype="boolean")
    # Un par que termina en t+1 (después del origen) con los mismos hogares no
    # puede cambiar nada: solo cuentan los pares que terminan en t.
    futuro = _pares_05((ANIO, TRIMESTRE), (ANIO, 3), 1, HOGARES, ratio_t=rng.uniform(0.1, 9, 6))
    futuro_panel = _panel((ANIO, TRIMESTRE), (ANIO, 3), 1, HOGARES, n_altas=7)
    return InsumosTrimestre(
        anio=ins.anio, trimestre=ins.trimestre, h=ins.h, pares=pares, distancia=ins.distancia, hogar=ins.hogar,
        individual=ins.individual, pares_previos=pd.concat([ins.pares_previos, futuro], ignore_index=True),
        panel_previos=pd.concat([ins.panel_previos, futuro_panel], ignore_index=True),
        individual_previo={**ins.individual_previo, (ANIO, 3): _individual(99)},
        contexto_aglomerado=ins.contexto_aglomerado)


def test_perturbar_t_mas_h_no_cambia_ninguna_feature(insumos):
    original = construir_features_trimestre(insumos)
    for semilla in range(3):
        perturbado = construir_features_trimestre(_perturbar_futuro(insumos, semilla))
        columnas = _columnas_catalogo(original)
        pd.testing.assert_frame_equal(original[columnas], perturbado[columnas], check_dtype=False)


def test_reporte_na_separa_estructural(insumos):
    features = construir_features_trimestre(insumos)
    reporte = reporte_na(features).set_index("feature")
    fila = reporte.loc["variacion_simetrica_hasta_t_c"]
    assert fila["pct_na_estructural_todos"] == pytest.approx(100 * 3 / 6)
    assert fila["pct_na_dato_todos"] == pytest.approx(100 * 1 / 3)
    assert reporte.loc["era", "pct_na_todos"] == 0


def test_esquema_real_de_06_features_sin_prohibidas():
    raiz = Path(__file__).resolve().parents[1] / "data" / "06_features"
    archivos = sorted(raiz.glob("pares_h*/ANO4=*/TRIMESTRE=*/data.parquet"))
    if not archivos:
        pytest.skip("data/06_features no existe todavía")
    import pyarrow.parquet as pq
    columnas = pq.read_schema(archivos[0]).names
    x = construir_matriz_X(pd.DataFrame(columns=columnas))
    assert nombres_prohibidos(x.columns) == []
    assert set(f.nombre for f in CATALOGO_FEATURES) <= set(columnas)


def test_jubilacion_continua_suma_jubilacion_y_residuo(insumos):
    f = construir_features_trimestre(insumos)
    esperado = f["share_jubilacion_t"] + f["share_residuo_no_laboral_t"]
    assert np.allclose(f["share_jubilacion_continua_t"], esperado, equal_nan=True)
    assert "cuestionario_2023t4_t" in f and "cuestionario_2023t4_t" not in construir_matriz_X(f)


def test_motivo_previa_na(insumos):
    previos = insumos.pares_previos.copy()
    previos["imputado_t"], previos["imputado_th"] = False, False
    previos.loc[(previos["CODUSU"] == "v1") & (previos["h"] == 1), "imputado_th"] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        extra = _pares_05((ANIO, 1), (ANIO, TRIMESTRE), 1, ["v3", "v4", "v5"], valido_c=[False] * 3,
                          ratio_t=[0.0, 1.0, np.nan], ratio_th=[0.0, 1.2, 1.0])
    extra.loc[extra["CODUSU"] == "v4", "identidad_confirmada"] = False
    extra["imputado_t"], extra["imputado_th"] = False, False
    pares = insumos.pares.copy()
    # Como en 05: `gap_previo` queda en NA si la variación del par previo es NA (v3, v5).
    pares.loc[pares["CODUSU"] == "v4", "gap_previo"] = 0
    insumos.pares = pares
    insumos.pares_previos = pd.concat([previos, extra], ignore_index=True)

    f = construir_features_trimestre(insumos).set_index("CODUSU")
    assert f["motivo_previa_na_t"].to_dict() == {
        "v0": "disponible", "v1": "ingreso_no_declarado", "v2": "disponible", "v3": "itf_cero_ambas_puntas",
        "v4": "sin_identidad", "v5": "ingreso_no_declarado"}
    assert f.loc["v1", "motivo_previa_na_detalle_t"] == "imputada_indec"
    assert f.loc["v5", "motivo_previa_na_detalle_t"] == "no_respuesta_ingreso"
    assert (f.loc[["v0", "v2", "v3", "v4"], "motivo_previa_na_detalle_t"]
            == f.loc[["v0", "v2", "v3", "v4"], "motivo_previa_na_t"]).all()
    assert ((f["motivo_previa_na_t"] == "disponible") == f["variacion_simetrica_previa_c"].notna()).all()
    # El par previo se encuentra aunque `gap_previo` de 05 esté en NA.
    assert pd.isna(f.loc["v3", "gap_previo"]) and f.loc["v3", "ratio_linea_pobreza_previo"] == 0.0
    assert pd.isna(f.loc["v4", "ratio_linea_pobreza_previo"])


def test_motivo_previa_primera_visita(insumos):
    f = construir_features_trimestre(insumos).set_index("CODUSU")
    assert (f.loc[["v3", "v4", "v5"], "motivo_previa_na_t"] == "primera_visita").all()


def test_reporte_na_contexto_sin_pares_que_terminan_en_t_es_estructural(insumos):
    insumos.pares_previos = insumos.pares_previos[insumos.pares_previos["h"] != 1]
    insumos.pares = insumos.pares.assign(gap_previo=pd.array([pd.NA] * len(insumos.pares), dtype="Int8"))
    insumos.contexto_aglomerado = None
    features = construir_features_trimestre(insumos)
    assert (features["n_pares_terminan_en_t"] == 0).all()
    reporte = reporte_na(features).set_index("feature")
    for feature in ("variacion_comun_trimestre_simetrica_rezagada_t", "gini_ipcf_aglo_t", "tasa_pobreza_aglo_t"):
        assert reporte.loc[feature, "pct_na_estructural_todos"] == 100.0
        assert np.isnan(reporte.loc[feature, "pct_na_dato_todos"])


def test_reporte_na_contexto_con_pares_que_terminan_en_t_es_dato(insumos):
    insumos.contexto_aglomerado = None
    reporte = reporte_na(construir_features_trimestre(insumos)).set_index("feature")
    assert reporte.loc["gini_ipcf_aglo_t", "pct_na_dato_todos"] == 100.0
    assert reporte.loc["gini_ipcf_aglo_t", "pct_na_estructural_todos"] == 0.0
