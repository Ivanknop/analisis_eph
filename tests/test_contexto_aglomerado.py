import numpy as np
import pandas as pd
import pytest

from contexto_aglomerado import (
    agregados_aglomerado_trimestre, componente_comun_rezagado, gini_anio_movil, periodo, tabla_hogares_gini,
    tasas_anio_movil, ventana_anio_movil,
)
from desigualdad import gini_ponderado


def test_periodo_correlativo_entre_anios():
    assert periodo(2010, 1) - periodo(2009, 4) == 1


def test_ventana_salta_trimestres_no_publicados():
    disponibles = [(2015, 1), (2015, 2), (2016, 2), (2016, 3)]
    assert ventana_anio_movil((2016, 2), disponibles) == [(2016, 2)]
    assert ventana_anio_movil((2016, 3), disponibles) == [(2016, 2), (2016, 3)]
    assert ventana_anio_movil((2015, 2), disponibles + [(2014, 2), (2014, 3)]) == [(2014, 3), (2015, 1), (2015, 2)]


def _hogar_individual(ipcf, aglomerado, pondih=None):
    n = len(ipcf)
    hogar = pd.DataFrame({"CODUSU": [f"v{i}" for i in range(n)], "NRO_HOGAR": 1, "AGLOMERADO": aglomerado, "IPCF": ipcf})
    if pondih is not None:
        hogar["PONDIH"] = pondih
    individual = pd.DataFrame({"CODUSU": np.repeat(hogar["CODUSU"], 2), "NRO_HOGAR": 1, "PONDERA": 10})
    return hogar, individual


def test_tabla_gini_peso_por_era():
    hogar, individual = _hogar_individual([100.0, 300.0], [1, 1], pondih=[5, 0])
    regular = tabla_hogares_gini(hogar, individual, historico=False)
    assert regular["peso_personas"].tolist() == [10.0, 0.0]
    historica = tabla_hogares_gini(hogar.drop(columns="PONDIH"), individual, historico=True)
    assert historica["peso_personas"].tolist() == [20.0, 20.0]
    assert historica["ipcf_relativo"].tolist() == pytest.approx([0.5, 1.5])


def test_gini_anio_movil_junta_ventana_y_aplica_piso():
    rng = np.random.default_rng(0)
    tablas, eras = {}, {}
    for q in [(2018, 1), (2018, 2)]:
        hogar, individual = _hogar_individual(rng.uniform(1, 100, 300), [2] * 250 + [3] * 50)
        tablas[q] = tabla_hogares_gini(hogar, individual, historico=True)
        eras[q] = "historica"
    resultado = gini_anio_movil(tablas, eras, (2018, 2), min_hogares=400).set_index("AGLOMERADO")
    juntos = pd.concat(tablas.values())
    juntos = juntos[juntos["AGLOMERADO"] == 2]
    assert resultado.loc[2, "n_hogares_gini_ventana"] == 500
    assert resultado.loc[2, "gini_ipcf_aglo"] == pytest.approx(gini_ponderado(juntos["ipcf_relativo"], juntos["peso_personas"]))
    assert np.isnan(resultado.loc[3, "gini_ipcf_aglo"]) and "menos de 400" in resultado.loc[3, "motivo_na_gini"]


def test_gini_anio_movil_na_si_cruza_era():
    hogar, individual = _hogar_individual(np.arange(1, 501, dtype=float), [2] * 500)
    tablas = {(2015, 2): tabla_hogares_gini(hogar, individual, True), (2016, 2): tabla_hogares_gini(hogar, individual, True)}
    eras = {(2015, 2): "historica", (2016, 2): "regional_2016"}
    # 2015T2 no entra en la ventana de 2016T2 (t-3 = 2015T3): no cruza.
    assert gini_anio_movil(tablas, eras, (2016, 2), min_hogares=1)["gini_ipcf_aglo"].notna().all()
    eras_falsas = {(2015, 2): "historica", (2015, 3): "regional_2016"}
    tablas_falsas = {(2015, 2): tablas[(2015, 2)], (2015, 3): tablas[(2016, 2)]}
    assert gini_anio_movil(tablas_falsas, eras_falsas, (2015, 3), min_hogares=1)["gini_ipcf_aglo"].isna().all()


def _distancia(aglomerado, pobre, indigente, peso, ratio_na=None):
    n = len(pobre)
    return pd.DataFrame({
        "AGLOMERADO": aglomerado, "pobre": pd.array(pobre, dtype="boolean"),
        "indigente": pd.array(indigente, dtype="boolean"), "peso_hogar": peso,
        "ratio_linea_pobreza": [np.nan if ratio_na and ratio_na[i] else 1.0 for i in range(n)],
    })


def _personas(aglomerado, estado, cat_ocup, pp07h, pondera):
    return pd.DataFrame({"AGLOMERADO": aglomerado, "ESTADO": estado, "CAT_OCUP": cat_ocup, "PP07H": pp07h,
                         "PONDERA": pondera})


def test_agregados_y_tasas_ponderadas():
    distancia = _distancia([1, 1, 1], [True, False, True], [True, False, False], [10, 30, 10],
                           ratio_na=[False, False, True])
    personas = _personas([1, 1, 1, 1], [1, 1, 2, 3], [3, 3, 0, 0], [1, 2, 0, 0], [10, 30, 20, 99])
    agregados = {(2018, 1): agregados_aglomerado_trimestre(distancia, personas)}
    fila = tasas_anio_movil(agregados, {(2018, 1): "historica"}, (2018, 1), min_hogares=1).iloc[0]
    assert fila["tasa_pobreza_aglo"] == pytest.approx(10 / 40)
    assert fila["tasa_indigencia_aglo"] == pytest.approx(10 / 40)
    assert fila["tasa_informalidad_aglo"] == pytest.approx(30 / 40)
    assert fila["tasa_desocupacion_aglo"] == pytest.approx(20 / 60)
    assert fila["n_hogares_ventana_aglo"] == 2


def test_tasas_piso_de_hogares():
    distancia = _distancia([1, 1], [True, False], [False, False], [1, 1])
    agregados = {(2018, 1): agregados_aglomerado_trimestre(distancia, _personas([1], [1], [3], [1], [1]))}
    fila = tasas_anio_movil(agregados, {(2018, 1): "historica"}, (2018, 1), min_hogares=3).iloc[0]
    assert np.isnan(fila["tasa_pobreza_aglo"]) and fila["n_hogares_ventana_aglo"] == 2


def test_componente_comun_rezagado_usa_solo_validos_c():
    pares = pd.DataFrame({
        "AGLOMERADO_th": [1, 1, 1, 2], "variacion_simetrica": [0.1, 0.2, 9.0, -0.5],
        "variacion_log": [0.1, 0.3, 9.0, np.nan], "par_valido_c": [True, True, False, True],
        "par_valido_log_c": [True, True, False, False], "peso_hogar_era_t": [1.0, 3.0, 100.0, 1.0],
    })
    escalares, tabla = componente_comun_rezagado(pares)
    assert escalares["variacion_comun_trimestre_simetrica_rezagada"] == pytest.approx(0.2)
    tabla = tabla.set_index("AGLOMERADO")
    assert tabla.loc[1, "variacion_comun_aglomerado_log_rezagada"] == pytest.approx(0.3)
    assert np.isnan(tabla.loc[2, "variacion_comun_aglomerado_log_rezagada"])
