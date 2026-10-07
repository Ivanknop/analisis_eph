"""Tests de `src/trayectoria_distancia.py`. DataFrames sintéticos chicos --
mismo criterio que `tests/test_panel.py`."""
import numpy as np
import pandas as pd
import pytest

from trayectoria_distancia import (
    agregar_componente_comun_y_propio,
    agregar_tendencia_previa,
    calcular_variacion_simetrica,
    clasificar_direccion,
    clasificar_transicion,
    construir_pares_distancia,
    descomponer_variacion,
    mediana_ponderada,
)


class TestCalcularVariacionSimetrica:
    def test_variacion_simple(self):
        r = calcular_variacion_simetrica(pd.Series([1.0]), pd.Series([2.0]))
        assert r.iloc[0] == pytest.approx(2 / 3)

    def test_admite_un_lado_cero(self):
        r = calcular_variacion_simetrica(pd.Series([0.0]), pd.Series([1.0]))
        assert r.iloc[0] == pytest.approx(2.0)

    def test_ambos_cero_da_na(self):
        r = calcular_variacion_simetrica(pd.Series([0.0]), pd.Series([0.0]))
        assert pd.isna(r.iloc[0])


class TestDescomponerVariacion:
    def test_suma_exacta_a_variacion_log(self):
        itf_t, itf_th = pd.Series([1000.0]), pd.Series([1500.0])
        cbt_t, cbt_th = pd.Series([200.0]), pd.Series([220.0])
        unidades_t, unidades_th = pd.Series([2.0]), pd.Series([2.5])
        d = descomponer_variacion(itf_t, itf_th, cbt_t, cbt_th, unidades_t, unidades_th)

        variacion_log = (np.log(itf_th / (cbt_th * unidades_th)) - np.log(itf_t / (cbt_t * unidades_t))).iloc[0]
        suma = (d["aporte_ingreso"] + d["aporte_precios"] + d["aporte_composicion"]).iloc[0]
        assert suma == pytest.approx(variacion_log)

    def test_itf_cero_en_un_lado_da_na(self):
        d = descomponer_variacion(pd.Series([0.0]), pd.Series([1000.0]), pd.Series([200.0]), pd.Series([200.0]),
                                   pd.Series([2.0]), pd.Series([2.0]))
        assert pd.isna(d["aporte_ingreso"].iloc[0])
        assert pd.isna(d["aporte_precios"].iloc[0])
        assert pd.isna(d["aporte_composicion"].iloc[0])


class TestMedianaPonderada:
    def test_mediana_simple(self):
        assert mediana_ponderada(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 1.0, 1.0])) == pytest.approx(2.0)

    def test_peso_desplaza_la_mediana(self):
        # peso 10 en el valor 1.0 -- la mitad del peso acumulado cae ahí.
        r = mediana_ponderada(pd.Series([1.0, 2.0, 3.0]), pd.Series([10.0, 1.0, 1.0]))
        assert r == pytest.approx(1.0)

    def test_sin_datos_validos_da_nan(self):
        assert np.isnan(mediana_ponderada(pd.Series([np.nan]), pd.Series([1.0])))


class TestAgregarComponenteComunYPropio:
    def test_comun_es_la_mediana_del_grupo_y_propia_es_la_diferencia(self):
        pares = pd.DataFrame({
            "ANO4_t": [2020, 2020, 2020], "TRIMESTRE_t": [1, 1, 1],
            "variacion_log": [0.1, 0.3, 0.5], "PONDIH_t": [1.0, 1.0, 1.0],
        })
        r = agregar_componente_comun_y_propio(pares, "variacion_log", "PONDIH_t",
                                               ["ANO4_t", "TRIMESTRE_t"], "trimestre_log")
        assert r["variacion_comun_trimestre_log"].iloc[0] == pytest.approx(0.3)
        assert r["variacion_propia_trimestre_log"].tolist() == pytest.approx([-0.2, 0.0, 0.2])


class TestClasificarDireccion:
    def test_umbral_default(self):
        r = clasificar_direccion(pd.Series([0.2, -0.2, 0.05, -0.05, np.nan]))
        assert r.tolist() == ["mejora", "empeora", "estable", "estable", pd.NA]


class TestClasificarTransicion:
    def test_las_cuatro_categorias_y_na(self):
        pobre_t = pd.Series([False, True, True, False, pd.NA], dtype="boolean")
        pobre_th = pd.Series([True, False, True, False, True], dtype="boolean")
        r = clasificar_transicion(pobre_t, pobre_th)
        assert r.tolist() == ["cae", "sale", "sigue_pobre", "sigue_no_pobre", pd.NA]


def _distancia(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "REGION", "AGLOMERADO", "era",
                "ITF", "PONDIH", "unidades_ae", "cbt_ae", "ratio_lp", "log_ratio_lp", "pobre"]
    return pd.DataFrame(filas, columns=columnas)


class TestConstruirParesDistancia:
    def test_pares_con_variaciones_y_derivados(self):
        distancia_t = _distancia([
            ["V1", 1, 2020, 1, 1, 32, "regional_2016", 1000.0, 100, 2.0, 200.0, 2.5, np.log(2.5), False],
        ])
        distancia_th = _distancia([
            ["V1", 1, 2020, 2, 1, 32, "regional_2016", 1500.0, 100, 2.0, 200.0, 3.75, np.log(3.75), False],
        ])
        pares_hogar = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1],
                                     "identidad_confirmada": [True]})
        r = construir_pares_distancia(distancia_t, distancia_th, pares_hogar).iloc[0]
        assert r["variacion_simetrica"] == pytest.approx((3.75 - 2.5) / ((2.5 + 3.75) / 2))
        assert r["variacion_log"] == pytest.approx(np.log(3.75) - np.log(2.5))
        assert r["diferencia_absoluta"] == pytest.approx(1.25)
        assert r["direccion"] == "mejora"
        assert r["transicion"] == "sigue_no_pobre"
        assert r["aporte_precios"] == pytest.approx(0.0)
        assert r["aporte_composicion"] == pytest.approx(0.0)
        assert r["par_valido"] == True  # noqa: E712

    def test_par_valido_false_si_identidad_no_confirmada(self):
        distancia_t = _distancia([
            ["V1", 1, 2020, 1, 1, 32, "regional_2016", 1000.0, 100, 2.0, 200.0, 2.5, np.log(2.5), False],
        ])
        distancia_th = _distancia([
            ["V1", 1, 2020, 2, 1, 32, "regional_2016", 1500.0, 100, 2.0, 200.0, 3.75, np.log(3.75), False],
        ])
        pares_hogar = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1],
                                     "identidad_confirmada": [False]})
        r = construir_pares_distancia(distancia_t, distancia_th, pares_hogar).iloc[0]
        assert r["par_valido"] == False  # noqa: E712


class TestAgregarTendenciaPrevia:
    def test_busca_el_tramo_t_menos_1_gap_cero(self):
        pares_h1_t_menos_1 = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1],
            "ANO4_t": [2020], "TRIMESTRE_t": [1], "ANO4_th": [2020], "TRIMESTRE_th": [2],
            "variacion_simetrica": [0.4],
        })
        pares_objetivo = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "ANO4_t": [2020], "TRIMESTRE_t": [2],
        })
        r = agregar_tendencia_previa(pares_objetivo, pares_h1_t_menos_1, None)
        assert r["variacion_simetrica_previa"].iloc[0] == pytest.approx(0.4)
        assert r["gap_previo"].iloc[0] == 0

    def test_busca_el_tramo_t_menos_4_gap_tres_si_no_hay_t_menos_1(self):
        # hueco de 2-2-2: el hogar no está en t-1, pero su propio h1 de hace
        # 4 trimestres (1ra->2da visita) se empareja por identidad de origen.
        pares_h1_t_menos_4 = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1], "AGLOMERADO_t": [32],
            "ANO4_t": [2019], "TRIMESTRE_t": [3], "ANO4_th": [2019], "TRIMESTRE_th": [4],
            "variacion_simetrica": [0.4],
        })
        pares_objetivo = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "AGLOMERADO_t": [32], "ANO4_t": [2020], "TRIMESTRE_t": [3],
        })
        r = agregar_tendencia_previa(pares_objetivo, None, pares_h1_t_menos_4)
        assert r["variacion_simetrica_previa"].iloc[0] == pytest.approx(0.4)
        assert r["gap_previo"].iloc[0] == 3

    def test_prioriza_t_menos_1_sobre_t_menos_4_si_ambos_estan(self):
        pares_h1_t_menos_1 = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1],
            "ANO4_t": [2020], "TRIMESTRE_t": [2], "ANO4_th": [2020], "TRIMESTRE_th": [3],
            "variacion_simetrica": [0.4],
        })
        pares_h1_t_menos_4 = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1], "AGLOMERADO_t": [32],
            "ANO4_t": [2019], "TRIMESTRE_t": [4], "ANO4_th": [2020], "TRIMESTRE_th": [1],
            "variacion_simetrica": [0.9],
        })
        pares_objetivo = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "AGLOMERADO_t": [32], "ANO4_t": [2020], "TRIMESTRE_t": [3],
        })
        r = agregar_tendencia_previa(pares_objetivo, pares_h1_t_menos_1, pares_h1_t_menos_4)
        assert r["variacion_simetrica_previa"].iloc[0] == pytest.approx(0.4)
        assert r["gap_previo"].iloc[0] == 0

    def test_sin_observacion_anterior_da_na(self):
        pares_objetivo = pd.DataFrame({
            "CODUSU": ["V1"], "NRO_HOGAR_t": [1], "AGLOMERADO_t": [32], "ANO4_t": [2020], "TRIMESTRE_t": [1],
        })
        r = agregar_tendencia_previa(pares_objetivo, None, None)
        assert pd.isna(r["variacion_simetrica_previa"].iloc[0])
        assert pd.isna(r["gap_previo"].iloc[0])
