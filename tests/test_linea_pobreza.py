"""Tests de `src/linea_pobreza.py`. Todo con DataFrames sintéticos chicos armados
a mano -- mismo criterio que tests/test_cascada.py."""
import numpy as np
import pandas as pd
import pytest

from linea_pobreza import calcular_distancia_lp


def _hogar(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "REGION", "AGLOMERADO", "ITF", "ingreso_no_declarado"]
    df = pd.DataFrame(filas, columns=columnas)
    df["ingreso_no_declarado"] = df["ingreso_no_declarado"].astype("boolean")
    return df


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]
    return pd.DataFrame(filas, columns=columnas)


def _canasta(region=1, cba_ae=1000.0, cbt_ae=2000.0):
    return pd.DataFrame({"REGION": [region], "cba_ae": [cba_ae], "cbt_ae": [cbt_ae]})


class TestCalcularDistanciaLp:
    def test_hogar_por_encima_de_la_linea(self):
        # 1 adulto equivalente (varón, 35) -> linea_pobreza = 2000, linea_indigencia = 1000.
        hogar = _hogar([["V1", 1, 1, 32, 3000, False]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert r["unidades_ae"] == pytest.approx(1.0)
        assert r["linea_pobreza"] == pytest.approx(2000.0)
        assert r["linea_indigencia"] == pytest.approx(1000.0)
        assert r["ratio_lp"] == pytest.approx(1.5)
        assert r["brecha_relativa_lp"] == pytest.approx(0.5)
        assert r["log_ratio_lp"] == pytest.approx(np.log(1.5))
        assert r["ratio_li"] == pytest.approx(3.0)
        assert r["pobre"] == False  # noqa: E712
        assert r["indigente"] == False  # noqa: E712
        assert r["ITF"] == 3000
        assert r["AGLOMERADO"] == 32

    def test_hogar_pobre_no_indigente(self):
        hogar = _hogar([["V1", 1, 1, 32, 1500, False]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert r["pobre"] == True  # noqa: E712
        assert r["indigente"] == False  # noqa: E712
        assert r["brecha_relativa_lp"] == pytest.approx(-0.25)

    def test_itf_cero_declarado_es_indigente_con_distancia_cero_y_log_nulo(self):
        hogar = _hogar([["V1", 1, 1, 32, 0, False]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert r["ratio_lp"] == pytest.approx(0.0)
        assert r["ratio_li"] == pytest.approx(0.0)
        assert pd.isna(r["log_ratio_lp"])
        assert pd.isna(r["log_ratio_li"])
        assert r["pobre"] == True  # noqa: E712
        assert r["indigente"] == True  # noqa: E712
        assert r["itf_cero_declarado"] == True  # noqa: E712

    def test_no_respuesta_confirmada_anula_toda_la_distancia(self):
        hogar = _hogar([["V1", 1, 1, 32, 0, True]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert pd.isna(r["ratio_lp"])
        assert pd.isna(r["brecha_relativa_lp"])
        assert pd.isna(r["log_ratio_lp"])
        assert pd.isna(r["pobre"])
        assert pd.isna(r["indigente"])
        # ITF==0 pero no_declarado=True -- no es un caso de "itf_cero_declarado"
        # (no se sabe si el 0 es real o un artefacto de la no respuesta).
        assert r["itf_cero_declarado"] == False  # noqa: E712

    def test_no_declarado_na_no_anula_la_distancia_era_historica(self):
        # D3/D25: en la era histórica ingreso_no_declarado es NA (no determinable),
        # no True -- la distancia se calcula igual con el ITF tal cual viene.
        hogar = _hogar([["V1", 1, 1, 32, 3000, pd.NA]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert r["ratio_lp"] == pytest.approx(1.5)
        assert r["pobre"] == False  # noqa: E712

    def test_region_sin_canasta_da_na(self):
        hogar = _hogar([["V1", 1, 99, 32, 3000, False]])
        individual = _individual([["V1", 1, 1, 35]])
        r = calcular_distancia_lp(hogar, individual, _canasta()).iloc[0]
        assert pd.isna(r["linea_pobreza"])
        assert pd.isna(r["ratio_lp"])
        assert pd.isna(r["log_ratio_lp"])
