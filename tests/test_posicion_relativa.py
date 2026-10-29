"""Tests de `src/posicion_relativa.py`. DataFrames sintéticos chicos armados
a mano -- mismo criterio que tests/test_linea_pobreza.py."""
import numpy as np
import pandas as pd
import pytest

from posicion_relativa import calcular_peso_hogar, calcular_posicion_relativa


def _hogar(filas, con_pondih=True):
    columnas = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "ITF", "ingreso_no_declarado"]
    if con_pondih:
        columnas = columnas[:5] + ["PONDIH"] + columnas[5:]
    df = pd.DataFrame(filas, columns=columnas)
    df["ingreso_no_declarado"] = df["ingreso_no_declarado"].astype("boolean")
    return df


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "CH04", "CH06", "PONDERA"]
    return pd.DataFrame(filas, columns=columnas)


class TestCalcularPesoHogar:
    def test_regular_usa_pondih(self):
        hogar = _hogar([["V1", 1, 2024, 1, 32, 1500, 3000, False]])
        individual = _individual([["V1", 1, 1, 35, 999]])
        peso = calcular_peso_hogar(hogar, individual, es_historico=False)
        assert peso[("V1", 1)] == pytest.approx(1500.0)

    def test_historica_usa_pondera_del_individual(self):
        hogar = _hogar([["V1", 1, 2010, 1, 32, 3000, False]], con_pondih=False)
        individual = _individual([["V1", 1, 1, 35, 700], ["V1", 1, 2, 33, 700]])
        peso = calcular_peso_hogar(hogar, individual, es_historico=True)
        assert peso[("V1", 1)] == pytest.approx(700.0)


class TestCalcularPosicionRelativa:
    def test_posicion_relativa_a_la_mediana_del_aglomerado(self):
        # 3 hogares, mismo aglomerado y trimestre, 1 adulto equivalente cada uno
        # (varón, 35 -> unidades_ae=1.0) -- y_ae = ITF. Mediana ponderada = 2000.
        hogar = _hogar([
            ["V1", 1, 2024, 1, 32, 1000, 1000, False],
            ["V2", 1, 2024, 1, 32, 1000, 2000, False],
            ["V3", 1, 2024, 1, 32, 1000, 4000, False],
        ])
        individual = _individual([
            ["V1", 1, 1, 35, 1], ["V2", 1, 1, 35, 1], ["V3", 1, 1, 35, 1],
        ])
        r = calcular_posicion_relativa(hogar, individual, es_historico=False)
        r = r.set_index("CODUSU")
        assert r.loc["V1", "mediana_aglomerado_y_ae"] == pytest.approx(2000.0)
        assert r.loc["V2", "posicion_relativa_aglomerado"] == pytest.approx(1.0)
        assert r.loc["V3", "posicion_relativa_aglomerado"] == pytest.approx(2.0)
        assert r.loc["V1", "era"] == "regular"
        assert r.loc["V1", "hot_deck_disponible"] == True  # noqa: E712

    def test_itf_cero_declarado_cuenta_para_la_mediana_y_da_y_ae_cero(self):
        hogar = _hogar([
            ["V1", 1, 2024, 1, 32, 1000, 0, False],
            ["V2", 1, 2024, 1, 32, 1000, 2000, False],
        ])
        individual = _individual([["V1", 1, 1, 35, 1], ["V2", 1, 1, 35, 1]])
        r = calcular_posicion_relativa(hogar, individual, es_historico=False).set_index("CODUSU")
        assert r.loc["V1", "y_ae"] == pytest.approx(0.0)
        assert pd.isna(r.loc["V1", "log_y_ae"])
        # mediana ponderada de [0, 2000] con pesos iguales -> 0 (sin interpolar,
        # mediana_ponderada toma el valor donde el acumulado cruza la mitad).
        assert r.loc["V1", "mediana_aglomerado_y_ae"] == pytest.approx(0.0)

    def test_no_respuesta_confirmada_anula_y_ae_y_se_excluye_de_la_mediana(self):
        hogar = _hogar([
            ["V1", 1, 2024, 1, 32, 1000, 0, True],
            ["V2", 1, 2024, 1, 32, 1000, 2000, False],
        ])
        individual = _individual([["V1", 1, 1, 35, 1], ["V2", 1, 1, 35, 1]])
        r = calcular_posicion_relativa(hogar, individual, es_historico=False).set_index("CODUSU")
        assert pd.isna(r.loc["V1", "y_ae"])
        assert pd.isna(r.loc["V1", "posicion_relativa_aglomerado"])
        # con V1 excluido, la mediana del aglomerado queda en el único y_ae de V2.
        assert r.loc["V2", "mediana_aglomerado_y_ae"] == pytest.approx(2000.0)

    def test_no_declarado_na_no_anula_y_ae_era_historica(self):
        # D3: en la era histórica ingreso_no_declarado es NA, no True -- no
        # enmascara (fillna(False) en la máscara).
        hogar = _hogar([["V1", 1, 2010, 1, 32, 3000, pd.NA]], con_pondih=False)
        individual = _individual([["V1", 1, 1, 35, 700]])
        r = calcular_posicion_relativa(hogar, individual, es_historico=True).set_index("CODUSU")
        assert r.loc["V1", "y_ae"] == pytest.approx(3000.0)
        assert r.loc["V1", "era"] == "historica"
        assert r.loc["V1", "hot_deck_disponible"] == False  # noqa: E712

    def test_unidades_ae_indeterminable_da_na(self):
        hogar = _hogar([["V1", 1, 2024, 1, 32, 1000, 3000, False]])
        individual = _individual([["V1", 1, 9, 35, 1]])  # CH04=9 -> sexo inválido
        r = calcular_posicion_relativa(hogar, individual, es_historico=False).set_index("CODUSU")
        assert pd.isna(r.loc["V1", "unidades_ae"])
        assert pd.isna(r.loc["V1", "y_ae"])
