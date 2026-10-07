"""Tests de `src/adulto_equivalente.py`. Todo con Series/DataFrames sintéticos
chicos armados a mano -- mismo criterio que tests/test_panel.py."""
import pandas as pd
import pytest

from adulto_equivalente import calcular_unidades_ae_hogar, calcular_unidades_equivalentes


class TestCalcularUnidadesEquivalentes:
    def test_menor_de_un_anio_ch06_menos_uno(self):
        # D19: CH06 nunca vale 0, los menores de 1 año están codificados -1.
        r = calcular_unidades_equivalentes(pd.Series([1, 2]), pd.Series([-1, -1]))
        assert r.tolist() == [0.35, 0.35]

    def test_varon_adulto_referencia(self):
        r = calcular_unidades_equivalentes(pd.Series([1]), pd.Series([35]))
        assert r.iloc[0] == pytest.approx(1.00)

    def test_mujer_adulta(self):
        r = calcular_unidades_equivalentes(pd.Series([2]), pd.Series([35]))
        assert r.iloc[0] == pytest.approx(0.77)

    def test_bordes_de_tramo_adolescente(self):
        # El ejemplo del codebook INDEC: mujeres de 46 a 60 años = 76% del varón
        # de la misma edad -- confirma que el tramo 46-60 toma el valor correcto.
        r = calcular_unidades_equivalentes(pd.Series([2, 1]), pd.Series([46, 46]))
        assert r.tolist() == [0.76, 1.00]

    def test_mayor_de_75_sin_techo(self):
        # CH06 real llega hasta 110 en los datos -- el último tramo no tiene techo.
        r = calcular_unidades_equivalentes(pd.Series([1, 1]), pd.Series([76, 110]))
        assert r.tolist() == [0.74, 0.74]

    def test_sexo_invalido_es_na(self):
        r = calcular_unidades_equivalentes(pd.Series([9]), pd.Series([30]))
        assert pd.isna(r.iloc[0])

    def test_edad_fuera_de_rango_es_na(self):
        r = calcular_unidades_equivalentes(pd.Series([1]), pd.Series([-2]))
        assert pd.isna(r.iloc[0])


class TestCalcularUnidadesAeHogar:
    def _individual(self, filas):
        columnas = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]
        return pd.DataFrame(filas, columns=columnas)

    def test_suma_por_hogar(self):
        # Hogar de 4: varón 35 (1.00) + mujer 31 (0.77) + hijo de 6 (0.64) +
        # hija de 8 (0.68) -- mismo ejemplo que trae el documento INDEC (3,09).
        individual = self._individual([
            ["V1", 1, 1, 35], ["V1", 1, 2, 31], ["V1", 1, 1, 6], ["V1", 1, 2, 8],
        ])
        r = calcular_unidades_ae_hogar(individual)
        assert r.loc[("V1", 1)] == pytest.approx(3.09)

    def test_distintos_hogares_no_se_mezclan(self):
        individual = self._individual([["V1", 1, 1, 35], ["V2", 1, 2, 35]])
        r = calcular_unidades_ae_hogar(individual)
        assert r.loc[("V1", 1)] == pytest.approx(1.00)
        assert r.loc[("V2", 1)] == pytest.approx(0.77)

    def test_un_integrante_indeterminable_anula_el_hogar(self):
        individual = self._individual([["V1", 1, 1, 35], ["V1", 1, 9, 30]])
        r = calcular_unidades_ae_hogar(individual)
        assert pd.isna(r.loc[("V1", 1)])
