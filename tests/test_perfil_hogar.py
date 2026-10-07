"""Tests de `src/perfil_hogar.py`. El módulo no tenía tests antes de esta ronda --
se agregan solo para `clasificar_situacion_escolar` (D22), no retroactivos a las
funciones existentes."""
import pandas as pd

from perfil_hogar import clasificar_situacion_escolar


class TestClasificarSituacionEscolar:
    def test_termino_pisa_a_ch10_aunque_diga_no_asiste(self):
        r = clasificar_situacion_escolar(ch10=pd.Series([2]), ch06=pd.Series([17]), nivel_ed=pd.Series([4]))
        assert r.iloc[0] == "termino"

    def test_asiste(self):
        r = clasificar_situacion_escolar(ch10=pd.Series([1]), ch06=pd.Series([10]), nivel_ed=pd.Series([3]))
        assert r.iloc[0] == "asiste"

    def test_no_asiste_no_completo(self):
        r = clasificar_situacion_escolar(ch10=pd.Series([3]), ch06=pd.Series([10]), nivel_ed=pd.Series([3]))
        assert r.iloc[0] == "no_asiste_no_completo"

    def test_fuera_de_rango_de_edad_da_na(self):
        r = clasificar_situacion_escolar(ch10=pd.Series([3]), ch06=pd.Series([5]), nivel_ed=pd.Series([1]))
        assert pd.isna(r.iloc[0])

    def test_ch10_ns_nr_sin_terminar_da_na(self):
        r = clasificar_situacion_escolar(ch10=pd.Series([9]), ch06=pd.Series([10]), nivel_ed=pd.Series([3]))
        assert pd.isna(r.iloc[0])
