"""Tests de `src/eph_client.py`.
"""
import pytest

from eph_client import (
    TrimestreNoPublicado,
    UrlDesconocida,
    _nombre_archivo,
    _nombre_archivo_historico,
)


class TestNombreArchivo:
    def test_patron_regular(self):
        assert _nombre_archivo(2017, 2) == "EPH_usu_2_Trim_2017_txt.zip"
        assert _nombre_archivo(2023, 4) == "EPH_usu_4_Trim_2023_txt.zip"

    def test_nombre_irregular_confirmado(self):
        assert _nombre_archivo(2016, 3) == "EPH_usu_3erTrim_2016_txt.zip"

    def test_trimestre_sin_patron_confirmado_levanta_url_desconocida(self):
        with pytest.raises(UrlDesconocida):
            _nombre_archivo(2012, 1)


class TestNombreArchivoHistorico:
    def test_2011_2013_es_zip(self):
        assert _nombre_archivo_historico(2011, 1) == "t111_dbf.zip"
        assert _nombre_archivo_historico(2013, 4) == "t413_dbf.zip"

    def test_2014_2015_es_rar(self):
        assert _nombre_archivo_historico(2014, 1) == "t114_dbf.rar"
        assert _nombre_archivo_historico(2015, 1) == "t115_dbf.rar"

    def test_2014_t2_es_la_excepcion_confirmada_en_zip(self):
        # único trimestre de 2014-2015 que no sigue la regla general
        assert _nombre_archivo_historico(2014, 2) == "t214_dbf.zip"

