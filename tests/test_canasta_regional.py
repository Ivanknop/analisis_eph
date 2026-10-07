"""Tests de `src/canasta_regional.py` con fixtures sintéticas chicas --
`descargar_canastas` se prueba contra archivos locales vía `file://`, sin red."""
import json

import pandas as pd
import pytest

from canasta_regional import (
    REGION_POR_COLUMNA,
    _meses_ventana,
    agregar_trimestral,
    cargar_canasta_larga,
    construir_canasta_trimestral,
    descargar_canastas,
)


class TestDescargarCanastas:
    def test_baja_y_escribe_manifest(self, tmp_path):
        fixture = tmp_path / "fixture_origen.csv"
        fixture.write_text("indice_tiempo,gran_buenos_aires\n2016-04-01,100.0\n")
        urls = {"cba": fixture.as_uri()}
        destino = tmp_path / "destino"

        manifest = descargar_canastas(destino, urls=urls)

        assert (destino / "cba_regional_crudo.csv").read_text() == fixture.read_text()
        assert (destino / "manifest.json").exists()
        assert json.loads((destino / "manifest.json").read_text()) == manifest
        assert manifest["archivos"]["cba_regional_crudo.csv"]["url"] == urls["cba"]

    def test_url_invalida_propaga_el_error_sin_reintentar(self, tmp_path):
        urls = {"cba": (tmp_path / "no_existe.csv").as_uri()}
        with pytest.raises(Exception):
            descargar_canastas(tmp_path / "destino", urls=urls)


class TestCargarCanastaLarga:
    def test_formato_largo_y_mapeo_de_region(self, tmp_path):
        csv = tmp_path / "cba.csv"
        csv.write_text(
            "indice_tiempo,gran_buenos_aires,cuyo\n"
            "2016-04-01,100.0,90.0\n"
            "2016-05-01,105.0,95.0\n"
        )
        r = cargar_canasta_larga(csv, "cba_ae")
        assert set(r.columns) == {"REGION", "ANO4", "MES", "cba_ae"}
        fila_gba_abril = r[(r["REGION"] == 1) & (r["MES"] == 4)].iloc[0]
        assert fila_gba_abril["cba_ae"] == pytest.approx(100.0)
        assert set(r["REGION"]) == {REGION_POR_COLUMNA["gran_buenos_aires"], REGION_POR_COLUMNA["cuyo"]}


class TestMesesVentana:
    def test_mismo_trimestre(self):
        assert _meses_ventana(2018, 2, "a_mismo_trimestre") == [(2018, 4), (2018, 5), (2018, 6)]

    def test_un_mes_atras_dentro_del_mismo_anio(self):
        assert _meses_ventana(2018, 2, "b_un_mes_atras") == [(2018, 3), (2018, 4), (2018, 5)]

    def test_un_mes_atras_cruza_anio(self):
        # T1 (ene-feb-mar) con la ventana corrida -> dic del año anterior.
        assert _meses_ventana(2018, 1, "b_un_mes_atras") == [(2017, 12), (2018, 1), (2018, 2)]


class TestAgregarTrimestral:
    def _largo(self):
        # Replica el borde real de la serie (D25): arranca en abril 2016, no hay
        # marzo 2016.
        filas = []
        for anio, mes, valor in [(2016, 4, 100), (2016, 5, 110), (2016, 6, 120)]:
            filas.append({"REGION": 1, "ANO4": anio, "MES": mes, "cba_ae": valor})
        return pd.DataFrame(filas)

    def test_promedio_mismo_trimestre(self):
        r = agregar_trimestral(self._largo(), "cba_ae", "a_mismo_trimestre")
        fila = r[(r["ANO4"] == 2016) & (r["TRIMESTRE"] == 2)].iloc[0]
        assert fila["cba_ae"] == pytest.approx((100 + 110 + 120) / 3)
        assert fila["canasta_ventana_parcial"] == False  # noqa: E712

    def test_ventana_parcial_en_el_borde_de_la_serie(self):
        # La ventana "b" de 2016T2 pide marzo 2016, que no está en la serie
        # (D25: la serie oficial arranca en abril 2016) -- la fixture lo replica.
        r = agregar_trimestral(self._largo(), "cba_ae", "b_un_mes_atras")
        fila = r[(r["ANO4"] == 2016) & (r["TRIMESTRE"] == 2)].iloc[0]
        assert fila["cba_ae"] == pytest.approx((100 + 110) / 2)  # solo abril y mayo
        assert fila["canasta_ventana_parcial"] == True  # noqa: E712


class TestConstruirCanastaTrimestral:
    def test_junta_cba_y_cbt(self, tmp_path):
        (tmp_path / "cba_regional_crudo.csv").write_text(
            "indice_tiempo,gran_buenos_aires\n2016-04-01,100.0\n2016-05-01,100.0\n2016-06-01,100.0\n")
        (tmp_path / "cbt_regional_crudo.csv").write_text(
            "indice_tiempo,gran_buenos_aires\n2016-04-01,200.0\n2016-05-01,200.0\n2016-06-01,200.0\n")
        r = construir_canasta_trimestral(tmp_path, "a_mismo_trimestre")
        fila = r[(r["ANO4"] == 2016) & (r["TRIMESTRE"] == 2) & (r["REGION"] == 1)].iloc[0]
        assert fila["cba_ae"] == pytest.approx(100.0)
        assert fila["cbt_ae"] == pytest.approx(200.0)
