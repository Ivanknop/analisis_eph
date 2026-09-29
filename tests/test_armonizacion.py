"""Tests de `src/armonizacion.py`. Todo con datos sintéticos -- nada acá lee
`data/eph_cache/`."""
from unittest.mock import patch

import pandas as pd
import pytest

from armonizacion import (
    aglomerados_esperados,
    aglomerados_faltantes,
    armonizar,
    construir_inventario,
    es_historico,
    flags_trimestre,
    pendiente_rar,
    posibles_renombres,
    registrar_inventario,
    tipos_inconsistentes,
    trimestres_publicados,
    variables_con_hueco,
)


class TestTrimestresPublicados:
    def test_excluye_no_publicados(self):
        trimestres = trimestres_publicados()
        assert (2007, 3) not in trimestres
        assert (2015, 3) not in trimestres
        assert (2015, 4) not in trimestres
        assert (2016, 1) not in trimestres

    def test_incluye_extremos(self):
        trimestres = trimestres_publicados()
        assert (2003, 3) in trimestres
        assert (2025, 4) in trimestres
        assert (2003, 1) not in trimestres  # la EPH continua arranca en 2003T3

    def test_total_86(self):
        assert len(trimestres_publicados()) == 86


class TestEsHistorico:
    def test_2015t2_es_historico(self):
        assert es_historico(2015, 2) is True

    def test_2016t1_no_es_historico(self):
        assert es_historico(2016, 1) is False


class TestPendienteRar:
    def test_no_historico_nunca_pendiente(self):
        assert pendiente_rar(2020, 1) is False

    def test_historico_zip_no_pendiente(self):
        # 2013 es zip según _EXTENSION_HISTORICA de eph_client -- no depende de unar.
        assert pendiente_rar(2013, 4) is False

    def test_historico_rar_sin_unar(self):
        with patch("armonizacion.shutil.which", return_value=None):
            assert pendiente_rar(2014, 1) is True

    def test_historico_rar_con_unar_instalado(self):
        with patch("armonizacion.shutil.which", return_value="/usr/bin/unar"):
            assert pendiente_rar(2014, 1) is False


class TestArmonizar:
    def test_mayusculas_y_dtypes_individual(self):
        df = pd.DataFrame({
            "codusu": ["123", "123"],
            "nro_hogar": [1, 1],
            "componente": [1, 2],
            "ano4": [2020, 2020],
            "trimestre": [1, 1],
            "aglomerado": [2, 2],
            "p21": [-9, 50000],
        })
        resultado = armonizar(df, "individual")
        assert list(resultado.columns) == ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "AGLOMERADO", "P21"]
        assert str(resultado["CODUSU"].dtype) == "string"
        assert str(resultado["COMPONENTE"].dtype) == "Int16"
        # -9 no se toca -- no es imputación, es un valor de no-respuesta que se deja igual.
        assert resultado["P21"].tolist() == [-9, 50000]

    def test_hogar_sin_componente(self):
        df = pd.DataFrame({
            "CODUSU": ["1"], "NRO_HOGAR": [1], "ANO4": [2020], "TRIMESTRE": [1], "AGLOMERADO": [2],
        })
        resultado = armonizar(df, "hogar")
        assert "COMPONENTE" not in resultado.columns
        assert str(resultado["AGLOMERADO"].dtype) == "Int16"

    def test_tipo_invalido_levanta_error(self):
        with pytest.raises(ValueError):
            armonizar(pd.DataFrame(), "otra_cosa")


class TestInventario:
    def test_registrar_y_construir(self):
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        registros = registrar_inventario(df, 2020, 1, "hogar")
        inventario = construir_inventario(registros)
        assert set(inventario["variable"]) == {"A", "B"}
        assert (inventario["tipo"] == "hogar").all()

    def test_variables_con_hueco(self):
        registros = (
            registrar_inventario(pd.DataFrame({"A": [1], "B": [1]}), 2020, 1, "hogar")
            + registrar_inventario(pd.DataFrame({"A": [1]}), 2020, 2, "hogar")
        )
        inventario = construir_inventario(registros)
        huecos = variables_con_hueco(inventario)
        assert set(huecos["variable"]) == {"B"}
        fila_b = huecos[huecos["variable"] == "B"].iloc[0]
        assert fila_b["trimestres_presente"] == 1
        assert fila_b["trimestres_verificados"] == 2

    def test_tipos_inconsistentes(self):
        registros = (
            registrar_inventario(pd.DataFrame({"IPCF": pd.array([1], dtype="Int64")}), 2003, 3, "hogar")
            + registrar_inventario(pd.DataFrame({"IPCF": [1.5]}), 2013, 4, "hogar")
            + registrar_inventario(pd.DataFrame({"ESTABLE": [1]}), 2003, 3, "hogar")
            + registrar_inventario(pd.DataFrame({"ESTABLE": [2]}), 2013, 4, "hogar")
        )
        inventario = construir_inventario(registros)
        inconsistentes = tipos_inconsistentes(inventario)
        assert set(inconsistentes["variable"]) == {"IPCF"}
        assert inconsistentes.iloc[0]["cantidad_tipos"] == 2


class TestPosiblesRenombres:
    def test_detecta_por_esqueleto_compartido(self):
        registros = (
            registrar_inventario(pd.DataFrame({"V2_M": [1]}), 2023, 2, "hogar")
            + registrar_inventario(pd.DataFrame({"V2_02_M": [1], "V2_03_M": [1]}), 2023, 4, "hogar")
        )
        inventario = construir_inventario(registros)
        candidatos = posibles_renombres(inventario)
        assert set(candidatos["variable_posible_reemplazo"]) == {"V2_02_M", "V2_03_M"}
        assert (candidatos["variable_anterior"] == "V2_M").all()

    def test_sin_relacion_no_propone_nada(self):
        registros = (
            registrar_inventario(pd.DataFrame({"CH04": [1]}), 2023, 2, "individual")
            + registrar_inventario(pd.DataFrame({"NIVEL_ED": [1]}), 2023, 4, "individual")
        )
        inventario = construir_inventario(registros)
        candidatos = posibles_renombres(inventario)
        assert candidatos.empty


class TestAglomerados:
    def test_esperados_por_era_sin_hardcode(self):
        conjuntos = {
            (2006, 1): {2, 3, 4},
            (2006, 2): {2, 3, 4},
            (2006, 3): {2, 3, 4, 5},
            (2006, 4): {2, 3, 4, 5},
        }
        esperados = aglomerados_esperados(conjuntos)
        assert esperados["pre"] == {2, 3, 4}
        assert esperados["post"] == {2, 3, 4, 5}

    def test_faltantes_detecta_hueco_puntual(self):
        # 8 presente en 9 de 10 trimestres (90%, cruza el umbral por defecto) --
        # falta solo en 2019T3, que es el caso puntual a detectar.
        conjuntos = {(2019, t): {2, 3, 8} for t in range(1, 11)}
        conjuntos[(2019, 3)] = {2, 3}
        esperados = aglomerados_esperados(conjuntos, corte=(2006, 3))
        faltantes = aglomerados_faltantes({2, 3}, 2019, 3, esperados, corte=(2006, 3))
        assert faltantes == [8]

    def test_sin_faltantes(self):
        esperados = {"post": {2, 3, 8}}
        assert aglomerados_faltantes({2, 3, 8}, 2020, 1, esperados) == []


class TestFlagsTrimestre:
    def test_periodo_intervenido(self):
        flags = flags_trimestre(2010, 1, [])
        assert flags["periodo_intervenido"] is True
        assert flags["encuesta_telefonica"] is False

    def test_no_intervenido_fuera_de_rango(self):
        flags = flags_trimestre(2006, 4, [])
        assert flags["periodo_intervenido"] is False

    def test_encuesta_telefonica_2020t2(self):
        flags = flags_trimestre(2020, 2, [])
        assert flags["encuesta_telefonica"] is True

    def test_aglomerados_faltantes_pasa_directo(self):
        flags = flags_trimestre(2019, 3, [8])
        assert flags["aglomerados_faltantes"] == [8]
