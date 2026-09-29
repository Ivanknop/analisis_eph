"""Tests de `src/pipeline_meta.py`. Todo con `tmp_path` -- nada acá toca
`data/` real."""
import time

import pandas as pd
import pytest

from pipeline_meta import (
    actualizar_resumen_trimestre,
    obtener_commit_git,
    particion_desactualizada,
    particion_existe,
    registrar_corrida,
    ruta_particion,
)


class TestObtenerCommitGit:
    def test_repo_sin_git_da_sin_commit(self, tmp_path):
        # tmp_path no es un repo git -- `git rev-parse` falla ahí.
        assert obtener_commit_git(tmp_path) == "sin commit"


class TestParticionExisteYRuta:
    def test_no_existe(self, tmp_path):
        assert particion_existe(tmp_path, "hogar", 2020, 1) is False

    def test_existe(self, tmp_path):
        p = ruta_particion(tmp_path, "hogar", 2020, 1)
        p.parent.mkdir(parents=True)
        p.write_text("x")
        assert particion_existe(tmp_path, "hogar", 2020, 1) is True

    def test_ruta_tiene_el_formato_esperado(self, tmp_path):
        p = ruta_particion(tmp_path, "individual", 2019, 3)
        assert p == tmp_path / "individual" / "ANO4=2019" / "TRIMESTRE=3" / "data.parquet"


class TestParticionDesactualizada:
    def test_destino_no_existe_es_desactualizado(self, tmp_path):
        origen = tmp_path / "origen.parquet"
        origen.write_text("x")
        destino = tmp_path / "destino.parquet"
        assert particion_desactualizada(origen, destino) is True

    def test_origen_mas_nuevo_es_desactualizado(self, tmp_path):
        destino = tmp_path / "destino.parquet"
        destino.write_text("viejo")
        time.sleep(0.01)
        origen = tmp_path / "origen.parquet"
        origen.write_text("nuevo")
        assert particion_desactualizada(origen, destino) is True

    def test_destino_mas_nuevo_no_esta_desactualizado(self, tmp_path):
        origen = tmp_path / "origen.parquet"
        origen.write_text("viejo")
        time.sleep(0.01)
        destino = tmp_path / "destino.parquet"
        destino.write_text("nuevo")
        assert particion_desactualizada(origen, destino) is False


class TestRegistrarCorrida:
    # El identificador de notebook usa el nombre completo del archivo
    # (`01_ingesta_armonizacion`, no `"01"` a secas) a propósito: un valor
    # puramente numérico con cero a la izquierda sufre el mismo problema que
    # `DECCFR="01"` (ver src/tipado_nucleo.py) apenas alguien relee el CSV sin
    # `dtype=str` -- pandas lo infiere como entero y pierde el cero.
    def test_crea_archivo_si_no_existe(self, tmp_path):
        archivo = tmp_path / "corridas.csv"
        registrar_corrida(archivo, "01_ingesta_armonizacion", {"TRIMESTRES": None}, 81, 162, 93.3, "abc1234")
        df = pd.read_csv(archivo)
        assert len(df) == 1
        assert df.iloc[0]["notebook"] == "01_ingesta_armonizacion"
        assert df.iloc[0]["commit_git"] == "abc1234"

    def test_append_no_pisa_filas_previas(self, tmp_path):
        archivo = tmp_path / "corridas.csv"
        registrar_corrida(archivo, "01_ingesta_armonizacion", {}, 81, 162, 93.3, "abc1234")
        registrar_corrida(archivo, "01b_capa_tipada_nucleo", {}, 81, 162, 89.9, "abc1234")
        df = pd.read_csv(archivo)
        assert len(df) == 2
        assert list(df["notebook"]) == ["01_ingesta_armonizacion", "01b_capa_tipada_nucleo"]


class TestActualizarResumenTrimestre:
    def test_crea_archivo_si_no_existe(self, tmp_path):
        archivo = tmp_path / "resumen.csv"
        actualizar_resumen_trimestre(archivo, [{"anio": 2020, "trimestre": 1, "tipo": "hogar", "filas": 100}])
        df = pd.read_csv(archivo)
        assert len(df) == 1

    def test_upsert_reemplaza_sin_duplicar(self, tmp_path):
        archivo = tmp_path / "resumen.csv"
        actualizar_resumen_trimestre(archivo, [{"anio": 2020, "trimestre": 1, "tipo": "hogar", "filas": 100}])
        actualizar_resumen_trimestre(archivo, [{"anio": 2020, "trimestre": 1, "tipo": "hogar", "filas": 999}])
        df = pd.read_csv(archivo)
        assert len(df) == 1
        assert df.iloc[0]["filas"] == 999

    def test_upsert_conserva_otras_claves(self, tmp_path):
        archivo = tmp_path / "resumen.csv"
        actualizar_resumen_trimestre(archivo, [{"anio": 2020, "trimestre": 1, "tipo": "hogar", "filas": 100}])
        actualizar_resumen_trimestre(archivo, [{"anio": 2020, "trimestre": 2, "tipo": "hogar", "filas": 200}])
        df = pd.read_csv(archivo)
        assert len(df) == 2
        assert set(df["trimestre"]) == {1, 2}

    def test_lista_vacia_no_hace_nada(self, tmp_path):
        archivo = tmp_path / "resumen.csv"
        actualizar_resumen_trimestre(archivo, [])
        assert not archivo.exists()
