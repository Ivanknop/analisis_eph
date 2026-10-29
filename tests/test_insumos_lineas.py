"""Tests de `src/insumos_lineas.py` -- fixtures sintéticas chicas con la
forma real de cada archivo descargado (confirmada contra los insumos
efectivamente bajados esta ronda). Nada de red: `descargar_insumo` se prueba
inyectando una sesión falsa."""
import json

import pandas as pd
import pytest

from insumos_lineas import (
    VALORES_DIC_2006_METODOLOGIA_22,
    actualizar_manifiesto,
    cargar_ipc_gba_base2008_capitulos,
    cargar_ipc_gba_historico,
    cargar_ipc_provincias_cifra,
    descargar_insumo,
    escribir_valores_dic_2006,
)


class _RespuestaFalsa:
    def __init__(self, contenido: bytes):
        self.content = contenido

    def raise_for_status(self):
        pass


class _SesionFalsa:
    def __init__(self, contenido: bytes = b"contenido"):
        self.contenido = contenido
        self.llamadas = 0

    def get(self, url, headers=None, timeout=None):
        self.llamadas += 1
        return _RespuestaFalsa(self.contenido)


class _SesionQueFalla:
    def get(self, *a, **kw):
        raise AssertionError("no debería pedir red: el archivo ya está cacheado")


class TestDescargarInsumo:
    def test_descarga_si_no_esta_cacheado(self, tmp_path):
        sesion = _SesionFalsa(b"hola mundo")
        archivo = descargar_insumo(tmp_path, "x.csv", "http://ejemplo/x.csv", session=sesion)
        assert archivo.read_bytes() == b"hola mundo"
        assert sesion.llamadas == 1

    def test_no_redescarga_si_ya_esta_cacheado(self, tmp_path):
        (tmp_path / "x.csv").write_bytes(b"ya estaba")
        archivo = descargar_insumo(tmp_path, "x.csv", "http://ejemplo/x.csv", session=_SesionQueFalla())
        assert archivo.read_bytes() == b"ya estaba"

    def test_force_refresh_vuelve_a_descargar(self, tmp_path):
        (tmp_path / "x.csv").write_bytes(b"viejo")
        sesion = _SesionFalsa(b"nuevo")
        archivo = descargar_insumo(tmp_path, "x.csv", "http://ejemplo/x.csv",
                                    session=sesion, force_refresh=True)
        assert archivo.read_bytes() == b"nuevo"
        assert sesion.llamadas == 1


class TestActualizarManifiesto:
    def test_registra_url_sha256_y_bytes(self, tmp_path):
        (tmp_path / "x.csv").write_bytes(b"contenido de prueba")
        manifest = actualizar_manifiesto(tmp_path, {"x.csv": "http://ejemplo/x.csv"})
        assert manifest["x.csv"]["url"] == "http://ejemplo/x.csv"
        assert manifest["x.csv"]["bytes"] == len(b"contenido de prueba")
        assert len(manifest["x.csv"]["sha256"]) == 64
        assert json.loads((tmp_path / "manifest.json").read_text()) == manifest

    def test_preserva_entradas_previas_no_tocadas(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"viejo.csv": {"url": "http://old"}}))
        (tmp_path / "x.csv").write_bytes(b"nuevo")
        manifest = actualizar_manifiesto(tmp_path, {"x.csv": "http://ejemplo/x.csv"})
        assert "viejo.csv" in manifest
        assert "x.csv" in manifest


class TestEscribirValoresDic2006:
    def test_escribe_los_5_conceptos_con_pagina_y_cuadro(self, tmp_path):
        destino = tmp_path / "dic_2006.csv"
        df = escribir_valores_dic_2006(destino)
        assert len(df) == len(VALORES_DIC_2006_METODOLOGIA_22) == 5
        assert set(df["concepto"]) == {
            "cba_ae_metodologia_nueva", "cbt_ae_metodologia_nueva", "ice_metodologia_nueva",
            "cba_ae_metodologia_anterior", "cbt_ae_metodologia_anterior",
        }
        assert (df["pagina"] == 15).all()
        assert (df["cuadro"] == "Cuadro 5").all()
        releido = pd.read_csv(destino)
        assert len(releido) == 5

    def test_cba_nueva_cbt_nueva_son_mayores_que_la_metodologia_anterior(self):
        # Chequeo de sentido (D.T.: metodología 2016 reconoce canastas más
        # caras que la ENGHo 1985/86 para el mismo mes) -- no valida el valor
        # exacto, solo que no se invirtió un signo al transcribir.
        df = pd.DataFrame(VALORES_DIC_2006_METODOLOGIA_22).set_index("concepto")["valor"]
        assert df["cba_ae_metodologia_nueva"] > df["cba_ae_metodologia_anterior"]
        assert df["cbt_ae_metodologia_nueva"] > df["cbt_ae_metodologia_anterior"]


class TestCargarIpcGbaHistorico:
    def test_formato_comun_y_filtra_otros_capitulos(self, tmp_path):
        csv = tmp_path / "ipc_historico.csv"
        csv.write_text(
            "indice_tiempo,nivel_general,alimentos_bebidas,indumentaria\n"
            "2003-01-01,139.38,153.46,142.86\n"
            "2003-02-01,140.17,155.30,141.76\n"
        )
        r = cargar_ipc_gba_historico(csv)
        assert set(r.columns) == {"mes", "region", "serie", "valor", "unidad", "fuente"}
        assert set(r["serie"]) == {"ipc_gba_nivel_general", "ipc_gba_alimentos_bebidas"}
        assert (r["region"] == "GBA").all()
        fila = r[(r["serie"] == "ipc_gba_nivel_general") & (r["mes"] == pd.Timestamp("2003-01-01"))].iloc[0]
        assert fila["valor"] == pytest.approx(139.38)

    def test_descarta_filas_sin_dato(self, tmp_path):
        csv = tmp_path / "ipc_historico.csv"
        csv.write_text(
            "indice_tiempo,nivel_general,alimentos_bebidas\n"
            "1943-01-01,2.05e-12,\n"  # alimentos_bebidas no existía en 1943 (NA real del archivo)
        )
        r = cargar_ipc_gba_historico(csv)
        assert set(r["serie"]) == {"ipc_gba_nivel_general"}


class TestCargarIpcGbaBase2008Capitulos:
    def test_mapea_columnas_ipc_al_formato_comun(self, tmp_path):
        csv = tmp_path / "ipc_capitulos.csv"
        csv.write_text(
            "indice_tiempo,ipc_nivel_general,ipc_alimentos_bebidas,ipc_indumentaria\n"
            "2005-01-01,100.0,101.0,99.0\n"
        )
        r = cargar_ipc_gba_base2008_capitulos(csv)
        assert set(r["serie"]) == {"ipc_gba_nivel_general", "ipc_gba_alimentos_bebidas"}
        assert len(r) == 2


class TestCargarIpcProvinciasCifra:
    def test_lee_la_hoja_con_titulo_y_header_en_fila_2(self, tmp_path):
        # Replica la forma real del .xlsx de CIFRA: fila 0 = título, fila 1 en
        # blanco, fila 2 = encabezado ("Período", "Índice enero 2014=100").
        xlsx = tmp_path / "ipc_provincias.xlsx"
        bruto = pd.DataFrame({
            0: ["IPC Provincias-CIFRA", None, "Período", "2007-01-01", "2007-02-01"],
            1: [None, None, "Índice enero 2014=100", 22.031248, 22.284217],
        })
        bruto.to_excel(xlsx, sheet_name="Hoja1", header=False, index=False)
        r = cargar_ipc_provincias_cifra(xlsx)
        assert len(r) == 2
        assert set(r["serie"]) == {"ipc_provincias_cifra_nivel_general"}
        assert r.iloc[0]["valor"] == pytest.approx(22.031248)
        assert r.iloc[0]["mes"] == pd.Timestamp("2007-01-01")
