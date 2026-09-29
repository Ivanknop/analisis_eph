"""Tests de `src/manifiesto_insumos.py`. Todo con `tmp_path` -- nada acá
descarga ni toca `data/eph_cache/` real."""
import hashlib

from manifiesto_insumos import calcular_sha256, construir_manifiesto, url_fuente


class TestCalcularSha256:
    def test_coincide_con_hashlib_directo(self, tmp_path):
        archivo = tmp_path / "x.zip"
        archivo.write_bytes(b"contenido de prueba")
        assert calcular_sha256(archivo) == hashlib.sha256(b"contenido de prueba").hexdigest()

    def test_distinto_contenido_distinto_hash(self, tmp_path):
        a = tmp_path / "a.zip"
        b = tmp_path / "b.zip"
        a.write_bytes(b"uno")
        b.write_bytes(b"dos")
        assert calcular_sha256(a) != calcular_sha256(b)


class TestUrlFuente:
    def test_regular_usa_url_base_mas_nombre(self):
        url = url_fuente(2023, 4, "EPH_usu_4_Trim_2023_txt.zip")
        assert url == "https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_usu_4_Trim_2023_txt.zip"

    def test_historico_con_captura_wayback_usa_esa_captura(self):
        url = url_fuente(2011, 1, "t111_dbf.zip")
        assert url.startswith("http://web.archive.org/web/20171018105514if_/")
        assert url.endswith("t111_dbf.zip")

    def test_historico_sin_captura_registrada_da_none(self):
        assert url_fuente(2007, 4, "t407_dbf.zip") is not None
        assert url_fuente(2007, 3, "no_deberia_existir.zip") is None


class TestConstruirManifiesto:
    def _crear_cache(self, tmp_path, anio, trimestre, nombre, contenido=b"x"):
        archivo = tmp_path / str(anio) / f"trim{trimestre}" / nombre
        archivo.parent.mkdir(parents=True)
        archivo.write_bytes(contenido)
        return archivo

    def test_cache_vacia_da_manifiesto_vacio(self, tmp_path):
        manifiesto = construir_manifiesto(tmp_path)
        assert len(manifiesto) == 0
        assert list(manifiesto.columns) == ["anio", "trimestre", "archivo", "url", "fecha_descarga", "sha256"]

    def test_una_fila_por_archivo_cacheado(self, tmp_path):
        self._crear_cache(tmp_path, 2023, 4, "EPH_usu_4_Trim_2023_txt.zip")
        self._crear_cache(tmp_path, 2011, 1, "t111_dbf.zip")
        manifiesto = construir_manifiesto(tmp_path)
        assert len(manifiesto) == 2
        assert set(manifiesto["anio"]) == {2023, 2011}

    def test_hash_coincide_con_el_contenido_real(self, tmp_path):
        archivo = self._crear_cache(tmp_path, 2023, 4, "EPH_usu_4_Trim_2023_txt.zip", contenido=b"datos reales")
        manifiesto = construir_manifiesto(tmp_path)
        assert manifiesto.iloc[0]["sha256"] == calcular_sha256(archivo)

    def test_ordenado_por_anio_y_trimestre(self, tmp_path):
        self._crear_cache(tmp_path, 2023, 4, "b.zip")
        self._crear_cache(tmp_path, 2011, 1, "a.zip")
        manifiesto = construir_manifiesto(tmp_path)
        assert list(manifiesto["anio"]) == [2011, 2023]
