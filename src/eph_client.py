"""Cliente de descarga+caché de microdatos trimestrales de la EPH (INDEC),
2003T3-2025 (histórico vía Wayback Machine hasta 2015T2, 2016+ vía URL
regular de INDEC)."""
# 2003T3 es el piso real: la EPH continua (panel rotativo trimestral, el
# formato que este cliente asume) reemplazó a la EPH puntual/"onda" (bianual,
# cuestionario distinto) a mediados de 2003 -- no hay captura de continua
# anterior a ese trimestre.
from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dbfread import DBF

_URL_BASE = "https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/"
_WAYBACK_URL_BASE = "http://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/"

_URLS_IRREGULARES = {
    (2016, 2): "EPH_usu_2doTrim_2016_txt.zip",
    (2016, 3): "EPH_usu_3erTrim_2016_txt.zip",
    (2016, 4): "EPH_usu_4toTrim_2016_txt.zip",
    (2017, 1): "EPH_usu_1er_trim_2017_txt.zip",
}

_WAYBACK_DBF = {
    (2003, 3): "20140908170023",
    (2003, 4): "20140908165933",
    (2004, 1): "20140908165841",
    (2004, 2): "20140908165749",
    (2004, 3): "20140908165659",
    (2004, 4): "20140908165601",
    (2005, 1): "20140908165451",
    (2005, 2): "20140908165356",
    (2005, 3): "20140908165248",
    (2005, 4): "20140908165105",
    (2006, 1): "20140908165010",
    (2006, 2): "20140908164925",
    (2006, 3): "20140908164836",
    (2006, 4): "20140908164743",
    (2007, 1): "20140908164648",
    (2007, 2): "20140908164550",
    (2007, 4): "20140908164447",
    (2008, 1): "20140908164402",
    (2008, 2): "20140908164304",
    (2008, 3): "20140908164133",
    (2008, 4): "20140908164039",
    (2009, 1): "20140908163927",
    (2009, 2): "20140908163830",
    (2009, 3): "20140908163710",
    (2009, 4): "20140908163610",
    (2010, 1): "20140908163507",
    (2010, 2): "20140908163427",
    (2010, 3): "20140908163328",
    (2010, 4): "20140908163236",
    (2011, 1): "20171018105514",
    (2011, 2): "20171018105001",
    (2011, 3): "20171018152650",
    (2011, 4): "20171018160408",
    (2012, 1): "20171018123400",
    (2012, 2): "20171018141654",
    (2012, 3): "20171018131133",
    (2012, 4): "20171018102400",
    (2013, 1): "20171018130429",
    (2013, 2): "20171018134505",
    (2013, 3): "20171018115205",
    (2013, 4): "20171018105846",
    (2014, 1): "20171018104417",
    (2014, 2): "20171018135407",
    (2014, 3): "20171018102630",
    (2014, 4): "20171018131344",
    (2015, 1): "20171018094820",
    (2015, 2): "20171018112910",
}

TRIMESTRES_NO_PUBLICADOS = {(2015, 3), (2015, 4), (2016, 1), (2007, 3)}

_EXTENSION_HISTORICA = {
    (2014, 1): "rar",
    (2014, 2): "zip",
    (2014, 3): "rar",
    (2014, 4): "rar",
    (2015, 1): "rar",
    (2015, 2): "rar",
}


class TrimestreNoPublicado(Exception):
    pass


class UrlDesconocida(Exception):
    """No hay un patrón de URL confirmado para este (año, trimestre); pasar `url` explícito."""


def _nombre_archivo(anio: int, trimestre: int) -> str:
    if (anio, trimestre) in _URLS_IRREGULARES:
        return _URLS_IRREGULARES[(anio, trimestre)]
    if anio > 2017 or (anio == 2017 and trimestre >= 2):
        return f"EPH_usu_{trimestre}_Trim_{anio}_txt.zip"
    raise UrlDesconocida(
        f"No hay un patrón de URL confirmado para {anio} T{trimestre}. "
        "Pasar `url` explícito a `descargar_trimestre` ."
    )


def _nombre_archivo_historico(anio: int, trimestre: int) -> str:
    """Nombre del archivo DBF histórico 2011-2015 en Wayback Machine —
    `t<trimestre><año 2 dígitos>_dbf.<zip|rar>`.
    """
    yy = anio % 100
    extension = _EXTENSION_HISTORICA.get((anio, trimestre), "zip" if anio <= 2013 else "rar")
    return f"t{trimestre}{yy:02d}_dbf.{extension}"


def _extraer_historico(archivo_path: Path, destino: Path) -> None:
    """Extrae un archivo histórico DBF (.zip/.rar) a `destino`."""
    # Los .zip se extraen con el módulo estándar `zipfile`, sin dependencias
    # externas. Los .rar sí necesitan `unar` instalado en el sistema.
    if archivo_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archivo_path) as z:
            z.extractall(destino)
        return
    try:
        resultado = subprocess.run(
            ["unar", "-quiet", "-output-directory", str(destino), str(archivo_path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"'unar' no está instalado (¿'apt-get install unar'?), no se puede "
            f"extraer {archivo_path}: {e}"
        ) from e
    if resultado.returncode != 0:
        raise RuntimeError(
            f"unar no pudo extraer {archivo_path} (¿está instalado? "
            f"'apt-get install unar'): {resultado.stderr}"
        )


class EphClient:
    def __init__(
        self, cache_dir: Path | str = Path("data/eph_cache"), timeout: float = 60.0
    ):
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.timeout = timeout

    def descargar_trimestre(
        self, anio: int, trimestre: int, url: str | None = None, force_refresh: bool = False
    ) -> Path:
        if (anio, trimestre) in TRIMESTRES_NO_PUBLICADOS:
            raise TrimestreNoPublicado(
                f"INDEC no publicó la EPH para {anio} T{trimestre} "
                "(emergencia estadística o no relevamiento)."
            )

        nombre = url.rsplit("/", 1)[-1] if url else _nombre_archivo(anio, trimestre)
        cache_path = self.cache_dir / str(anio) / f"trim{trimestre}" / nombre

        if cache_path.exists() and not force_refresh:
            return cache_path

        descarga_url = url or (_URL_BASE + nombre)
        response = self.session.get(descarga_url, timeout=self.timeout)
        response.raise_for_status()
        if response.headers.get("Content-Type", "").startswith("text/html"):
            raise UrlDesconocida(
                f"{descarga_url} no devolvió un zip "
                "(INDEC devuelve una página HTML en vez de 404)."
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return cache_path

    def leer_base(self, zip_path: Path, tipo: str) -> pd.DataFrame:
        if tipo not in ("individual", "hogar"):
            raise ValueError("tipo debe ser 'individual' u 'hogar'")
        patrones = ("ind", "personas") if tipo == "individual" else ("hog",)
        with zipfile.ZipFile(zip_path) as z:
            nombres = [
                n
                for n in z.namelist()
                if any(p in n.lower() for p in patrones) and n.lower().endswith((".txt", ".csv"))
            ]
            if not nombres:
                raise FileNotFoundError(f"No se encontró la base '{tipo}' dentro de {zip_path}")
            with z.open(nombres[0]) as f:
                return pd.read_csv(f, sep=";", decimal=",", encoding="latin-1", low_memory=False)

    def descargar_trimestre_historico(
        self, anio: int, trimestre: int, force_refresh: bool = False
    ) -> Path:
        """Descarga (o lee de caché) el .zip/.rar con las bases DBF de un
        trimestre 2011-2015, vía Wayback Machine.
        """
        if (anio, trimestre) in TRIMESTRES_NO_PUBLICADOS:
            raise TrimestreNoPublicado(
                f"INDEC no publicó la EPH para {anio} T{trimestre}."
            )
        timestamp = _WAYBACK_DBF.get((anio, trimestre))
        if timestamp is None:
            raise UrlDesconocida(
                f"No hay una captura de Wayback Machine confirmada para {anio} T{trimestre} "
            )
        nombre = _nombre_archivo_historico(anio, trimestre)
        cache_path = self.cache_dir / str(anio) / f"trim{trimestre}" / nombre

        if cache_path.exists() and not force_refresh:
            return cache_path

        url_wayback = f"http://web.archive.org/web/{timestamp}if_/{_WAYBACK_URL_BASE}{nombre}"
        response = self.session.get(url_wayback, timeout=self.timeout)
        response.raise_for_status()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return cache_path

    def leer_base_historica(self, archivo_path: Path, tipo: str) -> pd.DataFrame:
        """Lee la base `individual` u `hogar` de un .zip/.rar histórico
        (2003T3-2015, formato DBF)."""
        # Patrón de nombre corto -- `ind`/`hog`, no la palabra completa: hasta
        # 2009T4 los DBF vienen como `Ind_t*.DBF`/`Hog_t*.DBF`, desde 2010T1
        # (incl. 2011-2015) como `Individual_t*.dbf`/`Hogar_t*.dbf`.
        if tipo not in ("individual", "hogar"):
            raise ValueError("tipo debe ser 'individual' u 'hogar'")
        patron = "ind" if tipo == "individual" else "hog"
        with tempfile.TemporaryDirectory() as tmp:
            _extraer_historico(Path(archivo_path), Path(tmp))
            candidatos = [
                p for p in Path(tmp).rglob("*")
                if p.suffix.lower() == ".dbf" and patron in p.name.lower()
            ]
            if not candidatos:
                raise FileNotFoundError(f"No se encontró la base '{tipo}' dentro de {archivo_path}")
            tabla = DBF(str(candidatos[0]), encoding="latin-1", char_decode_errors="ignore")
            return pd.DataFrame(iter(tabla))

