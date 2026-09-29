"""Manifiesto de procedencia de `data/eph_cache/`: URL de origen, fecha de
descarga y hash SHA-256 de cada archivo crudo cacheado, para poder verificar
más adelante que se trabajó con los mismos insumos.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from armonizacion import es_historico
from eph_client import _URL_BASE, _WAYBACK_DBF, _WAYBACK_URL_BASE

COLUMNAS = ("anio", "trimestre", "archivo", "url", "fecha_descarga", "sha256")


def calcular_sha256(archivo: Path) -> str:
    hasher = hashlib.sha256()
    with open(archivo, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def url_fuente(anio: int, trimestre: int, nombre_archivo: str) -> str | None:
    """URL desde donde se descargó `nombre_archivo` para `(anio, trimestre)`.
    `None` si es histórico y no hay captura de Wayback Machine registrada
    para ese trimestre (no debería pasar para un archivo ya cacheado)."""
    if es_historico(anio, trimestre):
        timestamp = _WAYBACK_DBF.get((anio, trimestre))
        if timestamp is None:
            return None
        return f"http://web.archive.org/web/{timestamp}if_/{_WAYBACK_URL_BASE}{nombre_archivo}"
    return _URL_BASE + nombre_archivo


def construir_manifiesto(cache_dir: Path) -> pd.DataFrame:
    """Recorre `cache_dir/<anio>/trim<trimestre>/<archivo>` y arma una fila
    por archivo cacheado. `fecha_descarga` es el mtime del archivo en disco
    -- proxy razonable porque `EphClient` solo escribe el archivo una vez, al
    descargarlo, pero no es un evento de descarga logueado aparte."""
    filas = []
    for archivo in sorted(cache_dir.glob("*/trim*/*")):
        if not archivo.is_file():
            continue
        anio = int(archivo.parents[1].name)
        trimestre = int(archivo.parents[0].name.removeprefix("trim"))
        filas.append({
            "anio": anio,
            "trimestre": trimestre,
            "archivo": archivo.name,
            "url": url_fuente(anio, trimestre, archivo.name),
            "fecha_descarga": pd.Timestamp(archivo.stat().st_mtime, unit="s").isoformat(),
            "sha256": calcular_sha256(archivo),
        })
    return pd.DataFrame(filas, columns=list(COLUMNAS)).sort_values(["anio", "trimestre"]).reset_index(drop=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    cache_dir = repo / "data" / "eph_cache"
    destino = repo / "data" / "meta" / "manifiesto_insumos.csv"

    manifiesto = construir_manifiesto(cache_dir)
    destino.parent.mkdir(parents=True, exist_ok=True)
    manifiesto.to_csv(destino, index=False)
    print(f"guardado: {destino} ({len(manifiesto)} archivos)")
    sin_url = manifiesto[manifiesto["url"].isna()]
    if len(sin_url):
        print(f"ADVERTENCIA: {len(sin_url)} archivos sin URL reconstruible (histórico sin captura Wayback):")
        print(sin_url[["anio", "trimestre", "archivo"]].to_string(index=False))


if __name__ == "__main__":
    main()
