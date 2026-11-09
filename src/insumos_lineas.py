"""Insumos de la retropolación Tornarolli (2003-2015): descarga+caché (mismo
patrón que `eph_client.py`/`canasta_regional.py`) y parseo a formato tabular
común (`mes, region, serie, valor, unidad, fuente`) de la Metodología INDEC
N°22, el IPC-GBA histórico y el IPC Provincias (CIFRA-CTA)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from manifiesto_insumos import calcular_sha256

# No calcula ninguna línea, backcast ni distancia -- eso vive en
# `canasta_retropolada.py`. Los insumos de la canasta oficial (canastas regionales,
# tabla de adulto equivalente, fuentes de contraste CEDLAS/INDEC
# histórico) siguen en `data/canastas_indec/` (`canasta_regional.py`,
# `adulto_equivalente.py`, `ponderadores.py`) -- no se tocan ni se
# redescargan acá.

_HEADERS = {"User-Agent": "Mozilla/5.0"}

URLS = {
    "metodologia_22_pobreza.pdf":
        "https://www.indec.gob.ar/ftp/cuadros/sociedad/EPH_metodologia_22_pobreza.pdf",
    "ipc_gba_historico_1943_2008_crudo.csv":
        "https://infra.datos.gob.ar/catalog/sspm/dataset/178/distribution/178.1/download/ipc-historico.csv",
    "ipc_gba_base2008_capitulos_crudo.csv":
        "https://infra.datos.gob.ar/catalog/sspm/dataset/96/distribution/96.5/download/"
        "indice-precios-al-consumidor-apertura-por-capitulos-base-2008-mensual.csv",
    "ipc_provincias_cifra_2007_2018_crudo.xlsx":
        "https://centrocifra.org.ar/wp-content/uploads/2023/08/IPC-Provincias-2007-2018.xlsx",
}

# Insumo 2 (D.T.): valor de CBA/CBT por adulto equivalente del GBA, diciembre
# 2006, extraído a mano de `metodologia_22_pobreza.pdf` (Cuadro 5, p.15,
# columna "Nuevo" -- ENGHo 1996/97 validada con 2004/05, la metodología vigente
# desde 2016). La columna "Anterior" (ENGHo 1985/86) se deja solo de contraste,
# nunca como ancla.
VALORES_DIC_2006_METODOLOGIA_22 = [
    dict(concepto="cba_adulto_equivalente_metodologia_nueva", valor=149.13,
         unidad="pesos corrientes por adulto equivalente por mes", region="GBA",
         periodo="2006-12", pagina=15, cuadro="Cuadro 5",
         fuente='INDEC (2016), Metodología INDEC N°22, columna "Nuevo" (CBA)'),
    dict(concepto="cbt_adulto_equivalente_metodologia_nueva", valor=389.28,
         unidad="pesos corrientes por adulto equivalente por mes", region="GBA",
         periodo="2006-12", pagina=15, cuadro="Cuadro 5",
         fuente='INDEC (2016), Metodología INDEC N°22, columna "Nuevo" (CBT)'),
    dict(concepto="ice_metodologia_nueva", valor=2.61,
         unidad="inversa del coeficiente de Engel (adimensional)", region="GBA",
         periodo="2006-12", pagina=15, cuadro="Cuadro 5",
         fuente='INDEC (2016), Metodología INDEC N°22, columna "Nuevo" (ICE)'),
    dict(concepto="cba_adulto_equivalente_metodologia_anterior", valor=134.14,
         unidad="pesos corrientes por adulto equivalente por mes", region="GBA",
         periodo="2006-12", pagina=15, cuadro="Cuadro 5",
         fuente='INDEC (2016), ídem, columna "Anterior" (CBA) -- ENGHo 1985/86, '
                "solo contraste, no usar como ancla"),
    dict(concepto="cbt_adulto_equivalente_metodologia_anterior", valor=291.08,
         unidad="pesos corrientes por adulto equivalente por mes", region="GBA",
         periodo="2006-12", pagina=15, cuadro="Cuadro 5",
         fuente='INDEC (2016), ídem, columna "Anterior" (CBT) -- ENGHo 1985/86, '
                "solo contraste, no usar como ancla"),
]

# Capítulos del CSV de infra.datos.gob.ar que nos interesan -> nombre de serie
# en el formato tabular común.
_CAPITULOS_IPC_GBA = {
    "nivel_general": "ipc_gba_nivel_general",
    "alimentos_bebidas": "ipc_gba_alimentos_bebidas",
}
_CAPITULOS_IPC_GBA_BASE2008 = {
    "ipc_nivel_general": "ipc_gba_nivel_general",
    "ipc_alimentos_bebidas": "ipc_gba_alimentos_bebidas",
}


def descargar_insumo(destino: Path, nombre_archivo: str, url: str,
                      session: requests.Session | None = None, force_refresh: bool = False) -> Path:
    """Descarga `url` a `destino/nombre_archivo` -- se salta la descarga si el
    archivo ya está cacheado (mismo criterio que `EphClient.descargar_trimestre`).
    No reintenta: propaga el error tal cual."""
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / nombre_archivo
    if archivo.exists() and not force_refresh:
        return archivo
    sesion = session or requests.Session()
    respuesta = sesion.get(url, headers=_HEADERS, timeout=30)
    respuesta.raise_for_status()
    archivo.write_bytes(respuesta.content)
    return archivo


def actualizar_manifiesto(destino: Path, archivos_url: dict[str, str]) -> dict:
    """Agrega/actualiza en `destino/manifest.json` la entrada de cada archivo
    de `archivos_url` ya presente en disco -- url, fecha de descarga (mtime del
    archivo, mismo proxy que `manifiesto_insumos.construir_manifiesto`) y
    sha256 (`manifiesto_insumos.calcular_sha256`, reusada tal cual)."""
    manifest_path = destino / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for nombre, url in archivos_url.items():
        archivo = destino / nombre
        manifest[nombre] = {
            "url": url,
            "fecha_descarga_utc": datetime.fromtimestamp(
                archivo.stat().st_mtime, tz=timezone.utc).isoformat(),
            "sha256": calcular_sha256(archivo),
            "bytes": archivo.stat().st_size,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def descargar_todos(destino: Path, urls: dict[str, str] = URLS,
                     force_refresh: bool = False) -> dict:
    """Descarga todos los insumos de `urls` a `destino` y actualiza el
    manifiesto -- orquesta `descargar_insumo` + `actualizar_manifiesto`."""
    session = requests.Session()
    for nombre, url in urls.items():
        descargar_insumo(destino, nombre, url, session=session, force_refresh=force_refresh)
    return actualizar_manifiesto(destino, urls)


# A diferencia del resto de los insumos (descargados), estos valores vienen de
# una lectura manual del PDF de Metodología N°22 -- se versionan en un CSV
# chico con página y cuadro citados en cada fila, no se recalculan.
def escribir_valores_dic_2006(destino_csv: Path,
                               filas: list[dict] = VALORES_DIC_2006_METODOLOGIA_22) -> pd.DataFrame:
    """Vuelca `VALORES_DIC_2006_METODOLOGIA_22` a `destino_csv`, con referencia
    a página y cuadro en cada fila."""
    df = pd.DataFrame(filas)
    df.to_csv(destino_csv, index=False)
    return df


def _formato_comun(mes: pd.Series, region: str, serie: pd.Series, valor: pd.Series,
                    unidad: str, fuente: str) -> pd.DataFrame:
    return pd.DataFrame({
        "mes": pd.to_datetime(mes).values,
        "region": region,
        "serie": serie.to_numpy() if isinstance(serie, pd.Series) else serie,
        "valor": pd.to_numeric(valor, errors="raise").astype(float).to_numpy(),
        "unidad": unidad,
        "fuente": fuente,
    })


def cargar_ipc_gba_historico(ruta_csv: Path) -> pd.DataFrame:
    """IPC-GBA histórico (empalme bases 1943/1960/1974/1988/1999, hasta
    abril-2008), formato tabular común -- nivel general y alimentos y
    bebidas, mensual (infra.datos.gob.ar, catálogo sspm, dataset 178).
    Insumo 3: cubre de sobra 2003-2006."""
    df = pd.read_csv(ruta_csv)
    largo = df.melt(id_vars="indice_tiempo", var_name="capitulo", value_name="valor")
    largo = largo[largo["capitulo"].isin(_CAPITULOS_IPC_GBA)].dropna(subset=["valor"])
    return _formato_comun(
        mes=largo["indice_tiempo"], region="GBA",
        serie=largo["capitulo"].map(_CAPITULOS_IPC_GBA), valor=largo["valor"],
        unidad="índice (empalme bases 1943/1960/1974/1988/1999, abril-2008=100)",
        fuente="INDEC, vía infra.datos.gob.ar, catálogo sspm dataset 178 (ipc-historico.csv)",
    ).sort_values(["serie", "mes"]).reset_index(drop=True)


def cargar_ipc_gba_base2008_capitulos(ruta_csv: Path) -> pd.DataFrame:
    """IPC-GBA por capítulos, base abril-2008=100 (infra.datos.gob.ar, dataset
    96, resource 96.5) -- mismo formato común, usado como cruce/contraste de
    `cargar_ipc_gba_historico` en el solape 2003-2006 (ambos empalman la
    misma serie oficial, por caminos de publicación distintos)."""
    df = pd.read_csv(ruta_csv)
    largo = df.melt(id_vars="indice_tiempo", var_name="capitulo", value_name="valor")
    largo = largo[largo["capitulo"].isin(_CAPITULOS_IPC_GBA_BASE2008)].dropna(subset=["valor"])
    return _formato_comun(
        mes=largo["indice_tiempo"], region="GBA",
        serie=largo["capitulo"].map(_CAPITULOS_IPC_GBA_BASE2008), valor=largo["valor"],
        unidad="índice (base abril-2008=100)",
        fuente="INDEC, vía infra.datos.gob.ar, catálogo sspm dataset 96 (apertura por capítulos)",
    ).sort_values(["serie", "mes"]).reset_index(drop=True)


def cargar_ipc_provincias_cifra(ruta_xlsx: Path) -> pd.DataFrame:
    """IPC Provincias de CIFRA-CTA (Fernández y González, 2019, DT N°16),
    formato tabular común, un solo nivel general, enero 2007 a diciembre
    2018 (base enero 2014=100)."""
    # Un solo nivel porque es un promedio ponderado de IPC provinciales que no
    # desagregan capítulos de forma uniforme -- no hay alimentos_bebidas acá.
    # Ya encadena internamente provincias + CABA (desde julio 2012, ver nota
    # metodológica citada en `estado_insumos.md`); no hace falta rearmar el empalme.
    df = pd.read_excel(ruta_xlsx, sheet_name="Hoja1", header=2).dropna()
    df.columns = ["mes", "valor"]
    return _formato_comun(
        mes=df["mes"], region="nacional (promedio ponderado de provincias + CABA desde jul-2012)",
        serie="ipc_provincias_cifra_nivel_general", valor=df["valor"],
        unidad="índice (enero 2014=100)",
        fuente="CIFRA-CTA, a partir de Fernández, A.L. y M. González (2019), "
               "\"Cambios metodológicos en la medición de la pobreza en Argentina\", DT N°16",
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    destino = repo / "data" / "insumos_lineas"
    descargar_todos(destino)
    escribir_valores_dic_2006(destino / "dic_2006_gba_metodologia22.csv")
    print(f"insumos descargados y manifiesto actualizado en {destino}")


if __name__ == "__main__":
    main()
