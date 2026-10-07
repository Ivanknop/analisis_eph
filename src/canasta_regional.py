"""Canasta básica alimentaria y total (CBA/CBT) por región, por adulto
equivalente -- serie oficial INDEC redistribuida por datos.gob.ar (D25).
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

URLS_CBA_CBT = {
    "cba": "https://infra.datos.gob.ar/catalog/sspm/dataset/445/distribution/445.1/download/canasta-basica-alimentaria-regiones-del-pais.csv",
    "cbt": "https://infra.datos.gob.ar/catalog/sspm/dataset/446/distribution/446.1/download/canasta-basica-total-regiones-del-pais.csv",
}

# Columna del CSV de datos.gob.ar -> código REGION de la EPH.
REGION_POR_COLUMNA = {
    "gran_buenos_aires": 1, "noroeste": 40, "noreste": 41,
    "cuyo": 42, "pampeana": 43, "patagonia": 44,
}

MetodoVentana = Literal["a_mismo_trimestre", "b_un_mes_atras"]


def descargar_canastas(destino: Path, urls: dict[str, str] = URLS_CBA_CBT) -> dict:
    """Baja CBA y CBT regional a `destino/{cba,cbt}_regional_crudo.csv` +
    `destino/manifest.json`. No reintenta: propaga el error tal cual (D25)."""
    destino.mkdir(parents=True, exist_ok=True)
    manifest = {"descargado_utc": datetime.now(timezone.utc).isoformat(), "archivos": {}}
    for nombre, url in urls.items():
        archivo = destino / f"{nombre}_regional_crudo.csv"
        # datos.gob.ar devuelve 403 sin User-Agent (bloquea el default de urllib,
        # "Python-urllib/x.y") -- no es una restricción de acceso real al dataset
        # público, solo un filtro básico anti-bot.
        pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(pedido, timeout=30) as respuesta:
            contenido = respuesta.read()
        archivo.write_bytes(contenido)
        manifest["archivos"][archivo.name] = {"url": url, "bytes": len(contenido)}
    (destino / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def cargar_canasta_larga(ruta_csv: Path, variable: str) -> pd.DataFrame:
    """Formato largo (`REGION, ANO4, MES, variable`) a partir del CSV crudo de
    datos.gob.ar (columnas `indice_tiempo` + una por región)."""
    df = pd.read_csv(ruta_csv, parse_dates=["indice_tiempo"])
    largo = df.melt(id_vars="indice_tiempo", var_name="columna_region", value_name=variable)
    largo["REGION"] = largo["columna_region"].map(REGION_POR_COLUMNA)
    largo["ANO4"] = largo["indice_tiempo"].dt.year
    largo["MES"] = largo["indice_tiempo"].dt.month
    return largo.dropna(subset=["REGION", variable])[["REGION", "ANO4", "MES", variable]]


def _meses_ventana(anio: int, trimestre: int, metodo_ventana: MetodoVentana) -> list[tuple[int, int]]:
    primer_mes = (trimestre - 1) * 3 + 1
    offsets = (0, 1, 2) if metodo_ventana == "a_mismo_trimestre" else (-1, 0, 1)
    meses = []
    for offset in offsets:
        indice = anio * 12 + (primer_mes - 1) + offset
        meses.append((indice // 12, indice % 12 + 1))
    return meses


def agregar_trimestral(largo: pd.DataFrame, variable: str, metodo_ventana: MetodoVentana) -> pd.DataFrame:
    """Promedio mensual->trimestral de `variable` por `REGION`, según
    `metodo_ventana` (D25). `canasta_ventana_parcial=True` si la ventana
    pedida tiene menos de 3 meses con dato."""
    # metodo_ventana: "a_mismo_trimestre" = promedio de los 3 meses del
    # trimestre; "b_un_mes_atras" = corrido un mes atrás (mes de referencia
    # real del ingreso). canasta_ventana_parcial sale de contar cuántos
    # meses matchearon -- hoy solo pasa en 2016T2 con "b_un_mes_atras"
    # (pide marzo 2016, la serie arranca en abril 2016), no está hardcodeado.
    anios = range(int(largo["ANO4"].min()), int(largo["ANO4"].max()) + 1)
    mapping = pd.DataFrame([
        {"ANO4": anio, "TRIMESTRE": trimestre, "ANO4_mes": anio_mes, "MES": mes}
        for anio in anios for trimestre in (1, 2, 3, 4)
        for anio_mes, mes in _meses_ventana(anio, trimestre, metodo_ventana)
    ])
    combinado = mapping.merge(
        largo.rename(columns={"ANO4": "ANO4_mes"}), on=["ANO4_mes", "MES"], how="inner")
    agregado = combinado.groupby(["ANO4", "TRIMESTRE", "REGION"])[variable].agg(["mean", "size"]).reset_index()
    agregado["canasta_ventana_parcial"] = agregado["size"] < 3
    return agregado.rename(columns={"mean": variable}).drop(columns="size")


def construir_canasta_trimestral(destino: Path, metodo_ventana: MetodoVentana) -> pd.DataFrame:
    """CBA/CBT trimestral por región (`ANO4, TRIMESTRE, REGION, cba_ae, cbt_ae,
    canasta_ventana_parcial`), a partir de los CSV crudos ya descargados en
    `destino` (`descargar_canastas`)."""
    cba = agregar_trimestral(cargar_canasta_larga(destino / "cba_regional_crudo.csv", "cba_ae"),
                              "cba_ae", metodo_ventana)
    cbt = agregar_trimestral(cargar_canasta_larga(destino / "cbt_regional_crudo.csv", "cbt_ae"),
                              "cbt_ae", metodo_ventana)
    combinado = cba.merge(cbt, on=["ANO4", "TRIMESTRE", "REGION"], how="outer",
                           suffixes=("", "_cbt"))
    # CBA y CBT comparten la misma cobertura mensual (D25, sección 2) -- si alguna
    # vez difiere, mejor que falle acá a que el flag quede ambiguo en silencio.
    assert (combinado["canasta_ventana_parcial"] == combinado["canasta_ventana_parcial_cbt"]).all()
    return combinado.drop(columns="canasta_ventana_parcial_cbt")
