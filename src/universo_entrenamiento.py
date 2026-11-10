"""Universo de pares para las Fases C y D: validez, era y cortes desde
`data/06_features/`; objetivo desde `data/05_trayectoria_distancia/` por `CLAVE_PAR`.
"""
# Fuente única del universo (`.claude/decisiones.md`): la 05 no trae `par_valido_c` ni
# las marcas de corte en la era regular; `data/06_features/` sí, verificada par a par
# contra 05 + regla C (0 diferencias, 2026-11-10).
from __future__ import annotations

from pathlib import Path

import pandas as pd

import constantes as cte
import cortes_trayectoria as ct
import panel
from features import CLAVE_PAR

COLUMNAS_VALIDEZ = ["par_valido", *dict.fromkeys(cte.COLUMNA_VALIDEZ.values())]
COLUMNAS_UNIVERSO = [*CLAVE_PAR, "ANO4_th", "TRIMESTRE_th", "h", "era", "AGLOMERADO_t", "identidad_confirmada",
                     *COLUMNAS_VALIDEZ, *ct.FLAGS_CORTE, "peso_hogar_era_t", "peso_normalizado_trimestre"]
FLAGS_POR_SENSIBILIDAD = {
    "con_marcados": [],
    "sin_cortes_puntuales": ct.FLAGS_PUNTUALES,
    "sin_cortes_ni_intervenido": [*ct.FLAGS_PUNTUALES, "par_periodo_intervenido"],
}


def particiones(raiz: Path, h: int) -> list[Path]:
    """Archivos `pares_h{h}/ANO4=*/TRIMESTRE=*/data.parquet` bajo `raiz`, ordenados."""
    return sorted((Path(raiz) / f"pares_h{h}").glob("ANO4=*/TRIMESTRE=*/data.parquet"))


def normalizar_validez(universo: pd.DataFrame) -> pd.DataFrame:
    """Copia con `COLUMNAS_VALIDEZ` en `bool`: NA pasa a `False`."""
    # Los NA son pares históricos con jefe/a ambiguo/a en `t` (`identidad_confirmada` NA,
    # `panel.identidad_nucleo`): 4 en h=1 y 1 en h=4. Quedan fuera de todo universo, como en
    # `06d`; `identidad_confirmada` conserva el NA para poder contarlos.
    resultado = universo.copy()
    for columna in COLUMNAS_VALIDEZ:
        resultado[columna] = resultado[columna].astype("boolean").fillna(False).astype(bool)
    return resultado


def leer_universo(raiz_features: Path, h: int, columnas_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Todos los pares de `raiz_features/pares_h{h}` con `COLUMNAS_UNIVERSO` (+
    `columnas_extra`) y la validez normalizada (`normalizar_validez`)."""
    columnas = [*COLUMNAS_UNIVERSO, *columnas_extra]
    partes = [pd.read_parquet(ruta, columns=columnas) for ruta in particiones(raiz_features, h)]
    if not partes:
        raise FileNotFoundError(f"sin particiones en {raiz_features}/pares_h{h}")
    return normalizar_validez(pd.concat(partes, ignore_index=True))


def unir_objetivo(universo: pd.DataFrame, raiz_trayectoria: Path, h: int,
                  columnas_objetivo: tuple[str, ...] = ("variacion_simetrica", "variacion_log")) -> pd.DataFrame:
    """`universo` con `columnas_objetivo` de la capa 05 por `CLAVE_PAR` (uno a
    uno). Error si algún par del universo no está en la 05."""
    partes = [pd.read_parquet(ruta, columns=[*CLAVE_PAR, *columnas_objetivo]) for ruta in particiones(raiz_trayectoria, h)]
    objetivo = pd.concat(partes, ignore_index=True)
    unido = universo.merge(objetivo, on=CLAVE_PAR, how="left", validate="one_to_one", indicator=True)
    faltantes = int((unido["_merge"] != "both").sum())
    if faltantes:
        raise ValueError(f"h={h}: {faltantes} pares del universo sin fila en la 05")
    return unido.drop(columns="_merge")


def en_universo(pares: pd.DataFrame, universo: str) -> pd.Series:
    """Máscara de los pares de `universo` (`constantes.UNIVERSOS`) según el año de origen."""
    if universo not in cte.ANIO_DESDE_UNIVERSO:
        raise ValueError(f"universo desconocido: {universo}")
    return pares["ANO4_t"].astype(int) >= cte.ANIO_DESDE_UNIVERSO[universo]


def es_valido(pares: pd.DataFrame, objetivo: str, criterio_imputacion: str) -> pd.Series:
    """Máscara de validez para `objetivo` y `criterio_imputacion` (`constantes.COLUMNA_VALIDEZ`)."""
    return pares[cte.COLUMNA_VALIDEZ[(objetivo, criterio_imputacion)]].astype(bool)


def sin_cortes(pares: pd.DataFrame, sensibilidad: str) -> pd.Series:
    """Máscara de los pares sin ninguna marca de `FLAGS_POR_SENSIBILIDAD[sensibilidad]`."""
    flags = FLAGS_POR_SENSIBILIDAD[sensibilidad]
    if not flags:
        return pd.Series(True, index=pares.index)
    return ~pares[flags].astype(bool).any(axis=1)


def conteos_validos(pares: pd.DataFrame, holdout_desde: tuple[int, int] = (2023, 1)) -> pd.DataFrame:
    """Pares válidos por universo, h, era, partición, objetivo y criterio de
    imputación (formato largo). Las combinaciones sin pares no aparecen."""
    con_particion = pares.assign(particion=panel.particion_train_holdout(pares, holdout_desde))
    filas = []
    for universo in cte.UNIVERSOS:
        sub = con_particion[en_universo(con_particion, universo)]
        for (objetivo, criterio), columna in cte.COLUMNA_VALIDEZ.items():
            n = sub[sub[columna].astype(bool)].groupby(["h", "era", "particion"]).size().rename("n_pares_validos")
            filas.append(n.reset_index().assign(universo=universo, objetivo=objetivo, criterio_imputacion=criterio))
    columnas = ["universo", "h", "era", "particion", "objetivo", "criterio_imputacion", "n_pares_validos"]
    return pd.concat(filas, ignore_index=True)[columnas]
