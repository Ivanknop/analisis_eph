"""Armonización estructural de microdatos EPH cacheados: nombres de columna,
tipos de claves, e inventario de variable x trimestre x tipo real.
"""
# La separación histórico (DBF, 2003T3-2015T2) vs. regular (texto, 2016T1+)
# usa el mismo corte de año que ya asume `eph_client.py` internamente -- ver
# `es_historico`, no se reimplementa acá.
from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from eph_client import (
    EphClient,
    TRIMESTRES_NO_PUBLICADOS,
    TrimestreNoPublicado,
    UrlDesconocida,
    _EXTENSION_HISTORICA,
)

CLAVES_INDIVIDUAL = ("CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "AGLOMERADO")
CLAVES_HOGAR = ("CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO")

DTYPES_CLAVES = {
    "CODUSU": "string",
    "NRO_HOGAR": "Int16",
    "COMPONENTE": "Int16",
    "ANO4": "Int16",
    "TRIMESTRE": "Int16",
    "AGLOMERADO": "Int16",
}

_CORTE_AGLOMERADOS = (2006, 3)


def trimestres_publicados(anio_inicio: int = 2003, trimestre_inicio: int = 3,
                           anio_fin: int = 2025, trimestre_fin: int = 4) -> list[tuple[int, int]]:
    """Todos los (año, trimestre) en el rango publicado, excluyendo
    `TRIMESTRES_NO_PUBLICADOS` (importado de `eph_client`, ya incluye 2007T3
    y 2015T3-2016T1)."""
    trimestres = []
    anio, trimestre = anio_inicio, trimestre_inicio
    while (anio, trimestre) <= (anio_fin, trimestre_fin):
        if (anio, trimestre) not in TRIMESTRES_NO_PUBLICADOS:
            trimestres.append((anio, trimestre))
        trimestre += 1
        if trimestre > 4:
            trimestre = 1
            anio += 1
    return trimestres


def es_historico(anio: int, _trimestre: int) -> bool:
    """Mismo corte que usa `eph_client.py` internamente: 2003-2015 vía DBF/Wayback."""
    return anio <= 2015


def pendiente_rar(anio: int, trimestre: int) -> bool:
    """True si el trimestre es histórico, su archivo es `.rar`, y no hay
    `unar` disponible en este entorno."""
    # Se recalcula contra el entorno real (no una lista fija) para que deje
    # de aplicar apenas se instale `unar`.
    if not es_historico(anio, trimestre):
        return False
    extension = _EXTENSION_HISTORICA.get((anio, trimestre), "zip" if anio <= 2013 else "rar")
    return extension == "rar" and shutil.which("unar") is None


def leer_trimestre(client: EphClient, anio: int, trimestre: int, tipo: str) -> pd.DataFrame:
    """Lectura completa (no solo encabezado) de la base `tipo` de un
    trimestre. Deja pasar `TrimestreNoPublicado`/`UrlDesconocida`/
    `RuntimeError` -- el llamador decide qué hacer con cada una."""
    if es_historico(anio, trimestre):
        archivo = client.descargar_trimestre_historico(anio, trimestre)
        return client.leer_base_historica(archivo, tipo)
    archivo = client.descargar_trimestre(anio, trimestre)
    return client.leer_base(archivo, tipo)


def armonizar(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Nombres de columna a MAYÚSCULAS; claves de vinculación con tipo fijo
    (`DTYPES_CLAVES`). No toca ningún otro valor -- nada de -9/9/etc. se
    recodifica acá."""
    if tipo not in ("individual", "hogar"):
        raise ValueError("tipo debe ser 'individual' u 'hogar'")
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    claves = CLAVES_INDIVIDUAL if tipo == "individual" else CLAVES_HOGAR
    for clave in claves:
        if clave in df.columns:
            df[clave] = df[clave].astype(DTYPES_CLAVES[clave])
    return df


def registrar_inventario(df: pd.DataFrame, anio: int, trimestre: int, tipo: str) -> list[dict]:
    """Una fila por variable presente en `df`, con su `tipo_dato` real (el
    dtype tal cual esté al momento de llamar, no de un esquema declarado)."""
    return [
        {
            "tipo": tipo,
            "anio": anio,
            "trimestre": trimestre,
            "variable": columna,
            "tipo_dato": str(df[columna].dtype),
        }
        for columna in df.columns
    ]


def construir_inventario(registros: list[dict]) -> pd.DataFrame:
    """Formato largo (tidy): una fila por (tipo, variable, año, trimestre)
    con el `tipo_dato` observado."""
    # La ausencia de una variable en un trimestre se deriva de que no hay fila
    # para esa combinación -- no se materializan filas `presente=False`.
    return pd.DataFrame(registros, columns=["tipo", "anio", "trimestre", "variable", "tipo_dato"])


def variables_con_hueco(inventario_df: pd.DataFrame) -> pd.DataFrame:
    """Variables presentes en menos del 100% de los trimestres verificados
    para su `tipo` (individual/hogar)."""
    filas = []
    for tipo, grupo_tipo in inventario_df.groupby("tipo"):
        total_trimestres = grupo_tipo[["anio", "trimestre"]].drop_duplicates().shape[0]
        for variable, grupo_var in grupo_tipo.groupby("variable"):
            presentes = grupo_var[["anio", "trimestre"]].drop_duplicates().shape[0]
            if presentes < total_trimestres:
                filas.append({
                    "tipo": tipo,
                    "variable": variable,
                    "trimestres_presente": presentes,
                    "trimestres_verificados": total_trimestres,
                })
    return pd.DataFrame(filas, columns=["tipo", "variable", "trimestres_presente", "trimestres_verificados"])


def tipos_inconsistentes(inventario_df: pd.DataFrame) -> pd.DataFrame:
    """Variables cuyo `tipo_dato` observado cambia entre los trimestres donde
    están presentes (ej. `IPCF` pasando de tipo entero a float entre eras)."""
    filas = []
    for (tipo, variable), grupo in inventario_df.groupby(["tipo", "variable"]):
        tipos_vistos = sorted(grupo["tipo_dato"].unique())
        if len(tipos_vistos) > 1:
            filas.append({
                "tipo": tipo,
                "variable": variable,
                "tipos_observados": tipos_vistos,
                "cantidad_tipos": len(tipos_vistos),
            })
    return pd.DataFrame(filas, columns=["tipo", "variable", "tipos_observados", "cantidad_tipos"])


def _esqueleto(nombre: str) -> str:
    """Nombre sin dígitos, con guiones bajos colapsados -- para agrupar
    variantes de una misma variable entre trimestres (ej. `V2_M` y
    `V2_02_M` comparten esqueleto `V_M`)."""
    sin_digitos = re.sub(r"\d+", "", nombre)
    return re.sub(r"_+", "_", sin_digitos).strip("_")


def posibles_renombres(inventario_df: pd.DataFrame) -> pd.DataFrame:
    """Heurística de nombre, NO semántica: agrupa por `_esqueleto()` las
    variables que desaparecen/aparecen entre trimestres y propone pares."""
    # Requiere revisión humana -- puede haber falsos positivos (dos variables
    # no relacionadas que reducen al mismo esqueleto) y falsos negativos
    # (renames que cambian de raíz).
    filas = []
    for tipo, grupo_tipo in inventario_df.groupby("tipo"):
        trimestres = sorted(grupo_tipo[["anio", "trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        variables_por_trimestre = {
            t: set(grupo_tipo[(grupo_tipo["anio"] == t[0]) & (grupo_tipo["trimestre"] == t[1])]["variable"])
            for t in trimestres
        }
        for anterior, siguiente in zip(trimestres, trimestres[1:]):
            desaparecidas = variables_por_trimestre[anterior] - variables_por_trimestre[siguiente]
            aparecidas = variables_por_trimestre[siguiente] - variables_por_trimestre[anterior]
            por_esqueleto_aparecidas: dict[str, list[str]] = {}
            for var in aparecidas:
                por_esqueleto_aparecidas.setdefault(_esqueleto(var), []).append(var)
            for var_vieja in desaparecidas:
                candidatas = por_esqueleto_aparecidas.get(_esqueleto(var_vieja), [])
                for var_nueva in candidatas:
                    filas.append({
                        "tipo": tipo,
                        "anio_anterior": anterior[0], "trimestre_anterior": anterior[1],
                        "anio_siguiente": siguiente[0], "trimestre_siguiente": siguiente[1],
                        "variable_anterior": var_vieja,
                        "variable_posible_reemplazo": var_nueva,
                    })
    columnas = ["tipo", "anio_anterior", "trimestre_anterior", "anio_siguiente",
                "trimestre_siguiente", "variable_anterior", "variable_posible_reemplazo"]
    return pd.DataFrame(filas, columns=columnas)


def aglomerados_esperados(conjuntos_por_trimestre: dict[tuple[int, int], set[int]],
                           corte: tuple[int, int] = _CORTE_AGLOMERADOS,
                           umbral: float = 0.9) -> dict[str, set[int]]:
    """Códigos de aglomerado "esperados" por era (pre/post `corte`): los que
    aparecen en al menos `umbral` de los trimestres de esa era."""
    eras: dict[str, list[tuple[int, int]]] = {"pre": [], "post": []}
    for trimestre in conjuntos_por_trimestre:
        eras["pre" if trimestre < corte else "post"].append(trimestre)

    resultado = {}
    for era, trimestres_era in eras.items():
        if not trimestres_era:
            resultado[era] = set()
            continue
        conteo = Counter()
        for t in trimestres_era:
            conteo.update(conjuntos_por_trimestre[t])
        total = len(trimestres_era)
        resultado[era] = {codigo for codigo, veces in conteo.items() if veces / total >= umbral}
    return resultado


def aglomerados_faltantes(codigos_presentes: set[int], anio: int, trimestre: int,
                           esperados: dict[str, set[int]],
                           corte: tuple[int, int] = _CORTE_AGLOMERADOS) -> list[int]:
    era = "pre" if (anio, trimestre) < corte else "post"
    return sorted(esperados.get(era, set()) - codigos_presentes)


def flags_trimestre(anio: int, trimestre: int, aglomerados_faltantes_lista: list[int]) -> dict:
    return {
        "anio": anio,
        "trimestre": trimestre,
        "periodo_intervenido": 2007 <= anio <= 2015,
        "encuesta_telefonica": (anio, trimestre) == (2020, 2),
        "aglomerados_faltantes": list(aglomerados_faltantes_lista),
        "pendiente_rar": pendiente_rar(anio, trimestre),
    }


def tabla_flags(registros: list[dict]) -> pd.DataFrame:
    columnas = ["anio", "trimestre", "periodo_intervenido", "encuesta_telefonica",
                "aglomerados_faltantes", "pendiente_rar"]
    return pd.DataFrame(registros, columns=columnas)
