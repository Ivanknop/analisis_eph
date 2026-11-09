"""Distancia de cada hogar a la línea de pobreza/indigencia: es
el objeto de estudio -- `pobre`/`indigente` quedan como derivados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from adulto_equivalente import calcular_unidades_adulto_equivalente_hogar

COLUMNAS_HOGAR = ["CODUSU", "NRO_HOGAR", "REGION", "AGLOMERADO", "ITF", "ingreso_no_declarado"]


def _log_seguro(ratio: pd.Series) -> pd.Series:
    """`log(ratio)`, `NA` donde `ratio` es nulo o `<= 0` (evita `-inf` en
    `log(0)`, caso `itf_cero_declarado`)."""
    valores = ratio.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_valores = np.log(valores)
    log_valores[valores <= 0] = np.nan
    return pd.Series(log_valores, index=ratio.index, dtype="Float64")


def tasa_ponderada(df: pd.DataFrame, columna: str, peso: str) -> float:
    """Promedio ponderado de `columna` (booleana o 0/1) por `peso`, en
    porcentaje -- descarta filas con `columna` o `peso` nulos."""
    valido = df[columna].notna() & df[peso].notna()
    return 100 * (df.loc[valido, columna].astype(float) * df.loc[valido, peso]).sum() / df.loc[valido, peso].sum()


def calcular_distancia_linea_pobreza(hogar: pd.DataFrame, individual: pd.DataFrame,
                           canasta_region_trimestre: pd.DataFrame) -> pd.DataFrame:
    """Distancia de cada hogar a la línea de pobreza/indigencia, por hogar
    y trimestre."""
    unidades_adulto_equivalente = calcular_unidades_adulto_equivalente_hogar(individual)

    resultado = hogar[COLUMNAS_HOGAR].copy()
    idx = pd.MultiIndex.from_frame(resultado[["CODUSU", "NRO_HOGAR"]])
    resultado["unidades_adulto_equivalente"] = unidades_adulto_equivalente.reindex(idx).to_numpy()

    canasta = canasta_region_trimestre.set_index("REGION")[["cba_adulto_equivalente", "cbt_adulto_equivalente"]]
    resultado = resultado.merge(canasta, left_on="REGION", right_index=True, how="left")

    resultado["linea_indigencia"] = resultado["cba_adulto_equivalente"] * resultado["unidades_adulto_equivalente"]
    resultado["linea_pobreza"] = resultado["cbt_adulto_equivalente"] * resultado["unidades_adulto_equivalente"]

    itf = pd.to_numeric(resultado["ITF"], errors="coerce")
    # NA del flag (era histórica) != no respuesta confirmada -- solo
    # `True` anula la distancia, `NA` la deja calcularse con el ITF tal cual viene.
    no_respuesta_confirmada = (resultado["ingreso_no_declarado"] == True).fillna(False)  # noqa: E712

    ratio_linea_pobreza = (itf / resultado["linea_pobreza"]).mask(no_respuesta_confirmada)
    ratio_linea_indigencia = (itf / resultado["linea_indigencia"]).mask(no_respuesta_confirmada)
    resultado["ratio_linea_pobreza"] = ratio_linea_pobreza
    resultado["ratio_linea_indigencia"] = ratio_linea_indigencia
    resultado["brecha_relativa_linea_pobreza"] = ratio_linea_pobreza - 1
    resultado["brecha_relativa_linea_indigencia"] = ratio_linea_indigencia - 1
    resultado["log_ratio_linea_pobreza"] = _log_seguro(ratio_linea_pobreza)
    resultado["log_ratio_linea_indigencia"] = _log_seguro(ratio_linea_indigencia)

    pobre = (ratio_linea_pobreza < 1).astype("boolean")
    pobre[ratio_linea_pobreza.isna()] = pd.NA
    indigente = (ratio_linea_indigencia < 1).astype("boolean")
    indigente[ratio_linea_indigencia.isna()] = pd.NA
    resultado["pobre"] = pobre
    resultado["indigente"] = indigente

    ingreso_no_declarado = resultado["ingreso_no_declarado"].astype("boolean")
    es_cero = (itf == 0).astype("boolean")
    es_cero[itf.isna()] = pd.NA
    resultado["itf_cero_declarado"] = es_cero & ~ingreso_no_declarado

    return resultado[["CODUSU", "NRO_HOGAR", "REGION", "AGLOMERADO", "ITF", "unidades_adulto_equivalente",
                       "cba_adulto_equivalente", "cbt_adulto_equivalente", "linea_indigencia", "linea_pobreza",
                       "ratio_linea_pobreza", "brecha_relativa_linea_pobreza", "log_ratio_linea_pobreza",
                       "ratio_linea_indigencia", "brecha_relativa_linea_indigencia", "log_ratio_linea_indigencia",
                       "pobre", "indigente", "ingreso_no_declarado", "itf_cero_declarado"]]
