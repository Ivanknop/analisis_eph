"""Distancia de cada hogar a la línea de pobreza/indigencia (D33-D34): es
el objeto de estudio -- `pobre`/`indigente` quedan como derivados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from adulto_equivalente import calcular_unidades_ae_hogar

COLUMNAS_HOGAR = ["CODUSU", "NRO_HOGAR", "REGION", "AGLOMERADO", "ITF", "ingreso_no_declarado"]


def _log_seguro(ratio: pd.Series) -> pd.Series:
    """`log(ratio)`, `NA` donde `ratio` es nulo o `<= 0` (evita `-inf` en
    `log(0)`, caso `itf_cero_declarado`)."""
    valores = ratio.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_valores = np.log(valores)
    log_valores[valores <= 0] = np.nan
    return pd.Series(log_valores, index=ratio.index, dtype="Float64")


def calcular_distancia_lp(hogar: pd.DataFrame, individual: pd.DataFrame,
                           canasta_region_trimestre: pd.DataFrame) -> pd.DataFrame:
    """Distancia de cada hogar a la línea de pobreza/indigencia, por hogar
    y trimestre."""
    unidades_ae = calcular_unidades_ae_hogar(individual)

    resultado = hogar[COLUMNAS_HOGAR].copy()
    idx = pd.MultiIndex.from_frame(resultado[["CODUSU", "NRO_HOGAR"]])
    resultado["unidades_ae"] = unidades_ae.reindex(idx).to_numpy()

    canasta = canasta_region_trimestre.set_index("REGION")[["cba_ae", "cbt_ae"]]
    resultado = resultado.merge(canasta, left_on="REGION", right_index=True, how="left")

    resultado["linea_indigencia"] = resultado["cba_ae"] * resultado["unidades_ae"]
    resultado["linea_pobreza"] = resultado["cbt_ae"] * resultado["unidades_ae"]

    itf = pd.to_numeric(resultado["ITF"], errors="coerce")
    # NA del flag (era histórica, D3) != no respuesta confirmada -- solo
    # `True` anula la distancia, `NA` la deja calcularse con el ITF tal cual viene.
    no_respuesta_confirmada = (resultado["ingreso_no_declarado"] == True).fillna(False)  # noqa: E712

    ratio_lp = (itf / resultado["linea_pobreza"]).mask(no_respuesta_confirmada)
    ratio_li = (itf / resultado["linea_indigencia"]).mask(no_respuesta_confirmada)
    resultado["ratio_lp"] = ratio_lp
    resultado["ratio_li"] = ratio_li
    resultado["brecha_relativa_lp"] = ratio_lp - 1
    resultado["brecha_relativa_li"] = ratio_li - 1
    resultado["log_ratio_lp"] = _log_seguro(ratio_lp)
    resultado["log_ratio_li"] = _log_seguro(ratio_li)

    pobre = (ratio_lp < 1).astype("boolean")
    pobre[ratio_lp.isna()] = pd.NA
    indigente = (ratio_li < 1).astype("boolean")
    indigente[ratio_li.isna()] = pd.NA
    resultado["pobre"] = pobre
    resultado["indigente"] = indigente

    ingreso_no_declarado = resultado["ingreso_no_declarado"].astype("boolean")
    es_cero = (itf == 0).astype("boolean")
    es_cero[itf.isna()] = pd.NA
    resultado["itf_cero_declarado"] = es_cero & ~ingreso_no_declarado

    return resultado[["CODUSU", "NRO_HOGAR", "REGION", "AGLOMERADO", "ITF", "unidades_ae",
                       "cba_ae", "cbt_ae", "linea_indigencia", "linea_pobreza",
                       "ratio_lp", "brecha_relativa_lp", "log_ratio_lp",
                       "ratio_li", "brecha_relativa_li", "log_ratio_li",
                       "pobre", "indigente", "ingreso_no_declarado", "itf_cero_declarado"]]
