"""Trayectorias de posición relativa de ingreso (medición B): variación
entre `t` y `t+h` de `posicion_relativa.calcular_posicion_relativa`, sobre
los pares del panel ya vinculados (`data/03_panel`, todas las eras).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trayectoria_distancia import agregar_componente_comun_y_propio, calcular_variacion_simetrica

COLUMNAS_POSICION = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "era",
                      "ITF", "unidades_ae", "y_ae", "log_y_ae", "peso_hogar",
                      "posicion_relativa_nacional", "posicion_relativa_aglomerado",
                      "ingreso_no_declarado", "hot_deck_disponible"]


def _preparar_lado(posicion: pd.DataFrame, sufijo: str) -> pd.DataFrame:
    renombradas = {c: f"{c}_{sufijo}" for c in COLUMNAS_POSICION if c != "CODUSU"}
    renombradas["NRO_HOGAR"] = f"NRO_HOGAR_{sufijo}"
    return posicion[COLUMNAS_POSICION].rename(columns=renombradas)


def diferencia_log_y_ae(y_ae_t: pd.Series, y_ae_th: pd.Series) -> pd.Series:
    """`log(y_ae_th) - log(y_ae_t)`, `NA` si algún lado es nulo o `<= 0`."""
    a = pd.to_numeric(y_ae_t, errors="coerce")
    b = pd.to_numeric(y_ae_th, errors="coerce")
    definido = (a > 0) & (b > 0)
    resultado = pd.Series(np.nan, index=a.index, dtype="Float64")
    resultado[definido] = np.log(b[definido].astype(float)) - np.log(a[definido].astype(float))
    return resultado


def construir_pares_relativa(posicion_t: pd.DataFrame, posicion_th: pd.DataFrame,
                              pares_hogar: pd.DataFrame) -> pd.DataFrame:
    """Par hogar `t`->`t+h` de la medición B: une la Capa 1 en ambas puntas
    sobre `pares_hogar` (de `data/03_panel`), calcula `variacion_log_b`, la
    variación simétrica de la posición relativa, el componente común/propio
    y los filtros `par_valido_log`/`par_valido_simetrica`."""
    resultado = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "identidad_confirmada"]].merge(
        _preparar_lado(posicion_t, "t"), on=["CODUSU", "NRO_HOGAR_t"], how="left")
    resultado = resultado.merge(_preparar_lado(posicion_th, "th"), on=["CODUSU", "NRO_HOGAR_th"], how="left")

    resultado["variacion_log_b"] = diferencia_log_y_ae(resultado["y_ae_t"], resultado["y_ae_th"])
    resultado["variacion_simetrica_posicion_nacional_b"] = calcular_variacion_simetrica(
        resultado["posicion_relativa_nacional_t"], resultado["posicion_relativa_nacional_th"])
    resultado["variacion_simetrica_posicion_aglomerado_b"] = calcular_variacion_simetrica(
        resultado["posicion_relativa_aglomerado_t"], resultado["posicion_relativa_aglomerado_th"])

    # variacion_comun_b queda en pesos corrientes (incluye inflación local,
    # no se cancela como en variacion_propia_b) -- solo insumo intermedio,
    # nunca se interpreta como tendencia de empobrecimiento/mejora.
    resultado = agregar_componente_comun_y_propio(
        resultado, "variacion_log_b", "peso_hogar_t", ["ANO4_t", "TRIMESTRE_t", "AGLOMERADO_t"], "b")

    # Dos filtros, uno por objetivo -- mismo criterio y mismos nombres que
    # `trayectoria_distancia.construir_pares_distancia` (A): cada variación
    # exige que SU PROPIA variación no sea nula (`ITF=0`/`y_ae=0` admite la
    # simétrica, no el log).
    resultado["par_valido_log"] = (
        (resultado["identidad_confirmada"] == True) & resultado["variacion_log_b"].notna()  # noqa: E712
    )
    resultado["par_valido_simetrica"] = (
        (resultado["identidad_confirmada"] == True)  # noqa: E712
        & resultado["variacion_simetrica_posicion_aglomerado_b"].notna()
    )
    return resultado


def contar_exclusiones_y_ae(pares: pd.DataFrame) -> dict:
    """Desagrega por qué `variacion_log_b` es `NA`: `y_ae` nulo
    (`unidades_ae`/`ITF`/no respuesta) vs. `y_ae<=0` entre los no nulos, en
    cada punta del par."""
    y_ae_t = pd.to_numeric(pares["y_ae_t"], errors="coerce")
    y_ae_th = pd.to_numeric(pares["y_ae_th"], errors="coerce")
    return {
        "n_pares": len(pares),
        "n_sin_variacion_log_b": int(pares["variacion_log_b"].isna().sum()),
        "n_y_ae_t_nulo": int(y_ae_t.isna().sum()),
        "n_y_ae_th_nulo": int(y_ae_th.isna().sum()),
        "n_y_ae_t_no_positivo": int(((y_ae_t <= 0) & y_ae_t.notna()).sum()),
        "n_y_ae_th_no_positivo": int(((y_ae_th <= 0) & y_ae_th.notna()).sum()),
    }
