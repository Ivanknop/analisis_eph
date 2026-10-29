"""Posición relativa de ingreso por adulto equivalente (medición B, sin
línea de pobreza oficial): `y_ae = ITF/unidades_ae` y su posición frente a
la mediana ponderada nacional y de aglomerado-trimestre.
"""
# A diferencia de linea_pobreza.calcular_distancia_lp (medición A), este
# módulo no depende de ninguna canasta regional -- por eso cubre toda la
# serie 2003T3-2025T4 y no solo 2016T2 en adelante.
from __future__ import annotations

import numpy as np
import pandas as pd

from adulto_equivalente import calcular_unidades_ae_hogar
from retropolacion_pobreza import peso_hogar_historico
from trayectoria_distancia import mediana_ponderada

COLUMNAS_HOGAR = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "ITF", "ingreso_no_declarado"]


def _log_seguro(valor: pd.Series) -> pd.Series:
    """`log(valor)`, `NA` donde `valor` es nulo o `<= 0` (evita `-inf` en
    `log(0)`, caso de `ITF` declarado en 0 -- mismo patrón que
    `linea_pobreza._log_seguro`, duplicado a propósito: no se importa un
    símbolo privado entre módulos)."""
    valores = valor.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_valores = np.log(valores)
    log_valores[valores <= 0] = np.nan
    return pd.Series(log_valores, index=valor.index, dtype="Float64")


def calcular_peso_hogar(hogar: pd.DataFrame, individual: pd.DataFrame, es_historico: bool) -> pd.Series:
    """Ponderador de hogar unificado entre eras, indexado por
    `(CODUSU, NRO_HOGAR)`: `PONDIH` en la era regular,
    `retropolacion_pobreza.peso_hogar_historico` en la histórica (no existe
    `PONDIH` antes de 2016)."""
    if es_historico:
        return peso_hogar_historico(individual)
    idx = pd.MultiIndex.from_frame(hogar[["CODUSU", "NRO_HOGAR"]])
    return pd.Series(pd.to_numeric(hogar["PONDIH"], errors="coerce").to_numpy(), index=idx)


def calcular_posicion_relativa(hogar: pd.DataFrame, individual: pd.DataFrame, es_historico: bool) -> pd.DataFrame:
    """Capa 1 de la medición B, por hogar, para un trimestre. Misma
    precondición que `linea_pobreza.calcular_distancia_lp`: las medianas
    (nacional y por aglomerado) se calculan sobre todas las filas recibidas,
    no agrupan internamente por `ANO4`/`TRIMESTRE` -- se llama una vez por
    trimestre."""
    unidades_ae = calcular_unidades_ae_hogar(individual)
    peso_hogar_hogar = calcular_peso_hogar(hogar, individual, es_historico)

    resultado = hogar[COLUMNAS_HOGAR].copy()
    idx = pd.MultiIndex.from_frame(resultado[["CODUSU", "NRO_HOGAR"]])
    resultado["unidades_ae"] = unidades_ae.reindex(idx).to_numpy()
    resultado["peso_hogar"] = peso_hogar_hogar.reindex(idx).to_numpy()
    resultado["era"] = "historica" if es_historico else "regular"
    resultado["hot_deck_disponible"] = pd.array([not es_historico] * len(resultado), dtype="boolean")

    itf = pd.to_numeric(resultado["ITF"], errors="coerce")
    # Misma máscara que linea_pobreza.calcular_distancia_lp aplica a
    # ratio_lp: NA del flag (era histórica, D3) != no respuesta confirmada
    # -- solo `True` enmascara. `ITF==0` declarado NO se enmascara (D3): es
    # un `y_ae=0` real, no no-respuesta -- y por eso sí entra a la mediana.
    no_respuesta_confirmada = (resultado["ingreso_no_declarado"] == True).fillna(False)  # noqa: E712

    resultado["y_ae"] = (itf / resultado["unidades_ae"]).mask(no_respuesta_confirmada)
    resultado["log_y_ae"] = _log_seguro(resultado["y_ae"])

    # mediana_ponderada descarta NA internamente -- al calcularse sobre
    # y_ae ya enmascarado, excluye automáticamente la no respuesta
    # confirmada (y a los hogares sin unidades_ae determinable).
    resultado["mediana_nacional_y_ae"] = mediana_ponderada(resultado["y_ae"], resultado["peso_hogar"])
    mediana_aglomerado = resultado.groupby("AGLOMERADO").apply(
        lambda g: mediana_ponderada(g["y_ae"], g["peso_hogar"]), include_groups=False,
    )
    resultado["mediana_aglomerado_y_ae"] = resultado["AGLOMERADO"].map(mediana_aglomerado)

    resultado["posicion_relativa_nacional"] = resultado["y_ae"] / resultado["mediana_nacional_y_ae"]
    resultado["posicion_relativa_aglomerado"] = resultado["y_ae"] / resultado["mediana_aglomerado_y_ae"]

    return resultado[["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "era",
                       "ITF", "unidades_ae", "y_ae", "log_y_ae", "peso_hogar",
                       "mediana_nacional_y_ae", "mediana_aglomerado_y_ae",
                       "posicion_relativa_nacional", "posicion_relativa_aglomerado",
                       "ingreso_no_declarado", "hot_deck_disponible"]]
