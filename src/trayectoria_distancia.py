"""Trayectorias de distancia a la línea de pobreza (D33-D34): variación
entre `t` y `t+h` de `linea_pobreza.calcular_distancia_lp`, sobre los pares
del panel ya vinculados (D9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# No usa `src/trayectorias.py` (D12/D13, intocable esta ronda): esto encadena
# un solo par por vez, no las 4 visitas.
COLUMNAS_DISTANCIA = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "REGION", "AGLOMERADO", "era",
                       "ITF", "PONDIH", "unidades_ae", "cbt_ae", "ratio_lp", "log_ratio_lp", "pobre"]


def _preparar_lado(distancia: pd.DataFrame, sufijo: str) -> pd.DataFrame:
    renombradas = {c: f"{c}_{sufijo}" for c in COLUMNAS_DISTANCIA if c != "CODUSU"}
    renombradas["NRO_HOGAR"] = f"NRO_HOGAR_{sufijo}"
    return distancia[COLUMNAS_DISTANCIA].rename(columns=renombradas)


def calcular_variacion_simetrica(ratio_t: pd.Series, ratio_th: pd.Series) -> pd.Series:
    """Variación porcentual simétrica `(th - t) / ((|t| + |th|) / 2)` -- admite
    `ratio=0` en un lado; `NA` si ambos lados son `0` (0/0, caso raro)."""
    a, b = ratio_t.astype(float), ratio_th.astype(float)
    denominador = (a.abs() + b.abs()) / 2
    return ((b - a) / denominador.replace(0, np.nan)).astype("Float64")


def descomponer_variacion(itf_t: pd.Series, itf_th: pd.Series, cbt_ae_t: pd.Series, cbt_ae_th: pd.Series,
                           unidades_ae_t: pd.Series, unidades_ae_th: pd.Series) -> pd.DataFrame:
    """`aporte_ingreso + aporte_precios + aporte_composicion == variacion_log`,
    solo definido con `ITF > 0` en ambas puntas (si no, `log(ITF)` no existe)."""
    definido = (itf_t.astype(float) > 0) & (itf_th.astype(float) > 0)

    def _log_diff(a: pd.Series, b: pd.Series) -> pd.Series:
        resultado = pd.Series(np.nan, index=a.index, dtype="Float64")
        resultado[definido] = np.log(b[definido].astype(float)) - np.log(a[definido].astype(float))
        return resultado

    return pd.DataFrame({
        "aporte_ingreso": _log_diff(itf_t, itf_th),
        "aporte_precios": -_log_diff(cbt_ae_t, cbt_ae_th),
        "aporte_composicion": -_log_diff(unidades_ae_t, unidades_ae_th),
    })


def clasificar_direccion(variacion_simetrica: pd.Series, umbral: float = 0.10) -> pd.Series:
    """`"mejora"` (`variacion > umbral`), `"empeora"` (`< -umbral`), `"estable"`
    (dentro de `±umbral`) -- default `umbral=0.10` (D33, sensibilidad ±5/±20%
    reportada aparte, no en esta función)."""
    resultado = pd.Series(pd.NA, index=variacion_simetrica.index, dtype="object")
    valido = variacion_simetrica.notna()
    resultado[valido & (variacion_simetrica > umbral)] = "mejora"
    resultado[valido & (variacion_simetrica < -umbral)] = "empeora"
    resultado[valido & variacion_simetrica.between(-umbral, umbral)] = "estable"
    return resultado


def clasificar_transicion(pobre_t: pd.Series, pobre_th: pd.Series) -> pd.Series:
    """`"cae"` (no pobre -> pobre), `"sale"` (pobre -> no pobre), `"sigue_pobre"`,
    `"sigue_no_pobre"` -- `NA` si algún lado de `pobre` es nulo."""
    pt, pth = pobre_t.astype("boolean"), pobre_th.astype("boolean")
    valido = (pt.notna() & pth.notna()).to_numpy(dtype=bool)
    pt_np, pth_np = pt.fillna(False).to_numpy(dtype=bool), pth.fillna(False).to_numpy(dtype=bool)

    resultado = pd.Series(pd.NA, index=pt.index, dtype="object")
    resultado[valido & ~pt_np & pth_np] = "cae"
    resultado[valido & pt_np & ~pth_np] = "sale"
    resultado[valido & pt_np & pth_np] = "sigue_pobre"
    resultado[valido & ~pt_np & ~pth_np] = "sigue_no_pobre"
    return resultado


def mediana_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    """Mediana ponderada (sin interpolar -- el valor donde el peso acumulado
    cruza la mitad del total). `NaN` si no hay ningún par válido."""
    valido = valores.notna() & pesos.notna() & (pesos.astype(float) > 0)
    if not valido.any():
        return np.nan
    v = valores[valido].astype(float).to_numpy()
    w = pesos[valido].astype(float).to_numpy()
    orden = np.argsort(v)
    v, w = v[orden], w[orden]
    acumulado = np.cumsum(w)
    idx = np.searchsorted(acumulado, acumulado[-1] / 2)
    return float(v[min(idx, len(v) - 1)])


def agregar_componente_comun_y_propio(pares: pd.DataFrame, columna_variacion: str, columna_peso: str,
                                       columnas_grupo: list[str], sufijo: str) -> pd.DataFrame:
    """Mediana ponderada de `columna_variacion` por `columnas_grupo` como
    `variacion_comun_{sufijo}`, y `variacion_propia_{sufijo}` (el residuo)."""
    # variacion_comun_* usa información de t+h (agrega sobre el resultado):
    # sirve para descomponer el objetivo en componente de época/zona vs.
    # propia del hogar, nunca como feature de un modelo que predice adelante.
    comunes = pares.groupby(columnas_grupo, dropna=False).apply(
        lambda g: mediana_ponderada(g[columna_variacion], g[columna_peso]), include_groups=False,
    ).rename(f"variacion_comun_{sufijo}")
    resultado = pares.merge(comunes, on=columnas_grupo, how="left")
    resultado[f"variacion_propia_{sufijo}"] = resultado[columna_variacion] - resultado[f"variacion_comun_{sufijo}"]
    return resultado


def agregar_tendencia_previa(pares: pd.DataFrame, pares_h1_t_menos_1: pd.DataFrame | None,
                              pares_h1_t_menos_4: pd.DataFrame | None) -> pd.DataFrame:
    """Agrega `variacion_simetrica_previa` (candidata de feature, nunca
    objetivo) y `gap_previo`: última observación anterior disponible del
    mismo hogar, no solo `t-1`."""
    # Por la rotación 2-2-2 (D9/D12), un hogar solo puede tener una
    # observación anterior en 2 posiciones posibles, nunca las dos a la vez:
    # t-1 (gap_previo=0, 2ª visita en t) o t-4 (gap_previo=3, 3ª visita, tras
    # el hueco de 2 trimestres). En la 1ª visita (recién entra) ninguna
    # existe -- NA estructural, no un dato faltante.
    resultado = pares.copy()
    resultado["variacion_simetrica_previa"] = pd.array([pd.NA] * len(pares), dtype="Float64")
    resultado["gap_previo"] = pd.array([pd.NA] * len(pares), dtype="Int8")

    if pares_h1_t_menos_1 is not None and len(pares_h1_t_menos_1):
        candidato_1 = pares_h1_t_menos_1[
            ["CODUSU", "NRO_HOGAR_th", "ANO4_th", "TRIMESTRE_th", "variacion_simetrica"]
        ].rename(columns={"NRO_HOGAR_th": "NRO_HOGAR_t", "ANO4_th": "ANO4_t", "TRIMESTRE_th": "TRIMESTRE_t",
                           "variacion_simetrica": "_previa_1"})
        resultado = resultado.merge(candidato_1, on=["CODUSU", "NRO_HOGAR_t", "ANO4_t", "TRIMESTRE_t"], how="left")
        encontrado = resultado["_previa_1"].notna()
        resultado.loc[encontrado, "variacion_simetrica_previa"] = resultado.loc[encontrado, "_previa_1"]
        resultado.loc[encontrado, "gap_previo"] = 0
        resultado = resultado.drop(columns="_previa_1")

    if pares_h1_t_menos_4 is not None and len(pares_h1_t_menos_4):
        # `pares_h1_t_menos_4` ya viene acotado a la partición de origen `t-4`
        # (la llama quien orquesta, no busca en todo el historial) -- se
        # empareja por identidad de hogar en el origen (`CODUSU, NRO_HOGAR_t,
        # AGLOMERADO_t`), sin exigir igualdad de trimestre: el origen de ese
        # tramo es `t-4` por construcción, no `t`.
        candidato_4 = pares_h1_t_menos_4[
            ["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t", "variacion_simetrica"]
        ].rename(columns={"variacion_simetrica": "_previa_4"})
        resultado = resultado.merge(candidato_4, on=["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t"], how="left")
        sin_match_previo = resultado["variacion_simetrica_previa"].isna() & resultado["_previa_4"].notna()
        resultado.loc[sin_match_previo, "variacion_simetrica_previa"] = resultado.loc[sin_match_previo, "_previa_4"]
        resultado.loc[sin_match_previo, "gap_previo"] = 3
        resultado = resultado.drop(columns="_previa_4")

    return resultado


def construir_pares_distancia(distancia_t: pd.DataFrame, distancia_th: pd.DataFrame,
                               pares_hogar: pd.DataFrame) -> pd.DataFrame:
    """Par hogar `t`->`t+h` con distancia en ambas puntas y sus variaciones
    (D33/D34)."""
    resultado = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "identidad_confirmada"]].merge(
        _preparar_lado(distancia_t, "t"), on=["CODUSU", "NRO_HOGAR_t"], how="left")
    resultado = resultado.merge(_preparar_lado(distancia_th, "th"), on=["CODUSU", "NRO_HOGAR_th"], how="left")

    resultado["variacion_simetrica"] = calcular_variacion_simetrica(resultado["ratio_lp_t"], resultado["ratio_lp_th"])
    resultado["variacion_log"] = resultado["log_ratio_lp_th"] - resultado["log_ratio_lp_t"]
    resultado["diferencia_absoluta"] = resultado["ratio_lp_th"].astype(float) - resultado["ratio_lp_t"].astype(float)

    descomposicion = descomponer_variacion(
        resultado["ITF_t"], resultado["ITF_th"], resultado["cbt_ae_t"], resultado["cbt_ae_th"],
        resultado["unidades_ae_t"], resultado["unidades_ae_th"])
    resultado = pd.concat([resultado, descomposicion], axis=1)

    resultado["direccion"] = clasificar_direccion(resultado["variacion_simetrica"])
    resultado["transicion"] = clasificar_transicion(resultado["pobre_t"], resultado["pobre_th"])
    # `identidad_confirmada` puede ser `False` (otro hogar en la misma vivienda,
    # D10) sin que la fila se descarte -- `par_valido` marca cuáles sirven para
    # medir variación real, sin perder las demás (quedan para auditar aparte).
    resultado["par_valido"] = (resultado["identidad_confirmada"] == True) & resultado["variacion_simetrica"].notna()  # noqa: E712
    return resultado
