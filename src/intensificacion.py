"""Intensificación de la informalidad (D16/D20): si sube la proporción de activos
precarios de un hogar entre `t` y `t+h`, sobre activos emparejados por sexo+edad."""
# `panel._consumir_offset` es privada -- se duplica acá (restringida a ESTADO∈{1,2})
# en vez de importarse, mismo criterio que usa el resto del repo para mecanismos de
# negocio (la excepción real es `bootstrap.py`, que sí se comparte por ser inferencia
# estadística, no una regla de negocio).
from __future__ import annotations

import numpy as np
import pandas as pd

from panel import TOLERANCIA_EDAD_POR_HORIZONTE

TRATAMIENTOS_INDEPENDIENTE = ("informal", "categoria_propia", "excluir", "cuenta_propia_precario")

# PENDIENTE de confirmación de Iván (D16/D20) -- se implementan y reportan las 4, sin
# elegir ninguna. `cuenta_propia_precario` (Beccaria et al. 2022; Mandelman y Montes
# 2009) trata solo a cuenta propia como precario -- patrón no.
TRATAMIENTO_INDEPENDIENTE_DEFAULT = "categoria_propia"


def _consumir_offset(restante_t: pd.Series, restante_th: pd.Series, offset: int) -> tuple[pd.Series, pd.Series]:
    t_df = restante_t[restante_t > 0].reset_index()
    if t_df.empty:
        return restante_t, restante_th
    t_df["CH06_buscada"] = t_df["CH06"] + offset
    th_df = restante_th[restante_th > 0].reset_index().rename(columns={"CH06": "CH06_buscada", "restante": "restante_th"})
    if th_df.empty:
        return restante_t, restante_th
    combinado = t_df.merge(th_df, on=["CODUSU", "NRO_HOGAR", "CH04", "CH06_buscada"], how="inner")
    if combinado.empty:
        return restante_t, restante_th
    combinado["matched"] = combinado[["restante", "restante_th"]].min(axis=1)
    resta_t = combinado.set_index(["CODUSU", "NRO_HOGAR", "CH04", "CH06"])["matched"]
    restante_t = restante_t.subtract(resta_t, fill_value=0).rename("restante")
    resta_th = combinado.set_index(["CODUSU", "NRO_HOGAR", "CH04", "CH06_buscada"])["matched"]
    resta_th.index.names = restante_th.index.names
    restante_th = restante_th.subtract(resta_th, fill_value=0).rename("restante")
    return restante_t, restante_th


def clasificar_precariedad_persona(individual: pd.DataFrame) -> pd.Series:
    """Por persona activa: `"formal"`/`"informal_asalariado"`/`"patron"`/
    `"cuenta_propia"`/`"desocupado"` según `CAT_OCUP`/`PP07H`/`ESTADO`. `NA`
    si no se puede clasificar."""
    # patrón y cuenta propia se distinguen (no "independiente" combinado) porque D20
    # agrega un tratamiento de independientes que solo trata a cuenta propia como
    # precario (Beccaria et al. 2022; Mandelman y Montes 2009).
    estado = pd.to_numeric(individual["ESTADO"], errors="coerce")
    cat_ocup = pd.to_numeric(individual["CAT_OCUP"], errors="coerce")
    pp07h = pd.to_numeric(individual["PP07H"], errors="coerce")
    categoria = pd.Series(pd.array([pd.NA] * len(individual), dtype="object"), index=individual.index)
    categoria[estado == 2] = "desocupado"
    categoria[cat_ocup == 1] = "patron"
    categoria[cat_ocup == 2] = "cuenta_propia"
    categoria[(cat_ocup == 3) & (pp07h == 2)] = "informal_asalariado"
    categoria[(cat_ocup == 3) & (pp07h == 1)] = "formal"
    return categoria


def _activos(individual: pd.DataFrame, columnas_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    estado = pd.to_numeric(individual["ESTADO"], errors="coerce")
    columnas = ["CODUSU", "NRO_HOGAR", "CH04", "CH06", *columnas_extra]
    activo = individual.loc[estado.isin([1, 2]), columnas].copy()
    activo["categoria_precariedad"] = clasificar_precariedad_persona(individual).loc[activo.index]
    return activo


def _asignar_estado(rel: pd.DataFrame, col_nro_hogar: str, estables_por_bucket: pd.Series,
                     tamanio_original: pd.Series, etiqueta_no_estable: str) -> pd.DataFrame:
    """Por bucket (hogar, `CH04`, `CH06`): `"estable"` a las primeras
    `estables_por_bucket` filas, `etiqueta_no_estable` al resto. Marca
    `empate` si `tamanio_original > 1`."""
    # En un bucket con empate, el orden de asignación es arbitrario -- no identifica
    # una transición real, solo cuenta cuántos activos del bucket están estables
    # para el agregado `share_precario` (la matriz de transición, que sí necesita
    # identidad, excluye los buckets con empate).
    rel = rel.copy()
    orden = rel.groupby(["CODUSU", col_nro_hogar, "CH04", "CH06"]).cumcount()
    idx = pd.MultiIndex.from_frame(
        rel[["CODUSU", col_nro_hogar, "CH04", "CH06"]].rename(columns={col_nro_hogar: "NRO_HOGAR"})
    )
    n_estables = estables_por_bucket.reindex(idx).fillna(0).to_numpy()
    tamanio = tamanio_original.reindex(idx).fillna(0).to_numpy()
    rel["estado_persona"] = np.where(orden.to_numpy() < n_estables, "estable", etiqueta_no_estable)
    rel["empate"] = tamanio > 1
    return rel


def emparejar_activos(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                       pares_hogar: pd.DataFrame, h: int,
                       columnas_extra_th: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
    """Empareja activos por sexo+edad (ventana D10). Devuelve
    `{"t": ..., "th": ...}` con `categoria_precariedad`, `estado_persona`
    (`"estable"`/`"alta"`/`"baja"`) y `empate`."""
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[h]
    claves = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates()

    t_rel = _activos(individual_t).merge(
        claves[["CODUSU", "NRO_HOGAR_t"]].drop_duplicates(),
        left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_t"],
    ).drop(columns=["NRO_HOGAR"])
    th_rel = _activos(individual_th, columnas_extra_th).merge(
        claves[["CODUSU", "NRO_HOGAR_th"]].drop_duplicates(),
        left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_th"],
    ).drop(columns=["NRO_HOGAR"])

    bucket_t = t_rel.groupby(["CODUSU", "NRO_HOGAR_t", "CH04", "CH06"]).size().rename("restante")
    bucket_t.index.names = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]
    bucket_th = th_rel.groupby(["CODUSU", "NRO_HOGAR_th", "CH04", "CH06"]).size().rename("restante")
    bucket_th.index.names = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]
    tamanio_original_t, tamanio_original_th = bucket_t.copy(), bucket_th.copy()

    restante_t, restante_th = bucket_t, bucket_th
    centro = round((minimo + maximo) / 2)
    for offset in sorted(range(minimo, maximo + 1), key=lambda o: abs(o - centro)):
        restante_t, restante_th = _consumir_offset(restante_t, restante_th, offset)

    estables_t = (tamanio_original_t - restante_t).rename("n_estables")
    estables_th = (tamanio_original_th - restante_th).rename("n_estables")

    t_rel = _asignar_estado(t_rel, "NRO_HOGAR_t", estables_t, tamanio_original_t, "baja")
    th_rel = _asignar_estado(th_rel, "NRO_HOGAR_th", estables_th, tamanio_original_th, "alta")
    return {"t": t_rel, "th": th_rel, "claves": claves, "h": h}


def _activos_por_tratamiento(rel: pd.DataFrame, tratamiento_independiente: str) -> pd.DataFrame:
    if tratamiento_independiente == "excluir":
        return rel[~rel["categoria_precariedad"].isin(["patron", "cuenta_propia"])]
    return rel


def _es_precario(categoria_precariedad: pd.Series, tratamiento_independiente: str) -> pd.Series:
    precario = categoria_precariedad.isin(["informal_asalariado", "desocupado"])
    if tratamiento_independiente == "informal":
        precario = precario | categoria_precariedad.isin(["patron", "cuenta_propia"])
    elif tratamiento_independiente == "cuenta_propia_precario":
        precario = precario | (categoria_precariedad == "cuenta_propia")
    return precario


def calcular_share_precario(emparejados: dict[str, pd.DataFrame],
                             tratamiento_independiente: str = TRATAMIENTO_INDEPENDIENTE_DEFAULT) -> pd.DataFrame:
    """Por hogar-par: `share_precario_t`/`_th` sobre activos estables (según
    `tratamiento_independiente`) y `n_activos_estables_t`/`_th`. `NA` sin
    activos estables en esa punta."""
    if tratamiento_independiente not in TRATAMIENTOS_INDEPENDIENTE:
        raise ValueError(f"tratamiento_independiente debe ser uno de {TRATAMIENTOS_INDEPENDIENTE}")

    filas = []
    for lado, col_nro_hogar in (("t", "NRO_HOGAR_t"), ("th", "NRO_HOGAR_th")):
        rel = emparejados[lado]
        estables = rel[rel["estado_persona"] == "estable"]
        estables = _activos_por_tratamiento(estables, tratamiento_independiente)
        precario = _es_precario(estables["categoria_precariedad"], tratamiento_independiente)
        conteo = estables.assign(_precario=precario).groupby(["CODUSU", col_nro_hogar]).agg(
            n_activos_estables=("categoria_precariedad", "size"), n_precario=("_precario", "sum"),
        ).reset_index().rename(columns={col_nro_hogar: "NRO_HOGAR"})
        conteo["lado"] = lado
        filas.append(conteo)

    largo = pd.concat(filas, ignore_index=True)
    ancho = largo.pivot_table(index=["CODUSU", "NRO_HOGAR"], columns="lado",
                               values=["n_activos_estables", "n_precario"], fill_value=0)
    ancho.columns = [f"{col}_{lado}" for col, lado in ancho.columns]
    ancho = ancho.reset_index()
    for lado in ("t", "th"):
        for col in (f"n_activos_estables_{lado}", f"n_precario_{lado}"):
            if col not in ancho.columns:
                ancho[col] = 0
        n, p = ancho[f"n_activos_estables_{lado}"], ancho[f"n_precario_{lado}"]
        ancho[f"share_precario_{lado}"] = np.where(n > 0, p / n.replace(0, np.nan), np.nan)

    resultado = emparejados["claves"].copy()
    resultado = resultado.merge(ancho.rename(columns={"NRO_HOGAR": "NRO_HOGAR_t"}),
                                 on=["CODUSU", "NRO_HOGAR_t"], how="left")
    for col in ("n_activos_estables_t", "n_activos_estables_th", "n_precario_t", "n_precario_th"):
        if col not in resultado.columns:
            resultado[col] = 0
        resultado[col] = resultado[col].fillna(0).astype(int)
    for col in ("share_precario_t", "share_precario_th"):
        if col not in resultado.columns:
            resultado[col] = np.nan
    return resultado[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "n_activos_estables_t",
                       "n_activos_estables_th", "share_precario_t", "share_precario_th"]]


def calcular_intensificacion(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                              pares_hogar: pd.DataFrame, h: int,
                              tratamiento_independiente: str = TRATAMIENTO_INDEPENDIENTE_DEFAULT) -> pd.DataFrame:
    """Por hogar-par: `share_precario_t`/`_th`, `n_activos_t` (D16), `estado`
    (`"intensifica"`/`"no_intensifica"`/`"sin_activos_estables"`)."""
    emparejados = emparejar_activos(individual_t, individual_th, pares_hogar, h)
    share = calcular_share_precario(emparejados, tratamiento_independiente)

    activos_t_tratamiento = _activos_por_tratamiento(emparejados["t"], tratamiento_independiente)
    n_activos_t = activos_t_tratamiento.groupby(["CODUSU", "NRO_HOGAR_t"]).size().rename("n_activos_t")
    share = share.merge(n_activos_t, on=["CODUSU", "NRO_HOGAR_t"], how="left")
    share["n_activos_t"] = share["n_activos_t"].fillna(0).astype(int)

    sin_activos_estables = (share["n_activos_estables_t"] == 0) | (share["n_activos_estables_th"] == 0)
    intensifica = share["share_precario_th"] > share["share_precario_t"]

    estado = pd.Series("no_intensifica", index=share.index, dtype="object")
    estado[intensifica] = "intensifica"
    estado[sin_activos_estables] = "sin_activos_estables"
    share["intensificacion"] = intensifica.where(~sin_activos_estables)
    share["estado"] = estado
    return share


def matriz_transicion_persona(emparejados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transición de `categoria_precariedad` persona a persona entre activos
    estables, solo para buckets sin empate (D16/D20)."""
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[emparejados["h"]]
    columnas_t = ["CODUSU", "NRO_HOGAR_t", "CH04", "CH06", "categoria_precariedad"]
    columnas_th = ["CODUSU", "NRO_HOGAR_th", "CH04", "CH06", "categoria_precariedad"]
    restante_t = emparejados["t"][(emparejados["t"]["estado_persona"] == "estable") & ~emparejados["t"]["empate"]][columnas_t].copy()
    restante_th = emparejados["th"][(emparejados["th"]["estado_persona"] == "estable") & ~emparejados["th"]["empate"]][columnas_th].copy()
    restante_t["_id"] = range(len(restante_t))
    restante_th["_id"] = range(len(restante_th))

    # Cada bucket (hogar, CH04, CH06) ya tiene como máximo 1 persona de cada lado
    # (los empates se filtraron arriba) -- el merge por offset no puede producir
    # más de un candidato por persona, así que no hace falta resolver ambigüedad acá.
    pares = []
    centro = round((minimo + maximo) / 2)
    for offset in sorted(range(minimo, maximo + 1), key=lambda o: abs(o - centro)):
        candidatos = restante_t.merge(
            restante_th, left_on=["CODUSU", "NRO_HOGAR_t", "CH04"], right_on=["CODUSU", "NRO_HOGAR_th", "CH04"],
            suffixes=("_t", "_th"),
        )
        candidatos = candidatos[candidatos["CH06_th"] == candidatos["CH06_t"] + offset]
        if candidatos.empty:
            continue
        pares.append(candidatos)
        restante_t = restante_t[~restante_t["_id"].isin(candidatos["_id_t"])]
        restante_th = restante_th[~restante_th["_id"].isin(candidatos["_id_th"])]

    columnas_resultado = ["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "CH04",
                           "categoria_precariedad_t", "categoria_precariedad_th"]
    if not pares:
        return pd.DataFrame(columns=columnas_resultado).rename(
            columns={"categoria_precariedad_t": "categoria_t", "categoria_precariedad_th": "categoria_th"})
    resultado = pd.concat(pares, ignore_index=True)[columnas_resultado]
    return resultado.rename(columns={"categoria_precariedad_t": "categoria_t", "categoria_precariedad_th": "categoria_th"})
