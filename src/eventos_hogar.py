"""Eventos de composición que distorsionan la medición de cambios en la
situación laboral del hogar: menores que entran a trabajar, jubilaciones
en hogares mixtos, tipología de perceptores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from panel import TOLERANCIA_EDAD_POR_HORIZONTE
from perfil_hogar import clasificar_precariedad_persona, clasificar_situacion_escolar, kleene_any_hogar, to_bool

# `CAT_INAC`/`CH10` se traen de `01_armonizado` ad hoc (fuera de
# `ESQUEMA_NUCLEO`), como `NIVEL_ED` en `pipeline_meta`.

# A revisar.
EDAD_CORTE_MENOR = 18


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


def calcular_se_jubila(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                        pares_hogar: pd.DataFrame, h: int) -> pd.DataFrame:
    """Por hogar-par: cantidad de activos de `t` para los que existe en
    `t+h` alguien del mismo hogar/sexo recién jubilado, edad dentro de la
    ventana de tolerancia por horizonte."""
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[h]
    claves = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates()

    estado_t = pd.to_numeric(individual_t["ESTADO"], errors="coerce")
    activos_t = individual_t.loc[estado_t.isin([1, 2]), ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]].merge(
        claves[["CODUSU", "NRO_HOGAR_t"]].drop_duplicates(),
        left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_t"])

    cat_inac_th = pd.to_numeric(individual_th["CAT_INAC"], errors="coerce")
    jubilados_th = individual_th.loc[cat_inac_th == 1, ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]].rename(
        columns={"CH06": "CH06_th"}).merge(
        claves[["CODUSU", "NRO_HOGAR_th"]].drop_duplicates(),
        left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_th"])

    # Excluir de jubilados_th a quien ya tenía CAT_INAC==1 en t (mismo hogar, sexo,
    # edad en ventana D6) -- medido en datos reales (T0 2026-10-05): ~1.3% de los
    # matches usaban como candidato a un jubilado preexistente, no a quien realmente
    # transicionó (un activo de t nunca puede ser ese candidato -- CAT_INAC==1
    # implica ESTADO∉{1,2} -- así que no se pierde ningún caso real).
    cat_inac_t = pd.to_numeric(individual_t["CAT_INAC"], errors="coerce")
    ya_jubilados_t = individual_t.loc[cat_inac_t == 1, ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]].rename(
        columns={"CH06": "CH06_t_previo"}).merge(
        claves[["CODUSU", "NRO_HOGAR_t"]].drop_duplicates(),
        left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_t"])
    ya_era_jubilado = jubilados_th.merge(
        ya_jubilados_t, left_on=["CODUSU", "NRO_HOGAR_th", "CH04"], right_on=["CODUSU", "NRO_HOGAR_t", "CH04"])
    ya_era_jubilado["diferencia"] = ya_era_jubilado["CH06_th"] - ya_era_jubilado["CH06_t_previo"]
    ya_era_jubilado = ya_era_jubilado[ya_era_jubilado["diferencia"].between(minimo, maximo)]
    claves_a_excluir = ya_era_jubilado[["CODUSU", "NRO_HOGAR_th", "CH04", "CH06_th"]].drop_duplicates()
    jubilados_th = jubilados_th.merge(
        claves_a_excluir.assign(_ya_jubilado=True), on=["CODUSU", "NRO_HOGAR_th", "CH04", "CH06_th"], how="left")
    jubilados_th = jubilados_th[jubilados_th["_ya_jubilado"].isna()].drop(columns=["_ya_jubilado"])

    candidatos = activos_t.merge(jubilados_th, left_on=["CODUSU", "NRO_HOGAR_t", "CH04"],
                                  right_on=["CODUSU", "NRO_HOGAR_th", "CH04"])
    candidatos["diferencia"] = candidatos["CH06_th"] - candidatos["CH06"]
    candidatos = candidatos[candidatos["diferencia"].between(minimo, maximo)]
    bucket_con_jubilacion = candidatos[["CODUSU", "NRO_HOGAR_t", "CH04", "CH06"]].drop_duplicates()
    n_se_jubila = bucket_con_jubilacion.groupby(["CODUSU", "NRO_HOGAR_t"]).size().rename("n_se_jubila")

    resultado = claves.copy()
    resultado = resultado.merge(n_se_jubila, on=["CODUSU", "NRO_HOGAR_t"], how="left")
    resultado["n_se_jubila"] = resultado["n_se_jubila"].fillna(0).astype(int)
    return resultado[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "n_se_jubila"]]


def _clasificar_tipo_alta(altas_menor: pd.DataFrame, individual_t: pd.DataFrame,
                           pares_hogar: pd.DataFrame, h: int) -> pd.Series:
    """`"inactivo_a_activo"` si ya había alguien del mismo hogar/sexo inactivo
    en `t` con edad en la ventana D6; `"entra_activo"` si la persona es
    nueva en el hogar."""
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[h]
    estado_t = pd.to_numeric(individual_t["ESTADO"], errors="coerce")
    inactivos_t = individual_t.loc[~estado_t.isin([1, 2]), ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]].rename(
        columns={"NRO_HOGAR": "NRO_HOGAR_t", "CH06": "CH06_t"})

    altas_con_id = altas_menor.reset_index(drop=True)
    altas_con_id["_id"] = altas_con_id.index
    altas_con_t = altas_con_id.merge(
        pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates(),
        on=["CODUSU", "NRO_HOGAR_th"], how="left")

    candidatos = altas_con_t.merge(inactivos_t, on=["CODUSU", "NRO_HOGAR_t", "CH04"], how="left")
    diferencia = candidatos["CH06"] - candidatos["CH06_t"]
    valido = candidatos["CH06_t"].notna() & diferencia.between(minimo, maximo)
    tiene_match = valido.groupby(candidatos["_id"]).any().reindex(altas_con_id["_id"], fill_value=False)

    tipo = np.where(tiene_match.to_numpy(), "inactivo_a_activo", "entra_activo")
    return pd.Series(tipo, index=altas_menor.index)


def calcular_alta_activo_menor(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                                pares_hogar: pd.DataFrame, h: int,
                                edad_corte: int = EDAD_CORTE_MENOR) -> pd.DataFrame:
    """Por hogar-par: altas de activo con `CH06 < edad_corte` en `t+h`,
    separadas por `situacion_escolar_th` y por `tipo_alta`."""
    # D19: la entrada de un menor al trabajo no cuenta como mejora del eje laboral,
    # se cruza siempre con si sigue en la escuela y con si ya estaba en el hogar.
    emparejados = emparejar_activos(individual_t, individual_th, pares_hogar, h,
                                     columnas_extra_th=("CH10", "NIVEL_ED"))
    altas = emparejados["th"][emparejados["th"]["estado_persona"] == "alta"].copy()
    altas_menor = altas[altas["CH06"] < edad_corte].copy()

    resultado = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates()
    if altas_menor.empty:
        resultado["n_alta_activo_menor"] = 0
        return resultado

    situacion = clasificar_situacion_escolar(altas_menor["CH10"], altas_menor["CH06"], altas_menor["NIVEL_ED"])
    altas_menor["situacion_escolar_th"] = situacion.fillna("sin_dato").astype(str)
    altas_menor["tipo_alta"] = _clasificar_tipo_alta(altas_menor, individual_t, pares_hogar, h).astype(str)

    conteo = altas_menor.groupby(
        ["CODUSU", "NRO_HOGAR_th", "situacion_escolar_th", "tipo_alta"]).size().rename("n").reset_index()
    conteo["columna"] = "n_alta_activo_menor_" + conteo["situacion_escolar_th"] + "_" + conteo["tipo_alta"]
    ancho = conteo.pivot_table(index=["CODUSU", "NRO_HOGAR_th"], columns="columna", values="n", fill_value=0)

    resultado = resultado.merge(ancho, on=["CODUSU", "NRO_HOGAR_th"], how="left")
    columnas_conteo = [c for c in resultado.columns if c.startswith("n_alta_activo_menor_")]
    for col in columnas_conteo:
        resultado[col] = resultado[col].fillna(0).astype(int)
    resultado["n_alta_activo_menor"] = resultado[columnas_conteo].sum(axis=1) if columnas_conteo else 0
    return resultado


def calcular_tipologia_perceptores(hogar: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar: `"solo_laborales"`/`"solo_previsionales"`/`"mixto"`/
    `"sin_perceptores"` según ocupación e ingreso declarado por percepción
    (`V2`), no por monto. `NA` si es ambiguo."""
    idx = hogar.set_index(["CODUSU", "NRO_HOGAR"]).index
    algun_ocupado = kleene_any_hogar(individual, to_bool(individual["ESTADO"], [1])).reindex(idx)
    v2 = pd.to_numeric(hogar["V2"], errors="coerce")
    v2.index = idx

    ocupado_si = (algun_ocupado == True).fillna(False)  # noqa: E712
    ocupado_no = (algun_ocupado == False).fillna(False)  # noqa: E712

    tipologia = pd.Series(pd.array([pd.NA] * len(idx), dtype="object"), index=idx)
    tipologia[(ocupado_si & (v2 == 2)).to_numpy()] = "solo_laborales"
    tipologia[(ocupado_no & (v2 == 1)).to_numpy()] = "solo_previsionales"
    tipologia[(ocupado_si & (v2 == 1)).to_numpy()] = "mixto"
    tipologia[(ocupado_no & (v2 == 2)).to_numpy()] = "sin_perceptores"
    return tipologia.rename("tipologia_perceptores").reset_index()
