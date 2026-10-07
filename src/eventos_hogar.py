"""Eventos de composición que distorsionan la intensificación (D19/D20):
menores que entran a trabajar, jubilaciones en hogares mixtos, tipología
de perceptores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from intensificacion import emparejar_activos
from panel import TOLERANCIA_EDAD_POR_HORIZONTE
from perfil_hogar import clasificar_situacion_escolar, kleene_any_hogar, to_bool

# `CAT_INAC`/`CH10` se traen de `01_armonizado` ad hoc (fuera de
# `ESQUEMA_NUCLEO`), como `NIVEL_ED` en `pipeline_meta`.

# PENDIENTE de confirmación de Iván (D19).
EDAD_CORTE_MENOR = 18


def calcular_se_jubila(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                        pares_hogar: pd.DataFrame, h: int) -> pd.DataFrame:
    """Por hogar-par: cantidad de activos de `t` para los que existe en
    `t+h` alguien del mismo hogar/sexo recién jubilado, edad en D6 (D19)."""
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
