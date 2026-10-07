"""Cascada de casos para los targets Y1 (D16) e Y2 (D14-b): orquesta el
resto de `src/` y agrega las piezas nuevas de D24.
"""
# Funciones puras (reciben DataFrames/valores, no tocan `data/`), mismo criterio
# que panel.py/intensificacion.py -- testeables con fixtures sintéticas chicas.
from __future__ import annotations

import numpy as np
import pandas as pd

from armonizacion import es_historico, trimestres_publicados
from panel import sumar_trimestres, clasificar_viviendas_no_vinculadas
from perfil_hogar import calcular_situacion_laboral, kleene_any_hogar, to_bool

# Rotación 2-2-2 del panel: ~50 pp de `vivienda_sin_par` son salida por diseño de
# muestra, no atrición real -- misma constante que usó D9 (no se deriva un modelo
# de rotación por trimestre, D24 reusa el valor ya confirmado).
PCT_ROTACION_TEORICA = 50.0

TRATAMIENTOS_PERSISTENCIA = ("principal", "robustez")

# D1: aglomerados que INDEC no relevó en ese trimestre puntual -- un origen cuyo
# destino cae acá tiene 100% de `vivienda_sin_par` en ese aglomerado por
# construcción, no por atrición real.
AGLOMERADOS_FALTANTES_D1 = {(2019, 3): {8}, (2020, 3): {31}}


def aglomerados_excluidos_d1(anio_th: int, trimestre_th: int) -> frozenset[int]:
    """Aglomerados a excluir del cálculo de atrición para un destino
    `(anio_th, trimestre_th)` (D1). Vacío en cualquier otro destino."""
    return frozenset(AGLOMERADOS_FALTANTES_D1.get((anio_th, trimestre_th), set()))


def tiene_destino_valido(anio_t: int, trimestre_t: int, h: int) -> bool:
    """`False` si `t+h` no está publicado o si cruza el corte de formato de
    `CODUSU` entre era histórica y regular (D7): en ambos casos la
    vinculación no mide atrición real (D24)."""
    anio_th, trimestre_th = sumar_trimestres(anio_t, trimestre_t, h)
    if (anio_th, trimestre_th) not in set(trimestres_publicados()):
        return False
    return es_historico(anio_t, trimestre_t) == es_historico(anio_th, trimestre_th)


def clasificar_vinculacion_trimestre(hogar_t: pd.DataFrame, hogar_th: pd.DataFrame,
                                      pares_hogar: pd.DataFrame,
                                      pct_rotacion_teorica: float = PCT_ROTACION_TEORICA,
                                      aglomerados_excluidos_atricion: frozenset[int] = frozenset()) -> dict:
    """Resumen de `panel.clasificar_viviendas_no_vinculadas` para un origen,
    a nivel vivienda y a nivel hogar (D24: la cascada parte de hogares, no
    de viviendas)."""
    clasificacion = clasificar_viviendas_no_vinculadas(hogar_t, hogar_th, pares_hogar)
    conteos = clasificacion["estado_vinculacion"].value_counts()
    n_viviendas_t = len(clasificacion)
    # Nivel vivienda: vinculada/hogar_reemplazado/vivienda_sin_par siempre
    # suman n_viviendas_t (ver assert abajo).
    n_vinculada = int(conteos.get("vinculada", 0))
    n_hogar_reemplazado = int(conteos.get("hogar_reemplazado", 0))
    n_vivienda_sin_par = int(conteos.get("vivienda_sin_par", 0))
    assert n_vinculada + n_hogar_reemplazado + n_vivienda_sin_par == n_viviendas_t

    # % de vivienda_sin_par crudo y neto de la rotación teórica (D9),
    # excluyendo los aglomerados faltantes puntuales (D1, no atrición real).
    aglomerado_por_vivienda = hogar_t[["CODUSU", "AGLOMERADO"]].drop_duplicates("CODUSU")
    clasificacion = clasificacion.merge(aglomerado_por_vivienda, on="CODUSU", how="left")
    para_atricion = clasificacion[~clasificacion["AGLOMERADO"].isin(aglomerados_excluidos_atricion)]
    n_viviendas_atricion = len(para_atricion)
    n_sin_par_atricion = int((para_atricion["estado_vinculacion"] == "vivienda_sin_par").sum())
    pct_crudo = 100 * n_sin_par_atricion / n_viviendas_atricion if n_viviendas_atricion else np.nan
    pct_neto = pct_crudo - pct_rotacion_teorica if pd.notna(pct_crudo) else np.nan

    hogares = hogar_t[["CODUSU", "NRO_HOGAR"]].drop_duplicates()
    hogares = hogares.merge(clasificacion[["CODUSU", "estado_vinculacion"]], on="CODUSU", how="left")
    matcheados = pares_hogar[["CODUSU", "NRO_HOGAR_t"]].drop_duplicates().rename(columns={"NRO_HOGAR_t": "NRO_HOGAR"})
    hogares = hogares.merge(matcheados.assign(_matcheado=True), on=["CODUSU", "NRO_HOGAR"], how="left")
    hogares["_matcheado"] = hogares["_matcheado"].fillna(False).astype(bool)

    # Nivel hogar: todo hogar de t cae en exactamente una de estas 4
    # categorías (ver assert abajo); n_sin_match_multihogar es una vivienda
    # vinculada donde este hogar puntual no es el que matcheó (D9).
    n_hogares_en_t = len(hogares)
    n_hogares_matcheados = int(hogares["_matcheado"].sum())
    n_hogares_sin_par = int((hogares["estado_vinculacion"] == "vivienda_sin_par").sum())
    n_hogares_reemplazada = int((hogares["estado_vinculacion"] == "hogar_reemplazado").sum())
    n_sin_match_multihogar = int(
        ((hogares["estado_vinculacion"] == "vinculada").to_numpy() & ~hogares["_matcheado"].to_numpy()).sum())
    assert n_hogares_sin_par + n_hogares_reemplazada + n_sin_match_multihogar + n_hogares_matcheados == n_hogares_en_t

    return {
        "n_viviendas_t": n_viviendas_t, "n_vinculada": n_vinculada,
        "n_hogar_reemplazado": n_hogar_reemplazado, "n_vivienda_sin_par": n_vivienda_sin_par,
        "pct_vivienda_sin_par_crudo": pct_crudo, "pct_atricion_neta_rotacion": pct_neto,
        "n_hogares_en_vivienda_sin_par": n_hogares_sin_par,
        "n_hogares_en_vivienda_reemplazada": n_hogares_reemplazada,
        "n_hogares_sin_match_multihogar": n_sin_match_multihogar,
        "n_hogares_matcheados": n_hogares_matcheados,
    }


def clasificar_trabajo_lexicografico(emparejados: dict[str, pd.DataFrame],
                                      share: pd.DataFrame) -> pd.Series:
    """Eje "trabajo" en 3 niveles (D24): compara `share_precario_th` vs.
    `_t`; si empatan, desempata por la tasa de desocupados estables.
    `"sin_activos_estables"` si falta algún lado."""
    resultado = pd.Series("igual", index=share.index, dtype="object")
    sube = share["share_precario_th"] > share["share_precario_t"]
    baja = share["share_precario_th"] < share["share_precario_t"]
    resultado[sube] = "empeora"
    resultado[baja] = "mejora"

    empate = ~sube & ~baja & share["share_precario_t"].notna() & share["share_precario_th"].notna()
    desocupados_estables = {}
    for lado, col_nro in (("t", "NRO_HOGAR_t"), ("th", "NRO_HOGAR_th")):
        rel = emparejados[lado]
        estables = rel[rel["estado_persona"] == "estable"]
        desocupados_estables[lado] = (estables["categoria_precariedad"] == "desocupado").groupby(
            [estables["CODUSU"], estables[col_nro]]).sum()

    claves_t = pd.MultiIndex.from_frame(share[["CODUSU", "NRO_HOGAR_t"]])
    claves_th = pd.MultiIndex.from_frame(share[["CODUSU", "NRO_HOGAR_th"]])
    n_desoc_t = desocupados_estables["t"].reindex(claves_t).fillna(0).to_numpy()
    n_desoc_th = desocupados_estables["th"].reindex(claves_th).fillna(0).to_numpy()
    desoc_share_t = n_desoc_t / share["n_activos_estables_t"].replace(0, np.nan).to_numpy()
    desoc_share_th = n_desoc_th / share["n_activos_estables_th"].replace(0, np.nan).to_numpy()

    sube_desoc = empate.to_numpy() & (desoc_share_th > desoc_share_t)
    baja_desoc = empate.to_numpy() & (desoc_share_th < desoc_share_t)
    resultado[sube_desoc] = "empeora"
    resultado[baja_desoc] = "mejora"

    sin_activos_estables = (share["n_activos_estables_t"] == 0) | (share["n_activos_estables_th"] == 0)
    resultado[sin_activos_estables] = "sin_activos_estables"
    return resultado


def activos_sin_empleo_formal(individual: pd.DataFrame, idx: pd.MultiIndex) -> pd.Series:
    """D14 definición b, por hogar: `True` si hay algún activo y ninguno es
    asalariado formal."""
    # Reimplementada acá (no vive en src/, solo en el notebook 06,
    # intocable) sobre las mismas funciones de perfil_hogar.
    situacion = calcular_situacion_laboral(individual).set_index(
        ["CODUSU", "NRO_HOGAR"])["situacion_laboral"].reindex(idx)
    ningun_formal = (situacion != "algun_asalariado_formal").astype("boolean")
    ningun_formal[situacion.isna()] = pd.NA
    algun_activo = kleene_any_hogar(individual, to_bool(individual["ESTADO"], [1, 2])).reindex(idx)
    resultado = (algun_activo & ningun_formal).astype("boolean")
    resultado[algun_activo.isna() | (algun_activo.fillna(False) & ningun_formal.isna())] = pd.NA
    return resultado


def calcular_persistencia(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                           pares_hogar: pd.DataFrame, h: int) -> pd.DataFrame:
    """Y2 (D14-b) por hogar-par: estado en `t+h` por prioridad
    `sale_por_empleo` > `pasa_a_sin_activos` > `permanece`. Dos variantes de
    D14: `persiste_principal` y `persiste_robustez`."""
    claves = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates().reset_index(drop=True)
    idx_t = pd.MultiIndex.from_frame(claves[["CODUSU", "NRO_HOGAR_t"]].rename(columns={"NRO_HOGAR_t": "NRO_HOGAR"}))
    idx_th = pd.MultiIndex.from_frame(claves[["CODUSU", "NRO_HOGAR_th"]].rename(columns={"NRO_HOGAR_th": "NRO_HOGAR"}))

    resultado = claves.copy()
    resultado["activos_sin_empleo_formal_t"] = activos_sin_empleo_formal(individual_t, idx_t).to_numpy()

    situacion_th = calcular_situacion_laboral(individual_th).set_index(
        ["CODUSU", "NRO_HOGAR"])["situacion_laboral"].reindex(idx_th)
    sale_por_empleo = (situacion_th == "algun_asalariado_formal").astype("boolean")
    sale_por_empleo[situacion_th.isna()] = pd.NA
    algun_activo_th = kleene_any_hogar(individual_th, to_bool(individual_th["ESTADO"], [1, 2])).reindex(idx_th)
    pasa_a_sin_activos = (algun_activo_th == False)  # noqa: E712 -- Kleene: NA se preserva

    # Orden de asignación = prioridad ascendente (lo de mayor prioridad pisa al final).
    estado = pd.Series("permanece", index=claves.index, dtype="object")
    estado[pasa_a_sin_activos.fillna(False).to_numpy()] = "pasa_a_sin_activos"
    estado[sale_por_empleo.fillna(False).to_numpy()] = "sale_por_empleo"
    indeterminado = (sale_por_empleo.isna().to_numpy() & ~pasa_a_sin_activos.fillna(False).to_numpy()) | (
        pasa_a_sin_activos.isna().to_numpy() & ~sale_por_empleo.fillna(False).to_numpy())
    estado = estado.astype("object")
    estado[indeterminado] = pd.NA
    resultado["estado_th"] = estado.to_numpy()

    poblacion = (resultado["activos_sin_empleo_formal_t"] == True).to_numpy()  # noqa: E712
    resultado["persiste_principal"] = pd.array([pd.NA] * len(resultado), dtype="boolean")
    resultado["persiste_robustez"] = pd.array([pd.NA] * len(resultado), dtype="boolean")
    es_sale = (resultado["estado_th"] == "sale_por_empleo").to_numpy()
    es_permanece = (resultado["estado_th"] == "permanece").to_numpy()
    es_sin_activos = (resultado["estado_th"] == "pasa_a_sin_activos").to_numpy()
    resultado.loc[poblacion & es_sale, ["persiste_principal", "persiste_robustez"]] = False
    resultado.loc[poblacion & es_permanece, ["persiste_principal", "persiste_robustez"]] = True
    # persiste_principal cuenta pasa_a_sin_activos como "no sale" (positivo);
    # persiste_robustez lo deja NA en vez de True.
    resultado.loc[poblacion & es_sin_activos, "persiste_principal"] = True
    return resultado[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "activos_sin_empleo_formal_t",
                       "estado_th", "persiste_principal", "persiste_robustez"]]


def _metricas_y1(poblacion: pd.DataFrame, sufijo: str) -> dict:
    con_activos = poblacion[(poblacion["n_activos_estables_t"] > 0) & (poblacion["n_activos_estables_th"] > 0)]
    n_poblacion = len(con_activos)
    metricas = {f"n_poblacion_y1{sufijo}": n_poblacion}
    n_empeora_d16 = int((con_activos["trabajo_d16"] == "empeora").sum())
    metricas[f"n_y1_empeora_d16{sufijo}"] = n_empeora_d16
    metricas[f"tasa_y1_d16{sufijo}"] = 100 * n_empeora_d16 / n_poblacion if n_poblacion else np.nan
    for nivel in ("empeora", "igual", "mejora"):
        metricas[f"n_y1_{nivel}_lexicografica{sufijo}"] = int((con_activos["trabajo_lexicografico"] == nivel).sum())
    metricas[f"tasa_y1_lexicografica{sufijo}"] = (
        100 * metricas[f"n_y1_empeora_lexicografica{sufijo}"] / n_poblacion if n_poblacion else np.nan)
    return metricas


def _metricas_y2(poblacion: pd.DataFrame, sufijo: str) -> dict:
    metricas = {f"n_poblacion_y2{sufijo}": int(poblacion["persiste_principal"].notna().sum())}
    for variante in TRATAMIENTOS_PERSISTENCIA:
        columna = f"persiste_{variante}"
        n_con_dato = int(poblacion[columna].notna().sum())
        n_persiste = int((poblacion[columna] == True).sum())  # noqa: E712
        n_sale = n_con_dato - n_persiste
        metricas[f"n_y2_persiste_{variante}{sufijo}"] = n_persiste
        metricas[f"n_y2_poblacion_{variante}{sufijo}"] = n_con_dato
        metricas[f"tasa_y2_{variante}{sufijo}"] = 100 * n_persiste / n_con_dato if n_con_dato else np.nan
        # Versión invertida (positivo = sale_por_empleo) -- mismo denominador que
        # persiste_{variante}, es su complemento exacto dentro de esa variante.
        metricas[f"n_y2_sale_{variante}{sufijo}"] = n_sale
        metricas[f"tasa_y2_sale_{variante}{sufijo}"] = 100 * n_sale / n_con_dato if n_con_dato else np.nan
    return metricas


def construir_fila_cascada(hogares_en_t: int, estado_destino: str,
                            vinculacion: dict | None, tabla_pares: pd.DataFrame | None) -> dict:
    """Una fila de `cascada_trimestral.csv` (D24). `vinculacion`/
    `tabla_pares` son `None` cuando `estado_destino != "valido"`."""
    fila = {"hogares_en_t": hogares_en_t, "estado_destino": estado_destino}
    if estado_destino != "valido" or tabla_pares is None or vinculacion is None:
        return fila

    assert vinculacion["n_vinculada"] + vinculacion["n_hogar_reemplazado"] + vinculacion["n_vivienda_sin_par"] \
        == vinculacion["n_viviendas_t"]
    assert (vinculacion["n_hogares_en_vivienda_sin_par"] + vinculacion["n_hogares_en_vivienda_reemplazada"]
            + vinculacion["n_hogares_sin_match_multihogar"] + vinculacion["n_hogares_matcheados"]) == hogares_en_t

    fila.update(vinculacion)
    fila["n_pares"] = len(tabla_pares)
    assert fila["n_pares"] == vinculacion["n_hogares_matcheados"]
    identidad = tabla_pares["identidad_confirmada"]
    fila["n_identidad_confirmada"] = int((identidad == True).sum())  # noqa: E712
    fila["n_identidad_no_confirmada"] = int((identidad == False).sum())  # noqa: E712

    # Y1/Y2 sin sufijo NO excluyen el ingreso no declarado por default (D24:
    # pedido de Iván, corrige el D24 original). Las columnas `..._excl_ingreso`
    # abajo quedan reservadas para la futura grilla 3x3 (eje ingreso).
    confirmados = tabla_pares[identidad == True]  # noqa: E712
    excluido_ingreso = (confirmados["ingreso_no_declarado_t"] == True) | (  # noqa: E712
        confirmados["ingreso_no_declarado_th"] == True)  # noqa: E712
    fila["n_excluido_ingreso_no_declarado"] = int(excluido_ingreso.sum())
    elegibles = confirmados[~excluido_ingreso.fillna(False)]
    fila["n_elegibles_post_ingreso"] = len(elegibles)
    assert fila["n_excluido_ingreso_no_declarado"] + fila["n_elegibles_post_ingreso"] == fila["n_identidad_confirmada"]

    fila.update(_metricas_y1(confirmados, ""))
    fila.update(_metricas_y2(confirmados, ""))
    fila.update(_metricas_y1(elegibles, "_excl_ingreso"))
    fila.update(_metricas_y2(elegibles, "_excl_ingreso"))
    return fila


def clasificar_fold(pares: pd.DataFrame, anio_test: int, trimestre_test: int, embargo: int = 1) -> pd.Series:
    """Walk-forward con embargo (D24): `"test"` si el origen coincide,
    `"train"` si el destino cae `embargo`+ antes, `"descartado"` si no."""
    limite_anio, limite_trimestre = sumar_trimestres(anio_test, trimestre_test, -embargo)
    es_test = (pares["ANO4_t"] == anio_test) & (pares["TRIMESTRE_t"] == trimestre_test)
    destino_antes_del_limite = (pares["ANO4_th"] < limite_anio) | (
        (pares["ANO4_th"] == limite_anio) & (pares["TRIMESTRE_th"] <= limite_trimestre))
    resultado = pd.Series("descartado", index=pares.index, dtype="object")
    resultado[destino_antes_del_limite] = "train"
    resultado[es_test] = "test"
    return resultado


def frecuencias_vivienda(hogar: pd.DataFrame, columnas: list[str],
                          por_aglomerado: bool = False) -> pd.DataFrame:
    """Frecuencia (`n` y `pct`, `NA` incluido) de cada valor observado de
    `columnas`, por `(ANO4, TRIMESTRE)` y opcionalmente por `AGLOMERADO`
    (D24)."""
    claves = ["ANO4", "TRIMESTRE", "AGLOMERADO"] if por_aglomerado else ["ANO4", "TRIMESTRE"]
    filas = []
    for columna in columnas:
        conteo = hogar.groupby(claves)[columna].value_counts(dropna=False).rename("n").reset_index()
        conteo = conteo.rename(columns={columna: "valor"})
        conteo["variable"] = columna
        filas.append(conteo)
    resultado = pd.concat(filas, ignore_index=True)
    resultado["pct"] = 100 * resultado["n"] / resultado.groupby([*claves, "variable"])["n"].transform("sum")
    return resultado[[*claves, "variable", "valor", "n", "pct"]]
