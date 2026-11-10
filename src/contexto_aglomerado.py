"""Contexto del aglomerado en `t` (año móvil `t-3..t`): tasas de pobreza,
indigencia, informalidad y desocupación, Gini del IPCF y componente común
rezagado de la variación de distancia. Funciones puras, no leen `data/`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from desigualdad import gini_ponderado, normalizar_por_media_ponderada
from trayectoria_distancia import mediana_ponderada

# Mismo piso que el Gini por aglomerado de `decisiones.md`; se aplica igual a las tasas.
MIN_HOGARES_VENTANA = 400
LARGO_VENTANA = 4
TRIMESTRE = tuple[int, int]


def periodo(anio: int, trimestre: int) -> int:
    """Índice correlativo de trimestre (`anio*4 + trimestre - 1`)."""
    return int(anio) * 4 + int(trimestre) - 1


def ventana_anio_movil(trimestre: TRIMESTRE, disponibles) -> list[TRIMESTRE]:
    """Trimestres de `disponibles` dentro de `t-3..t` (los no publicados
    simplemente no están)."""
    fin = periodo(*trimestre)
    return sorted(q for q in disponibles if fin - LARGO_VENTANA < periodo(*q) <= fin)


def tabla_hogares_gini(hogar: pd.DataFrame, individual: pd.DataFrame, historico: bool) -> pd.DataFrame:
    """Una fila por hogar: `AGLOMERADO`, `ipcf`, `peso_personas` (suma de
    `PONDERA` en la histórica, `PONDIH * n_personas` en la regular) e
    `ipcf_relativo` a la media ponderada nacional del trimestre."""
    # Misma receta que la sección de desigualdad de `05_composicion_quintiles`
    # (commit 994a4b5), que escribió `data/meta/gini_aglomerado.csv`: así el
    # Gini se puede validar contra ese archivo.
    integrantes = individual.groupby(["CODUSU", "NRO_HOGAR"]).agg(
        n_personas=("PONDERA", "size"), pondera_hogar=("PONDERA", lambda s: pd.to_numeric(s, errors="coerce").sum()))
    tabla = hogar[["CODUSU", "NRO_HOGAR", "AGLOMERADO", "IPCF"] + (["PONDIH"] if "PONDIH" in hogar else [])].merge(
        integrantes.reset_index(), on=["CODUSU", "NRO_HOGAR"], how="left")
    if historico:
        tabla["peso_personas"] = tabla["pondera_hogar"].astype(float)
    else:
        tabla["peso_personas"] = pd.to_numeric(tabla["PONDIH"], errors="coerce").astype(float) * tabla["n_personas"]
    tabla["ipcf"] = pd.to_numeric(tabla["IPCF"], errors="coerce").astype(float)
    tabla["ipcf_relativo"] = normalizar_por_media_ponderada(tabla["ipcf"], tabla["peso_personas"])
    return tabla[["CODUSU", "NRO_HOGAR", "AGLOMERADO", "ipcf", "ipcf_relativo", "peso_personas", "n_personas"]]


def _era_unica(eras: dict[TRIMESTRE, str], ventana: list[TRIMESTRE]) -> bool:
    return len({eras[q] for q in ventana}) == 1


def gini_anio_movil(tablas: dict[TRIMESTRE, pd.DataFrame], eras: dict[TRIMESTRE, str], trimestre: TRIMESTRE,
                    min_hogares: int = MIN_HOGARES_VENTANA) -> pd.DataFrame:
    """Por aglomerado: Gini del `ipcf_relativo` ponderado por personas,
    juntando los hogares de la ventana `t-3..t`. `NA` (con `motivo_na`) si la
    ventana cruza de era o tiene menos de `min_hogares` hogares válidos."""
    ventana = ventana_anio_movil(trimestre, tablas)
    datos = pd.concat([tablas[q] for q in ventana], ignore_index=True)
    datos = datos[datos["AGLOMERADO"].notna()]
    cruza_era = not _era_unica(eras, ventana)
    filas = []
    for aglomerado, sub in datos.groupby("AGLOMERADO"):
        valido = sub[(sub["peso_personas"] > 0) & sub["ipcf_relativo"].notna()]
        motivo = ("ventana cruza el quiebre de era" if cruza_era else
                  f"menos de {min_hogares} hogares en la ventana" if len(valido) < min_hogares else None)
        gini = np.nan if motivo else gini_ponderado(valido["ipcf_relativo"], valido["peso_personas"])
        filas.append({"AGLOMERADO": int(aglomerado), "gini_ipcf_aglo": gini, "n_hogares_gini_ventana": len(valido),
                      "motivo_na_gini": motivo})
    return pd.DataFrame(filas, columns=["AGLOMERADO", "gini_ipcf_aglo", "n_hogares_gini_ventana", "motivo_na_gini"])


def agregados_aglomerado_trimestre(distancia: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    """Por aglomerado de un trimestre: sumas ponderadas para las tasas de
    pobreza e indigencia (hogares, `peso_hogar_era`, imputados adentro) y de
    informalidad y desocupación (personas, `PONDERA`)."""
    # Variante A (imputados adentro), como pide `decisiones.md` para tasas y
    # niveles: el contexto describe al aglomerado, no la trayectoria del hogar.
    peso = (pd.to_numeric(distancia["PONDIH"], errors="coerce") if "PONDIH" in distancia
            else pd.to_numeric(distancia["peso_hogar"], errors="coerce")).astype(float)
    con_ratio = distancia["ratio_linea_pobreza"].notna() & (peso > 0)
    hogares = pd.DataFrame({
        "AGLOMERADO": distancia["AGLOMERADO"],
        "n_hogares": con_ratio.astype(int),
        "w_hogares": peso.where(con_ratio, 0.0),
        "w_pobre": peso.where(con_ratio & (distancia["pobre"] == True).fillna(False), 0.0),  # noqa: E712
        "w_indigente": peso.where(con_ratio & (distancia["indigente"] == True).fillna(False), 0.0),  # noqa: E712
    }).groupby("AGLOMERADO").sum()

    estado = pd.to_numeric(individual["ESTADO"], errors="coerce")
    cat_ocup = pd.to_numeric(individual["CAT_OCUP"], errors="coerce")
    pp07h = pd.to_numeric(individual["PP07H"], errors="coerce")
    pondera = pd.to_numeric(individual["PONDERA"], errors="coerce").astype(float).fillna(0.0)
    asalariado = (estado == 1) & (cat_ocup == 3) & pp07h.isin([1, 2])
    personas = pd.DataFrame({
        "AGLOMERADO": individual["AGLOMERADO"],
        "w_asalariados": pondera.where(asalariado, 0.0),
        "w_asalariados_informales": pondera.where(asalariado & (pp07h == 2), 0.0),
        "w_activos": pondera.where(estado.isin([1, 2]), 0.0),
        "w_desocupados": pondera.where(estado == 2, 0.0),
    }).groupby("AGLOMERADO").sum()
    resultado = hogares.join(personas, how="outer").fillna(0.0)
    resultado["n_hogares"] = resultado["n_hogares"].astype(int)
    return resultado.reset_index()


def tasas_anio_movil(agregados: dict[TRIMESTRE, pd.DataFrame], eras: dict[TRIMESTRE, str], trimestre: TRIMESTRE,
                     min_hogares: int = MIN_HOGARES_VENTANA) -> pd.DataFrame:
    """Por aglomerado: tasas de pobreza, indigencia, informalidad asalariada
    y desocupación sumando la ventana `t-3..t`. `NA` si cruza de era o junta
    menos de `min_hogares` hogares con ratio."""
    ventana = ventana_anio_movil(trimestre, agregados)
    suma = pd.concat([agregados[q] for q in ventana], ignore_index=True).groupby("AGLOMERADO").sum()
    resultado = pd.DataFrame({
        "tasa_pobreza_aglo": suma["w_pobre"] / suma["w_hogares"].replace(0, np.nan),
        "tasa_indigencia_aglo": suma["w_indigente"] / suma["w_hogares"].replace(0, np.nan),
        "tasa_informalidad_aglo": suma["w_asalariados_informales"] / suma["w_asalariados"].replace(0, np.nan),
        "tasa_desocupacion_aglo": suma["w_desocupados"] / suma["w_activos"].replace(0, np.nan),
        "n_hogares_ventana_aglo": suma["n_hogares"].astype(int),
        "n_trimestres_ventana": len(ventana),
    })
    insuficiente = (resultado["n_hogares_ventana_aglo"] < min_hogares) | (not _era_unica(eras, ventana))
    columnas_tasa = [c for c in resultado.columns if c.startswith("tasa_")]
    resultado.loc[insuficiente, columnas_tasa] = np.nan
    return resultado.reset_index()


def componente_comun_rezagado(pares_que_terminan_en_t: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Mediana ponderada (`peso_hogar_era_t` del par) de `variacion_simetrica`
    y `variacion_log` de los pares que terminan en `t`, sobre `par_valido_c` /
    `par_valido_log_c`: escalar del trimestre y tabla por `AGLOMERADO`."""
    # Se recalcula en vez de leer `variacion_comun_*` de 05: esa mediana se armó
    # sobre todos los pares, incluidos los sin identidad y los imputados.
    pares = pares_que_terminan_en_t
    configuracion = {"simetrica": ("variacion_simetrica", "par_valido_c"), "log": ("variacion_log", "par_valido_log_c")}
    escalares, por_aglomerado = {}, []
    for sufijo, (columna, filtro) in configuracion.items():
        validos = pares[(pares[filtro] == True).fillna(False)]  # noqa: E712
        escalares[f"variacion_comun_trimestre_{sufijo}_rezagada"] = mediana_ponderada(validos[columna], validos["peso_hogar_era_t"])
        medianas = pd.Series({
            aglomerado: mediana_ponderada(g[columna], g["peso_hogar_era_t"])
            for aglomerado, g in validos.groupby("AGLOMERADO_th")}, dtype=float)
        por_aglomerado.append(medianas.rename(f"variacion_comun_aglomerado_{sufijo}_rezagada"))
    tabla = pd.concat(por_aglomerado, axis=1) if por_aglomerado else pd.DataFrame()
    tabla.index.name = "AGLOMERADO"
    return escalares, tabla.reset_index()
