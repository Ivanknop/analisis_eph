"""Panel de hogares: vinculación de viviendas/hogares entre un trimestre de
origen `t` y un trimestre de destino `t+h`, sobre el Parquet ya tipado del
notebook 01b. No construye ninguna etiqueta de deterioro -- solo el panel
vinculado con sus flags (ver `notebooks/02_panel_hogares.ipynb`).
"""
# Todas las funciones son puras (reciben DataFrames/sets, no tocan `data/`)
# para poder testearlas con fixtures sintéticas chicas (`tests/test_panel.py`).
from __future__ import annotations

import hashlib

import pandas as pd

# Ventana de `diferencia_edad` aceptada como "misma persona" por horizonte --
# PROVISORIA (D6). Centrada en el desplazamiento esperado, no en 0: en h=4
# pasa un año (moda observada en +1, 83.5% de los componentes con mismo sexo),
# no en 0 (6.3%) -- ver diagnóstico de la tarea. h=1 cubre 92.92% de los
# componentes con mismo sexo; h=4 cubre 93.22%.
TOLERANCIA_EDAD_POR_HORIZONTE = {1: (-1, 1), 4: (0, 2)}

# Fechas públicas de asunción (día de la asunción, no del trimestre) --
# [confirmado, hecho público]. La conversión a "trimestre de mandato" es una
# CONVENCIÓN de este módulo, no un hallazgo empírico -- ver `mandato_presidencial`.
# Todas las asunciones reales caen el 10/12 (Q4) salvo Nestor Kirchner (25/5,
# Q2) -- bajo la regla "primer día del trimestre", el primer trimestre COMPLETO
# de cada mandato es el T1 del año siguiente a la asunción (10/12 es posterior
# al 1/10, así que ese Q4 todavía empieza con el presidente saliente). Por eso
# el "desde" de la tabla no coincide con el año de asunción.
MANDATO_PRESIDENCIAL = [
    ((2003, 3), "Nestor Kirchner"),    # asume 2003-05-25 -- 2003T3 es el primer trimestre completo
    ((2008, 1), "CFK I"),              # asume 2007-12-10 -- 2007T4 sigue siendo de Kirchner
    ((2012, 1), "CFK II"),             # asume 2011-12-10 -- 2011T4 sigue siendo de CFK I
    ((2016, 1), "Macri"),              # asume 2015-12-10 -- 2015T4 sigue siendo de CFK II
    ((2020, 1), "Alberto Fernandez"),  # asume 2019-12-10 -- 2019T4 sigue siendo de Macri
    ((2024, 1), "Milei"),              # asume 2023-12-10 -- 2023T4 sigue siendo de Alberto Fernandez
]


def sumar_trimestres(anio: int, trimestre: int, n: int) -> tuple[int, int]:
    """Desplaza `(anio, trimestre)` `n` trimestres hacia adelante (`n > 0`)."""
    indice = (anio * 4 + (trimestre - 1)) + n
    return indice // 4, indice % 4 + 1


def mandato_presidencial(anio: int, trimestre: int) -> str:
    """Presidente en funciones el primer día de `(anio, trimestre)`, según la
    convención de `MANDATO_PRESIDENCIAL` (fechas de asunción reales)."""
    # **Convención**, no un hecho: el traspaso real es el 10/12, dentro del
    # Q4 del año de cambio de mando -- ese trimestre de transición queda
    # asignado acá al presidente saliente, no se reparte el trimestre.
    actual = "Nestor Kirchner"  # antes de la primera asunción en la tabla no debería llamarse
    for (anio_desde, trimestre_desde), nombre in MANDATO_PRESIDENCIAL:
        if (anio, trimestre) >= (anio_desde, trimestre_desde):
            actual = nombre
        else:
            break
    return actual


def regimen_ingreso_par(historico_t: bool, historico_th: bool) -> str:
    """`"historico"` si ambos extremos son pre-2016, `"regular"` si ambos son
    post-2016, `"mixto"` si el par cruza el corte (empíricamente posible para
    algunos pares `h=4` alrededor de 2015-2016, ver plan de la tarea)."""
    if historico_t and historico_th:
        return "historico"
    if not historico_t and not historico_th:
        return "regular"
    return "mixto"


def tasa_vinculacion_viviendas(codusu_t: set[str], codusu_th: set[str]) -> dict:
    """% de viviendas (`CODUSU`) de `t` que también aparecen en `t+h`."""
    n_viviendas_t = len(codusu_t)
    n_vinculadas = len(codusu_t & codusu_th)
    tasa = (n_vinculadas / n_viviendas_t) if n_viviendas_t else None
    return {"n_viviendas_t": n_viviendas_t, "n_vinculadas": n_vinculadas, "tasa": tasa}


def vincular_hogares(hogar_t: pd.DataFrame, hogar_th: pd.DataFrame) -> pd.DataFrame:
    """Join exacto por `CODUSU+NRO_HOGAR+AGLOMERADO` entre hogares de `t` y
    `t+h` -- identidad de hogar, no solo de vivienda (ver D9)."""
    # Match por CODUSU solo (versión anterior) arma producto cruzado cuando
    # una vivienda tiene más de un hogar en t o en t+h (confirmado: ~0.9% de
    # las combinaciones vivienda-trimestre en los datos reales). Requerir
    # también NRO_HOGAR deja afuera los hogares que cambian de número entre
    # t y t+h, pero el diagnóstico de la tarea mostró que >96% de esos casos
    # no tiene composición coincidente -- no parecen el mismo hogar
    # renumerado. Quedan reportados aparte por
    # `clasificar_viviendas_no_vinculadas`, no se pierden en silencio.
    claves = ["CODUSU", "NRO_HOGAR", "AGLOMERADO"]
    pares = hogar_t[claves].merge(hogar_th[claves], on=claves)
    pares = pares.rename(columns={"NRO_HOGAR": "NRO_HOGAR_t"})
    pares["NRO_HOGAR_th"] = pares["NRO_HOGAR_t"]
    return pares


def clasificar_viviendas_no_vinculadas(hogar_t: pd.DataFrame, hogar_th: pd.DataFrame,
                                        pares_hogar: pd.DataFrame) -> pd.DataFrame:
    """Por cada vivienda (`CODUSU`) presente en `t`: `"vinculada"` (tiene un
    hogar con match exacto en `t+h`), `"hogar_reemplazado"` (la vivienda
    sigue en `t+h` pero ningún hogar matcheó exacto) o `"vivienda_sin_par"`
    (la vivienda ya no está en `t+h`)."""
    codusu_th = set(hogar_th["CODUSU"])
    codusu_vinculadas = set(pares_hogar["CODUSU"])
    resultado = hogar_t[["CODUSU"]].drop_duplicates().reset_index(drop=True)
    resultado["estado_vinculacion"] = "vinculada"
    en_th_no_vinculada = (~resultado["CODUSU"].isin(codusu_vinculadas)) & resultado["CODUSU"].isin(codusu_th)
    sin_par = ~resultado["CODUSU"].isin(codusu_th)
    resultado.loc[en_th_no_vinculada, "estado_vinculacion"] = "hogar_reemplazado"
    resultado.loc[sin_par, "estado_vinculacion"] = "vivienda_sin_par"
    return resultado


def validar_jefe(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                  pares_hogar: pd.DataFrame) -> pd.DataFrame:
    """Para cada hogar vinculado (mismo `CODUSU`+`NRO_HOGAR` en ambos lados),
    compara el/la jefe/a (`CH03==1`) de cada lado. `jefe_ambiguo=True` si algún
    lado no tiene exactamente una persona con `CH03==1`."""
    # `cambio_jefatura` es un flag INDEPENDIENTE (no entra en `identidad_nucleo`):
    # un cambio de sexo en la jefatura es un evento real y esperable (jefe/a
    # que se va, pareja que pasa a encabezar el hogar), no evidencia de que el
    # hogar sea "otro". Tampoco se elige arbitrariamente entre varias personas
    # cuando hay ambigüedad -- eso queda marcado, no resuelto.
    # Vectorizado con merges -- un .iterrows() + filtro booleano por fila acá sería
    # O(n_pares * n_individuos) y no escala más allá de un puñado de trimestres
    # (confirmado: colgó al probar con 9 orígenes reales).
    jefe_t = individual_t[individual_t["CH03"] == 1]
    jefe_th = individual_th[individual_th["CH03"] == 1]

    conteo_t = jefe_t.groupby(["CODUSU", "NRO_HOGAR"]).size().rename("n_jefe_t")
    conteo_th = jefe_th.groupby(["CODUSU", "NRO_HOGAR"]).size().rename("n_jefe_th")

    # Si hay más de un jefe/a en un hogar (no debería, pero no se asume), nos
    # quedamos con uno cualquiera para el merge de atributos -- da igual cuál:
    # ese caso ya queda marcado `jefe_ambiguo=True` y sus atributos se invalidan
    # más abajo, nunca se usan para clasificar nada.
    jefe_t_unico = jefe_t.drop_duplicates(["CODUSU", "NRO_HOGAR"])[["CODUSU", "NRO_HOGAR", "CH04", "CH06"]]
    jefe_th_unico = jefe_th.drop_duplicates(["CODUSU", "NRO_HOGAR"])[["CODUSU", "NRO_HOGAR", "CH04", "CH06"]]

    resultado = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].copy()
    resultado = resultado.merge(conteo_t, left_on=["CODUSU", "NRO_HOGAR_t"], right_index=True, how="left")
    resultado = resultado.merge(conteo_th, left_on=["CODUSU", "NRO_HOGAR_th"], right_index=True, how="left")
    resultado[["n_jefe_t", "n_jefe_th"]] = resultado[["n_jefe_t", "n_jefe_th"]].fillna(0)
    resultado["jefe_ambiguo"] = (resultado["n_jefe_t"] != 1) | (resultado["n_jefe_th"] != 1)

    resultado = resultado.merge(
        jefe_t_unico.rename(columns={"CH04": "CH04_t", "CH06": "CH06_t"}),
        left_on=["CODUSU", "NRO_HOGAR_t"], right_on=["CODUSU", "NRO_HOGAR"], how="left",
    ).drop(columns=["NRO_HOGAR"])
    resultado = resultado.merge(
        jefe_th_unico.rename(columns={"CH04": "CH04_th", "CH06": "CH06_th"}),
        left_on=["CODUSU", "NRO_HOGAR_th"], right_on=["CODUSU", "NRO_HOGAR"], how="left",
    ).drop(columns=["NRO_HOGAR"])

    # dtypes nullable ANTES de asignar NA por ambigüedad -- bool/Int8 planos no
    # aceptan NaN (TypeError), a diferencia de "boolean"/"Int64".
    resultado["mismo_sexo_jefe"] = (resultado["CH04_t"] == resultado["CH04_th"]).astype("boolean")
    resultado["diferencia_edad_jefe"] = resultado["CH06_th"].astype("Int64") - resultado["CH06_t"].astype("Int64")
    resultado.loc[resultado["jefe_ambiguo"], ["mismo_sexo_jefe", "diferencia_edad_jefe"]] = pd.NA

    resultado["cambio_jefatura"] = resultado["jefe_ambiguo"] | (resultado["mismo_sexo_jefe"] == False)  # noqa: E712
    return resultado[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "jefe_ambiguo",
                       "mismo_sexo_jefe", "diferencia_edad_jefe", "cambio_jefatura"]]


def emparejar_componentes(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                           pares_hogar: pd.DataFrame) -> pd.DataFrame:
    """Empareja TODOS los miembros (no solo jefe/a) de cada hogar vinculado por
    número de `COMPONENTE` presente en ambos lados. Formato largo: una fila por
    componente matcheado, con `mismo_sexo` y `diferencia_edad`."""
    # Diagnóstico PREVIO (`tasa_coincidencia_componente`): ¿alcanza con matchear
    # por número de COMPONENTE para identificar a la misma persona? La
    # identidad del panel (`identidad_nucleo`) y el conteo de altas/bajas
    # (`contar_altas_bajas`) NO usan esto -- matchean por sexo+edad, no por
    # COMPONENTE (ver D10: la reasignación de COMPONENTE entre visitas solo
    # explica ~7% de los desajustes).
    columnas = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH04", "CH06"]
    izq = pares_hogar.merge(
        individual_t[columnas], left_on=["CODUSU", "NRO_HOGAR_t"], right_on=["CODUSU", "NRO_HOGAR"],
    ).drop(columns=["NRO_HOGAR"])
    emparejados = izq.merge(
        individual_th[columnas], left_on=["CODUSU", "NRO_HOGAR_th", "COMPONENTE"],
        right_on=["CODUSU", "NRO_HOGAR", "COMPONENTE"], suffixes=("_t", "_th"),
    ).drop(columns=["NRO_HOGAR"])
    emparejados["mismo_sexo"] = emparejados["CH04_t"] == emparejados["CH04_th"]
    emparejados["diferencia_edad"] = emparejados["CH06_th"].astype("Int64") - emparejados["CH06_t"].astype("Int64")
    return emparejados[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "COMPONENTE", "mismo_sexo", "diferencia_edad"]]


def tasa_coincidencia_componente(componentes_emparejados: pd.DataFrame) -> dict:
    """Diagnóstico PREVIO a usar `COMPONENTE` como base de matcheo de miembros
    (pedido explícito de la tarea): ¿de verdad identifica a la misma persona
    entre visitas? Se reporta el número, no se asume de antemano."""
    n = len(componentes_emparejados)
    tasa_mismo_sexo = componentes_emparejados["mismo_sexo"].mean() if n else None
    return {
        "n_componentes_matcheados": n,
        "tasa_mismo_sexo": tasa_mismo_sexo,
        "distribucion_diferencia_edad": componentes_emparejados["diferencia_edad"].dropna().tolist(),
    }


def identidad_nucleo(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                      pares_hogar: pd.DataFrame, h: int) -> pd.DataFrame:
    """Identidad del hogar-par por núcleo (D10): `identidad_confirmada` es
    `True` si el/la jefe/a de `t` aparece en el núcleo de `t+h` (`CH03` 1 o 2
    -- permite intercambio de rol) con sexo y edad dentro de
    `TOLERANCIA_EDAD_POR_HORIZONTE[h]`. `NA` si el/la jefe/a de `t` es
    ambiguo/a."""
    # Reemplaza la clasificación por proporción de componentes (D9): esa regla
    # exigía que coincidiera casi TODO el hogar, lo que penalizaba
    # mecánicamente a los hogares grandes (ver D10) sin relación con si el
    # hogar es "el mismo". La composición completa queda aparte, como flag,
    # en `contar_altas_bajas` -- no entra acá, mismo criterio que
    # `cambio_jefatura` (independiente, no descalifica la identidad).
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[h]
    jefe_t = individual_t[individual_t["CH03"] == 1]
    conteo_jefe_t = jefe_t.groupby(["CODUSU", "NRO_HOGAR"]).size().rename("n_jefe_t")
    jefe_t_unico = jefe_t.drop_duplicates(["CODUSU", "NRO_HOGAR"])[["CODUSU", "NRO_HOGAR", "CH04", "CH06"]]
    nucleo_th = individual_th[individual_th["CH03"].isin([1, 2])][["CODUSU", "NRO_HOGAR", "CH04", "CH06"]]

    resultado = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].copy()
    resultado = resultado.merge(conteo_jefe_t, left_on=["CODUSU", "NRO_HOGAR_t"], right_index=True, how="left")
    resultado["n_jefe_t"] = resultado["n_jefe_t"].fillna(0)
    jefe_ambiguo = resultado["n_jefe_t"] != 1

    resultado = resultado.merge(
        jefe_t_unico.rename(columns={"CH04": "CH04_jefe_t", "CH06": "CH06_jefe_t"}),
        left_on=["CODUSU", "NRO_HOGAR_t"], right_on=["CODUSU", "NRO_HOGAR"], how="left",
    ).drop(columns=["NRO_HOGAR"])

    candidatos = resultado.merge(
        nucleo_th.rename(columns={"NRO_HOGAR": "NRO_HOGAR_th", "CH04": "CH04_th", "CH06": "CH06_th"}),
        on=["CODUSU", "NRO_HOGAR_th"], how="left",
    )
    candidatos["diferencia"] = candidatos["CH06_th"].astype("Int64") - candidatos["CH06_jefe_t"].astype("Int64")
    candidatos["coincide"] = (candidatos["CH04_th"] == candidatos["CH04_jefe_t"]) & candidatos["diferencia"].between(minimo, maximo)
    encontrado = candidatos.groupby(["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"])["coincide"].any()

    resultado = resultado.merge(encontrado.rename("_encontrado"), on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"], how="left")
    resultado["identidad_confirmada"] = resultado["_encontrado"].fillna(False).astype("boolean")
    resultado.loc[jefe_ambiguo, "identidad_confirmada"] = pd.NA
    return resultado[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "identidad_confirmada"]]


def _consumir_offset(restante_t: pd.Series, restante_th: pd.Series, offset: int) -> tuple[pd.Series, pd.Series]:
    """Empareja lo que pueda entre `restante_t`/`restante_th` (Series indexadas
    por `CODUSU, NRO_HOGAR, CH04, CH06`, valor = cantidad sin matchear) con
    `CH06_th == CH06_t + offset`, y resta lo matcheado de ambos lados."""
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


def contar_altas_bajas(individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                        pares_hogar: pd.DataFrame, h: int) -> pd.DataFrame:
    """`n_altas`/`n_bajas` de integrantes por hogar-par (D10, composición como
    flag aparte de la identidad): empareja personas por sexo y edad dentro de
    `TOLERANCIA_EDAD_POR_HORIZONTE[h]` -- no por `COMPONENTE` (ver D10: la
    reasignación de `COMPONENTE` solo explica ~7% de los desajustes)."""
    # Matching goloso por offset de edad dentro de la ventana, probando primero
    # el offset más cercano al centro (el desplazamiento esperado) -- no
    # garantiza el matching máximo exacto en el caso raro de que dos personas
    # del mismo sexo con edades distintas compitan por la misma persona del
    # otro lado, pero es el criterio razonable para un flag descriptivo.
    minimo, maximo = TOLERANCIA_EDAD_POR_HORIZONTE[h]
    claves = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"]].drop_duplicates()

    t_rel = individual_t.merge(claves[["CODUSU", "NRO_HOGAR_t"]].drop_duplicates(),
                                left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_t"])
    th_rel = individual_th.merge(claves[["CODUSU", "NRO_HOGAR_th"]].drop_duplicates(),
                                  left_on=["CODUSU", "NRO_HOGAR"], right_on=["CODUSU", "NRO_HOGAR_th"])

    restante_t = t_rel.groupby(["CODUSU", "NRO_HOGAR_t", "CH04", "CH06"]).size().rename("restante")
    restante_t.index.names = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]
    restante_th = th_rel.groupby(["CODUSU", "NRO_HOGAR_th", "CH04", "CH06"]).size().rename("restante")
    restante_th.index.names = ["CODUSU", "NRO_HOGAR", "CH04", "CH06"]

    centro = round((minimo + maximo) / 2)
    for offset in sorted(range(minimo, maximo + 1), key=lambda o: abs(o - centro)):
        restante_t, restante_th = _consumir_offset(restante_t, restante_th, offset)

    n_bajas = restante_t[restante_t > 0].groupby(["CODUSU", "NRO_HOGAR"]).sum().rename("n_bajas")
    n_altas = restante_th[restante_th > 0].groupby(["CODUSU", "NRO_HOGAR"]).sum().rename("n_altas")

    resultado = claves.copy()
    resultado = resultado.merge(n_bajas, left_on=["CODUSU", "NRO_HOGAR_t"], right_index=True, how="left")
    resultado = resultado.merge(n_altas, left_on=["CODUSU", "NRO_HOGAR_th"], right_index=True, how="left")
    resultado[["n_altas", "n_bajas"]] = resultado[["n_altas", "n_bajas"]].fillna(0).astype(int)
    resultado["cambio_composicion"] = (resultado["n_altas"] + resultado["n_bajas"]) > 0
    return resultado


def _hay_ocupado(individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar (`CODUSU, NRO_HOGAR`), `True` si algún componente tiene
    `ESTADO==1`. `groupby(...).apply(lambda ...)` (versión anterior) tardaba
    ~2.3s por trimestre por el overhead de Python por grupo -- con `.any()`
    directo sobre la serie booleana (sin `.apply`) es ~100x más rápido."""
    return (individual["ESTADO"] == 1).groupby(
        [individual["CODUSU"], individual["NRO_HOGAR"]]
    ).any().reset_index(name="hay_ocupado")


def construir_pares(hogar_t: pd.DataFrame, hogar_th: pd.DataFrame,
                     individual_t: pd.DataFrame, individual_th: pd.DataFrame,
                     anio_t: int, trimestre_t: int, h: int,
                     historico_t: bool, historico_th: bool) -> pd.DataFrame:
    """Arma la tabla final de pares hogar-a-hogar para el trimestre de origen
    `(anio_t, trimestre_t)` y horizonte `h`, sin etiqueta -- ver el esquema de
    columnas en `notebooks/02_panel_hogares.ipynb`."""
    # Requiere `hogar_*` con `CODUSU, NRO_HOGAR, ingreso_no_declarado` e
    # `individual_*` con `CODUSU, NRO_HOGAR, COMPONENTE, CH03, CH04, CH06, ESTADO`.
    anio_th, trimestre_th = sumar_trimestres(anio_t, trimestre_t, h)

    pares = vincular_hogares(hogar_t, hogar_th)
    if pares.empty:
        return pares.assign(**{c: pd.Series(dtype="object") for c in (
            "h", "ANO4_t", "TRIMESTRE_t", "ANO4_th", "TRIMESTRE_th", "identidad_confirmada",
            "ingreso_declarado_t", "ingreso_declarado_th", "hay_ocupado_t", "hay_ocupado_th",
            "regimen_ingreso", "mandato_presidencial_t", "mandato_presidencial_th",
            "diferencia_edad_jefe", "mismo_sexo_jefe", "cambio_jefatura", "jefe_ambiguo",
            "n_altas", "n_bajas", "cambio_composicion",
        )})

    jefe_df = validar_jefe(individual_t, individual_th, pares)
    identidad_df = identidad_nucleo(individual_t, individual_th, pares, h)
    composicion_df = contar_altas_bajas(individual_t, individual_th, pares, h)

    resultado = pares.merge(jefe_df, on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"])
    resultado = resultado.merge(identidad_df, on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"])
    resultado = resultado.merge(composicion_df, on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"])

    ingreso_t = hogar_t[["CODUSU", "NRO_HOGAR", "ingreso_no_declarado"]].rename(
        columns={"NRO_HOGAR": "NRO_HOGAR_t", "ingreso_no_declarado": "ingreso_declarado_t"})
    ingreso_th = hogar_th[["CODUSU", "NRO_HOGAR", "ingreso_no_declarado"]].rename(
        columns={"NRO_HOGAR": "NRO_HOGAR_th", "ingreso_no_declarado": "ingreso_declarado_th"})
    resultado = resultado.merge(ingreso_t, on=["CODUSU", "NRO_HOGAR_t"]).merge(ingreso_th, on=["CODUSU", "NRO_HOGAR_th"])
    # ingreso_no_declarado -> ingreso_declarado: invertido (NA se preserva).
    resultado["ingreso_declarado_t"] = ~resultado["ingreso_declarado_t"]
    resultado["ingreso_declarado_th"] = ~resultado["ingreso_declarado_th"]

    ocupado_t = _hay_ocupado(individual_t).rename(columns={"NRO_HOGAR": "NRO_HOGAR_t", "hay_ocupado": "hay_ocupado_t"})
    ocupado_th = _hay_ocupado(individual_th).rename(columns={"NRO_HOGAR": "NRO_HOGAR_th", "hay_ocupado": "hay_ocupado_th"})
    resultado = resultado.merge(ocupado_t, on=["CODUSU", "NRO_HOGAR_t"]).merge(ocupado_th, on=["CODUSU", "NRO_HOGAR_th"])

    resultado["h"] = h
    resultado["ANO4_t"] = anio_t
    resultado["TRIMESTRE_t"] = trimestre_t
    resultado["ANO4_th"] = anio_th
    resultado["TRIMESTRE_th"] = trimestre_th
    resultado["regimen_ingreso"] = regimen_ingreso_par(historico_t, historico_th)
    resultado["mandato_presidencial_t"] = mandato_presidencial(anio_t, trimestre_t)
    resultado["mandato_presidencial_th"] = mandato_presidencial(anio_th, trimestre_th)

    return resultado


def hash_esquema_panel() -> str:
    """Hash estable de los umbrales/convenciones de este módulo que afectan el
    resultado (`TOLERANCIA_EDAD_POR_HORIZONTE`, `MANDATO_PRESIDENCIAL`)."""
    # Mismo propósito que `tipado_nucleo.hash_esquema()`: registrar en
    # `corridas.csv` con qué versión de estos umbrales se corrió el panel.
    contenido = repr((TOLERANCIA_EDAD_POR_HORIZONTE, MANDATO_PRESIDENCIAL))
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:12]
