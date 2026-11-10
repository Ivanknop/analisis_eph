"""Marcas de imputación de ingresos de la era histórica (`IDIMPH`/`IDIMPP`,
2003T3-2015T2) y fracción del `ITF` que es imputada, como intervalo
`[fraccion_min, fraccion_max]` (diseño de registro INDEC 2009, pp. 12, 28-30).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tipado_nucleo import parsear_monetario

POSICIONES = (1, 2, 3, 4, 5)
CODIGO_SIN_IMPUTAR = "00000"

# Rango de dígitos definido en el diseño de registro 2009 (IDIMPP pp. 28-29;
# IDIMPH p. 12 es binario en todas las posiciones).
DIGITOS_VALIDOS_IDIMPP = {1: set(range(6)), 2: set(range(4)), 3: {0, 1}, 4: set(range(8)), 5: {0, 1}}
DIGITOS_VALIDOS_IDIMPH = {pos: {0, 1} for pos in POSICIONES}

GRUPO_AMPLIO_NO_LABORAL = ("V21_M", "V3_M", "V4_M", "V5_M", "V8_M", "V9_M", "V10_M",
                           "V11_M", "V12_M", "V18_M", "V19_AM")
V_NO_LABORALES_LISTADAS = ("V2_M",) + GRUPO_AMPLIO_NO_LABORAL
RESIDUO_NO_LABORAL = "RESIDUO_NO_LABORAL"
RESPALDO_OCUPACION_PRINCIPAL = "P21"
PLAN_EMPLEO = "PLAN_EMPLEO"
COMPONENTES_I1A = ("PP08D1", "PP08F1", "PP08F2")
# PP08J1/J2/J3 entran a P47T por fuera de P21 (ver comentario de INCLUSIONES).
COMPONENTES_I2 = ("P21", "PP08J1", "PP08J2", "PP08J3", "TOT_P12", "T_VI")
VARIABLES_PLAN_EMPLEO = {"PJ1_1": "P21", "PJ2_1": "TOT_P12", "PJ3_1": "V5_M"}

VARIABLES_MONTO = ("P21", "PP06C", "PP06D", "PP08D1", "PP08D4", "PP08F1", "PP08F2", "PP08J1", "PP08J2",
                   "PP08J3", "TOT_P12", "P47T", "T_VI") + V_NO_LABORALES_LISTADAS

# Tipos: "exacto_si_i1" = exacto solo si la persona cumple I1 (si no, respaldo
# P21, cota superior); "exacto" -> entra en min y max; "cota_superior" e
# "inferido" -> solo en max.
_POS1_PESOS = (("PP08D1",), "exacto_si_i1")
_POS1_BONOS = (("PP08D4", "PP08F1", "PP08F2"), "exacto_si_i1")
_POS2_AGUINALDO = (("PP08J1",), "exacto")
_POS2_BONIFICACIONES = (("PP08J2", "PP08J3"), "exacto")
_POS4_JUBILACION = (("V2_M",), "exacto")
_POS4_GRUPO_AMPLIO = (GRUPO_AMPLIO_NO_LABORAL, "cota_superior")
_POS4_OTROS = ((RESIDUO_NO_LABORAL,), "inferido")

MAPEO_IDIMPP: dict[tuple[int, int], list[tuple[tuple[str, ...], str]]] = {
    (1, 1): [_POS1_PESOS],                                   # p. 28
    (1, 2): [_POS1_BONOS],
    (1, 3): [_POS1_PESOS, _POS1_BONOS],
    (1, 4): [(("PP06C",), "exacto")],                        # p. 20
    (1, 5): [(("PP06D",), "exacto")],
    (2, 1): [_POS2_AGUINALDO],                               # p. 28, montos p. 22
    (2, 2): [_POS2_BONIFICACIONES],
    (2, 3): [_POS2_AGUINALDO, _POS2_BONIFICACIONES],
    (3, 1): [(("TOT_P12",), "exacto")],                      # p. 26
    (4, 1): [_POS4_JUBILACION],                              # p. 29, montos pp. 26-27
    (4, 2): [_POS4_GRUPO_AMPLIO],
    (4, 3): [_POS4_JUBILACION, _POS4_GRUPO_AMPLIO],
    (4, 4): [_POS4_OTROS],
    (4, 5): [_POS4_JUBILACION, _POS4_OTROS],
    (4, 6): [_POS4_GRUPO_AMPLIO, _POS4_OTROS],
    (4, 7): [_POS4_JUBILACION, _POS4_GRUPO_AMPLIO, _POS4_OTROS],
    (5, 1): [((PLAN_EMPLEO,), "inferido")],                  # p. 29; sin monto propio (p. 28)
}

# P21 contiene los montos habituales de la ocupación principal (p. 25): si
# entra P21 en una fracción, D1/D4/F1/F2 no se suman aparte. PP08J1/J2/J3
# (aguinaldo, bonificaciones no habituales, retroactivos) quedan FUERA de P21:
# verificado en los 47 trimestres históricos (05d), con J*>0 nunca están dentro
# de P21 y en 86-100% de los casos entran a P47T por fuera de él -- por eso la
# pos. 2 nunca se descarta por inclusión. PP08D1 ⊇ PP08D4 solo vale para quien
# cumple I1a (con I1b, D4 está fuera de D1) -- se aplica por persona.
INCLUSIONES = {"P21": {"PP08D1", "PP08D4", "PP08F1", "PP08F2"}}
INCLUSIONES_SOLO_I1A = {"PP08D1": {"PP08D4"}}

TRAMOS_FRACCION = ("0", "(0,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1]", ">1")
TOLERANCIA_PESOS = 1.0


def limpiar_monto(serie: pd.Series) -> pd.Series:
    """Monto numérico (`Float64`): `-7`/`-8` -> `0`, `-9` -> `NA`, resto como
    lo deja `tipado_nucleo.parsear_monetario`."""
    # p. 30: -7 "no tenía esa ocupación" y -8 "no tuvo ingresos" en PP06C/PP06D
    # son ausencia de monto, no no-respuesta; -9 sí es Ns/Nr.
    valores = parsear_monetario(serie)
    valores = valores.mask(valores.isin([-7, -8]), 0.0)
    return valores.mask(valores == -9)


def contar_codigos_especiales(df: pd.DataFrame, variables: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Por variable presente en `df`: cantidad de `-7`, `-8`, `-9` y `NA` sobre
    el valor crudo parseado (antes de `limpiar_monto`)."""
    filas = []
    for variable in variables:
        if variable not in df:
            continue
        crudo = parsear_monetario(df[variable])
        filas.append({"variable": variable, "n_menos7": int((crudo == -7).sum()), "n_menos8": int((crudo == -8).sum()),
                      "n_menos9": int((crudo == -9).sum()), "n_na": int(crudo.isna().sum())})
    return pd.DataFrame(filas, columns=["variable", "n_menos7", "n_menos8", "n_menos9", "n_na"])


def leer_marcas_imputacion(anio: int, trimestre: int, ruta_armonizado, tipo: str,
                           columnas_extra: list[str] | None = None) -> pd.DataFrame:
    """Lee de `01_armonizado` las claves, la marca (`IDIMPH`/`IDIMPP`/`IMPUTA`
    según `tipo` y era), los montos y `PJ*` presentes, más `columnas_extra`."""
    archivo = ruta_armonizado / tipo / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet"
    disponibles = pq.read_schema(archivo).names
    if tipo == "hogar":
        deseadas = ["CODUSU", "NRO_HOGAR", "AGLOMERADO", "IDIMPH", "ITF"]
    else:
        deseadas = (["CODUSU", "NRO_HOGAR", "COMPONENTE", "IDIMPP", "IMPUTA"] + list(VARIABLES_MONTO)
                    + list(VARIABLES_PLAN_EMPLEO))
    deseadas += columnas_extra or []
    return pd.read_parquet(archivo, columns=[c for c in dict.fromkeys(deseadas) if c in disponibles])


def descomponer_codigo_imputacion(codigo: pd.Series, digitos_validos: dict[int, set[int]] = DIGITOS_VALIDOS_IDIMPP) -> pd.DataFrame:
    """`pos_1..pos_5` (`Int8`), `alguna_imputacion` y `codigo_invalido` (largo
    != 5, no numérico o dígito fuera de `digitos_validos`); inválido -> `NA`."""
    texto = codigo.astype("string").str.strip()
    resultado = pd.DataFrame(index=codigo.index)
    invalido = ~texto.str.fullmatch(r"\d{5}").fillna(False).astype(bool)
    for pos in POSICIONES:
        digito = pd.to_numeric(texto.str[pos - 1], errors="coerce").astype("Int8")
        invalido |= ~digito.isin(list(digitos_validos[pos])).fillna(False).astype(bool)
        resultado[f"pos_{pos}"] = digito
    for pos in POSICIONES:
        resultado.loc[invalido, f"pos_{pos}"] = pd.NA
    alguna = (texto != CODIGO_SIN_IMPUTAR).astype("boolean")
    alguna[invalido] = pd.NA
    resultado["alguna_imputacion"] = alguna
    resultado["codigo_invalido"] = invalido
    return resultado


def verificar_idimph_contra_idimpp(hogar: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar y posición: `IDIMPH` contra `any(IDIMPP_pos != 0)` de sus
    miembros; `discrepa_pos_k` es `NA` si algún código es inválido."""
    pos_hogar = descomponer_codigo_imputacion(hogar["IDIMPH"], DIGITOS_VALIDOS_IDIMPH)
    pos_persona = descomponer_codigo_imputacion(individual["IDIMPP"])
    claves = ["CODUSU", "NRO_HOGAR"]
    marcas = individual[claves].copy()
    for pos in POSICIONES:
        marcas[f"p{pos}"] = (pos_persona[f"pos_{pos}"] != 0).astype("boolean")
    marcas["invalido"] = pos_persona["codigo_invalido"]
    agregado = marcas.groupby(claves).agg({**{f"p{p}": "max" for p in POSICIONES}, "invalido": "any"})
    resultado = hogar[claves].copy()
    agregado = agregado.reindex(pd.MultiIndex.from_frame(resultado[claves]))
    for pos in POSICIONES:
        marca_hogar = (pos_hogar[f"pos_{pos}"] == 1).astype("boolean")
        marca_miembros = pd.array(agregado[f"p{pos}"].to_numpy(), dtype="boolean")
        discrepa = pd.Series(marca_hogar.to_numpy() != marca_miembros, index=resultado.index, dtype="boolean")
        sin_verificar = pos_hogar["codigo_invalido"].to_numpy() | agregado["invalido"].fillna(True).to_numpy(dtype=bool)
        discrepa[sin_verificar] = pd.NA
        resultado[f"discrepa_pos_{pos}"] = discrepa
    resultado["codigo_hogar_invalido"] = pos_hogar["codigo_invalido"].to_numpy()
    return resultado


def _montos_limpios(individual: pd.DataFrame) -> pd.DataFrame:
    montos = pd.DataFrame(index=individual.index)
    for variable in VARIABLES_MONTO:
        montos[variable] = limpiar_monto(individual[variable]) if variable in individual else pd.Series(pd.NA, index=individual.index, dtype="Float64")
    residuo = montos["T_VI"] - montos[list(V_NO_LABORALES_LISTADAS)].sum(axis=1, skipna=False)
    montos[RESIDUO_NO_LABORAL] = residuo.clip(lower=0)
    montos["_residuo_sin_monto"] = residuo <= 0
    return montos


def verificar_identidad_ocupacion_principal(individual: pd.DataFrame) -> pd.DataFrame:
    """I1a: `P21 == PP08D1+PP08F1+PP08F2` (±$1; `PP08J*` van por fuera de `P21`). I1b: I1a
    + `PP08D4`. Independiente: `P21 == PP06C` o `PP06D`. `variante_i1` = primera
    que se cumple (a, b, independiente) o `NA`."""
    montos = _montos_limpios(individual)
    suma_a = montos[list(COMPONENTES_I1A)].sum(axis=1, skipna=False)
    p21 = montos["P21"]
    cumple_a = ((p21 - suma_a).abs() <= TOLERANCIA_PESOS).fillna(False)
    cumple_b = ((p21 - (suma_a + montos["PP08D4"])).abs() <= TOLERANCIA_PESOS).fillna(False)
    cumple_indep = (((p21 - montos["PP06C"]).abs() <= TOLERANCIA_PESOS)
                    | ((p21 - montos["PP06D"]).abs() <= TOLERANCIA_PESOS)).fillna(False)
    variante = pd.Series(pd.NA, index=individual.index, dtype="object")
    variante[cumple_indep.to_numpy(dtype=bool)] = "independiente"
    variante[cumple_b.to_numpy(dtype=bool)] = "b"
    variante[cumple_a.to_numpy(dtype=bool)] = "a"
    return pd.DataFrame({"cumple_i1": variante.isin(["a", "b"]), "variante_i1": variante}, index=individual.index)


def verificar_identidades_ingreso(hogar: pd.DataFrame, individual: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(I2 por persona: `P47T == P21+PP08J1+PP08J2+PP08J3+TOT_P12+T_VI`, I3 por
    hogar: `ITF == Σ P47T`), ±$1 -- `NA` si falta algún monto."""
    montos = _montos_limpios(individual)
    i2 = ((montos["P47T"] - montos[list(COMPONENTES_I2)].sum(axis=1, skipna=False)).abs() <= TOLERANCIA_PESOS)
    suma_p47t = montos["P47T"].groupby([individual["CODUSU"], individual["NRO_HOGAR"]]).sum(min_count=1)
    suma_p47t = suma_p47t.reindex(pd.MultiIndex.from_frame(hogar[["CODUSU", "NRO_HOGAR"]])).to_numpy()
    i3 = pd.Series((limpiar_monto(hogar["ITF"]).to_numpy(dtype=float, na_value=np.nan) - suma_p47t.astype(float)),
                   index=hogar.index).abs() <= TOLERANCIA_PESOS
    return i2.astype("boolean"), i3.astype("boolean")


def _marcas_variables(individual: pd.DataFrame, identidad: pd.DataFrame, posiciones: pd.DataFrame,
                      solo_posicion: int | None = None) -> dict:
    variables = set(VARIABLES_MONTO) | {RESIDUO_NO_LABORAL}
    def vacio() -> pd.Series:
        return pd.Series(False, index=individual.index)
    en_min = {v: vacio() for v in variables}
    en_max = {v: vacio() for v in variables}
    cumple_i1 = identidad["cumple_i1"].to_numpy(dtype=bool)
    usa_respaldo, usa_cota, usa_inferido, plan_sin_pj = vacio(), vacio(), vacio(), vacio()

    for (pos, digito), entradas in MAPEO_IDIMPP.items():
        if solo_posicion is not None and pos != solo_posicion:
            continue
        marcado = (posiciones[f"pos_{pos}"] == digito).fillna(False).to_numpy(dtype=bool)
        if not marcado.any():
            continue
        for vars_entrada, tipo in entradas:
            if tipo == "exacto_si_i1":
                exacto, respaldo = marcado & cumple_i1, marcado & ~cumple_i1
                for v in vars_entrada:
                    en_min[v] |= exacto
                    en_max[v] |= exacto
                en_max[RESPALDO_OCUPACION_PRINCIPAL] |= respaldo
                usa_respaldo |= respaldo
            elif tipo == "exacto":
                for v in vars_entrada:
                    en_min[v] |= marcado
                    en_max[v] |= marcado
            elif vars_entrada == (PLAN_EMPLEO,):
                alguna_pj = np.zeros(len(individual), dtype=bool)
                for marca_pj, destino in VARIABLES_PLAN_EMPLEO.items():
                    if marca_pj not in individual:
                        continue
                    tiene = (pd.to_numeric(individual[marca_pj], errors="coerce") == 1).fillna(False).to_numpy(dtype=bool)
                    en_max[destino] |= marcado & tiene
                    alguna_pj |= tiene
                plan_sin_pj |= marcado & ~alguna_pj
                usa_inferido |= marcado
            else:
                for v in vars_entrada:
                    en_max[v] |= marcado
                if tipo == "cota_superior":
                    usa_cota |= marcado
                else:
                    usa_inferido |= marcado

    es_i1a = (identidad["variante_i1"] == "a").fillna(False).to_numpy(dtype=bool)
    for marcas in (en_min, en_max):
        for contenedora, contenidas in INCLUSIONES.items():
            for v in contenidas:
                marcas[v] &= ~marcas[contenedora]
        for contenedora, contenidas in INCLUSIONES_SOLO_I1A.items():
            for v in contenidas:
                marcas[v] &= ~(marcas[contenedora] & es_i1a)
    return {"en_min": en_min, "en_max": en_max, "usa_respaldo_p21": usa_respaldo,
            "usa_cota_superior": usa_cota, "usa_inferido": usa_inferido, "plan_sin_pj": plan_sin_pj}


def _menos9_en_marcadas(individual: pd.DataFrame, marcas: dict) -> pd.Series:
    resultado = pd.Series(False, index=individual.index)
    for variable, marcada in marcas.items():
        if variable in individual and marcada.any():
            resultado |= marcada & (parsear_monetario(individual[variable]) == -9).fillna(False).astype(bool)
    return resultado


def _sumar_marcadas(marcas: dict, montos: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    total = pd.Series(0.0, index=montos.index)
    hay_na = pd.Series(False, index=montos.index)
    for variable, marcada in marcas.items():
        if not marcada.any():
            continue
        valor = montos[variable].astype(float)
        hay_na |= marcada & valor.isna()
        total += valor.where(marcada, 0.0).fillna(0.0)
    return total, hay_na


def monto_imputado_persona(individual: pd.DataFrame) -> pd.DataFrame:
    """Por persona: `monto_imputado_min`/`_max` (unión de variables según
    `MAPEO_IDIMPP` respetando inclusiones; `NA` si falta un monto o pos. 5 sin
    `PJ*` en max) y marcas de origen y de causa del `NA`."""
    montos = _montos_limpios(individual)
    identidad = verificar_identidad_ocupacion_principal(individual)
    posiciones = descomponer_codigo_imputacion(individual["IDIMPP"])
    marcas = _marcas_variables(individual, identidad, posiciones)
    monto_min, na_min = _sumar_marcadas(marcas["en_min"], montos)
    monto_max, na_max = _sumar_marcadas(marcas["en_max"], montos)
    menos9 = _menos9_en_marcadas(individual, marcas["en_max"])
    plan_sin_pj = marcas["plan_sin_pj"]
    return pd.DataFrame({
        "monto_imputado_min": monto_min.mask(na_min).astype("Float64"),
        "monto_imputado_max": monto_max.mask(na_max | plan_sin_pj).astype("Float64"),
        "usa_respaldo_p21": marcas["usa_respaldo_p21"], "usa_cota_superior": marcas["usa_cota_superior"],
        "usa_inferido": marcas["usa_inferido"],
        "max_na_plan_sin_pj": plan_sin_pj,
        "max_na_menos9": na_max & menos9 & ~plan_sin_pj,
        "max_na_otra": na_max & ~menos9 & ~plan_sin_pj,
        "alguna_imputacion": posiciones["alguna_imputacion"],
        "codigo_invalido": posiciones["codigo_invalido"],
        "variante_i1": identidad["variante_i1"],
    }, index=individual.index)


def verificar_monto_marcado(individual: pd.DataFrame) -> pd.DataFrame:
    """Por persona y posición marcada: monto mapeado (en max) igual a `0`,
    `-9` crudo o `NA`; `pos4_marcado_sin_monto` (residuo <= 0 con dígito 4) y
    `pos5_sin_pj`."""
    montos = _montos_limpios(individual)
    identidad = verificar_identidad_ocupacion_principal(individual)
    posiciones = descomponer_codigo_imputacion(individual["IDIMPP"])
    resultado = pd.DataFrame(index=individual.index)
    for pos in POSICIONES:
        marcas = _marcas_variables(individual, identidad, posiciones, solo_posicion=pos)
        monto, hay_na = _sumar_marcadas(marcas["en_max"], montos)
        menos9 = _menos9_en_marcadas(individual, marcas["en_max"])
        marcado = (posiciones[f"pos_{pos}"] > 0).fillna(False).astype(bool)
        con_monto_definido = marcado & ~hay_na & ~marcas["plan_sin_pj"]
        resultado[f"pos{pos}_marcado_monto_cero"] = con_monto_definido & (monto == 0)
        resultado[f"pos{pos}_marcado_monto_menos9"] = marcado & menos9
        resultado[f"pos{pos}_marcado_monto_na"] = marcado & hay_na & ~menos9
        if pos == 5:
            resultado["pos5_sin_pj"] = marcas["plan_sin_pj"]
    digito4 = posiciones["pos_4"].isin([4, 5, 6, 7]).fillna(False).astype(bool)
    resultado["pos4_marcado_sin_monto"] = digito4 & montos["_residuo_sin_monto"].fillna(False).astype(bool)
    return resultado


def fraccion_itf_imputada(hogar: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar: `fraccion_min`/`fraccion_max` = Σ monto imputado / `ITF`
    (`NA` si `ITF==0` o algún monto de esa fracción es `NA`), `alguna_imputacion`
    según `IDIMPH` y causa del `NA` de max."""
    personas = monto_imputado_persona(individual)
    claves = [individual["CODUSU"], individual["NRO_HOGAR"]]
    columnas_any = ["max_na_plan_sin_pj", "max_na_menos9", "max_na_otra", "usa_respaldo_p21",
                    "usa_cota_superior", "usa_inferido"]
    agregado = personas[columnas_any].groupby(claves).any()
    for lado in ("min", "max"):
        monto = personas[f"monto_imputado_{lado}"].astype(float)
        suma = monto.fillna(0.0).groupby(claves).sum()
        algun_na = monto.isna().groupby(claves).any()
        agregado[f"monto_imputado_{lado}"] = suma.mask(algun_na)
    resultado = hogar[["CODUSU", "NRO_HOGAR"]].copy()
    agregado = agregado.reindex(pd.MultiIndex.from_frame(resultado))
    itf = limpiar_monto(hogar["ITF"]).astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for lado in ("min", "max"):
            monto = pd.to_numeric(agregado[f"monto_imputado_{lado}"], errors="coerce").to_numpy(dtype=float)
            fraccion = np.where(itf > 0, monto / itf, np.nan)
            resultado[f"monto_imputado_{lado}"] = pd.array(monto, dtype="Float64")
            resultado[f"fraccion_{lado}"] = pd.array(fraccion, dtype="Float64")
    for columna in ("max_na_plan_sin_pj", "max_na_menos9", "max_na_otra", "usa_respaldo_p21",
                    "usa_cota_superior", "usa_inferido"):
        resultado[columna] = agregado[columna].fillna(False).to_numpy(dtype=bool)
    resultado["alguna_imputacion"] = descomponer_codigo_imputacion(hogar["IDIMPH"], DIGITOS_VALIDOS_IDIMPH)[
        "alguna_imputacion"].to_numpy()
    resultado["itf_cero"] = itf == 0
    return resultado


def tramo_fraccion(fraccion: pd.Series) -> pd.Series:
    """Tramo de la fracción imputada: `"0"`, `"(0,0.1]"`, `"(0.1,0.25]"`,
    `"(0.25,0.5]"`, `"(0.5,1]"`, `">1"` (discrepancia) o `NA`."""
    f = fraccion.astype(float)
    condiciones = [f == 0, (f > 0) & (f <= 0.1), (f > 0.1) & (f <= 0.25), (f > 0.25) & (f <= 0.5),
                   (f > 0.5) & (f <= 1 + 1e-9), f > 1 + 1e-9]
    tramo = np.select(condiciones, TRAMOS_FRACCION, default="")
    return pd.Series(tramo, index=fraccion.index, dtype="object").replace("", pd.NA)


def clasificar_estado_imputacion_par(imp_t: pd.Series, imp_th: pd.Series) -> pd.Series:
    """`"ninguna"`, `"solo_t"`, `"solo_th"` o `"ambas"`; `NA` si falta una punta."""
    t, th = imp_t.astype("boolean"), imp_th.astype("boolean")
    valido = (t.notna() & th.notna()).to_numpy(dtype=bool)
    t_np, th_np = t.fillna(False).to_numpy(dtype=bool), th.fillna(False).to_numpy(dtype=bool)
    estado = np.select([~t_np & ~th_np, t_np & ~th_np, ~t_np & th_np, t_np & th_np],
                       ["ninguna", "solo_t", "solo_th", "ambas"], default="")
    resultado = pd.Series(estado, index=imp_t.index, dtype="object")
    resultado[~valido] = pd.NA
    return resultado


def resumen_dispersion(serie: pd.Series) -> dict:
    """`n`, desvío, IQR, p10 y p90 de los valores no nulos."""
    valores = serie.dropna().astype(float)
    if valores.empty:
        return {"n": 0, "desvio": np.nan, "iqr": np.nan, "p10": np.nan, "p90": np.nan}
    q10, q25, q75, q90 = np.quantile(valores, [0.10, 0.25, 0.75, 0.90])
    return {"n": int(len(valores)), "desvio": float(valores.std()), "iqr": float(q75 - q25),
            "p10": float(q10), "p90": float(q90)}


def pendiente_regresion_media(x: pd.Series, y: pd.Series) -> float:
    """Pendiente MCO de `y` sobre `x` (filas con ambos no nulos); `NaN` con
    menos de 2 puntos o `x` constante."""
    valido = x.notna() & y.notna()
    xv, yv = x[valido].astype(float).to_numpy(), y[valido].astype(float).to_numpy()
    if len(xv) < 2 or np.var(xv) == 0:
        return np.nan
    return float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv))


def reponderar_dentro_de_celda(df: pd.DataFrame, peso: str, excluir: pd.Series, celdas: list[str]) -> pd.Series:
    """Peso de los no excluidos escalado por `Σ_celda peso / Σ_celda peso
    no excluido` (conserva el total de cada celda); excluidos -> `0`."""
    w = df[peso].astype(float)
    excluir = excluir.fillna(False).astype(bool)
    total = w.groupby([df[c] for c in celdas]).transform("sum")
    retenido = w.where(~excluir, 0.0).groupby([df[c] for c in celdas]).transform("sum")
    factor = (total / retenido.replace(0, np.nan)).fillna(0.0)
    return (w * factor).where(~excluir, 0.0)
