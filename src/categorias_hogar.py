"""Categorizaciones de hogar como foto de un solo trimestre: composición,
situación laboral y precariedad, fuentes de ingreso, vivienda y estrategias.
Funciones puras sobre `hogar`/`individual` de un trimestre; no leen `data/`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from perfil_hogar import (bucket_nivel_ed, calcular_componentes_ancla, calcular_grupo_ingreso, calcular_info_jefe,
                          calcular_situacion_laboral, clasificar_precariedad_persona, clasificar_situacion_escolar)

CLAVE_HOGAR = ["CODUSU", "NRO_HOGAR"]

COLUMNAS_HOGAR = [
    "CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "ITF", "IPCF", "PONDIH", "DECCFR",
    "II1", "II2", "II5", "II5_1", "II7", "II8", "IV1", "IV3", "IV4", "IV6", "IV8", "IV10", "IV11",
    "IV12_1", "IV12_2", "IV12_3", "V2", "V13", "V14", "V15", "V16", "V17", "ingreso_no_declarado",
]
COLUMNAS_INDIVIDUAL_NUCLEO = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "AGLOMERADO", "CH03", "CH04", "CH06",
    "ESTADO", "CAT_OCUP", "PP07H", "PONDERA", "P21", "T_VI",
]

# Fuentes de ingreso por era de cuestionario. Desde 2023T4 INDEC abre V2, V5,
# V11, V21 y V22 en componentes `_0x_M`; sumarlos reproduce `T_VI` (100% en
# 2023T4-2025T4), por eso se armoniza sumando y no se marca solo el corte.
FUENTES_LABORALES = ["P21", "TOT_P12", "PP08J1", "PP08J2", "PP08J3"]
FUENTES_SOLO_OCUPADOS = ["PP08J1", "PP08J2", "PP08J3"]
FUENTES_OTROS_NO_LABORALES = ["V3_M", "V4_M", "V8_M", "V9_M", "V10_M", "V12_M", "V18_M", "V19_AM"]
FUENTES_HASTA_2023T3 = {"jubilacion": ["V2_M", "V21_M"], "transferencias": ["V5_M", "V11_M"]}
FUENTES_DESDE_2023T4 = {
    "jubilacion": ["V2_01_M", "V2_02_M", "V2_03_M", "V21_01_M", "V21_02_M", "V21_03_M",
                   "V22_01_M", "V22_02_M", "V22_03_M"],
    "transferencias": ["V5_01_M", "V5_02_M", "V5_03_M", "V11_01_M", "V11_02_M"],
}
TRIMESTRE_CUESTIONARIO_NUEVO = (2023, 4)

EDAD_MENOR_14, EDAD_MENOR_15, EDAD_MENOR_18, EDAD_MAYOR_65 = 14, 15, 18, 65
CALIFICACIONES_VALIDAS = (1, 2, 3, 4)
CALIFICACIONES_NO_PRECARIAS = (1, 2)


def fuentes_por_trimestre(anio: int, trimestre: int) -> dict[str, list[str]]:
    """Columnas de cada fuente (`laboral`, `jubilacion`, `transferencias`,
    `otros_no_laborales`) según el cuestionario vigente en el trimestre."""
    variables = FUENTES_DESDE_2023T4 if (anio, trimestre) >= TRIMESTRE_CUESTIONARIO_NUEVO else FUENTES_HASTA_2023T3
    return {"laboral": FUENTES_LABORALES, **variables, "otros_no_laborales": FUENTES_OTROS_NO_LABORALES}


def _numerico(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").astype(float)


def _por_hogar(individual: pd.DataFrame, valores: pd.Series | pd.DataFrame, agregacion: str = "sum"):
    return valores.groupby([individual["CODUSU"], individual["NRO_HOGAR"]]).agg(agregacion)


def calcular_estrategias(hogar: pd.DataFrame) -> pd.DataFrame:
    """`recurre_desahorro` (V13/V14/V17) y `recurre_endeudamiento_formal`
    (V15/V16), mismo índice que `hogar`."""
    v = {c: pd.to_numeric(hogar[c], errors="coerce") for c in ("V13", "V14", "V15", "V16", "V17")}
    recurre_desahorro = (v["V13"] == 1) | (v["V14"] == 1) | (v["V17"] == 1)
    recurre_endeudamiento_formal = (v["V15"] == 1) | (v["V16"] == 1)
    return pd.DataFrame({"recurre_desahorro": recurre_desahorro,
                         "recurre_endeudamiento_formal": recurre_endeudamiento_formal}, index=hogar.index)


def grupo_ingreso_en_trimestre(hogar: pd.DataFrame, claves: pd.DataFrame) -> pd.Series:
    """`calcular_grupo_ingreso` de `hogar`, alineado a las filas de `claves`
    (`CODUSU`, `NRO_HOGAR`). `NA` donde el hogar no está en `hogar`."""
    grupos = hogar[CLAVE_HOGAR].assign(grupo_ingreso=calcular_grupo_ingreso(hogar))
    return claves[CLAVE_HOGAR].merge(grupos, on=CLAVE_HOGAR, how="left")["grupo_ingreso"].set_axis(claves.index)


def composicion_hogar(individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar: conteos por edad, `share_dependientes`, y sexo/edad/nivel
    educativo/condición del jefe. Features del jefe en `NA` si `jefe_ambiguo`."""
    edad = _numerico(individual["CH06"])
    # CH06 == -1 es "menor de 1 año" en la EPH: cuenta como menor, no como faltante.
    conteos = pd.DataFrame({
        "n_miembros": 1,
        "n_menores_14": edad < EDAD_MENOR_14,
        "n_menores_15": edad < EDAD_MENOR_15,
        "n_menores_18": edad < EDAD_MENOR_18,
        "n_mayores_65": edad >= EDAD_MAYOR_65,
    }).astype(int)
    resultado = _por_hogar(individual, conteos)
    # Dependientes sobre el total y no sobre la población de 15-64: así un hogar
    # sin miembros en edad activa no queda en NA.
    resultado["share_dependientes"] = (resultado["n_menores_15"] + resultado["n_mayores_65"]) / resultado["n_miembros"]

    jefe = calcular_info_jefe(individual).set_index(CLAVE_HOGAR)
    es_jefe = individual["CH03"] == 1
    edad_jefe = edad[es_jefe].groupby([individual.loc[es_jefe, "CODUSU"], individual.loc[es_jefe, "NRO_HOGAR"]]).first()
    jefe["edad_jefe"] = edad_jefe.reindex(jefe.index)
    jefe["nivel_ed_jefe"] = _numerico(jefe["nivel_ed_jefe"]).where(lambda s: s.between(1, 7))
    jefe["sexo_jefe"] = _numerico(jefe["sexo_jefe"])
    columnas_jefe = ["sexo_jefe", "edad_jefe", "nivel_ed_jefe", "condicion_actividad_jefe"]
    jefe.loc[jefe["jefe_ambiguo"], columnas_jefe] = np.nan
    jefe["nivel_ed_jefe_bucket"] = bucket_nivel_ed(jefe["nivel_ed_jefe"])
    resultado = resultado.join(jefe[columnas_jefe + ["nivel_ed_jefe_bucket", "jefe_ambiguo"]], how="left")
    # Un hogar sin ninguna fila con CH03 == 1 no aparece en calcular_info_jefe.
    resultado["jefe_ambiguo"] = resultado["jefe_ambiguo"].fillna(True).astype(bool)
    return resultado.reset_index()


def hay_menor_no_asiste(individual: pd.DataFrame) -> pd.Series:
    """Por hogar: `True` si algún miembro de 6-17 no asiste y no completó el
    secundario; `False` sin menores en edad escolar; `NA` si no se puede decidir."""
    situacion = clasificar_situacion_escolar(individual["CH10"], individual["CH06"], individual["NIVEL_ED"])
    en_edad = _numerico(individual["CH06"]).between(6, 17)
    # No usa `kleene_any_hogar`: ahí un adulto con False alcanza para dar False
    # aunque un menor no tenga dato; acá basta un menor sin dato para no poder decidir.
    no_asiste = en_edad & (situacion == "no_asiste_no_completo").fillna(False).astype(bool)
    sin_dato = en_edad & situacion.isna()
    algun_no_asiste = _por_hogar(individual, no_asiste, "any")
    resultado = pd.Series(False, index=algun_no_asiste.index, dtype="boolean")
    resultado[_por_hogar(individual, sin_dato, "any")] = pd.NA
    resultado[algun_no_asiste] = True
    return resultado.rename("hay_menor_no_asiste")


def calificacion_ocupacion(pp04d_cod: pd.Series) -> pd.Series:
    """5º dígito de `PP04D_COD` (CNO): 1 profesional, 2 técnica, 3 operativa,
    4 no calificada. `NA` fuera de 1-4 (no especificada o sin código)."""
    # Se completa a 5 dígitos con ceros: los códigos de directivos (`05001`)
    # pierden el cero inicial al guardarse como número, y leer el dígito por
    # posición sin completarlo da un 5º dígito inexistente.
    codigo = pd.to_numeric(pp04d_cod, errors="coerce")
    valido = codigo.notna() & (codigo >= 0) & (codigo < 100000)
    digito = pd.Series(np.nan, index=pp04d_cod.index)
    digito[valido] = codigo[valido].astype("int64").astype(str).str.zfill(5).str[4].astype(int)
    return digito.where(digito.isin(CALIFICACIONES_VALIDAS))


def laboral_hogar(individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar: `situacion_laboral` (4 niveles), conteos de ocupados,
    desocupados y activos, `sin_activos` y `share_ocupados` (ocupados/miembros)."""
    estado = _numerico(individual["ESTADO"])
    conteos = pd.DataFrame({"n_miembros": 1, "n_ocupados": estado == 1, "n_desocupados": estado == 2}).astype(int)
    resultado = _por_hogar(individual, conteos)
    resultado["n_activos"] = resultado["n_ocupados"] + resultado["n_desocupados"]
    resultado["sin_activos"] = resultado["n_activos"] == 0
    resultado["share_ocupados"] = resultado["n_ocupados"] / resultado["n_miembros"]
    situacion = calcular_situacion_laboral(individual).set_index(CLAVE_HOGAR)["situacion_laboral"]
    resultado["situacion_laboral"] = situacion.reindex(resultado.index)
    return resultado.drop(columns="n_miembros").reset_index()


def precariedad_hogar(individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar: `share_precario_{A,B,C,D}` sobre los activos clasificables
    de cada variante (NA sin activos clasificables) y sus denominadores."""
    # A: asalariados informales + desocupados. B: A + cuenta propia. C: B + patrón.
    # D: A + cuenta propia de calificación operativa o no calificada. Se calculan
    # las cuatro; la elección queda para la Fase D (decisión del usuario 2026-11-10).
    categoria = clasificar_precariedad_persona(individual)
    clasificable = categoria.notna() & _numerico(individual["ESTADO"]).isin([1, 2])
    informal_o_desocupado = categoria.isin(["informal_asalariado", "desocupado"])
    cuenta_propia = categoria == "cuenta_propia"
    calificacion = calificacion_ocupacion(individual["PP04D_COD"]) if "PP04D_COD" in individual else pd.Series(np.nan, index=individual.index)
    cuenta_propia_no_calificada = cuenta_propia & ~calificacion.isin(CALIFICACIONES_NO_PRECARIAS) & calificacion.notna()
    clasificable_d = clasificable & ~(cuenta_propia & calificacion.isna())

    marcas = pd.DataFrame({
        "n_activos_clasificables": clasificable,
        "n_activos_clasificables_D": clasificable_d,
        "precario_A": clasificable & informal_o_desocupado,
        "precario_B": clasificable & (informal_o_desocupado | cuenta_propia),
        "precario_C": clasificable & (informal_o_desocupado | cuenta_propia | (categoria == "patron")),
        "precario_D": clasificable_d & (informal_o_desocupado | cuenta_propia_no_calificada),
    }).astype(int)
    suma = _por_hogar(individual, marcas)
    resultado = suma[["n_activos_clasificables", "n_activos_clasificables_D"]].copy()
    for variante in "ABC":
        resultado[f"share_precario_{variante}"] = suma[f"precario_{variante}"] / suma["n_activos_clasificables"].replace(0, np.nan)
    resultado["share_precario_D"] = suma["precario_D"] / suma["n_activos_clasificables_D"].replace(0, np.nan)
    return resultado.reset_index()


def fuentes_ingreso_hogar(individual: pd.DataFrame, fuentes: dict[str, list[str]], itf: pd.Series) -> pd.DataFrame:
    """Por hogar: `share_{fuente}` = suma de la fuente / `ITF`, más
    `share_residuo_no_laboral` (`T_VI` − no laborales listadas). NA si
    `ITF <= 0` o si algún monto usado es faltante o código negativo."""
    # `itf` indexado por (CODUSU, NRO_HOGAR). Los códigos negativos (-9 = no
    # sabe/no responde) se tratan como faltantes: no se suman como cero.
    resultado = pd.DataFrame(index=itf.index)
    denominador = _numerico(itf).where(lambda s: s > 0)
    no_ocupado = (_numerico(individual["ESTADO"]) != 1).to_numpy()
    no_laborales_listados = []
    for fuente, columnas in fuentes.items():
        montos = pd.DataFrame({c: _numerico(individual[c]) if c in individual else np.nan for c in columnas})
        montos = montos.where(montos >= 0)
        # En la era regular PP08J1-3 vienen vacíos fuera de los ocupados (la
        # pregunta solo se hace a ocupados): ahí es "no corresponde", no faltante.
        for c in FUENTES_SOLO_OCUPADOS:
            if c in montos and c in individual:
                vacio_no_corresponde = no_ocupado & _numerico(individual[c]).isna().to_numpy()
                montos.loc[vacio_no_corresponde, c] = 0.0
        total = _por_hogar(individual, montos.sum(axis=1), "sum")
        hay_faltante = _por_hogar(individual, montos.isna().any(axis=1), "any")
        total = total.where(~hay_faltante).reindex(itf.index)
        resultado[f"share_{fuente}"] = total / denominador
        if fuente != "laboral":
            no_laborales_listados.append(total)
    t_vi = _numerico(individual["T_VI"]).where(lambda s: s >= 0)
    t_vi_hogar = _por_hogar(individual, t_vi, "sum").where(~_por_hogar(individual, t_vi.isna(), "any")).reindex(itf.index)
    residuo = t_vi_hogar - sum(no_laborales_listados)
    resultado["share_residuo_no_laboral"] = residuo / denominador
    return resultado.rename_axis(CLAVE_HOGAR).reset_index()


def vivienda_hogar(hogar: pd.DataFrame, n_miembros: pd.Series) -> pd.DataFrame:
    """Componentes de vivienda y hábitat por hogar, índice V1
    (`ancla_con_deficit`) y V2 amplia (V1 + techo, agua, hoyo, villa).
    `hogar` indexado por (`CODUSU`, `NRO_HOGAR`)."""
    ancla = calcular_componentes_ancla(hogar, n_miembros)
    codigo = {c: pd.to_numeric(hogar[c], errors="coerce") for c in ("IV4", "IV6", "IV11", "IV12_1", "IV12_2", "IV12_3", "II7", "II8")}
    # IV4 == 9 es "N/S, propiedad horizontal" y el 0 de IV9-IV11 es "no
    # corresponde": ninguno cuenta como precario.
    resultado = pd.DataFrame({
        "hacinamiento_nbi": ancla["hacinamiento_nbi"].fillna(False).astype(bool),
        "calidad_materiales_deficitaria": ancla["calidad_materiales_deficitaria"].fillna(False).astype(bool),
        "condiciones_sanitarias_deficitarias": ancla["condiciones_sanitarias_deficitarias"].fillna(False).astype(bool),
        "techo_precario": codigo["IV4"].isin([6, 7]),
        "agua_fuera_terreno": codigo["IV6"] == 3,
        "desague_a_hoyo": codigo["IV11"] == 4,
        "villa": codigo["IV12_3"] == 1,
        "entorno_basural_o_inundable": (codigo["IV12_1"] == 1) | (codigo["IV12_2"] == 1),
        "tenencia_precaria": codigo["II7"].isin([6, 7]),
        "inquilino": codigo["II7"] == 3,
        "cocina_lena_kerosene": codigo["II8"] == 3,
    }, index=hogar.index)
    resultado["vivienda_precaria_v1"] = ancla["ancla_con_deficit"].astype(bool)
    resultado["vivienda_precaria_v2"] = resultado[
        ["vivienda_precaria_v1", "techo_precario", "agua_fuera_terreno", "desague_a_hoyo", "villa"]].any(axis=1)
    return resultado.rename_axis(CLAVE_HOGAR).reset_index()
