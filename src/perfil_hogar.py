
import numpy as np
import pandas as pd

def calcular_grupo_ingreso(hogar: pd.DataFrame) -> pd.Series:
    """Quintil (Q1-Q5) desde DECCFR, más `decil_0` e `ingreso_no_declarado` aparte
    (ver nota metodológica arriba). `NA` si DECCFR no es parseable como 0-10 o 12."""
    deccfr = pd.to_numeric(hogar["DECCFR"], errors="coerce")
    grupo = pd.Series(pd.array([pd.NA] * len(hogar), dtype="object"), index=hogar.index)
    valido = deccfr.between(1, 10)
    grupo[valido] = "Q" + (((deccfr[valido].astype("Int64") + 1) // 2)).astype(str)
    grupo[deccfr == 0] = "decil_0"
    grupo[deccfr == 12] = "ingreso_no_declarado"
    if "PONDIH" in hogar.columns:
        pondih = pd.to_numeric(hogar["PONDIH"], errors="coerce")
        grupo[(pondih == 0) & grupo.isna()] = "ingreso_no_declarado"
    return grupo


def calcular_situacion_laboral(individual: pd.DataFrame) -> pd.DataFrame:
    """4 estados por hogar, por prioridad (D-pendiente Parte E punto 5):
    algún_asalariado_formal > algún_independiente_sin_asalariado_formal >
    solo_asalariados_informales > sin_ocupados."""
    estado = pd.to_numeric(individual["ESTADO"], errors="coerce")
    cat_ocup = pd.to_numeric(individual["CAT_OCUP"], errors="coerce")
    pp07h = pd.to_numeric(individual["PP07H"], errors="coerce")
    ocupado = estado == 1
    formal = ocupado & (cat_ocup == 3) & (pp07h == 1)
    independiente = ocupado & cat_ocup.isin([1, 2])
    informal_asalariado = ocupado & (cat_ocup == 3) & (pp07h == 2)

    g = pd.DataFrame({"CODUSU": individual["CODUSU"], "NRO_HOGAR": individual["NRO_HOGAR"],
                       "formal": formal, "independiente": independiente, "informal_asalariado": informal_asalariado})
    ag = g.groupby(["CODUSU", "NRO_HOGAR"])[["formal", "independiente", "informal_asalariado"]].any()
    situacion = pd.Series("sin_ocupados", index=ag.index, dtype="object")
    situacion[ag["informal_asalariado"]] = "solo_asalariados_informales"
    situacion[ag["independiente"]] = "algun_independiente_sin_asalariado_formal"
    situacion[ag["formal"]] = "algun_asalariado_formal"
    return situacion.rename("situacion_laboral").reset_index()

def calcular_decil(hogar: pd.DataFrame) -> pd.Series:
    '''Entero 1-10 desde DECCFR -- NA si no es un decil válido (decil_0/12/no parseable).'''
    deccfr = pd.to_numeric(hogar["DECCFR"], errors="coerce")
    decil = pd.Series(pd.array([pd.NA] * len(hogar), dtype="Int8"), index=hogar.index)
    valido = deccfr.between(1, 10)
    decil[valido] = deccfr[valido].astype("Int8")
    return decil

def bucket_nivel_ed(nivel_ed: pd.Series) -> pd.Series:
    """2 buckets explícitos (para no fragmentar celdas en Parte E): 1-3 = secundario
    incompleto o menos, 4-6 = secundario completo o más. 7 (sin instrucción) entra en
    el primer bucket; 9 (Ns/Nr) y nulos quedan fuera (NA)."""
    n = pd.to_numeric(nivel_ed, errors="coerce")
    bucket = pd.Series(pd.array([pd.NA] * len(n), dtype="object"), index=n.index)
    bucket[n.isin([1, 2, 3, 7])] = "secundario incompleto o menos"
    bucket[n.isin([4, 5, 6])] = "secundario completo o más"
    return bucket

def to_bool(serie: pd.Series, codigos: list[int]) -> pd.Series:
    '''Nullable boolean: True si el valor numérico está en `codigos`, NA si el valor
    original no es numérico/está vacío, False en el resto -- a diferencia de
    `pd.to_numeric(...) == valor` '''
    num = pd.to_numeric(serie, errors="coerce")
    out = num.isin(codigos).astype("boolean")
    out[num.isna()] = pd.NA
    return out

def kleene_any_hogar(individual: pd.DataFrame, serie_bool: pd.Series) -> pd.Series:
    '''any() Kleene-aware agrupado por (CODUSU, NRO_HOGAR): True si algún miembro
    cumple, NA si nadie cumple pero todos tienen el dato faltante, False en el resto
    (distinto del `.any()` default de pandas, que trata NA como False).'''
    d = pd.DataFrame({"CODUSU": individual["CODUSU"], "NRO_HOGAR": individual["NRO_HOGAR"], "v": serie_bool})
    g = d.groupby(["CODUSU", "NRO_HOGAR"])
    algun_true = (d["v"].fillna(False) == True).groupby([d["CODUSU"], d["NRO_HOGAR"]]).any()
    todos_na = d["v"].isna().groupby([d["CODUSU"], d["NRO_HOGAR"]]).all()
    resultado = pd.Series(False, index=algun_true.index, dtype="boolean")
    resultado[todos_na] = pd.NA
    resultado[algun_true] = True
    return resultado


def calcular_componentes_ancla(hogar: pd.DataFrame, n_miembros: pd.Series) -> pd.DataFrame:
    '''Componentes del ancla habitacional (adaptación del criterio NBI del INDEC) y el
    flag combinado `ancla_con_deficit`. `hogar` debe estar indexado por
    `(CODUSU, NRO_HOGAR)`; `n_miembros` es una Series con el mismo índice (o un
    superconjunto -- se reindexa acá). Uso general: versión completa (6 columnas) de
    `calcular_componentes_ancla`, antes duplicada parcialmente (subconjunto de 4
    columnas, misma fórmula) en 04_trayectorias/06_poblacion_riesgo.'''
    n = n_miembros.reindex(hogar.index)
    ii1 = pd.to_numeric(hogar["II1"], errors="coerce")
    cuartos_dormir = pd.to_numeric(hogar["II2"], errors="coerce").fillna(0)
    tiene_ii5 = pd.to_numeric(hogar["II5"], errors="coerce") == 1
    cuartos_dormir = cuartos_dormir + pd.to_numeric(hogar["II5_1"], errors="coerce").where(tiene_ii5, 0).fillna(0)

    hacinamiento_nbi = (n / ii1.replace(0, np.nan)) > 3
    hacinamiento_para_dormir = (n / cuartos_dormir.replace(0, np.nan)) > 3

    iv1 = pd.to_numeric(hogar["IV1"], errors="coerce")
    iv3 = pd.to_numeric(hogar["IV3"], errors="coerce")
    calidad_materiales_deficitaria = iv1.isin([3, 4, 5]) | (iv3 == 3)

    iv8 = pd.to_numeric(hogar["IV8"], errors="coerce")
    iv10 = pd.to_numeric(hogar["IV10"], errors="coerce")
    condiciones_sanitarias_deficitarias = (iv8 == 2) | (iv10 == 3)

    entorno_deficitario = (
        (pd.to_numeric(hogar["IV12_1"], errors="coerce") == 1)
        | (pd.to_numeric(hogar["IV12_2"], errors="coerce") == 1)
        | (pd.to_numeric(hogar["IV12_3"], errors="coerce") == 1)
    )

    ancla_con_deficit = (
        hacinamiento_nbi.fillna(False) | calidad_materiales_deficitaria.fillna(False)
        | condiciones_sanitarias_deficitarias.fillna(False)
    )

    return pd.DataFrame({
        "hacinamiento_nbi": hacinamiento_nbi, "hacinamiento_para_dormir": hacinamiento_para_dormir,
        "calidad_materiales_deficitaria": calidad_materiales_deficitaria,
        "condiciones_sanitarias_deficitarias": condiciones_sanitarias_deficitarias,
        "entorno_deficitario": entorno_deficitario, "ancla_con_deficit": ancla_con_deficit,
    }, index=hogar.index)