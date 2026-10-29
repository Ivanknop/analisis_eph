"""Canasta GBA (CBA/CBT por adulto equivalente) 2003-01/2016-03, dos anclas
(Tornarolli 2018, CEDLAS DT 226): diciembre 2006 (Metodología INDEC N°22) y
abril 2016 (canasta oficial), extendida a las 6 regiones. No calcula
distancia por hogar."""
# Las validaciones y el ensamblado con la canasta oficial viven en
# `notebooks/05b_canastas_retropoladas.ipynb`, no en este módulo.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from canasta_regional import REGION_POR_COLUMNA, agregar_trimestral, cargar_canasta_larga

REGION_GBA = REGION_POR_COLUMNA["gran_buenos_aires"]

INICIO_RETROPOLACION = pd.Timestamp("2003-01-01")
ANCLA_DIC_2006 = pd.Timestamp("2006-12-01")
ANCLA_ABR_2016 = pd.Timestamp("2016-04-01")
MES_EMPALME_CIFRA = pd.Timestamp("2007-01-01")
FIN_RETROPOLACION = pd.Timestamp("2016-03-01")

ORIGEN_OFICIAL = "oficial"
ORIGEN_DOS_ANCLAS = "retropolada_2_anclas"
ORIGEN_UNA_ANCLA = "retropolada_1_ancla"

_CONCEPTO_POR_VARIABLE = {"cba_ae": "cba_ae_metodologia_nueva", "cbt_ae": "cbt_ae_metodologia_nueva"}


def cargar_ancla_dic_2006(ruta_csv: Path) -> dict[str, float]:
    """Ancla de diciembre 2006 (columna "Nuevo" de Metodología INDEC N°22) -- {"cba_ae": 149.13, "cbt_ae": 389.28}."""
    valores = pd.read_csv(ruta_csv).set_index("concepto")["valor"]
    return {variable: float(valores[concepto]) for variable, concepto in _CONCEPTO_POR_VARIABLE.items()}


def cargar_ancla_abr_2016(canastas_dir: Path) -> pd.DataFrame:
    """Ancla de abril 2016 por región, canasta oficial -- REGION, cba_ae, cbt_ae (6 filas)."""
    cba = cargar_canasta_larga(canastas_dir / "cba_regional_crudo.csv", "cba_ae")
    cbt = cargar_canasta_larga(canastas_dir / "cbt_regional_crudo.csv", "cbt_ae")
    cba_abr = cba[(cba["ANO4"] == 2016) & (cba["MES"] == 4)][["REGION", "cba_ae"]]
    cbt_abr = cbt[(cbt["ANO4"] == 2016) & (cbt["MES"] == 4)][["REGION", "cbt_ae"]]
    return cba_abr.merge(cbt_abr, on="REGION").reset_index(drop=True)


def encadenar_indice(indec: pd.Series, cifra: pd.Series, mes_empalme: pd.Timestamp = MES_EMPALME_CIFRA,
                      mes_ultimo_indec: pd.Timestamp = ANCLA_DIC_2006) -> tuple[pd.Series, float]:
    """Encadena IPC-GBA INDEC (hasta `mes_ultimo_indec`) con IPC Provincias CIFRA
    reescalado (desde `mes_empalme`), ambos indexados por mes. Devuelve el índice
    encadenado y la variación mensual INDEC usada en el empalme."""
    # El empalme reescala TODA la serie CIFRA (no solo desde mes_empalme) por un único
    # factor constante -- preserva el crecimiento propio de CIFRA mes a mes, no
    # reintroduce el nivel INDEC en ningún punto salvo el de empalme.
    variacion = indec[mes_empalme] / indec[mes_ultimo_indec] - 1
    nivel_empalme = indec[mes_ultimo_indec] * (1 + variacion)
    cifra_reescalado = cifra * (nivel_empalme / cifra[mes_empalme])
    indice = pd.concat([
        indec[indec.index <= mes_ultimo_indec],
        cifra_reescalado[cifra_reescalado.index >= mes_empalme],
    ]).sort_index()
    return indice, float(variacion)


def peso_lineal(mes: pd.Timestamp, inicio: pd.Timestamp = ANCLA_DIC_2006, fin: pd.Timestamp = ANCLA_ABR_2016) -> float:
    """w(mes): 1 en `inicio`, 0 en `fin`, lineal en meses calendario entre medio."""
    meses_totales = (fin.year - inicio.year) * 12 + (fin.month - inicio.month)
    meses_desde_inicio = (mes.year - inicio.year) * 12 + (mes.month - inicio.month)
    return 1 - meses_desde_inicio / meses_totales


def backcast_simple(valor_ancla: float, indice: pd.Series, mes_ancla: pd.Timestamp) -> pd.Series:
    """C(m) = valor_ancla x indice[m] / indice[mes_ancla], para todo m en `indice`."""
    return valor_ancla * indice / indice[mes_ancla]


def blend_dos_anclas(c_b: pd.Series, c_a: pd.Series, pesos: pd.Series) -> pd.Series:
    """w·C_B + (1-w)·C_A, elemento a elemento (mismo índice de meses en los tres)."""
    return pesos * c_b + (1 - pesos) * c_a


def construir_canasta_gba_mensual(valor_ancla_dic_2006: float, valor_ancla_abr_2016: float,
                                   indice: pd.Series, variable: str) -> pd.DataFrame:
    """Canasta GBA mensual 2003-01/2016-03 para `variable` (cba_ae o cbt_ae): un
    ancla en 2003-2006, dos anclas (más la versión de sensibilidad de un solo
    ancla) en 2007-01/2016-03. Devuelve REGION(=GBA), ANO4, MES, `variable`, origen."""
    tramo_2003_2006 = indice[(indice.index >= INICIO_RETROPOLACION) & (indice.index <= ANCLA_DIC_2006)]
    c_2003_2006 = backcast_simple(valor_ancla_dic_2006, tramo_2003_2006, ANCLA_DIC_2006)

    tramo_2007_2016 = indice[(indice.index >= MES_EMPALME_CIFRA) & (indice.index <= FIN_RETROPOLACION)]
    c_a = backcast_simple(valor_ancla_abr_2016, indice, ANCLA_ABR_2016).reindex(tramo_2007_2016.index)
    c_b = backcast_simple(valor_ancla_dic_2006, indice, ANCLA_DIC_2006).reindex(tramo_2007_2016.index)
    pesos = tramo_2007_2016.index.to_series().apply(peso_lineal)
    c_dos_anclas = blend_dos_anclas(c_b, c_a, pesos)

    partes = [
        pd.DataFrame({"mes": c_2003_2006.index, "valor": c_2003_2006.to_numpy(), "origen": ORIGEN_DOS_ANCLAS}),
        pd.DataFrame({"mes": c_dos_anclas.index, "valor": c_dos_anclas.to_numpy(), "origen": ORIGEN_DOS_ANCLAS}),
        pd.DataFrame({"mes": c_a.index, "valor": c_a.to_numpy(), "origen": ORIGEN_UNA_ANCLA}),
    ]
    resultado = pd.concat(partes, ignore_index=True)
    resultado["REGION"] = REGION_GBA
    resultado["ANO4"] = resultado["mes"].dt.year
    resultado["MES"] = resultado["mes"].dt.month
    return resultado.rename(columns={"valor": variable})[["REGION", "ANO4", "MES", variable, "origen"]]


def extender_regiones(canasta_gba: pd.DataFrame, ancla_abr_2016: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Extiende una canasta GBA (REGION, ANO4, MES, `variable`, origen) a las 6
    regiones vía el ratio de la canasta oficial en abril 2016: C_r(m) = C_GBA(m) x
    C_r(abr-2016) / C_GBA(abr-2016)."""
    ratio_regional = ancla_abr_2016.set_index("REGION")[variable]
    valor_gba_abr_2016 = ratio_regional.loc[REGION_GBA]
    partes = []
    for region, valor_abr_2016 in ratio_regional.items():
        extendida = canasta_gba.copy()
        extendida["REGION"] = region
        extendida[variable] = extendida[variable] * (valor_abr_2016 / valor_gba_abr_2016)
        partes.append(extendida)
    return pd.concat(partes, ignore_index=True)


def agregar_trimestral_por_origen(mensual: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Promedio mensual->trimestral ("a_mismo_trimestre", D25/D26) de `variable`,
    separado por `origen` -- reusa `canasta_regional.agregar_trimestral` por grupo
    (esa función no conoce la columna `origen`)."""
    partes = []
    for origen, grupo in mensual.groupby("origen"):
        trimestral = agregar_trimestral(grupo[["REGION", "ANO4", "MES", variable]], variable, "a_mismo_trimestre")
        trimestral["origen"] = origen
        partes.append(trimestral)
    return pd.concat(partes, ignore_index=True)


def semestres_historicos_computables(trimestres_disponibles: list[tuple[int, int]]) -> list[tuple]:
    """Semestres 2003-2015 con los 2 trimestres componentes disponibles --
    `(anio, semestre, componentes, es_movil)`. 2007-II usa el semestre móvil
    2007T4+2008T1 en vez de 2007T3+2007T4 (2007T3 no se publicó)."""
    disponibles = set(trimestres_disponibles)
    resultado = []
    for anio in range(2003, 2016):
        for semestre, (t1, t2) in ((1, (1, 2)), (2, (3, 4))):
            if anio == 2003 and semestre == 1:
                continue
            componentes = ((anio, t1), (anio, t2))
            if componentes[0] in disponibles and componentes[1] in disponibles:
                resultado.append((anio, semestre, componentes, False))
            elif anio == 2007 and semestre == 2:
                movil = ((2007, 4), (2008, 1))
                if movil[0] in disponibles and movil[1] in disponibles:
                    resultado.append((anio, semestre, movil, True))
    return resultado
