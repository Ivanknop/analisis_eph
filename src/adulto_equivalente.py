"""Unidades de adulto equivalente por sexo y edad (D25), tabla INDEC citada
junto a `TABLA_ADULTO_EQUIVALENTE`.
"""
from __future__ import annotations

import pandas as pd

# Fuente: INDEC, *Canasta básica alimentaria. Canasta básica total. Preguntas
# frecuentes* (Notas al pie N°3, junio de 2020, ISSN 2618-2319,
# ISBN 978-950-896-584-4), Tabla 1, p.6.
# (edad_min, edad_max) inclusive -> (unidades mujer, unidades varón). `-1` es la
# edad real de "menor de 1 año" en CH06 (D19: nunca vale 0), no un dato faltante.
# El último tramo (76+) no tiene techo -- CH06 llega hasta 110 en los datos reales.
TABLA_ADULTO_EQUIVALENTE = (
    ((-1, -1), (0.35, 0.35)),   # menor de 1 año
    ((1, 1), (0.37, 0.37)),
    ((2, 2), (0.46, 0.46)),
    ((3, 3), (0.51, 0.51)),
    ((4, 4), (0.55, 0.55)),
    ((5, 5), (0.60, 0.60)),
    ((6, 6), (0.64, 0.64)),
    ((7, 7), (0.66, 0.66)),
    ((8, 8), (0.68, 0.68)),
    ((9, 9), (0.69, 0.69)),
    ((10, 10), (0.70, 0.79)),
    ((11, 11), (0.72, 0.82)),
    ((12, 12), (0.74, 0.85)),
    ((13, 13), (0.76, 0.90)),
    ((14, 14), (0.76, 0.96)),
    ((15, 15), (0.77, 1.00)),
    ((16, 16), (0.77, 1.03)),
    ((17, 17), (0.77, 1.04)),
    ((18, 29), (0.76, 1.02)),
    ((30, 45), (0.77, 1.00)),
    ((46, 60), (0.76, 1.00)),
    ((61, 75), (0.67, 0.83)),
    ((76, 9999), (0.63, 0.74)),
)


def calcular_unidades_equivalentes(ch04: pd.Series, ch06: pd.Series) -> pd.Series:
    """Unidades de adulto equivalente por persona según `CH04`/`CH06`. `NA`
    si `ch04` no es 1/2 o `ch06` no cae en ningún tramo de la tabla."""
    sexo = pd.to_numeric(ch04, errors="coerce")
    edad = pd.to_numeric(ch06, errors="coerce")
    resultado = pd.Series([pd.NA] * len(edad), index=edad.index, dtype="Float64")
    for (edad_min, edad_max), (unidades_mujer, unidades_varon) in TABLA_ADULTO_EQUIVALENTE:
        en_tramo = edad.between(edad_min, edad_max)
        resultado[en_tramo & (sexo == 1)] = unidades_varon
        resultado[en_tramo & (sexo == 2)] = unidades_mujer
    return resultado


def calcular_unidades_ae_hogar(individual: pd.DataFrame) -> pd.Series:
    """Suma de `calcular_unidades_equivalentes` por hogar (`CODUSU, NRO_HOGAR`).
    `NA` si algún integrante del hogar no tiene unidades determinables -- no se
    trata a esa persona como `0` en silencio."""
    unidades = calcular_unidades_equivalentes(individual["CH04"], individual["CH06"])
    grupo = [individual["CODUSU"], individual["NRO_HOGAR"]]
    suma = unidades.groupby(grupo).sum(min_count=1)
    hogar_con_na = unidades.isna().groupby(grupo).any()
    suma[hogar_con_na] = pd.NA
    return suma
