"""Capa tipada de variables núcleo (ingresos, empleo, vivienda, estrategias del
hogar) sobre el Parquet ya armonizado del notebook 01. No toca `data/eph_cache/`
ni re-lee DBF, ni recodifica valores especiales (-9, 9, 0 "no corresponde").
"""
# Ver `ESQUEMA_NUCLEO` para los códigos especiales documentados por variable,
# citando la fuente (codebook `data/EPH_tot_urbano_estructura_bases_2025.pdf`
# cuando cubre la variable; "inferido" cuando no).
from __future__ import annotations

import re

import pandas as pd

_PATRON_COMA_DECIMAL = re.compile(r"\d,\d+$")
_PATRON_PUNTO_DECIMAL = re.compile(r"\d\.\d+$")

CODEBOOK = "data/EPH_tot_urbano_estructura_bases_2025.pdf"

# Códigos especiales generales (codebook, "Comentarios generales", pág. 31):
# 9/99/999/9999 = Ns/Nr salvo indicación contraria; montos de ingreso -9 = no
# respuesta; PP06C/PP06D además -7/-8; 0 = "no corresponde a la secuencia".
CODIGOS_ESPECIALES_GENERALES = {
    "no_sabe_no_responde": "9, 99, 999, 9999 según longitud del campo",
    "no_respuesta_monto_ingreso": -9,
    "no_correspondia_ocupacion_pp06c_pp06d": -7,
    "no_tuvo_ingresos_pp06c_pp06d": -8,
    "no_corresponde_secuencia": 0,
}

_VIVIENDA_CODIGOS_ESPECIALES = {"no_corresponde": 0}

ESQUEMA_NUCLEO: dict[str, dict] = {
    # --- Hogar: ingresos ---
    "IPCF": {"base": "hogar", "regla_parseo": "monetario",
             "fuente": f"{CODEBOOK} p.12 (N(12,2))"},
    "ITF": {"base": "hogar", "regla_parseo": "monetario",
            "fuente": f"{CODEBOOK} p.12 (N(12))"},
    "PONDIH": {"base": "hogar", "regla_parseo": "entero_directo", "dtype": "Int32",
               "fuente": f"{CODEBOOK} p.12/p.30 -- tipo inconsistente en el propio "
                         "codebook (C(6) en p.12, N(6) en p.30); tratado como numérico "
                         "porque así se comporta en los datos reales.",
               "codigos_especiales": {"no_corresponde_o_ausente_pre_2016": None}},
    # --- Hogar: deciles (no cubiertos por el codebook) ---
    **{
        v: {"base": "hogar", "regla_parseo": "entero_con_blancos",
            "fuente": "inferido -- no cubierto por el codebook (ver diagnóstico §0a/§2)",
            "codigos_especiales": {"00": "ingreso cero real (peso normal)",
                                    "12": "no clasificable (PONDIH=0 en la práctica)"}}
        for v in ("DECCFR", "DECIFR", "ADECCFR", "ADECIFR", "RDECCFR", "RDECIFR")
    },
    # --- Individual: ingresos y núcleo laboral ---
    "P21": {"base": "individual", "regla_parseo": "monetario",
            "fuente": f"{CODEBOOK} p.29",
            "codigos_especiales": {"no_respuesta": -9}},
    "T_VI": {"base": "individual", "regla_parseo": "monetario",
             "fuente": f"{CODEBOOK} p.30 (N(12), monto total de ingresos no laborales)"},
    "PONDERA": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int32",
                "fuente": f"{CODEBOOK} p.7/p.16 (N(6))"},
    "ESTADO": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
               "fuente": f"{CODEBOOK} p.18"},
    "CAT_OCUP": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
                 "fuente": f"{CODEBOOK} p.18", "codigos_especiales": {"ns_nr": 9}},
    "PP07H": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
              "fuente": f"{CODEBOOK} p.25"},
    "CH03": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
             "fuente": f"{CODEBOOK} p.16",
             "codigos_especiales": {"1": "jefe/a", "2": "conyuge/pareja",
                                     "3": "hijo/a/hijastro/a", "4": "yerno/nuera",
                                     "5": "nieto/a", "6": "madre/padre", "7": "suegro/a",
                                     "8": "hermano/a", "9": "otros familiares",
                                     "10": "no familiares"}},
    "CH04": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
             "fuente": f"{CODEBOOK} p.16"},
    "CH06": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
             "fuente": f"{CODEBOOK} p.16"},
    "NIVEL_ED": {"base": "individual", "regla_parseo": "entero_directo", "dtype": "Int8",
                 "fuente": f"{CODEBOOK} p.17", "codigos_especiales": {"ns_nr": 9}},
}

# Vivienda: las 27 variables verificadas (19 + 8), códigos confirmados contra el
# codebook (p.7-9) y contra los datos (2023T3 vs 2023T4, ver diagnóstico §0d/§3).
_VIVIENDA = (
    "II1", "II2", "II3", "II3_1", "II4_1", "II4_2", "II4_3", "II5", "II5_1",
    "II6", "II6_1", "II7", "II8", "II9",
    "IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7", "IV8", "IV9",
    "IV10", "IV11", "IV12_1", "IV12_2", "IV12_3",
)
for _v in _VIVIENDA:
    ESQUEMA_NUCLEO[_v] = {
        "base": "hogar", "regla_parseo": "entero_directo", "dtype": "Int8",
        "fuente": f"{CODEBOOK} p.7-9",
        "codigos_especiales": dict(_VIVIENDA_CODIGOS_ESPECIALES),
    }

# Variables V de hogar (sí/no/ns-nc/no corresponde) -- cada una tipada por
# separado, sin fusionar viejas y nuevas (ver diagnóstico §0e/§4).
_V_HOGAR = (
    "V1", "V2", "V2_01", "V2_02", "V2_03",
    "V21_01", "V21_02", "V21_03", "V22_01", "V22_02", "V22_03",
    "V3", "V4", "V5", "V5_1", "V5_2", "V5_3", "V5_01", "V5_02", "V5_03",
    "V6", "V7", "V8", "V9", "V10", "V11", "V11_1", "V11_2", "V11_01", "V11_02",
    "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19_A", "V19_B",
    "V21", "V22",
)
for _v in _V_HOGAR:
    ESQUEMA_NUCLEO[_v] = {
        "base": "hogar", "regla_parseo": "entero_directo", "dtype": "Int8",
        "fuente": f"{CODEBOOK} p.10-12 (sub-ítems 2023T4+); raíces pre-2023T4 no "
                  "cubiertas por el codebook",
    }

# Variables V*_M de individual -- montos, confirmado por el codebook (p.29-30).
_V_INDIVIDUAL_M = (
    "V2_M", "V2_01_M", "V2_02_M", "V2_03_M",
    "V21_01_M", "V21_02_M", "V21_03_M", "V22_01_M", "V22_02_M", "V22_03_M",
    "V3_M", "V4_M", "V5_M", "V5_01_M", "V5_02_M", "V5_03_M",
    "V8_M", "V9_M", "V10_M", "V11_M", "V11_01_M", "V11_02_M",
    "V12_M", "V18_M", "V19_AM", "V21_M",
)
for _v in _V_INDIVIDUAL_M:
    ESQUEMA_NUCLEO[_v] = {
        "base": "individual", "regla_parseo": "monetario",
        "fuente": f"{CODEBOOK} p.29-30",
    }


def hash_esquema() -> str:
    """Hash estable de `ESQUEMA_NUCLEO` -- cambia si se agrega/quita/redefine
    una variable. Sirve para detectar, en `data/meta/resumen_02_nucleo.csv`,
    con qué versión del esquema se tipó cada partición."""
    # Ordena las claves antes de hashear: el orden de iteración de un dict no
    # está garantizado entre procesos/versiones de Python, y sin ordenar el
    # hash no sería reproducible. Complementa a `particion_desactualizada`
    # (pipeline_meta.py), que solo compara mtimes de datos, no cambios de
    # código como este.
    import hashlib
    contenido = repr(sorted(ESQUEMA_NUCLEO.items()))
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:12]


def parsear_monetario(serie: pd.Series) -> pd.Series:
    """Cubre coma decimal (mayoría de los trimestres) y punto decimal (2021T1/T3,
    ver diagnóstico §1) con la misma regla -- reemplaza coma por punto antes de
    convertir, no-op si no hay coma."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype("Float64")
    return pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False), errors="coerce").astype("Float64")


def detectar_formato_decimal(serie: pd.Series) -> dict:
    """Para variables monetarias en texto: ¿el trimestre usa coma decimal,
    punto decimal, o mezcla ambos en la misma columna? Series ya numéricas no
    tienen ambigüedad de separador."""
    # La mezcla es la anomalía real (2021T1/T3 usan punto, el resto coma) --
    # esto verifica que cada trimestre sea internamente consistente, no lo
    # asume.
    if pd.api.types.is_numeric_dtype(serie):
        return {"tiene_coma": False, "tiene_punto": False, "mezcla": False}
    texto = serie.dropna().astype(str)
    tiene_coma = bool(texto.str.contains(_PATRON_COMA_DECIMAL, regex=True).any())
    tiene_punto = bool(texto.str.contains(_PATRON_PUNTO_DECIMAL, regex=True).any())
    return {"tiene_coma": tiene_coma, "tiene_punto": tiene_punto, "mezcla": tiene_coma and tiene_punto}


def parsear_entero_con_blancos(serie: pd.Series) -> pd.Series:
    """Para deciles: `str.strip()` primero (cubre `'  '` como nulo, ver
    diagnóstico §1), zero-padding no afecta el valor casteado."""
    limpio = serie.astype(str).str.strip().replace("", pd.NA).replace("nan", pd.NA)
    return pd.to_numeric(limpio, errors="coerce").astype("Int16")


def parsear_entero_directo(serie: pd.Series, dtype: str = "Int8") -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").astype(dtype)


def tipar_variable(df: pd.DataFrame, variable: str) -> pd.Series:
    if variable not in ESQUEMA_NUCLEO:
        raise KeyError(f"'{variable}' no está en ESQUEMA_NUCLEO")
    info = ESQUEMA_NUCLEO[variable]
    serie = df[variable]
    if info["regla_parseo"] == "monetario":
        return parsear_monetario(serie)
    if info["regla_parseo"] == "entero_con_blancos":
        return parsear_entero_con_blancos(serie)
    if info["regla_parseo"] == "entero_directo":
        return parsear_entero_directo(serie, info.get("dtype", "Int8"))
    raise ValueError(f"regla_parseo desconocida para '{variable}': {info['regla_parseo']}")


def reporte_no_parseables(serie_original: pd.Series, serie_tipada: pd.Series,
                           anio: int, trimestre: int, tipo: str, variable: str) -> list[dict]:
    """Valores no-nulos en el original que quedaron nulos tras tipar (falló el
    parseo, no es un nulo legítimo) -- agrupado por valor original, no fila por
    fila. Nunca se imputa ni se descarta en silencio: esto es lo que se reporta."""
    original_no_nulo = serie_original.notna() & (serie_original.astype(str).str.strip() != "")
    fallo = original_no_nulo & serie_tipada.isna()
    if not fallo.any():
        return []
    conteo = serie_original[fallo].astype(str).value_counts()
    return [
        {"anio": anio, "trimestre": trimestre, "tipo": tipo, "variable": variable,
         "valor_original": valor, "cantidad": int(cantidad)}
        for valor, cantidad in conteo.items()
    ]


def ingreso_no_declarado(hogar_df: pd.DataFrame, es_historico: bool) -> pd.Series:
    """`PONDIH == 0` en era regular (incluye `DECCFR == 12` y los casos de decil
    nulo con PONDIH=0, ver diagnóstico de la sección "Flag ingreso_no_declarado").
    `NA` en toda la era histórica -- no hay `PONDIH` de hogar antes de 2016."""
    if es_historico or "PONDIH" not in hogar_df.columns:
        return pd.Series(pd.NA, index=hogar_df.index, dtype="boolean")
    pondih = pd.to_numeric(hogar_df["PONDIH"], errors="coerce")
    flag = (pondih == 0).astype("boolean")
    if "DECCFR" in hogar_df.columns:
        deccfr = parsear_entero_con_blancos(hogar_df["DECCFR"])
        es_12 = deccfr == 12
        assert flag[es_12.fillna(False)].fillna(False).all(), (
            "hay hogares con DECCFR==12 y PONDIH!=0 -- la relación de subconjunto asumida no se cumple"
        )
    return flag
