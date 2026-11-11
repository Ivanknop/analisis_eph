"""Variables de hogar sin ingresos para modelar la no respuesta (vivienda, hogar,
composición, jefe con rama CAES y dimensiones del CNO), propensión de respuesta
fuera de fold, pesos inversos recortados e imputación de sensibilidad."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import validacion as va
from features import CLAVE_PAR

CLAVE_HOGAR_TRIMESTRE = ["CODUSU", "NRO_HOGAR", "AGLOMERADO", "ANO4", "TRIMESTRE"]
VARIABLES_VIVIENDA = ["IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7", "IV8", "IV9", "IV10", "IV11",
                      "IV12_1", "IV12_2", "IV12_3"]
VARIABLES_HOGAR = ["II1", "II2", "II3", "II3_1", "II4_1", "II4_2", "II4_3", "II5", "II5_1", "II6", "II6_1",
                   "II7", "II8", "II9"]
VARIABLES_COMPOSICION = ["IX_TOT", "IX_MEN10", "IX_MAYEQ10"]
VARIABLES_JEFE = ["CH04", "CH06", "CH07", "CH08", "NIVEL_ED", "ESTADO", "CAT_OCUP", "CAT_INAC", "PP04A", "PP04C",
                  "PP3E_TOT", "PP07G1", "PP07G2", "PP07G3", "PP07G4", "PP07H", "PP07I", "PP07K"]
VARIABLES_RAMA = ("PP04B_COD", "PP04B_CAES")
VARIABLES_NO_CORRESPONDE_CERO = ("PP04A", "PP04C", "PP07G1", "PP07G2", "PP07G3", "PP07G4", "PP07H", "PP07I", "PP07K")
# El CAES de la EPH cambia en 2012T1 (verificado en los microdatos: construcción pasa de
# 4500 a 4000 y hogares de 9500 a 9700 de un trimestre al siguiente). Entre 2012T1 y 2015T2
# la columna se llama PP04B_CAES; la armonización no la unificó con PP04B_COD.
INICIO_CAES_MERCOSUR = (2012, 1)
RAMAS = ["agro_extractiva", "industria", "electricidad_agua", "construccion", "comercio", "hoteles_restaurantes",
         "transporte_comunicaciones", "finanzas_inmobiliarias_empresariales", "administracion_publica",
         "ensenanza", "salud", "otros_servicios", "servicio_domestico"]
# Divisiones (2 primeros dígitos) -> gran rama. [inferido] de los códigos más frecuentes y
# de la estructura CIIU rev. 3 (CAES EPH 2003) y rev. 4 (CAES-MERCOSUR 1.0).
_DIVISIONES_CAES_2003 = [((1, 5), "agro_extractiva"), ((10, 14), "agro_extractiva"), ((15, 39), "industria"),
                         ((40, 41), "electricidad_agua"), ((45, 45), "construccion"), ((50, 54), "comercio"),
                         ((55, 55), "hoteles_restaurantes"), ((60, 64), "transporte_comunicaciones"),
                         ((65, 74), "finanzas_inmobiliarias_empresariales"), ((75, 75), "administracion_publica"),
                         ((80, 80), "ensenanza"), ((85, 85), "salud"), ((90, 93), "otros_servicios"),
                         ((95, 95), "servicio_domestico"), ((99, 99), "otros_servicios")]
_DIVISIONES_CAES_MERCOSUR = [((1, 9), "agro_extractiva"), ((10, 33), "industria"), ((35, 39), "electricidad_agua"),
                             ((40, 43), "construccion"), ((45, 48), "comercio"), ((49, 53), "transporte_comunicaciones"),
                             ((55, 56), "hoteles_restaurantes"), ((58, 63), "transporte_comunicaciones"),
                             ((64, 82), "finanzas_inmobiliarias_empresariales"), ((84, 84), "administracion_publica"),
                             ((85, 85), "ensenanza"), ((86, 88), "salud"), ((90, 96), "otros_servicios"),
                             ((97, 98), "servicio_domestico"), ((99, 99), "otros_servicios")]
CATEGORICAS = ["IV1", "IV3", "IV4", "IV6", "IV7", "IV8", "IV9", "IV10", "IV11", "II7", "II8", "II9", "REGION",
               "aglomerado_cat", "CH07_jefe", "ESTADO_jefe", "CAT_OCUP_jefe", "CAT_INAC_jefe", "PP04A_jefe",
               "rama_jefe", "cno_caracter_jefe", "cno_jerarquia_jefe", "cno_tecnologia_jefe", "cobertura_jefe"]
PARAMETROS_LGBM = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 2, "verbose": -1,
                   "seed": 0}
RONDAS_LGBM = 300
P_MIN_PESO = 0.05
PERCENTIL_RECORTE_PESO = 0.99
N_FOLDS_PROPENSION = 5
# Codebook INDEC (`data/EPH_tot_urbano_estructura_bases_2025.pdf`, variable AGLOMERADO).
NOMBRES_AGLOMERADO = {
    2: "Gran La Plata", 3: "Bahía Blanca-Cerri", 4: "Gran Rosario", 5: "Gran Santa Fe", 6: "Gran Paraná",
    7: "Posadas", 8: "Gran Resistencia", 9: "Comodoro Rivadavia-Rada Tilly", 10: "Gran Mendoza", 12: "Corrientes",
    13: "Gran Córdoba", 14: "Concordia", 15: "Formosa", 17: "Neuquén-Plottier", 18: "Santiago del Estero-La Banda",
    19: "Jujuy-Palpalá", 20: "Río Gallegos", 22: "Gran Catamarca", 23: "Gran Salta", 25: "La Rioja",
    26: "Gran San Luis", 27: "Gran San Juan", 29: "Gran Tucumán-Tafí Viejo", 30: "Santa Rosa-Toay",
    31: "Ushuaia-Río Grande", 32: "Ciudad Autónoma de Buenos Aires", 33: "Partidos del Gran Buenos Aires",
    34: "Mar del Plata", 36: "Río Cuarto", 38: "San Nicolás-Villa Constitución", 91: "Rawson-Trelew",
    93: "Viedma-Carmen de Patagones",
}
COLUMNAS_FUERA_PROPENSION = (*CLAVE_HOGAR_TRIMESTRE, "era", "n_jefes")
COLUMNAS_PARES_PROPENSION = [*CLAVE_PAR, "ANO4_th", "TRIMESTRE_th", "AGLOMERADO_t", "AGLOMERADO_th", "era_t", "identidad_confirmada",
                             "par_valido_simetrica", "par_valido_c"]


def rama_caes(codigo: pd.Series, anio: int, trimestre: int) -> pd.Series:
    """Gran rama (`RAMAS`) desde la división del código CAES vigente en el trimestre; `NA` si no clasifica."""
    numero = pd.to_numeric(codigo, errors="coerce")
    division = (numero // 100).where(numero >= 100)
    tabla = _DIVISIONES_CAES_MERCOSUR if (anio, trimestre) >= INICIO_CAES_MERCOSUR else _DIVISIONES_CAES_2003
    resultado = pd.Series(pd.NA, index=codigo.index, dtype="object")
    for (desde, hasta), rama in tabla:
        resultado[division.between(desde, hasta).fillna(False).to_numpy()] = rama
    return resultado


def dimensiones_cno(codigo: pd.Series) -> pd.DataFrame:
    """Carácter (dígitos 1-2), jerarquía (3), tecnología (4) y calificación (5) del
    CNO-2001 desde `PP04D_COD`, completado a 5 dígitos; `NA` fuera de rango."""
    # Igual que `categorias_hogar.calificacion_ocupacion`: el cero inicial se pierde
    # cuando el código se guarda como número (era regular).
    numero = pd.to_numeric(codigo, errors="coerce")
    valido = numero.notna() & (numero >= 0) & (numero < 100000) & (numero >= 1000)
    texto = numero.where(valido).astype("Int64").astype("string").str.zfill(5)
    resultado = pd.DataFrame({
        "cno_caracter": pd.to_numeric(texto.str[:2], errors="coerce"),
        "cno_jerarquia": pd.to_numeric(texto.str[2], errors="coerce"),
        "cno_tecnologia": pd.to_numeric(texto.str[3], errors="coerce"),
        "cno_calificacion": pd.to_numeric(texto.str[4], errors="coerce"),
    }, index=codigo.index)
    rangos = {"cno_jerarquia": (0, 3), "cno_tecnologia": (0, 3), "cno_calificacion": (1, 4), "cno_caracter": (1, 99)}
    for columna, (desde, hasta) in rangos.items():
        resultado[columna] = resultado[columna].where(resultado[columna].between(desde, hasta))
    return resultado


def cobertura_medica(ch08: pd.Series) -> pd.Series:
    """`obra_social` (contiene 1), `prepaga` (2 sin 1), `publica` (solo 3), `sin_cobertura` (4); `NA` en 9 o vacío."""
    digitos = pd.to_numeric(ch08, errors="coerce").astype("Int64").astype("string")
    condiciones = [digitos.str.contains("1"), digitos.str.contains("2"), digitos == "3", digitos == "4"]
    valores = np.select([c.fillna(False).to_numpy(dtype=bool) for c in condiciones],
                        ["obra_social", "prepaga", "publica", "sin_cobertura"], default="")
    return pd.Series(valores, index=ch08.index, dtype="object").replace("", pd.NA)


def _leer(ruta: Path, columnas) -> pd.DataFrame:
    disponibles = pq.read_schema(ruta).names
    return pd.read_parquet(ruta, columns=[c for c in dict.fromkeys(columnas) if c in disponibles])


def _numerico(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[columna], errors="coerce").astype(float)


def variables_hogar(anio: int, trimestre: int, ruta_armonizado: Path) -> pd.DataFrame:
    """Una fila por hogar del trimestre con vivienda, hogar, composición, agregados
    laborales y del jefe/a (NA si 0 o más de un `CH03 == 1`). Ningún monto de ingreso."""
    base = Path(ruta_armonizado)
    particion = f"ANO4={anio}/TRIMESTRE={trimestre}/data.parquet"
    hogar = _leer(base / "hogar" / particion, ["CODUSU", "NRO_HOGAR", "AGLOMERADO", "REGION", "MAS_500",
                                               *VARIABLES_VIVIENDA, *VARIABLES_HOGAR, *VARIABLES_COMPOSICION])
    individual = _leer(base / "individual" / particion, ["CODUSU", "NRO_HOGAR", "CH03", "PP04D_COD",
                                                         *VARIABLES_JEFE, *VARIABLES_RAMA])
    for columna in [c for c in hogar.columns if c not in ("CODUSU", "NRO_HOGAR", "AGLOMERADO", "MAS_500")]:
        hogar[columna] = _numerico(hogar, columna)
    # 2003T3-2009T2: IX_MEN10 viene en NA justo cuando no hay menores de 10 (IX_TOT == IX_MAYEQ10
    # en todos esos casos); sin corregir, el NA marca el período dentro de la era histórica.
    if {"IX_MEN10", "IX_TOT", "IX_MAYEQ10"} <= set(hogar.columns):
        sin_menores = hogar["IX_MEN10"].isna() & hogar["IX_TOT"].eq(hogar["IX_MAYEQ10"])
        hogar.loc[sin_menores, "IX_MEN10"] = 0.0
    # Copia del aglomerado para usarlo como predictor categórico sin tocar la clave de merge.
    hogar["aglomerado_cat"] = hogar["AGLOMERADO"]
    hogar["MAS_500"] = (hogar["MAS_500"].astype(str).str.upper() == "S").astype(float)
    columna_rama = next((c for c in VARIABLES_RAMA if c in individual), None)
    individual["rama"] = (rama_caes(individual[columna_rama], anio, trimestre) if columna_rama
                          else pd.Series(pd.NA, index=individual.index, dtype="object"))
    individual = individual.join(dimensiones_cno(individual["PP04D_COD"]))
    individual["cobertura"] = cobertura_medica(individual["CH08"])
    estado, cat_inac = _numerico(individual, "ESTADO"), _numerico(individual, "CAT_INAC")
    asalariado = _numerico(individual, "CAT_OCUP").eq(3) & estado.eq(1)
    agregados = pd.DataFrame({
        "CODUSU": individual["CODUSU"], "NRO_HOGAR": individual["NRO_HOGAR"],
        "n_ocupados": estado.eq(1), "n_desocupados": estado.eq(2), "n_inactivos": estado.eq(3),
        "n_jubilados": cat_inac.eq(1), "n_estudiantes": cat_inac.eq(3),
        "n_sin_cobertura": individual["cobertura"].eq("sin_cobertura"),
        "n_asalariados": asalariado, "n_asalariados_con_descuento": asalariado & _numerico(individual, "PP07H").eq(1),
        "n_jefes": _numerico(individual, "CH03").eq(1),
    }).groupby(["CODUSU", "NRO_HOGAR"]).sum().reset_index()
    jefes = individual[_numerico(individual, "CH03").eq(1)]
    jefes = jefes[~jefes.duplicated(["CODUSU", "NRO_HOGAR"], keep=False)]
    columnas_jefe = [c for c in [*VARIABLES_JEFE, "rama", "cno_caracter", "cno_jerarquia", "cno_tecnologia",
                                 "cno_calificacion", "cobertura"] if c in jefes]
    jefe = jefes[["CODUSU", "NRO_HOGAR", *columnas_jefe]].copy()
    for columna in [c for c in columnas_jefe if c not in ("rama", "cobertura", "CH08")]:
        jefe[columna] = _numerico(jefe, columna)
    # "No corresponde" es 0 en la era histórica y vacío en la regular; 0 no es respuesta
    # válida en estas preguntas. Las horas sí admiten 0: ahí manda la condición de ocupado.
    for columna in [c for c in VARIABLES_NO_CORRESPONDE_CERO if c in jefe]:
        jefe[columna] = jefe[columna].mask(jefe[columna].eq(0))
    if "PP3E_TOT" in jefe:
        jefe["PP3E_TOT"] = jefe["PP3E_TOT"].where(_numerico(jefe, "ESTADO").eq(1))
    jefe = jefe.drop(columns=["CH08"]).rename(columns={c: f"{c}_jefe" for c in columnas_jefe if c != "CH08"})
    if "rama_jefe" in jefe:
        jefe["rama_jefe"] = jefe["rama_jefe"].where(jefe["ESTADO_jefe"].eq(1))
    resultado = hogar.merge(agregados, on=["CODUSU", "NRO_HOGAR"], how="left").merge(jefe, on=["CODUSU", "NRO_HOGAR"], how="left")
    beneficios = resultado.reindex(columns=[f"PP07G{i}_jefe" for i in range(1, 5)])
    resultado["beneficios_jefe"] = beneficios.eq(1).sum(axis=1).where(_numerico(resultado, "CAT_OCUP_jefe").eq(3))
    resultado["ambientes_por_persona"] = _numerico(resultado, "II1") / resultado["IX_TOT"].replace(0, np.nan)
    resultado["share_ocupados"] = resultado["n_ocupados"] / resultado["IX_TOT"].replace(0, np.nan)
    return resultado.assign(ANO4=anio, TRIMESTRE=trimestre)


def a_categoricas(df: pd.DataFrame, columnas: list[str] = CATEGORICAS) -> pd.DataFrame:
    """Copia con `columnas` presentes como `category` (mismas categorías en todo el DataFrame)."""
    resultado = df.copy()
    for columna in [c for c in columnas if c in resultado]:
        resultado[columna] = resultado[columna].astype("string").astype("category")
    return resultado


def grupos_vivienda(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """Fold (0..n_folds-1) por vivienda (`CODUSU` + `AGLOMERADO`), estable entre corridas."""
    clave = df["CODUSU"].astype(str) + "_" + df["AGLOMERADO"].astype(str)
    return (pd.util.hash_pandas_object(clave, index=False).to_numpy() % n_folds).astype(int)


def _ajustar(X: pd.DataFrame, y: np.ndarray, objetivo: str, peso: np.ndarray | None = None,
             num_threads: int | None = None):
    import lightgbm as lgb

    parametros = {**PARAMETROS_LGBM, "objective": objetivo}
    if num_threads is not None:
        parametros["num_threads"] = num_threads
    datos = lgb.Dataset(X, label=y, weight=peso, categorical_feature="auto", free_raw_data=False)
    return lgb.train(parametros, datos, num_boost_round=RONDAS_LGBM)


def ajustar_lgbm(X: pd.DataFrame, y: pd.Series, objetivo: str = "binary"):
    """LightGBM (`PARAMETROS_LGBM`, 2 hilos) sobre todas las filas."""
    return _ajustar(X, y.astype(float).to_numpy(), objetivo)


def predecir_fuera_de_fold(X: pd.DataFrame, y: pd.Series, fold: np.ndarray, objetivo: str = "binary",
                           num_threads: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Predicción fuera de fold de LightGBM e importancia por ganancia (media entre folds, normalizada a 1)."""
    prediccion = pd.Series(np.nan, index=X.index)
    importancias = []
    for k in np.unique(fold):
        entrena, prueba = fold != k, fold == k
        modelo = _ajustar(X[entrena], y[entrena].astype(float).to_numpy(), objetivo, num_threads=num_threads)
        prediccion[prueba] = modelo.predict(X[prueba])
        importancias.append(pd.Series(modelo.feature_importance("gain"), index=X.columns))
    importancia = pd.concat(importancias, axis=1).mean(axis=1)
    return prediccion, (importancia / importancia.sum()).sort_values(ascending=False).rename("ganancia").to_frame()


def pesos_inversos(probabilidad: pd.Series, p_min: float = 0.05, percentil_recorte: float = 0.99) -> pd.Series:
    """`1 / max(p, p_min)`, recortado al percentil `percentil_recorte` de los pesos resultantes."""
    peso = 1.0 / probabilidad.astype(float).clip(lower=p_min)
    return peso.clip(upper=peso.quantile(percentil_recorte))


def resumen_pesos(peso: pd.Series) -> dict:
    """Cuantiles, máximo, CV y tamaño efectivo de Kish (`(Σw)²/Σw²`) relativo a `n`."""
    w = peso.dropna().astype(float)
    q = np.quantile(w, [0.01, 0.1, 0.5, 0.9, 0.99]) if len(w) else [np.nan] * 5
    return {"n": len(w), "p01": q[0], "p10": q[1], "p50": q[2], "p90": q[3], "p99": q[4], "max": w.max(),
            "cv": w.std() / w.mean(), "n_efectivo_relativo": (w.sum() ** 2 / (w ** 2).sum()) / len(w)}


def metricas_binarias(y: pd.Series, p: pd.Series) -> dict:
    """AUC, log loss, Brier, tasa observada y media predicha (filas con ambos no nulos)."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    valido = y.notna() & p.notna()
    yv, pv = y[valido].astype(int).to_numpy(), p[valido].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    auc = roc_auc_score(yv, pv) if len(np.unique(yv)) == 2 else np.nan
    return {"n": int(valido.sum()), "auc": auc, "log_loss": log_loss(yv, pv, labels=[0, 1]),
            "brier": brier_score_loss(yv, pv), "tasa_observada": yv.mean(), "media_predicha": pv.mean(),
            "acierto_umbral_0_5": ((pv >= 0.5) == yv).mean()}


def metricas_continuas(y: pd.Series, prediccion: pd.Series) -> dict:
    """RMSE, MAE, R² y sesgo medio (predicción − observado)."""
    valido = y.notna() & prediccion.notna()
    yv, pv = y[valido].astype(float).to_numpy(), prediccion[valido].astype(float).to_numpy()
    error = pv - yv
    return {"n": int(valido.sum()), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error ** 2) / np.sum((yv - yv.mean()) ** 2)), "sesgo": float(error.mean())}


def columnas_propension(variables: pd.DataFrame) -> list[str]:
    """Columnas de `variables_hogar` que entran al modelo de propensión (sin clave, era ni fecha)."""
    return [c for c in variables.columns if c not in COLUMNAS_FUERA_PROPENSION]


def leer_pares_05(ruta_trayectoria: Path, h: int, columnas=COLUMNAS_PARES_PROPENSION) -> pd.DataFrame:
    """Pares de `data/05_trayectoria_distancia/pares_h{h}` con las `columnas` presentes."""
    partes = []
    for ruta in sorted((Path(ruta_trayectoria) / f"pares_h{h}").glob("ANO4=*/TRIMESTRE=*/data.parquet")):
        disponibles = pq.read_schema(ruta).names
        partes.append(pd.read_parquet(ruta, columns=[c for c in columnas if c in disponibles]))
    return pd.concat(partes, ignore_index=True)


def marcar_perdidos_por_ingreso(pares: pd.DataFrame, clases: pd.DataFrame) -> pd.DataFrame:
    """Agrega `valido_actual` (criterio `imputacion_excluye`), `clase_t`/`clase_th` y
    `perdido_por_ingreso` (identidad confirmada, no válido, alguna punta no declarada)."""
    # Misma regla que `leer_pares_train` del 05f: par_valido_c en la histórica,
    # par_valido_simetrica en la regular (donde coinciden).
    historica = pares["era_t"] == "historica"
    resultado = pares.copy()
    resultado["valido_actual"] = np.where(historica, pares["par_valido_c"].astype("boolean").fillna(False),
                                          pares["par_valido_simetrica"].astype("boolean").fillna(False)).astype(bool)
    for lado in ("t", "th"):
        info = clases[[*CLAVE_HOGAR_TRIMESTRE, "clase"]].rename(columns={
            "NRO_HOGAR": f"NRO_HOGAR_{lado}", "AGLOMERADO": f"AGLOMERADO_{lado}", "ANO4": f"ANO4_{lado}",
            "TRIMESTRE": f"TRIMESTRE_{lado}", "clase": f"clase_{lado}"})
        resultado = resultado.merge(info, on=["CODUSU", f"NRO_HOGAR_{lado}", f"AGLOMERADO_{lado}", f"ANO4_{lado}",
                                              f"TRIMESTRE_{lado}"], how="left", validate="many_to_one")
    identidad = resultado["identidad_confirmada"].astype("boolean").fillna(False).astype(bool)
    resultado["perdido_por_ingreso"] = identidad & ~resultado["valido_actual"] & (
        (resultado["clase_t"] != "declarado") | (resultado["clase_th"] != "declarado"))
    return resultado


def unir_variables_t(pares: pd.DataFrame, variables: pd.DataFrame) -> pd.DataFrame:
    """`pares` con las columnas de `columnas_propension(variables)` del hogar en `t`."""
    columnas = columnas_propension(variables)
    en_t = variables[[*CLAVE_HOGAR_TRIMESTRE, *columnas]].rename(columns={
        "NRO_HOGAR": "NRO_HOGAR_t", "AGLOMERADO": "AGLOMERADO_t", "ANO4": "ANO4_t", "TRIMESTRE": "TRIMESTRE_t"})
    return pares.merge(en_t, on=["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t", "ANO4_t", "TRIMESTRE_t"], how="left",
                       validate="many_to_one")


def pesos_con_corte(probabilidad: pd.Series, corte: float, p_min: float = P_MIN_PESO) -> pd.Series:
    """`1 / max(p, p_min)` recortado en `corte` (el percentil se fija afuera, en train)."""
    return (1.0 / probabilidad.astype(float).clip(lower=p_min)).clip(upper=corte)


def _parametros_propension(num_threads: int) -> dict:
    return {**PARAMETROS_LGBM, "num_threads": num_threads, "objective": "binary"}


@va.solo_train
def ajustar_propension_fold(poblacion: pd.DataFrame, columnas: list[str], num_threads: int = 2,
                            n_folds: int = N_FOLDS_PROPENSION, p_min: float = P_MIN_PESO,
                            percentil_recorte: float = PERCENTIL_RECORTE_PESO) -> dict:
    """Propensión de par válido por era sobre `poblacion` (train del fold): pesos de los
    válidos por predicción fuera de fold por vivienda, modelo completo y corte p99 por era."""
    import lightgbm as lgb

    pesos, modelos, cortes, metricas = [], {}, {}, []
    for era, grupo in poblacion.groupby("era_t", sort=True):
        datos = a_categoricas(grupo.reset_index(drop=True), [c for c in CATEGORICAS if c in columnas])
        y = datos["valido_actual"].astype(int)
        fold = grupos_vivienda(datos.rename(columns={"AGLOMERADO_t": "AGLOMERADO"}), n_folds)
        probabilidad = pd.Series(np.nan, index=datos.index)
        parametros = _parametros_propension(num_threads)
        for k in np.unique(fold):
            entrena = fold != k
            modelo = lgb.train(parametros, lgb.Dataset(datos.loc[entrena, columnas], label=y[entrena].astype(float)),
                               num_boost_round=RONDAS_LGBM)
            probabilidad[~entrena] = modelo.predict(datos.loc[~entrena, columnas])
        validos = y == 1
        crudo = 1.0 / probabilidad[validos].clip(lower=p_min)
        cortes[era] = float(crudo.quantile(percentil_recorte))
        peso = pesos_con_corte(probabilidad[validos], cortes[era], p_min)
        pesos.append(pd.Series(peso.to_numpy(), index=pd.MultiIndex.from_frame(datos.loc[validos, CLAVE_PAR])))
        modelos[era] = lgb.train(parametros, lgb.Dataset(datos[columnas], label=y.astype(float)),
                                 num_boost_round=RONDAS_LGBM)
        metricas.append({"era": era, **metricas_binarias(y, probabilidad), **{f"peso_{k}": v for k, v in
                                                                            resumen_pesos(peso).items()}})
    return {"pesos": pd.concat(pesos) if pesos else pd.Series(dtype=float), "modelos": modelos, "cortes": cortes,
            "metricas": pd.DataFrame(metricas), "columnas": list(columnas), "p_min": p_min,
            "claves_ajuste": pd.MultiIndex.from_frame(poblacion[CLAVE_PAR])}


def pesos_propension_test(ajuste: dict, test: pd.DataFrame) -> pd.Series:
    """Peso de propensión de cada par de `test` (con las X en `t`) con el modelo y el
    corte de su era en `ajuste`; `NA` si la era no tiene modelo. Índice: `CLAVE_PAR`."""
    clave = pd.MultiIndex.from_frame(test[CLAVE_PAR])
    if clave.isin(ajuste["claves_ajuste"]).any():
        raise ValueError("pares del test entre las filas de ajuste de la propensión")
    peso = pd.Series(np.nan, index=test.index)
    columnas = ajuste["columnas"]
    for era, grupo in test.groupby("era_t", sort=True):
        if era not in ajuste["modelos"]:
            continue
        datos = a_categoricas(grupo, [c for c in CATEGORICAS if c in columnas])
        probabilidad = pd.Series(ajuste["modelos"][era].predict(datos[columnas]), index=grupo.index)
        peso[grupo.index] = pesos_con_corte(probabilidad, ajuste["cortes"][era], ajuste["p_min"])
    return pd.Series(peso.to_numpy(), index=clave)


def parametros_peso_propension(num_threads: int) -> tuple:
    """Parámetros del peso de `imputacion_excluye_reponderado_propension` para el manifest."""
    import json

    return (("modelo", "lightgbm_por_era_variables_hogar_t"), ("p_min", P_MIN_PESO),
            ("percentil_recorte", PERCENTIL_RECORTE_PESO), ("n_folds_vivienda", N_FOLDS_PROPENSION),
            ("rondas", RONDAS_LGBM), ("parametros_lgbm", json.dumps(_parametros_propension(num_threads), sort_keys=True)))
