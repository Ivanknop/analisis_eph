"""Validación de las Fases C y D: candado del holdout, folds walk-forward con
embargo, conjuntos de train por criterio de imputación, líneas base, métricas,
bootstrap por cluster y verificación de predicciones fuera de muestra."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

import constantes as cte
import imputacion_ingresos as ii
import linea_base_trayectoria as lb
import panel
import pipeline_meta as pm
import universo_entrenamiento as ue
from features import CLAVE_PAR

HOLDOUT_DESDE = (2023, 1)
ULTIMO_DESTINO_TRAIN = panel.sumar_trimestres(*HOLDOUT_DESDE, -1)
NOTEBOOK_HOLDOUT = "14_evaluacion_holdout"
EVENTO_APERTURA = "apertura_holdout"
COLUMNAS_FECHA = ["ANO4_t", "TRIMESTRE_t", "ANO4_th", "TRIMESTRE_th"]
CLAVE_VIVIENDA = ["CODUSU", "AGLOMERADO_t"]
CELDAS_REPONDERACION = ["AGLOMERADO_t", "ANO4_t", "TRIMESTRE_t"]
COLUMNAS_MODELADO = ("pobre_t", "ratio_linea_pobreza_t")
COLUMNAS_OBJETIVO = ("pobre_th", "variacion_simetrica", "variacion_log")

# Un ciclo estacional completo de orígenes: con menos, la línea base por decil de un fold
# queda estimada sobre 1-3 trimestres y el fold no es comparable con el resto.
MIN_TRIMESTRES_TRAIN = 4
CONTEXTO_FOLD = {
    2007: "hueco_2007t3", 2009: "crisis_internacional", 2014: "devaluacion_2014", 2016: "salto_de_era",
    2018: "crisis_cambiaria", 2019: "crisis_cambiaria", 2020: "pandemia",
}
LINEAS_BASE = ["persistencia", "media_train", "decil_ratio"]
# Recorte de continuas antes de estandarizar en el lineal: `ratio_linea_pobreza_t` y
# similares tienen cola derecha larga, que en un modelo lineal domina los coeficientes.
RECORTE_LINEAL = (0.005, 0.995)
# Recorte de probabilidades para el logit de la calibración: evita infinitos con
# predicciones 0/1 sin mover las que el modelo da en el rango usual.
EPS_LOGIT = 1e-6
# Coeficientes mayores indican separación (casi) perfecta: la logística no tiene
# máximo finito y el valor no es interpretable como calibración.
MAX_COEFICIENTE_CALIBRACION = 50.0


class FugaHoldoutError(RuntimeError):
    """Filas del holdout o del embargo en una función que solo admite train."""


def indice_trimestre(anio: pd.Series, trimestre: pd.Series) -> pd.Series:
    """`anio * 4 + trimestre - 1` como `int64`."""
    return anio.astype("int64") * 4 + trimestre.astype("int64") - 1


INDICE_HOLDOUT = HOLDOUT_DESDE[0] * 4 + HOLDOUT_DESDE[1] - 1
INDICE_ULTIMO_DESTINO_TRAIN = ULTIMO_DESTINO_TRAIN[0] * 4 + ULTIMO_DESTINO_TRAIN[1] - 1


def exigir_sin_holdout(pares: pd.DataFrame) -> None:
    """`FugaHoldoutError` si algún par tiene origen ≥ `HOLDOUT_DESDE` o destino
    posterior a `ULTIMO_DESTINO_TRAIN`; `TypeError` si faltan columnas de fecha."""
    faltan = [c for c in COLUMNAS_FECHA if c not in pares.columns]
    if faltan:
        raise TypeError(f"sin {faltan}: no se puede verificar que no haya holdout")
    origen = indice_trimestre(pares["ANO4_t"], pares["TRIMESTRE_t"])
    destino = indice_trimestre(pares["ANO4_th"], pares["TRIMESTRE_th"])
    n_holdout, n_embargo = int((origen >= INDICE_HOLDOUT).sum()), int((destino > INDICE_ULTIMO_DESTINO_TRAIN).sum())
    if n_holdout or n_embargo:
        raise FugaHoldoutError(f"{n_holdout} pares con origen en el holdout, {n_embargo} con destino fuera de train")


FUNCIONES_SOLO_TRAIN: dict[str, Callable] = {}


def solo_train(funcion: Callable) -> Callable:
    """Decorador: corre `exigir_sin_holdout` sobre cada `DataFrame` de pares (con
    `ANO4_t`) de los argumentos antes de llamar a `funcion`; la registra en
    `FUNCIONES_SOLO_TRAIN`."""
    @functools.wraps(funcion)
    def envoltura(*args, **kwargs):
        for valor in [*args, *kwargs.values()]:
            if isinstance(valor, pd.DataFrame) and "ANO4_t" in valor.columns:
                exigir_sin_holdout(valor)
        return funcion(*args, **kwargs)
    FUNCIONES_SOLO_TRAIN[funcion.__name__] = envoltura
    return envoltura


def fraccion_max_par(pares: pd.DataFrame) -> pd.Series:
    """Mayor `fraccion_max` de las dos puntas; punta no imputada vale 0, punta
    imputada sin fracción válida deja el par en NA."""
    # Misma regla que `C_parcial_{u}_max` del 05d (reproduce sus conteos par a par).
    lados = [pares[f"fraccion_max_{lado}"].astype(float).where(
        pares[f"imputado_{lado}"].astype("boolean").fillna(False).astype(bool), 0.0) for lado in ("t", "th")]
    return pd.concat(lados, axis=1).max(axis=1, skipna=False)


def _leer_pares(raiz_features: Path, raiz_trayectoria: Path, h: int, columnas_extra: tuple[str, ...]) -> pd.DataFrame:
    # `leer_universo` ya trae `COLUMNAS_UNIVERSO`: pedirlas otra vez duplica la columna
    # y rompe las máscaras por fila (`era` pasaba a ser un DataFrame).
    columnas = tuple(c for c in dict.fromkeys([*COLUMNAS_MODELADO, *columnas_extra]) if c not in ue.COLUMNAS_UNIVERSO)
    pares = ue.leer_universo(raiz_features, h, columnas)
    pares = ue.unir_objetivo(pares, raiz_trayectoria, h, COLUMNAS_OBJETIVO)
    pares = ue.unir_marcas_imputacion(pares, raiz_trayectoria, h)
    return pares.assign(particion=panel.particion_train_holdout(pares, HOLDOUT_DESDE),
                        fraccion_max_par=fraccion_max_par(pares))


def mascara_train(pares: pd.DataFrame, variable: str = "simetrica",
                  criterio_imputacion: str = cte.CRITERIO_IMPUTACION_PRINCIPAL) -> pd.Series:
    """Pares que entran al train según `criterio_imputacion`
    (`constantes.CRITERIOS_IMPUTACION_EXPERIMENTO`) para `variable` (`simetrica`/`log`)."""
    conserva = ue.es_valido(pares, variable, "imputacion_conserva")
    excluye = ue.es_valido(pares, variable, "imputacion_excluye")
    if criterio_imputacion in ("imputacion_excluye", *cte.CRITERIOS_REPONDERADOS):
        return excluye
    if criterio_imputacion == "imputacion_conserva":
        return conserva
    if criterio_imputacion == "imputacion_excluye_fraccion_max_0.5":
        return conserva & pares["fraccion_max_par"].le(cte.UMBRAL_FRACCION_IMPUTADA).fillna(False).astype(bool)
    raise ValueError(f"criterio de imputación desconocido: {criterio_imputacion}")


def mascara_evaluacion(pares: pd.DataFrame, variable: str = "simetrica", conjunto: str = "limpio") -> pd.Series:
    """`limpio`: válidos sin punta imputada (el mismo para todo criterio);
    `imputados`: válidos con alguna punta imputada (solo era histórica)."""
    excluye = ue.es_valido(pares, variable, "imputacion_excluye")
    if conjunto == "limpio":
        return excluye
    if conjunto == "imputados":
        return ue.es_valido(pares, variable, "imputacion_conserva") & ~excluye
    raise ValueError(f"conjunto de evaluación desconocido: {conjunto}")


def cargar_pares_modelado(raiz_features: Path, raiz_trayectoria: Path, h: int, columnas_extra: tuple[str, ...] = (),
                          variable: str = "simetrica",
                          criterio_imputacion: str = cte.CRITERIO_IMPUTACION_PRINCIPAL) -> pd.DataFrame:
    """Pares de train (destino ≤ `ULTIMO_DESTINO_TRAIN`) válidos para
    `criterio_imputacion`, con objetivo y marcas de imputación. Nunca holdout ni embargo."""
    pares = _leer_pares(raiz_features, raiz_trayectoria, h, columnas_extra)
    pares = pares[(pares["particion"] == "train") & mascara_train(pares, variable, criterio_imputacion)]
    exigir_sin_holdout(pares)
    return pares.reset_index(drop=True)


def _aperturas_previas(archivo_corridas: Path, h: int) -> int:
    if not Path(archivo_corridas).exists():
        return 0
    corridas = pd.read_csv(archivo_corridas, dtype=str)
    eventos = [json.loads(p) for p in corridas["parametros_json"].dropna()]
    return sum(1 for e in eventos if e.get("evento") == EVENTO_APERTURA and int(e.get("h", -1)) == h)


def abrir_holdout(raiz_features: Path, raiz_trayectoria: Path, h: int, *, notebook: str, motivo: str,
                  archivo_corridas: Path, repo: Path, columnas_extra: tuple[str, ...] = (),
                  variable: str = "simetrica", reapertura: str | None = None) -> pd.DataFrame:
    """Pares del holdout (conjunto limpio de `variable`). Solo desde `NOTEBOOK_HOLDOUT`;
    registra la apertura en `archivo_corridas` y exige `reapertura` si ya hubo una para `h`."""
    if notebook != NOTEBOOK_HOLDOUT:
        raise PermissionError(f"el holdout solo se abre desde {NOTEBOOK_HOLDOUT}, no desde {notebook}")
    if not motivo:
        raise ValueError("abrir el holdout exige un motivo")
    previas = _aperturas_previas(archivo_corridas, h)
    if previas and not reapertura:
        raise PermissionError(f"h={h}: el holdout ya se abrió {previas} vez/veces; pasar `reapertura` con la justificación")
    pares = _leer_pares(raiz_features, raiz_trayectoria, h, columnas_extra)
    holdout = pares[(pares["particion"] == "holdout") & mascara_evaluacion(pares, variable)].reset_index(drop=True)
    parametros = {"evento": EVENTO_APERTURA, "h": h, "motivo": motivo, "variable": variable,
                  "reapertura": reapertura, "aperturas_previas": previas}
    n_trimestres = holdout.groupby(["ANO4_t", "TRIMESTRE_t"]).ngroups
    pm.registrar_corrida(Path(archivo_corridas), notebook, parametros, n_trimestres, len(holdout), 0.0,
                         pm.obtener_commit_git(repo))
    return holdout


def poblacion_objetivo(pares: pd.DataFrame, objetivo_modelado: str) -> pd.Series:
    """Máscara de la población de `objetivo_modelado`: todos (`trayectoria`),
    `pobre_t == False` (`entrada`) o `pobre_t == True` (`permanencia`)."""
    if objetivo_modelado == "trayectoria":
        return pd.Series(True, index=pares.index)
    pobre_t = pares["pobre_t"].astype("boolean")
    if objetivo_modelado == "entrada":
        return (pobre_t == False).fillna(False).astype(bool)  # noqa: E712
    if objetivo_modelado == "permanencia":
        return (pobre_t == True).fillna(False).astype(bool)  # noqa: E712
    raise ValueError(f"objetivo de modelado desconocido: {objetivo_modelado}")


def columna_respuesta(objetivo_modelado: str, variable: str = "simetrica") -> str:
    """`variacion_{variable}` para `trayectoria`, `pobre_th` para entrada/permanencia."""
    return f"variacion_{variable}" if objetivo_modelado == "trayectoria" else "pobre_th"


def es_clasificacion(objetivo_modelado: str) -> bool:
    return objetivo_modelado != "trayectoria"


@solo_train
def conjunto_train(pares: pd.DataFrame, variable: str = "simetrica",
                   criterio_imputacion: str = cte.CRITERIO_IMPUTACION_PRINCIPAL,
                   ponderacion: str = "sin_ponderar") -> tuple[pd.DataFrame, pd.Series]:
    """Pares de train para `criterio_imputacion` y su peso de muestra (1, o
    `peso_normalizado_trimestre`; reescalado por celda en `_reponderado`)."""
    if ponderacion not in cte.PONDERACIONES:
        raise ValueError(f"ponderación desconocida: {ponderacion}")
    base = (pd.Series(1.0, index=pares.index) if ponderacion == "sin_ponderar"
            else pares["peso_normalizado_trimestre"].astype(float))
    seleccion = mascara_train(pares, variable, criterio_imputacion)
    if criterio_imputacion != "imputacion_excluye_reponderado":
        return pares[seleccion], base[seleccion]
    # Reparte el peso de los imputados excluidos entre los no imputados de su celda, sobre
    # la población de `imputacion_conserva` (la que el reponderado intenta representar).
    universo = mascara_train(pares, variable, "imputacion_conserva")
    tabla = pares.loc[universo, CELDAS_REPONDERACION].assign(_peso=base[universo])
    peso = ii.reponderar_dentro_de_celda(tabla, "_peso", ~seleccion[universo], CELDAS_REPONDERACION)
    return pares[seleccion], peso.loc[pares.index[seleccion]]


@solo_train
def generar_folds(pares: pd.DataFrame, h: int, min_trimestres_train: int = MIN_TRIMESTRES_TRAIN) -> pd.DataFrame:
    """Folds de ventana expansiva con test = año de origen Y: train con destino
    ≤ (Y−1)T4, embargo = origen < YT1 ≤ destino. Uno por fila, solo los que tienen
    test y al menos `min_trimestres_train` trimestres de origen en train."""
    origen = indice_trimestre(pares["ANO4_t"], pares["TRIMESTRE_t"])
    destino = indice_trimestre(pares["ANO4_th"], pares["TRIMESTRE_th"])
    filas = []
    for anio in sorted(pares["ANO4_t"].astype(int).unique()):
        inicio = anio * 4
        es_train, es_test = destino < inicio, pares["ANO4_t"].astype(int) == anio
        es_embargo = (origen < inicio) & (destino >= inicio)
        trimestres_train = int(origen[es_train].nunique())
        if trimestres_train < min_trimestres_train:
            continue
        eras_train, eras_test = sorted(pares.loc[es_train, "era"].unique()), sorted(pares.loc[es_test, "era"].unique())
        filas.append({
            "h": h, "anio_test": anio, "ultimo_destino_train": inicio - 1,
            "n_trimestres_origen_train": trimestres_train, "n_trimestres_origen_test": int(origen[es_test].nunique()),
            "n_train": int(es_train.sum()), "n_embargo": int(es_embargo.sum()), "n_test": int(es_test.sum()),
            "eras_train": "+".join(eras_train), "era_test": "+".join(eras_test),
            "contexto_fold": CONTEXTO_FOLD.get(anio, ""),
            "destino_en_contexto": CONTEXTO_FOLD.get(anio + 1, "") if h >= 4 else "",
            "periodo_intervenido": 2007 <= anio <= 2015,
        })
    return pd.DataFrame(filas)


@solo_train
def asignar_fold(pares: pd.DataFrame, anio_test: int) -> pd.Series:
    """`"train"`, `"embargo"`, `"test"` o `"fuera"` de cada par para el fold `anio_test`."""
    origen = indice_trimestre(pares["ANO4_t"], pares["TRIMESTRE_t"])
    destino = indice_trimestre(pares["ANO4_th"], pares["TRIMESTRE_th"])
    inicio = anio_test * 4
    rol = pd.Series("fuera", index=pares.index, dtype="object")
    rol[destino < inicio] = "train"
    rol[(origen < inicio) & (destino >= inicio)] = "embargo"
    rol[pares["ANO4_t"].astype(int) == anio_test] = "test"
    return rol


def sin_viviendas_de_train(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Máscara de los pares de `test` cuya vivienda (`CLAVE_VIVIENDA`) no aparece en `train`."""
    en_train = pd.MultiIndex.from_frame(train[CLAVE_VIVIENDA].drop_duplicates())
    return pd.Series(~pd.MultiIndex.from_frame(test[CLAVE_VIVIENDA]).isin(en_train), index=test.index)


@solo_train
def ajustar_linea_base(train: pd.DataFrame, objetivo_modelado: str, linea_base: str,
                       variable: str = "simetrica", n_cuantiles: int = 10) -> dict:
    """Línea base de `LINEAS_BASE` ajustada sin ponderar sobre `train` (ya
    restringido a la población del objetivo)."""
    y = train[columna_respuesta(objetivo_modelado, variable)].astype(float)
    modelo = {"objetivo_modelado": objetivo_modelado, "linea_base": linea_base, "media_train": float(y.mean())}
    if linea_base == "decil_ratio":
        modelo["decil"] = lb.ajustar_regresion_media_decil(train["ratio_linea_pobreza_t"], y, n_cuantiles)
    elif linea_base not in ("persistencia", "media_train"):
        raise ValueError(f"línea base desconocida: {linea_base}")
    return modelo


def predecir_linea_base(modelo: dict, pares: pd.DataFrame) -> pd.Series:
    """Predicción de `modelo` (`ajustar_linea_base`): variación o probabilidad de `pobre_th`."""
    if modelo["linea_base"] == "media_train":
        return pd.Series(modelo["media_train"], index=pares.index)
    if modelo["linea_base"] == "decil_ratio":
        return lb.predecir_regresion_media_decil(modelo["decil"], pares["ratio_linea_pobreza_t"])
    if es_clasificacion(modelo["objetivo_modelado"]):
        return pares["pobre_t"].astype(bool).astype(float)
    return pd.Series(0.0, index=pares.index)


@solo_train
def ajustar_preprocesamiento_lineal(train: pd.DataFrame, columnas_share: list[str], columnas_continuas: list[str],
                                    columnas_binarias: list[str] = (), columnas_categoricas: list[str] = (),
                                    recorte: tuple[float, float] = RECORTE_LINEAL) -> dict:
    """Parámetros estimados en `train`: valor neutro (0 en shares y binarias,
    mediana en continuas), indicador de faltante si hay NA, recorte por cuantiles
    de continuas, niveles one-hot y media/desvío para estandarizar."""
    # Preprocesamiento del modelo, no imputación de datos (`.claude/decisiones.md`);
    # gradient boosting usa NA nativo y no pasa por acá.
    numericas = {}
    for columna in [*columnas_share, *columnas_binarias, *columnas_continuas]:
        valores = pd.to_numeric(train[columna].astype("Float64"), errors="coerce").astype(float)
        continua = columna in columnas_continuas
        limites = (tuple(np.nanquantile(valores, recorte)) if continua and valores.notna().any() else (-np.inf, np.inf))
        neutro = float(np.nanmedian(valores)) if continua and valores.notna().any() else 0.0
        numericas[columna] = {"neutro": neutro, "limites": limites, "indicador": bool(valores.isna().any())}
    categoricas = {}
    for columna in columnas_categoricas:
        frecuencias = train[columna].astype("object").fillna("__na__").value_counts()
        categoricas[columna] = {"referencia": frecuencias.index[0], "niveles": sorted(frecuencias.index[1:])}
    preprocesamiento = {"numericas": numericas, "categoricas": categoricas}
    matriz = _matriz_lineal(preprocesamiento, train)
    media, desvio = matriz.mean(), matriz.std(ddof=0)
    constantes = desvio[desvio == 0].index.tolist()
    preprocesamiento.update(media=media.drop(constantes), desvio=desvio.drop(constantes), descartadas=constantes)
    return preprocesamiento


def _matriz_lineal(preprocesamiento: dict, pares: pd.DataFrame) -> pd.DataFrame:
    bloques = {}
    for columna, spec in preprocesamiento["numericas"].items():
        valores = pd.to_numeric(pares[columna].astype("Float64"), errors="coerce").astype(float)
        if spec["indicador"]:
            bloques[f"{columna}__na"] = valores.isna().astype(float)
        bloques[columna] = valores.clip(*spec["limites"]).fillna(spec["neutro"])
    for columna, spec in preprocesamiento["categoricas"].items():
        valores = pares[columna].astype("object").fillna("__na__")
        for nivel in spec["niveles"]:
            bloques[f"{columna}=={nivel}"] = (valores == nivel).astype(float)
    return pd.DataFrame(bloques, index=pares.index)


def aplicar_preprocesamiento_lineal(preprocesamiento: dict, pares: pd.DataFrame) -> pd.DataFrame:
    """Matriz estandarizada de `pares` con los parámetros de `ajustar_preprocesamiento_lineal`
    (niveles no vistos en train quedan en la referencia)."""
    matriz = _matriz_lineal(preprocesamiento, pares).drop(columns=preprocesamiento["descartadas"])
    return (matriz - preprocesamiento["media"]) / preprocesamiento["desvio"]


def _alinear(*series: pd.Series | None, peso: pd.Series | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    presentes = [s for s in series if s is not None]
    valido = pd.concat([s.notna() for s in presentes], axis=1).all(axis=1)
    if peso is not None:
        valido &= peso.notna()
    w = np.ones(int(valido.sum())) if peso is None else peso[valido].astype(float).to_numpy()
    return [None if s is None else s[valido].astype(float).to_numpy() for s in series], w


def metricas_regresion(y: pd.Series, prediccion: pd.Series, media_train: float,
                       prediccion_base: pd.Series | None = None, peso: pd.Series | None = None) -> dict:
    """MAE, RMSE, R² contra `media_train` y `skill_vs_base = 1 − MSE / MSE_base`
    (ponderados si hay `peso`), sobre las filas con todo no nulo."""
    (yv, pv, bv), w = _alinear(y, prediccion, prediccion_base, peso=peso)
    if len(yv) == 0 or w.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2_vs_media_train": np.nan, "skill_vs_base": np.nan}
    mse = np.sum(w * (yv - pv) ** 2) / w.sum()
    mse_media = np.sum(w * (yv - media_train) ** 2) / w.sum()
    resultado = {"n": int(len(yv)), "mae": float(np.sum(w * np.abs(yv - pv)) / w.sum()), "rmse": float(np.sqrt(mse)),
                 "r2_vs_media_train": float(1 - mse / mse_media) if mse_media > 0 else np.nan, "skill_vs_base": np.nan}
    if bv is not None:
        mse_base = np.sum(w * (yv - bv) ** 2) / w.sum()
        resultado["skill_vs_base"] = float(1 - mse / mse_base) if mse_base > 0 else np.nan
    return resultado


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS_LOGIT, 1 - EPS_LOGIT)
    return np.log(q / (1 - q))


def calibracion_logistica(y: np.ndarray, p: np.ndarray, w: np.ndarray, iteraciones: int = 50) -> tuple[float, float]:
    """Intercepto y pendiente de la logística ponderada de `y` sobre `logit(p)`
    (Newton-Raphson); `NaN` si `p` es constante o `y` tiene una sola clase."""
    # IRLS propio en vez de `LogisticRegression`: dos parámetros, sin penalización y con
    # pesos arbitrarios (Poisson del bootstrap) sin depender de la API de `penalty`.
    x = _logit(p)
    if np.ptp(x) == 0 or w.sum() == 0 or len(np.unique(y[w > 0])) < 2:
        return np.nan, np.nan
    diseno = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iteraciones):
        mu = expit(diseno @ beta)
        gradiente = diseno.T @ (w * (y - mu))
        hessiana = (diseno * (w * mu * (1 - mu))[:, None]).T @ diseno
        try:
            paso = np.linalg.solve(hessiana, gradiente)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta = beta + paso
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > MAX_COEFICIENTE_CALIBRACION:
            return np.nan, np.nan
        if np.max(np.abs(paso)) < 1e-8:
            return float(beta[0]), float(beta[1])
    return np.nan, np.nan


def curva_confiabilidad(y: pd.Series, probabilidad: pd.Series, peso: pd.Series | None = None,
                        n_bins: int = 10) -> pd.DataFrame:
    """Por bin de cuantil de `probabilidad`: peso, probabilidad media y tasa
    observada. Con ≤ `n_bins` valores distintos, un bin por valor."""
    (yv, pv), w = _alinear(y, probabilidad, peso=peso)
    if len(pv) == 0:
        return pd.DataFrame(columns=["bin", "n", "peso", "probabilidad_media", "tasa_observada"])
    valores = np.unique(pv)
    if len(valores) <= n_bins:
        grupo = np.searchsorted(valores, pv)
    else:
        cortes = np.unique(np.quantile(pv, np.linspace(0, 1, n_bins + 1))[1:-1])
        grupo = np.searchsorted(cortes, pv, side="right")
    tabla = pd.DataFrame({"bin": grupo, "w": w, "wy": w * yv, "wp": w * pv})
    agregado = tabla.groupby("bin").agg(n=("w", "size"), peso=("w", "sum"), wy=("wy", "sum"), wp=("wp", "sum"))
    agregado = agregado[agregado["peso"] > 0]
    return pd.DataFrame({"bin": agregado.index, "n": agregado["n"].to_numpy(), "peso": agregado["peso"].to_numpy(),
                         "probabilidad_media": (agregado["wp"] / agregado["peso"]).to_numpy(),
                         "tasa_observada": (agregado["wy"] / agregado["peso"]).to_numpy()})


def ece(curva: pd.DataFrame) -> float:
    """Error de calibración esperado: Σ (peso_bin / peso) · |tasa − probabilidad media|."""
    if curva.empty or curva["peso"].sum() == 0:
        return np.nan
    return float(np.sum(curva["peso"] * (curva["tasa_observada"] - curva["probabilidad_media"]).abs()) / curva["peso"].sum())


def _auc(y: np.ndarray, p: np.ndarray, w: np.ndarray, funcion: Callable) -> float:
    positivos = w[y == 1].sum()
    if np.ptp(p) == 0 or positivos == 0 or positivos == w.sum():
        return np.nan
    return float(funcion(y, p, sample_weight=w))


def metricas_clasificacion(y: pd.Series, probabilidad: pd.Series, probabilidad_base: pd.Series | None = None,
                           tasa_train: float | None = None, peso: pd.Series | None = None, n_bins: int = 10) -> dict:
    """Log loss, Brier, Brier skill (contra base y contra `tasa_train`), AUC-ROC,
    AUC-PR (cruda, relativa a la prevalencia y de la clase negativa), ECE y
    calibración en grande (media predicha, intercepto y pendiente)."""
    (yv, pv, bv), w = _alinear(y, probabilidad, probabilidad_base, peso=peso)
    vacio = dict.fromkeys(["log_loss", "brier", "brier_skill_vs_base", "brier_skill_vs_tasa_train", "auc_roc",
                           "auc_pr", "auc_pr_relativa", "auc_pr_clase_negativa", "ece", "prevalencia",
                           "media_predicha", "diferencia_media_predicha", "intercepto_calibracion",
                           "pendiente_calibracion"], np.nan)
    if len(yv) == 0 or w.sum() == 0:
        return {"n": 0, **vacio}
    total = w.sum()
    prevalencia, media_predicha = float(np.sum(w * yv) / total), float(np.sum(w * pv) / total)
    brier = float(np.sum(w * (pv - yv) ** 2) / total)
    # Con alguna probabilidad exactamente 0 o 1 (la persistencia) el log loss es infinito o
    # depende del recorte elegido: se deja NA en vez de reportar un número arbitrario.
    degenerada = bool(np.any((pv <= 0) | (pv >= 1)))
    log_loss = np.nan if degenerada else float(-np.sum(w * (yv * np.log(pv) + (1 - yv) * np.log(1 - pv))) / total)
    resultado = {**vacio, "n": int(len(yv)), "prevalencia": prevalencia, "media_predicha": media_predicha,
                 "diferencia_media_predicha": media_predicha - prevalencia, "brier": brier, "log_loss": log_loss}
    if bv is not None:
        brier_base = float(np.sum(w * (bv - yv) ** 2) / total)
        resultado["brier_skill_vs_base"] = 1 - brier / brier_base if brier_base > 0 else np.nan
    if tasa_train is not None:
        brier_tasa = float(np.sum(w * (tasa_train - yv) ** 2) / total)
        resultado["brier_skill_vs_tasa_train"] = 1 - brier / brier_tasa if brier_tasa > 0 else np.nan
    resultado["auc_roc"] = _auc(yv, pv, w, roc_auc_score)
    resultado["auc_pr"] = _auc(yv, pv, w, average_precision_score)
    resultado["auc_pr_clase_negativa"] = _auc(1 - yv, 1 - pv, w, average_precision_score)
    if prevalencia < 1:
        resultado["auc_pr_relativa"] = (resultado["auc_pr"] - prevalencia) / (1 - prevalencia)
    resultado["ece"] = ece(curva_confiabilidad(pd.Series(yv), pd.Series(pv), pd.Series(w), n_bins))
    resultado["intercepto_calibracion"], resultado["pendiente_calibracion"] = calibracion_logistica(yv, pv, w)
    return resultado


def bootstrap_metricas_cluster(datos: pd.DataFrame, columna_cluster: str | list[str],
                               funcion: Callable[[pd.DataFrame, pd.Series], dict], peso: pd.Series | None = None,
                               n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """`funcion(datos, peso)` observada e IC95% por bootstrap de cluster con pesos
    Poisson(1) por cluster (multiplicados por `peso`). Una fila por métrica."""
    # Mismo esquema que `bootstrap.py` (Poisson i.i.d. por cluster), pero genérico en la
    # métrica: AUC y calibración no se agregan por cluster, se recalculan con los pesos.
    base = pd.Series(1.0, index=datos.index) if peso is None else peso.astype(float)
    observado = funcion(datos, base)
    codigos = datos.groupby(columna_cluster, sort=False).ngroup().to_numpy()
    n_clusters = int(codigos.max()) + 1 if len(codigos) else 0
    rng = np.random.default_rng(seed)
    replicas = []
    for _ in range(n_boot):
        w_cluster = rng.poisson(1.0, size=n_clusters).astype(float)
        replicas.append(funcion(datos, base * w_cluster[codigos]))
    replicas = pd.DataFrame(replicas)
    filas = []
    for metrica, valor in observado.items():
        serie = replicas[metrica].astype(float).dropna() if metrica in replicas else pd.Series(dtype=float)
        bajo, alto = (np.percentile(serie, [2.5, 97.5]) if len(serie) else (np.nan, np.nan))
        filas.append({"metrica": metrica, "valor": valor, "ic95_low": float(bajo), "ic95_high": float(alto),
                      "n_clusters": n_clusters})
    return pd.DataFrame(filas)


def _metricas_linea_base(objetivo_modelado: str, y: pd.Series, prediccion: pd.Series, base: pd.Series,
                         persistencia: pd.Series, media_train: float, peso: pd.Series | None) -> dict:
    if es_clasificacion(objetivo_modelado):
        return metricas_clasificacion(y, prediccion, base, media_train, peso)
    resultado = metricas_regresion(y, prediccion, media_train, base, peso)
    resultado["skill_vs_persistencia"] = metricas_regresion(y, prediccion, media_train, persistencia, peso)["skill_vs_base"]
    return resultado


@solo_train
def evaluar_lineas_base(pares: pd.DataFrame, folds: pd.DataFrame, objetivo_modelado: str, variable: str = "simetrica",
                        n_boot: int = 200, seed: int = 42, agrupada: bool = False) -> pd.DataFrame:
    """Métricas de `LINEAS_BASE` en cada fold de `folds` (train por el criterio
    principal, test sobre el conjunto limpio), sin ponderar y ponderadas; IC solo
    para `decil_ratio`. Con `agrupada`, el test excluye las viviendas de su train."""
    poblacion = poblacion_objetivo(pares, objetivo_modelado)
    columna_y = columna_respuesta(objetivo_modelado, variable)
    filas = []
    for anio in folds["anio_test"]:
        rol = asignar_fold(pares, int(anio))
        train = pares[(rol == "train") & poblacion & mascara_train(pares, variable)]
        test = pares[(rol == "test") & poblacion & mascara_evaluacion(pares, variable)]
        evaluacion = cte.EVALUACIONES_AGRUPADAS[0]
        if agrupada:
            test, evaluacion = test[sin_viviendas_de_train(test, train)], cte.EVALUACIONES_AGRUPADAS[1]
        if train.empty or test.empty:
            continue
        modelos = {nombre: ajustar_linea_base(train, objetivo_modelado, nombre, variable) for nombre in LINEAS_BASE}
        predicciones = {nombre: predecir_linea_base(m, test) for nombre, m in modelos.items()}
        media_train = modelos["media_train"]["media_train"]
        y = test[columna_y].astype(float)
        for ponderacion in cte.PONDERACIONES:
            peso = None if ponderacion == "sin_ponderar" else test["peso_normalizado_trimestre"].astype(float)
            for nombre, prediccion in predicciones.items():
                if nombre == "decil_ratio" and n_boot:
                    def funcion(d: pd.DataFrame, w: pd.Series, _p=prediccion) -> dict:
                        return _metricas_linea_base(objetivo_modelado, y, _p, predicciones["decil_ratio"],
                                                    predicciones["persistencia"], media_train, w)
                    tabla = bootstrap_metricas_cluster(test, CLAVE_VIVIENDA, funcion, peso, n_boot, seed)
                else:
                    metricas = _metricas_linea_base(objetivo_modelado, y, prediccion, predicciones["decil_ratio"],
                                                    predicciones["persistencia"], media_train, peso)
                    tabla = pd.DataFrame({"metrica": list(metricas), "valor": list(metricas.values())})
                filas.append(tabla.assign(anio_test=int(anio), linea_base=nombre, ponderacion=ponderacion,
                                          evaluacion_agrupada=evaluacion, n_train=len(train)))
    if not filas:
        return pd.DataFrame()
    resultado = pd.concat(filas, ignore_index=True).assign(objetivo_modelado=objetivo_modelado, variable=variable)
    return resultado


def prevalencias(pares: pd.DataFrame, variable: str = "simetrica",
                 criterio_imputacion: str = cte.CRITERIO_IMPUTACION_PRINCIPAL,
                 por: tuple[str, ...] = ("particion", "era")) -> pd.DataFrame:
    """N y tasa de `pobre_th` para entrada y permanencia entre los pares válidos
    por `criterio_imputacion`, agrupados por `por`."""
    validos = pares[mascara_train(pares, variable, criterio_imputacion)]
    filas = []
    for objetivo in ("entrada", "permanencia"):
        sub = validos[poblacion_objetivo(validos, objetivo)]
        tabla = sub.groupby(list(por))["pobre_th"].agg(n="size", tasa_positiva="mean").reset_index()
        filas.append(tabla.assign(objetivo_modelado=objetivo, criterio_imputacion=criterio_imputacion))
    return pd.concat(filas, ignore_index=True)


def resumen_todas_particiones(raiz_features: Path, raiz_trayectoria: Path, h: int) -> dict[str, pd.DataFrame]:
    """Solo agregados de train, embargo y holdout: prevalencias por universo ×
    criterio × partición × era y por año de origen, y pérdida de la evaluación
    agrupada en el holdout. Ninguna fila de pares sale de esta función."""
    # Excepción documentada al candado (`.claude/decisiones.md`): conteos y tasas de la
    # etiqueta del holdout son descriptivos (diagnóstico de deriva), no se usan para elegir nada.
    pares = _leer_pares(raiz_features, raiz_trayectoria, h, ())
    por_particion, por_anio, perdida = [], [], []
    for universo in cte.UNIVERSOS:
        sub = pares[ue.en_universo(pares, universo)]
        for criterio in ("imputacion_excluye", "imputacion_conserva"):
            por_particion.append(prevalencias(sub, criterio_imputacion=criterio).assign(universo=universo))
            por_particion.append(prevalencias(sub, criterio_imputacion=criterio, por=("particion",))
                                 .assign(universo=universo, era="todas"))
            por_anio.append(prevalencias(sub, criterio_imputacion=criterio, por=("ANO4_t", "particion", "era"))
                            .assign(universo=universo))
        train = sub[(sub["particion"] == "train") & mascara_train(sub)]
        holdout = sub[(sub["particion"] == "holdout") & mascara_evaluacion(sub)]
        conserva = sin_viviendas_de_train(holdout, train)
        perdida.append({"universo": universo, "anio_test": "holdout", "n_test": len(holdout),
                        "n_test_sin_viviendas_train": int(conserva.sum())})
    return {"prevalencias": pd.concat(por_particion, ignore_index=True).assign(h=h),
            "prevalencias_anio": pd.concat(por_anio, ignore_index=True).assign(h=h),
            "perdida_agrupada_holdout": pd.DataFrame(perdida).assign(h=h)}


@solo_train
def perdida_agrupada_folds(pares: pd.DataFrame, folds: pd.DataFrame, variable: str = "simetrica") -> pd.DataFrame:
    """Por fold: pares del test limpio y cuántos quedan al excluir las viviendas de su train."""
    filas = []
    for anio in folds["anio_test"]:
        rol = asignar_fold(pares, int(anio))
        train = pares[(rol == "train") & mascara_train(pares, variable)]
        test = pares[(rol == "test") & mascara_evaluacion(pares, variable)]
        filas.append({"anio_test": int(anio), "n_test": len(test),
                      "n_test_sin_viviendas_train": int(sin_viviendas_de_train(test, train).sum())})
    return pd.DataFrame(filas)


COLUMNAS_PREDICCIONES = [*CLAVE_PAR, "h", "fold", "ultimo_destino_train", "evaluacion_agrupada_excluido",
                         "conjunto_evaluacion", "prediccion", "y_observado"]


def ruta_predicciones(raiz: Path, objetivo_modelado: str, h: int, configuracion_id: str) -> Path:
    """`raiz/{objetivo_modelado}/h={h}/configuracion={id}/data.parquet` (capa `07_predicciones`)."""
    return Path(raiz) / objetivo_modelado / f"h={h}" / f"configuracion={configuracion_id}" / "data.parquet"


def _salida_git(repo: Path, *argumentos: str) -> str | None:
    try:
        resultado = subprocess.run(["git", *argumentos], cwd=repo, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return resultado.stdout if resultado.returncode == 0 else None


def estado_arbol_git(repo: Path) -> dict:
    """`commit_git`, `arbol_sucio` (cambios sin commitear) y `hash_diff` (SHA-256 de
    `git diff HEAD`, `""` si no hay cambios). Solo comandos de lectura."""
    estado = _salida_git(repo, "status", "--porcelain")
    diff = _salida_git(repo, "diff", "HEAD")
    return {"commit_git": pm.obtener_commit_git(repo),
            "arbol_sucio": None if estado is None else bool(estado.strip()),
            "hash_diff": None if diff is None else (hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else "")}


def id_parametros(parametros: dict) -> str:
    """`configuracion_id`: 12 primeros hex del SHA-256 de `parametros` en JSON ordenado."""
    serializado = json.dumps(parametros, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serializado.encode()).hexdigest()[:12]


def manifest_configuracion(parametros: dict, repo: Path, notebook_origen: str, descripcion: str = "") -> dict:
    """Manifest de una configuración de `07_predicciones`: `configuracion_id` (hash
    de `parametros`), fecha, estado de git (`estado_arbol_git`) y origen."""
    return {"configuracion_id": id_parametros(parametros),
            "fecha": pd.Timestamp.now().isoformat(timespec="seconds"), **estado_arbol_git(repo),
            "parametros": parametros, "notebook_origen": notebook_origen, "descripcion": descripcion}


def verificar_predicciones_fuera_de_muestra(predicciones: pd.DataFrame) -> None:
    """`ValueError` si alguna predicción viene de un fold cuyo train podía incluir
    su par (destino ≤ `ultimo_destino_train`), si el origen no cae en el año del
    fold o el holdout, o si hay claves repetidas por fold."""
    faltan = [c for c in COLUMNAS_PREDICCIONES if c not in predicciones.columns]
    if faltan:
        raise ValueError(f"predicciones sin {faltan}")
    origen = indice_trimestre(predicciones["ANO4_t"], predicciones["TRIMESTRE_t"])
    destino = origen + predicciones["h"].astype("int64")
    ultimo = predicciones["ultimo_destino_train"].astype("int64")
    es_holdout = predicciones["fold"].astype(str) == "holdout"
    anio_fold = pd.to_numeric(predicciones["fold"].where(~es_holdout), errors="coerce")
    problemas = {
        "par_en_train_del_fold": int((destino <= ultimo).sum()),
        "origen_fuera_del_anio_del_fold": int((~es_holdout & (predicciones["ANO4_t"].astype(int) != anio_fold)).sum()),
        "ultimo_destino_no_corresponde": int((~es_holdout & (ultimo != anio_fold * 4 - 1)).sum()),
        "holdout_con_origen_en_train": int((es_holdout & (origen < INDICE_HOLDOUT)).sum()),
        "holdout_con_train_fuera_de_train": int((es_holdout & (ultimo > INDICE_ULTIMO_DESTINO_TRAIN)).sum()),
        "clave_repetida": int(predicciones.duplicated([*CLAVE_PAR, "h", "fold", "conjunto_evaluacion"]).sum()),
    }
    malos = {k: v for k, v in problemas.items() if v}
    if malos:
        raise ValueError(f"predicciones no fuera de muestra: {malos}")


def parametros_por_defecto(funcion: Callable) -> dict:
    """Valores por defecto de los parámetros de `funcion` (para auditar defaults)."""
    original = inspect.unwrap(funcion)
    return {n: p.default for n, p in inspect.signature(original).parameters.items() if p.default is not inspect._empty}
