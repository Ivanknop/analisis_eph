"""Modelos del bloque A (Fase D): matriz X, validación interna, LightGBM y lineal
regularizado con early stopping y reajuste, predicción, evaluación comparada por
bootstrap y capa `data/07_predicciones/` con reanudación por fold."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

import constantes as cte
import features as ft
import validacion as va
from features import CLAVE_PAR

SEMILLA = 20261110
# 6 hilos para dejar margen a otra sesión en el repo; la grilla corrió con 8. Con el número
# de hilos fijo el resultado es determinista.
HILOS = 6
VERSION_X = "bloque_a_v1"

PRECARIEDAD_FUERA_DE_X = ["share_precario_B_t", "share_precario_C_t", "share_precario_D_t"]
# En h=4 el par previo es el que termina en t: las dos columnas son idénticas
# [confirmado sobre data/06_features]; en h=1 no (coinciden 0,6%).
DUPLICADAS_POR_H = {4: ["variacion_simetrica_hasta_t_c"]}

NIVELES_CATEGORICOS = {
    "nivel_ed_jefe_bucket_t": ["secundario incompleto o menos", "secundario completo o más"],
    "situacion_laboral_t": cte.ORDEN_SITUACION_LABORAL,
    "condicion_actividad_jefe_t": ["ocupado", "desocupado", "inactivo"],
    "tipologia_perceptores_t": ["solo_laborales", "solo_previsionales", "mixto", "sin_perceptores"],
    "motivo_previa_na_t": ft.MOTIVOS_PREVIA,
    "era": cte.ERAS,
}

TECNICAS_MODELO = ["decil_ratio", "lineal_regularizado", "gradient_boosting"]
SUFIJO_SIN_AGLOMERADOS = "_sin_baja_respuesta"
CRITERIOS_BLOQUE_A = ["imputacion_excluye", "imputacion_conserva"]

PARAMETROS_GB_FIJOS = {
    "learning_rate": 0.05, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "deterministic": True, "force_row_wise": True, "num_threads": HILOS, "verbose": -1,
    "seed": SEMILLA, "bagging_seed": SEMILLA + 1, "feature_fraction_seed": SEMILLA + 2, "data_random_seed": SEMILLA + 3,
}
MAX_ITERACIONES_GB = 3000
PACIENCIA_GB = 100
GRILLA_GB = [{"num_leaves": hojas, "min_data_in_leaf": minimo, "feature_fraction": fraccion}
             for hojas in (15, 63) for minimo in (100, 1000) for fraccion in (0.6, 0.9)]
# Elección de la grilla (`elegir_hiperparametros_gb`, validación interna de los 6 últimos
# folds, 2026-11-10), registrada en `.claude/decisiones.md`. La grilla es plana: las
# diferencias entre combinaciones (0,0004–0,002) son menores que la variación entre folds.
HIPERPARAMETROS_GB: dict[tuple[str, int], dict] = {
    ("trayectoria", 1): {"num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9},
    ("trayectoria", 4): {"num_leaves": 15, "min_data_in_leaf": 100, "feature_fraction": 0.6},
    ("entrada", 1): {"num_leaves": 15, "min_data_in_leaf": 1000, "feature_fraction": 0.6},
    ("entrada", 4): {"num_leaves": 15, "min_data_in_leaf": 1000, "feature_fraction": 0.6},
    ("permanencia", 1): {"num_leaves": 15, "min_data_in_leaf": 100, "feature_fraction": 0.9},
    ("permanencia", 4): {"num_leaves": 15, "min_data_in_leaf": 100, "feature_fraction": 0.6},
}

# Penalización en unidades de pérdida sumada sobre filas (prior fijo, no se reescala al
# reajustar con más filas): ridge `alpha = fuerza`, lasso `alpha = fuerza / n` (sklearn
# promedia la pérdida), logística `C = 1 / fuerza`.
# Siete valores: en la prueba corta (h=1, fold 2020) la elegida cayó en los dos extremos
# de la grilla original de cinco (1 y 10.000).
FUERZAS_LINEAL = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
PENALIZACIONES_LINEAL = {"trayectoria": ["ridge", "lasso"], "clasificacion": ["l2", "l1"]}
# Elección de `elegir_penalizacion_lineal` (misma validación que la grilla de GB).
PENALIZACION_LINEAL: dict[tuple[str, int], str] = {
    ("trayectoria", 1): "lasso", ("trayectoria", 4): "lasso",
    ("entrada", 1): "l1", ("entrada", 4): "l1",
    ("permanencia", 1): "l1", ("permanencia", 4): "l2",
}

# Más filas en el reajuste admiten más árboles con el mismo learning rate; el tope evita
# extrapolar en los primeros folds, donde la validación interna es una parte grande del train.
TOPE_ESCALA_ITERACIONES = 1.5
UMBRAL_TRAIN_CHICO = 50_000
N_FOLDS_SELECCION = 6


def columnas_x(h: int) -> list[str]:
    """Lista blanca de `CATALOGO_FEATURES` del bloque A para `h`: precariedad A
    sola y sin las columnas duplicadas de ese `h`."""
    fuera = set(PRECARIEDAD_FUERA_DE_X) | set(DUPLICADAS_POR_H.get(h, []))
    columnas = [f.nombre for f in ft.CATALOGO_FEATURES if f.en_x and f.nombre not in fuera]
    prohibidas = ft.nombres_prohibidos(columnas)
    if prohibidas:
        raise ValueError(f"columnas prohibidas en X: {prohibidas}")
    return columnas


def columnas_extra_carga(h: int) -> tuple[str, ...]:
    """Columnas a pedir a `validacion.cargar_pares_modelado` para armar X."""
    return tuple(columnas_x(h))


def preparar_x_gb(pares: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """X para LightGBM: numéricas en `float` (NA nativo) y categóricas con los
    niveles fijos de `NIVELES_CATEGORICOS`. Error ante un nivel no previsto."""
    x = {}
    for columna in columnas:
        valores = pares[columna]
        if columna in NIVELES_CATEGORICOS:
            niveles = NIVELES_CATEGORICOS[columna]
            imprevistos = set(valores.dropna().astype(str).unique()) - set(niveles)
            if imprevistos:
                raise ValueError(f"{columna}: niveles no previstos {sorted(imprevistos)}")
            x[columna] = pd.Categorical(valores.astype("object").where(valores.notna(), None), categories=niveles)
        else:
            x[columna] = pd.to_numeric(valores.astype("Float64"), errors="coerce").astype(float)
    return pd.DataFrame(x, index=pares.index)


def clasificar_columnas_lineal(pares: pd.DataFrame, columnas: list[str]) -> dict[str, list[str]]:
    """Columnas por tipo para el preprocesamiento lineal: share (`share_*`),
    binaria (bool), categórica (`NIVELES_CATEGORICOS`) o continua."""
    tipos = {"share": [], "binaria": [], "categorica": [], "continua": []}
    for columna in columnas:
        if columna in NIVELES_CATEGORICOS:
            tipos["categorica"].append(columna)
        elif columna.startswith("share_"):
            tipos["share"].append(columna)
        elif str(pares[columna].dtype) in ("bool", "boolean"):
            tipos["binaria"].append(columna)
        else:
            tipos["continua"].append(columna)
    return tipos


@va.solo_train
def particion_validacion_interna(train_fold: pd.DataFrame, variable: str = "simetrica") -> tuple[pd.Series, pd.Series, int]:
    """Máscaras (train para elegir iteraciones, validación) dentro de `train_fold`:
    validación = último año de origen, solo pares limpios; embargo de h afuera."""
    anio_validacion = int(train_fold["ANO4_t"].max())
    rol = va.asignar_fold(train_fold, anio_validacion)
    return rol == "train", (rol == "test") & va.mascara_evaluacion(train_fold, variable), anio_validacion


def validacion_utilizable(train_fold: pd.DataFrame, mascara_es: pd.Series, mascara_validacion: pd.Series) -> bool:
    """`True` si el train para elegir iteraciones tiene al menos `MIN_TRIMESTRES_TRAIN`
    trimestres de origen y la validación no está vacía."""
    trimestres = train_fold.loc[mascara_es, ["ANO4_t", "TRIMESTRE_t"]].drop_duplicates()
    return len(trimestres) >= va.MIN_TRIMESTRES_TRAIN and bool(mascara_validacion.any())


def iteraciones_reajuste(mejor_iteracion: int, n_es: int, n_final: int,
                         tope: float = TOPE_ESCALA_ITERACIONES) -> int:
    """`round(mejor_iteracion × min(n_final / n_es, tope))`, al menos 1."""
    return max(1, int(round(mejor_iteracion * min(n_final / n_es, tope))))


def _objetivo_lgb(objetivo_modelado: str) -> dict:
    if va.es_clasificacion(objetivo_modelado):
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "regression", "metric": "l2"}


def _parametros_gb(objetivo_modelado: str, hiperparametros: dict) -> dict:
    return {**PARAMETROS_GB_FIJOS, **_objetivo_lgb(objetivo_modelado), **hiperparametros}


def parada_temprana_gb(x_es: pd.DataFrame, y_es: pd.Series, peso_es: pd.Series, x_val: pd.DataFrame,
                       y_val: pd.Series, objetivo_modelado: str, hiperparametros: dict) -> tuple[int, float]:
    """Mejor iteración y pérdida de validación (log loss o MSE) de LightGBM con
    early stopping (`PACIENCIA_GB`) sobre `x_val`."""
    parametros = _parametros_gb(objetivo_modelado, hiperparametros)
    datos = lgb.Dataset(x_es, y_es.astype(float), weight=peso_es.astype(float), free_raw_data=True)
    validacion = lgb.Dataset(x_val, y_val.astype(float), reference=datos)
    modelo = lgb.train(parametros, datos, num_boost_round=MAX_ITERACIONES_GB, valid_sets=[validacion],
                       callbacks=[lgb.early_stopping(PACIENCIA_GB, verbose=False)])
    metrica = parametros["metric"]
    return int(modelo.best_iteration), float(modelo.best_score["valid_0"][metrica])


def ajustar_gb_final(x: pd.DataFrame, y: pd.Series, peso: pd.Series, objetivo_modelado: str,
                     hiperparametros: dict, iteraciones: int) -> lgb.Booster:
    """LightGBM con `iteraciones` fijas sobre todo el train del fold."""
    datos = lgb.Dataset(x, y.astype(float), weight=peso.astype(float))
    return lgb.train(_parametros_gb(objetivo_modelado, hiperparametros), datos, num_boost_round=iteraciones)


def _estimador_lineal(objetivo_modelado: str, penalizacion: str, fuerza: float, n: int):
    if not va.es_clasificacion(objetivo_modelado):
        if penalizacion == "ridge":
            return Ridge(alpha=fuerza)
        if penalizacion == "lasso":
            return Lasso(alpha=fuerza / n, max_iter=5000, tol=1e-4, random_state=SEMILLA)
    elif penalizacion == "l2":
        return LogisticRegression(C=1 / fuerza, l1_ratio=0.0, solver="lbfgs", max_iter=1000)
    elif penalizacion == "l1":
        return LogisticRegression(C=1 / fuerza, l1_ratio=1.0, solver="liblinear", max_iter=1000, random_state=SEMILLA)
    raise ValueError(f"penalización {penalizacion} no corresponde a {objetivo_modelado}")


def ajustar_lineal(matriz: pd.DataFrame, y: pd.Series, peso: pd.Series, objetivo_modelado: str,
                   penalizacion: str, fuerza: float):
    """Estimador lineal regularizado ajustado sobre `matriz` ya preprocesada."""
    estimador = _estimador_lineal(objetivo_modelado, penalizacion, fuerza, len(matriz))
    objetivo = y.astype(int) if va.es_clasificacion(objetivo_modelado) else y.astype(float)
    estimador.fit(matriz.to_numpy(), objetivo.to_numpy(), sample_weight=peso.astype(float).to_numpy())
    return estimador


def _predecir_lineal(estimador, matriz: pd.DataFrame, objetivo_modelado: str) -> np.ndarray:
    if va.es_clasificacion(objetivo_modelado):
        return estimador.predict_proba(matriz.to_numpy())[:, 1]
    return estimador.predict(matriz.to_numpy())


def perdida(y: pd.Series, prediccion: np.ndarray, objetivo_modelado: str) -> float:
    """Log loss (clasificación, probabilidades recortadas a 1e-15) o MSE."""
    yv, pv = y.astype(float).to_numpy(), np.asarray(prediccion, dtype=float)
    if va.es_clasificacion(objetivo_modelado):
        p = np.clip(pv, 1e-15, 1 - 1e-15)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    return float(np.mean((yv - pv) ** 2))


@dataclass
class ModeloAjustado:
    """Modelo de un fold con todo lo necesario para predecir y el registro del ajuste."""
    tecnica: str
    objetivo_modelado: str
    columnas: list[str]
    estimador: object = None
    preprocesamiento: dict | None = None
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Configuracion:
    """Una configuración de `07_predicciones`: objetivo, variable, h, técnica,
    criterio, hiperparámetros (GB) o penalización (lineal) y columnas de X."""
    objetivo_modelado: str
    variable: str
    h: int
    tecnica: str
    criterio_imputacion: str
    columnas: tuple[str, ...]
    hiperparametros: tuple = ()
    penalizacion: str | None = None
    parametros_peso: tuple = ()
    aglomerados_fuera_train: tuple = ()

    def parametros(self) -> dict:
        parametros = {"objetivo_modelado": self.objetivo_modelado, "variable": self.variable, "h": self.h,
                      "tecnica": self.tecnica, "criterio_imputacion": self.criterio_imputacion,
                      "universo": "universo_2003_2025", "version_x": VERSION_X, "columnas_x": list(self.columnas),
                      "hiperparametros": dict(self.hiperparametros), "penalizacion": self.penalizacion,
                      "parametros_gb_fijos": PARAMETROS_GB_FIJOS if self.tecnica == "gradient_boosting" else None}
        # Solo si no están vacíos: así el hash (`configuracion_id`) de las configuraciones
        # del bloque A, escritas antes de que existieran estos campos, no cambia.
        if self.parametros_peso:
            parametros["parametros_peso"] = dict(self.parametros_peso)
        if self.aglomerados_fuera_train:
            parametros["aglomerados_fuera_train"] = list(self.aglomerados_fuera_train)
        return parametros

    @property
    def nombre(self) -> str:
        sufijo = SUFIJO_SIN_AGLOMERADOS if self.aglomerados_fuera_train else ""
        return f"{self.tecnica}|{self.criterio_imputacion}{sufijo}"


def _y(pares: pd.DataFrame, objetivo_modelado: str, variable: str) -> pd.Series:
    return pares[va.columna_respuesta(objetivo_modelado, variable)].astype(float)


@va.solo_train
def entrenar_fold(train_fold: pd.DataFrame, configuracion: Configuracion, fuerza: float | None = None,
                  peso: pd.Series | None = None) -> ModeloAjustado:
    """Ajusta `configuracion` sobre `train_fold` (ya filtrado), con `peso` de muestra
    (1 si es None): validación interna (sin ponderar) y reajuste final sobre todo
    `train_fold`. `None` si la validación interna no es utilizable."""
    objetivo, variable, columnas = configuracion.objetivo_modelado, configuracion.variable, list(configuracion.columnas)
    y = _y(train_fold, objetivo, variable)
    peso = pd.Series(1.0, index=train_fold.index) if peso is None else peso.astype(float).reindex(train_fold.index)
    if peso.isna().any() or (peso < 0).any():
        raise ValueError("pesos de train faltantes o negativos")
    if configuracion.tecnica == "decil_ratio":
        modelo = va.ajustar_linea_base(train_fold, objetivo, "decil_ratio", variable)
        return ModeloAjustado("decil_ratio", objetivo, columnas, modelo, info={"n_final": len(train_fold)})
    mascara_es, mascara_val, anio_validacion = particion_validacion_interna(train_fold, variable)
    info = {"anio_validacion": anio_validacion, "n_es": int(mascara_es.sum()), "n_validacion": int(mascara_val.sum()),
            "n_final": len(train_fold)}
    necesita_validacion = configuracion.tecnica == "gradient_boosting" or fuerza is None
    if necesita_validacion and not validacion_utilizable(train_fold, mascara_es, mascara_val):
        return None
    t0 = time.time()
    if configuracion.tecnica == "gradient_boosting":
        hiper = dict(configuracion.hiperparametros)
        x = preparar_x_gb(train_fold, columnas)
        mejor, perdida_val = parada_temprana_gb(x[mascara_es], y[mascara_es], peso[mascara_es], x[mascara_val],
                                                y[mascara_val], objetivo, hiper)
        iteraciones = iteraciones_reajuste(mejor, info["n_es"], info["n_final"])
        estimador = ajustar_gb_final(x, y, peso, objetivo, hiper, iteraciones)
        info.update(mejor_iteracion=mejor, n_iteraciones_final=iteraciones, perdida_validacion=perdida_val)
        return ModeloAjustado("gradient_boosting", objetivo, columnas, estimador, info={**info, "segundos": time.time() - t0})
    if configuracion.tecnica != "lineal_regularizado":
        raise ValueError(f"técnica desconocida: {configuracion.tecnica}")
    tipos = clasificar_columnas_lineal(train_fold, columnas)
    if fuerza is None:
        prep_es = va.ajustar_preprocesamiento_lineal(train_fold[mascara_es], tipos["share"], tipos["continua"],
                                                     tipos["binaria"], tipos["categorica"])
        matriz_es = va.aplicar_preprocesamiento_lineal(prep_es, train_fold[mascara_es])
        matriz_val = va.aplicar_preprocesamiento_lineal(prep_es, train_fold[mascara_val])
        perdidas = {}
        for candidata in FUERZAS_LINEAL:
            estimador = ajustar_lineal(matriz_es, y[mascara_es], peso[mascara_es], objetivo,
                                       configuracion.penalizacion, candidata)
            perdidas[candidata] = perdida(y[mascara_val], _predecir_lineal(estimador, matriz_val, objetivo), objetivo)
        fuerza = min(perdidas, key=perdidas.get)
        info.update(perdida_validacion=perdidas[fuerza], perdidas_validacion=perdidas,
                    fuerza_en_borde=fuerza in (FUERZAS_LINEAL[0], FUERZAS_LINEAL[-1]))
    preprocesamiento = va.ajustar_preprocesamiento_lineal(train_fold, tipos["share"], tipos["continua"],
                                                          tipos["binaria"], tipos["categorica"])
    matriz = va.aplicar_preprocesamiento_lineal(preprocesamiento, train_fold)
    estimador = ajustar_lineal(matriz, y, peso, objetivo, configuracion.penalizacion, fuerza)
    info.update(fuerza=fuerza, n_columnas_lineal=matriz.shape[1], segundos=time.time() - t0)
    return ModeloAjustado("lineal_regularizado", objetivo, columnas, estimador, preprocesamiento, info)


def predecir(modelo: ModeloAjustado, pares: pd.DataFrame) -> pd.Series:
    """Predicción de `modelo` sobre `pares`: variación o probabilidad de `pobre_th`."""
    if modelo.tecnica == "decil_ratio":
        return va.predecir_linea_base(modelo.estimador, pares)
    if modelo.tecnica == "gradient_boosting":
        return pd.Series(modelo.estimador.predict(preparar_x_gb(pares, modelo.columnas)), index=pares.index)
    matriz = va.aplicar_preprocesamiento_lineal(modelo.preprocesamiento, pares)
    return pd.Series(_predecir_lineal(modelo.estimador, matriz, modelo.objetivo_modelado), index=pares.index)


@va.solo_train
def datos_fold(pares: pd.DataFrame, anio_test: int, objetivo_modelado: str, variable: str,
               criterio_imputacion: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train del fold para el criterio, test del fold: limpio + imputados), ambos
    restringidos a la población de `objetivo_modelado`."""
    rol = va.asignar_fold(pares, anio_test)
    poblacion = va.poblacion_objetivo(pares, objetivo_modelado)
    train = pares[(rol == "train") & poblacion & va.mascara_train(pares, variable, criterio_imputacion)]
    test = pares[(rol == "test") & poblacion & (va.mascara_evaluacion(pares, variable)
                                                | va.mascara_evaluacion(pares, variable, "imputados"))]
    return train, test


@va.solo_train
def datos_fold_configuracion(pares: pd.DataFrame, anio_test: int, configuracion: Configuracion
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`datos_fold` de `configuracion`, con su train sin `aglomerados_fuera_train` (el test queda entero)."""
    train, test = datos_fold(pares, anio_test, configuracion.objetivo_modelado, configuracion.variable,
                             configuracion.criterio_imputacion)
    if configuracion.aglomerados_fuera_train:
        train = train[~train["AGLOMERADO_t"].astype(int).isin(configuracion.aglomerados_fuera_train)]
    return train, test


@va.solo_train
def peso_train_fold(pares: pd.DataFrame, anio_test: int, train: pd.DataFrame, configuracion: Configuracion,
                    pesos_externos: pd.Series | None = None) -> pd.Series | None:
    """Peso de muestra del train del fold: `None` (sin peso), el de `conjunto_train` para
    `_reponderado` o `pesos_externos` (índice `CLAVE_PAR`) para `_reponderado_propension`."""
    criterio = configuracion.criterio_imputacion
    if criterio == "imputacion_excluye_reponderado":
        rol = va.asignar_fold(pares, anio_test)
        universo = pares[(rol == "train") & va.poblacion_objetivo(pares, configuracion.objetivo_modelado)]
        if configuracion.aglomerados_fuera_train:
            universo = universo[~universo["AGLOMERADO_t"].astype(int).isin(configuracion.aglomerados_fuera_train)]
        seleccion, peso = va.conjunto_train(universo, configuracion.variable, criterio)
        if not seleccion.index.equals(train.index):
            raise ValueError("el train reponderado no coincide con el del fold")
        return peso
    if criterio == "imputacion_excluye_reponderado_propension":
        if pesos_externos is None:
            raise ValueError("el nivel de propensión exige `pesos_externos`")
        peso = pesos_externos.reindex(pd.MultiIndex.from_frame(train[CLAVE_PAR]))
        if peso.isna().any():
            raise ValueError(f"{int(peso.isna().sum())} pares de train sin peso de propensión")
        return pd.Series(peso.to_numpy(), index=train.index)
    return None


def resumen_peso(peso: pd.Series) -> dict:
    """Media, p99, máximo y tamaño efectivo de Kish relativo (`(Σw)²/Σw²/n`) de `peso`."""
    w = peso.astype(float)
    return {"peso_medio": float(w.mean()), "peso_p99": float(w.quantile(0.99)), "peso_max": float(w.max()),
            "kish_relativo": float(w.sum() ** 2 / (w ** 2).sum() / len(w))}


def directorio_configuracion(raiz: Path, configuracion: Configuracion, configuracion_id: str) -> Path:
    """Carpeta de la configuración en `07_predicciones` (`validacion.ruta_predicciones` sin el archivo)."""
    return va.ruta_predicciones(raiz, configuracion.objetivo_modelado, configuracion.h, configuracion_id).parent


def id_configuracion(configuracion: Configuracion) -> str:
    # Sin pasar por `manifest_configuracion`: el manifest consulta git (~20s con el filtro
    # nbstripout) y el id es solo el hash de los parámetros.
    return va.id_parametros(configuracion.parametros())


def ruta_fold(directorio: Path, fold: int | str) -> Path:
    return directorio / f"fold={fold}" / "data.parquet"


def fold_completo(directorio: Path, fold: int | str) -> bool:
    """`True` si el fold ya está escrito, tiene su `ajuste.json` y pasa el verificador."""
    ruta = ruta_fold(directorio, fold)
    if not ruta.exists() or not (ruta.parent / "ajuste.json").exists():
        return False
    try:
        va.verificar_predicciones_fuera_de_muestra(pd.read_parquet(ruta))
    except ValueError:
        return False
    return True


def tabla_predicciones(test: pd.DataFrame, train: pd.DataFrame, prediccion: pd.Series, configuracion: Configuracion,
                       anio_test: int) -> pd.DataFrame:
    """Filas de `07_predicciones` del fold (`validacion.COLUMNAS_PREDICCIONES`)."""
    limpio = va.mascara_evaluacion(test, configuracion.variable)
    tabla = test[CLAVE_PAR].copy()
    tabla["h"] = configuracion.h
    tabla["fold"] = str(anio_test)
    tabla["ultimo_destino_train"] = anio_test * 4 - 1
    tabla["evaluacion_agrupada_excluido"] = ~va.sin_viviendas_de_train(test, train)
    tabla["conjunto_evaluacion"] = np.where(limpio, "limpio", "imputados")
    tabla["prediccion"] = prediccion.astype(float)
    tabla["y_observado"] = _y(test, configuracion.objetivo_modelado, configuracion.variable)
    return tabla[va.COLUMNAS_PREDICCIONES].reset_index(drop=True)


def _escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, default=_serializable), encoding="utf-8")


def _serializable(valor):
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    return str(valor)


def leer_ajuste(directorio: Path, fold: int | str) -> dict:
    return json.loads((ruta_fold(directorio, fold).parent / "ajuste.json").read_text(encoding="utf-8"))


@va.solo_train
def correr_configuracion(pares: pd.DataFrame, folds: list[int], configuracion: Configuracion, raiz: Path, repo: Path,
                         notebook_origen: str, rehacer: bool = False,
                         fuerzas: dict[int, float] | None = None,
                         pesos_externos: dict[int, pd.Series] | None = None,
                         guardar_modelo: bool = False) -> pd.DataFrame:
    """Entrena y escribe en `raiz` cada fold de `folds` (con pesos de `peso_train_fold`
    y, si `guardar_modelo`, el booster), salteando los completos salvo `rehacer`.
    Una fila por fold con el registro del ajuste."""
    configuracion_id = id_configuracion(configuracion)
    directorio = directorio_configuracion(raiz, configuracion, configuracion_id)
    if rehacer and directorio.exists():
        shutil.rmtree(directorio)
    if not (directorio / "manifest.json").exists():
        _escribir_json(directorio / "manifest.json",
                       va.manifest_configuracion(configuracion.parametros(), repo, notebook_origen, configuracion.nombre))
    filas = []
    # Una vez por llamada, y solo si hay algo que escribir: `git diff HEAD` pasa por el
    # filtro nbstripout de cada notebook (~20s) y el árbol no cambia mientras corre.
    estado_git: dict = {}
    for anio in folds:
        base = {"configuracion_id": configuracion_id, "configuracion": configuracion.nombre, "anio_test": anio}
        if fold_completo(directorio, anio):
            filas.append({**base, **leer_ajuste(directorio, anio), "reanudado": True})
            continue
        # Una falla de un fold queda registrada y no corta el resto (la reanudación permite
        # rehacerlo); la fuga de holdout y el verificador, afuera del try, sí cortan todo.
        try:
            train, test = datos_fold_configuracion(pares, anio, configuracion)
            if fuerzas is not None and anio not in fuerzas:
                filas.append({**base, "omitido": "sin_fuerza_de_referencia", "reanudado": False})
                continue
            externos = None if pesos_externos is None else pesos_externos.get(anio)
            peso = peso_train_fold(pares, anio, train, configuracion, externos)
            modelo = entrenar_fold(train, configuracion, None if fuerzas is None else fuerzas[anio], peso)
            if modelo is None:
                filas.append({**base, "omitido": "sin_validacion_interna", "reanudado": False})
                continue
            tabla = tabla_predicciones(test, train, predecir(modelo, test), configuracion, anio)
        except va.FugaHoldoutError:
            raise
        except Exception as error:
            filas.append({**base, "error": f"{type(error).__name__}: {error}", "reanudado": False})
            continue
        va.verificar_predicciones_fuera_de_muestra(tabla)
        ruta = ruta_fold(directorio, anio)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_parquet(ruta, index=False)
        if not estado_git:
            estado_git = va.estado_arbol_git(repo)
        resumen = {}
        if peso is not None:
            train[CLAVE_PAR].assign(peso=peso.to_numpy()).to_parquet(ruta.parent / "pesos_train.parquet", index=False)
            resumen = resumen_peso(peso)
        if guardar_modelo and configuracion.tecnica == "gradient_boosting":
            modelo.estimador.save_model(str(ruta.parent / "modelo.txt"))
        ajuste = {**modelo.info, **resumen, "fraccion_historica_train": float((train["era"] == "historica").mean()),
                  "n_test_limpio": int((tabla["conjunto_evaluacion"] == "limpio").sum()),
                  "n_test_imputados": int((tabla["conjunto_evaluacion"] == "imputados").sum()),
                  **estado_git}
        _escribir_json(ruta.parent / "ajuste.json", ajuste)
        filas.append({**base, **ajuste, "reanudado": False})
    return pd.DataFrame(filas)


def leer_predicciones(raiz: Path, configuracion: Configuracion) -> pd.DataFrame:
    """Todas las predicciones escritas de `configuracion` (folds, sin holdout), verificadas."""
    directorio = directorio_configuracion(raiz, configuracion, id_configuracion(configuracion))
    partes = [pd.read_parquet(r) for r in sorted(directorio.glob("fold=*/data.parquet"))]
    if not partes:
        return pd.DataFrame(columns=va.COLUMNAS_PREDICCIONES)
    tabla = pd.concat(partes, ignore_index=True)
    tabla = tabla[tabla["fold"] != "holdout"]
    va.verificar_predicciones_fuera_de_muestra(tabla)
    return tabla


@va.solo_train
def elegir_hiperparametros_gb(pares: pd.DataFrame, folds: list[int], objetivo_modelado: str, h: int,
                              variable: str = "simetrica", grilla: list[dict] = GRILLA_GB) -> pd.DataFrame:
    """Pérdida de validación interna (y mejor iteración) de cada combinación de
    `grilla` en cada fold de `folds`, con `imputacion_excluye`. Una fila por
    combinación × fold; el test de los folds no se usa."""
    columnas = columnas_x(h)
    filas = []
    for anio in folds:
        train, _ = datos_fold(pares, anio, objetivo_modelado, variable, cte.CRITERIO_IMPUTACION_PRINCIPAL)
        mascara_es, mascara_val, anio_validacion = particion_validacion_interna(train, variable)
        x, y = preparar_x_gb(train, columnas), _y(train, objetivo_modelado, variable)
        peso = pd.Series(1.0, index=train.index)
        for indice, hiper in enumerate(grilla):
            t0 = time.time()
            mejor, perdida_val = parada_temprana_gb(x[mascara_es], y[mascara_es], peso[mascara_es], x[mascara_val],
                                                    y[mascara_val], objetivo_modelado, hiper)
            filas.append({"objetivo_modelado": objetivo_modelado, "variable": variable, "h": h, "anio_test": anio,
                          "anio_validacion": anio_validacion, "combinacion": indice, **hiper, "mejor_iteracion": mejor,
                          "perdida_validacion": perdida_val, "n_es": int(mascara_es.sum()),
                          "n_validacion": int(mascara_val.sum()), "segundos": time.time() - t0})
    return pd.DataFrame(filas)


def resumir_grilla_gb(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por combinación: pérdida media entre folds, su rango y mejor iteración
    media; `elegida` = menor pérdida media; `en_borde` si `num_leaves` o
    `min_data_in_leaf` de la elegida es un extremo de la grilla."""
    claves = ["objetivo_modelado", "variable", "h", "combinacion", "num_leaves", "min_data_in_leaf", "feature_fraction"]
    resumen = tabla.groupby(claves, as_index=False).agg(
        perdida_media=("perdida_validacion", "mean"), perdida_min=("perdida_validacion", "min"),
        perdida_max=("perdida_validacion", "max"), mejor_iteracion_media=("mejor_iteracion", "mean"),
        mejor_iteracion_max=("mejor_iteracion", "max"))
    resumen["elegida"] = resumen.groupby(["objetivo_modelado", "variable", "h"])["perdida_media"].transform("min") == resumen["perdida_media"]
    extremos = {p: (min(g[p] for g in GRILLA_GB), max(g[p] for g in GRILLA_GB)) for p in ("num_leaves", "min_data_in_leaf")}
    resumen["en_borde"] = resumen["elegida"] & np.logical_or.reduce(
        [resumen[p].isin(extremos[p]) for p in extremos])
    resumen["iteraciones_en_tope"] = resumen["mejor_iteracion_max"] >= MAX_ITERACIONES_GB
    return resumen


@va.solo_train
def elegir_penalizacion_lineal(pares: pd.DataFrame, folds: list[int], objetivo_modelado: str, h: int,
                               variable: str = "simetrica") -> pd.DataFrame:
    """Pérdida de validación interna de cada penalización × fuerza en cada fold
    de `folds`, con `imputacion_excluye`. Una fila por penalización × fuerza × fold."""
    columnas = columnas_x(h)
    clase = "clasificacion" if va.es_clasificacion(objetivo_modelado) else "trayectoria"
    filas = []
    for anio in folds:
        train, _ = datos_fold(pares, anio, objetivo_modelado, variable, cte.CRITERIO_IMPUTACION_PRINCIPAL)
        mascara_es, mascara_val, anio_validacion = particion_validacion_interna(train, variable)
        tipos = clasificar_columnas_lineal(train, columnas)
        prep = va.ajustar_preprocesamiento_lineal(train[mascara_es], tipos["share"], tipos["continua"],
                                                  tipos["binaria"], tipos["categorica"])
        matriz_es = va.aplicar_preprocesamiento_lineal(prep, train[mascara_es])
        matriz_val = va.aplicar_preprocesamiento_lineal(prep, train[mascara_val])
        y = _y(train, objetivo_modelado, variable)
        peso = pd.Series(1.0, index=matriz_es.index)
        for penalizacion in PENALIZACIONES_LINEAL[clase]:
            for fuerza in FUERZAS_LINEAL:
                t0 = time.time()
                estimador = ajustar_lineal(matriz_es, y[mascara_es], peso, objetivo_modelado, penalizacion, fuerza)
                filas.append({"objetivo_modelado": objetivo_modelado, "variable": variable, "h": h, "anio_test": anio,
                              "anio_validacion": anio_validacion, "penalizacion": penalizacion, "fuerza": fuerza,
                              "perdida_validacion": perdida(y[mascara_val], _predecir_lineal(estimador, matriz_val,
                                                                                            objetivo_modelado),
                                                            objetivo_modelado),
                              "segundos": time.time() - t0})
    return pd.DataFrame(filas)


def resumir_penalizacion_lineal(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por penalización: pérdida media entre folds de la mejor fuerza de cada fold;
    `elegida` = la menor; `fuerza_en_borde` = fracción de folds cuya mejor fuerza es extrema."""
    claves = ["objetivo_modelado", "variable", "h", "penalizacion"]
    mejor = tabla.loc[tabla.groupby([*claves, "anio_test"])["perdida_validacion"].idxmin()]
    resumen = mejor.groupby(claves, as_index=False).agg(
        perdida_media=("perdida_validacion", "mean"),
        fuerza_mediana=("fuerza", "median"),
        fuerza_en_borde=("fuerza", lambda f: float(f.isin([FUERZAS_LINEAL[0], FUERZAS_LINEAL[-1]]).mean())))
    resumen["elegida"] = resumen.groupby(["objetivo_modelado", "variable", "h"])["perdida_media"].transform("min") == resumen["perdida_media"]
    return resumen


def configuraciones_bloque_a(objetivo_modelado: str, variable: str, h: int) -> list[Configuracion]:
    """Configuraciones del bloque A para (objetivo, variable, h): técnicas × criterios
    (la trayectoria log solo con decil y GB). Exige `HIPERPARAMETROS_GB` y `PENALIZACION_LINEAL`."""
    # El decil sobre la log es la referencia del skill de GB en la variante log.
    tecnicas = ["decil_ratio", "gradient_boosting"] if variable == "log" else TECNICAS_MODELO
    columnas = tuple(columnas_x(h))
    configuraciones = []
    for tecnica in tecnicas:
        hiper, penalizacion = (), None
        if tecnica == "gradient_boosting":
            hiper = tuple(sorted(HIPERPARAMETROS_GB[(objetivo_modelado, h)].items()))
        elif tecnica == "lineal_regularizado":
            penalizacion = PENALIZACION_LINEAL[(objetivo_modelado, h)]
        for criterio in CRITERIOS_BLOQUE_A:
            configuraciones.append(Configuracion(objetivo_modelado, variable, h, tecnica, criterio, columnas, hiper,
                                                 penalizacion))
    return configuraciones


CRITERIOS_BLOQUE_C = ["imputacion_excluye_fraccion_max_0.5", "imputacion_excluye_reponderado",
                      "imputacion_excluye_reponderado_propension"]


def configuracion_gb(objetivo_modelado: str, h: int, criterio_imputacion: str,
                     aglomerados_fuera_train: tuple = ()) -> Configuracion:
    """GB simétrico con los hiperparámetros del bloque A para `criterio_imputacion`."""
    import no_respuesta_modelo as nm

    parametros_peso = (nm.parametros_peso_propension(HILOS)
                       if criterio_imputacion == "imputacion_excluye_reponderado_propension" else ())
    return Configuracion(objetivo_modelado, "simetrica", h, "gradient_boosting", criterio_imputacion,
                         tuple(columnas_x(h)), tuple(sorted(HIPERPARAMETROS_GB[(objetivo_modelado, h)].items())),
                         parametros_peso=parametros_peso, aglomerados_fuera_train=tuple(aglomerados_fuera_train))


def configuraciones_referencia(objetivo_modelado: str, h: int) -> list[Configuracion]:
    """Decil y GB de `imputacion_excluye` del bloque A (ya escritos, no se reentrenan)."""
    return [c for c in configuraciones_bloque_a(objetivo_modelado, "simetrica", h)
            if c.criterio_imputacion == cte.CRITERIO_IMPUTACION_PRINCIPAL and c.tecnica != "lineal_regularizado"]


def _metricas(objetivo_modelado: str, y: pd.Series, prediccion: pd.Series, base: pd.Series, media_train: float,
              peso: pd.Series | None) -> dict:
    if va.es_clasificacion(objetivo_modelado):
        return va.metricas_clasificacion(y, prediccion, base, media_train, peso)
    return va.metricas_regresion(y, prediccion, media_train, base, peso)


DIFERENCIAS_CLASIFICACION = ["log_loss", "brier", "auc_roc", "auc_pr_relativa", "pendiente_calibracion",
                             "intercepto_calibracion", "ece", "diferencia_media_predicha"]
DIFERENCIAS_REGRESION = ["mae", "rmse", "skill_vs_base"]


def evaluar_fold_comparado(predicciones: dict[str, pd.Series], y: pd.Series, objetivo_modelado: str,
                           medias_train: dict[str, float], vivienda: pd.DataFrame, peso: pd.Series | None = None,
                           n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Métricas de cada configuración (`"técnica|criterio"`) contra el decil de su
    criterio (o el de `excluye`) y diferencias pareadas criterio − excluye por técnica,
    con IC95% de un único bootstrap por vivienda (las mismas réplicas para todo)."""
    nombres = list(predicciones)
    diferencias = DIFERENCIAS_CLASIFICACION if va.es_clasificacion(objetivo_modelado) else DIFERENCIAS_REGRESION
    referencia = f"|{cte.CRITERIO_IMPUTACION_PRINCIPAL}"

    def decil_de(criterio: str) -> str:
        propio = f"decil_ratio|{criterio}"
        return propio if propio in predicciones else f"decil_ratio{referencia}"

    def todas(datos: pd.DataFrame, w: pd.Series) -> dict:
        por_configuracion = {}
        for nombre in nombres:
            criterio = nombre.split("|")[1]
            media = medias_train.get(nombre, medias_train.get(criterio))
            por_configuracion[nombre] = _metricas(objetivo_modelado, y, predicciones[nombre],
                                                  predicciones.get(decil_de(criterio)), media, w)
        resultado = {f"{n}#{m}": v for n, metricas in por_configuracion.items() for m, v in metricas.items()}
        for nombre, metricas in por_configuracion.items():
            tecnica, criterio = nombre.split("|")
            decil = por_configuracion.get(decil_de(criterio))
            if decil is not None and tecnica != "decil_ratio":
                for metrica in diferencias:
                    resultado[f"{nombre}#{metrica}__menos_decil"] = metricas[metrica] - decil[metrica]
        for nombre, metricas in por_configuracion.items():
            tecnica, criterio = nombre.split("|")
            excluye = por_configuracion.get(f"{tecnica}{referencia}")
            if excluye is None or nombre == f"{tecnica}{referencia}":
                continue
            # `imputacion_conserva` -> `conserva_menos_excluye`, el nombre que ya leen 09 y 10.
            etiqueta = f"{tecnica}|{criterio.removeprefix('imputacion_')}_menos_excluye"
            for metrica in diferencias:
                resultado[f"{etiqueta}#{metrica}"] = metricas[metrica] - excluye[metrica]
        return resultado

    tabla = va.bootstrap_metricas_cluster(vivienda, list(vivienda.columns), todas, peso, n_boot, seed)
    partes = tabla["metrica"].str.split("#", n=1, expand=True)
    tabla["configuracion"], tabla["metrica"] = partes[0], partes[1]
    return tabla


def resumen_estabilidad(metricas_fold: pd.DataFrame, valor: str = "valor") -> pd.DataFrame:
    """Por configuración y métrica: media, desvío, IQR, mínimo y máximo entre folds,
    excluidos los folds con `train_chico`; y la media en folds con y sin cada marca."""
    validos = metricas_fold[~metricas_fold["train_chico"].astype(bool)]
    claves = ["objetivo_modelado", "variable", "h", "configuracion", "metrica"]
    resumen = validos.groupby(claves)[valor].agg(
        media="mean", desvio="std", iqr=lambda s: s.quantile(0.75) - s.quantile(0.25), minimo="min", maximo="max",
        n_folds="size").reset_index()
    for marca in ("contexto_marcado", "periodo_intervenido", "salto_de_era"):
        medias = validos.groupby([*claves, marca])[valor].mean().unstack(marca)
        for flag, sufijo in ((True, "con"), (False, "sin")):
            if flag in medias:
                resumen = resumen.merge(medias[flag].rename(f"media_{sufijo}_{marca}").reset_index(), on=claves, how="left")
    return resumen


@va.solo_train
def preparar_evaluacion_fold(pares: pd.DataFrame, raiz: Path, configuraciones: list[Configuracion], anio_test: int,
                             conjunto: str = "limpio", ponderacion: str = "sin_ponderar",
                             aglomerados: tuple | None = None, excluir_aglomerados: tuple | None = None,
                             peso_test: pd.Series | None = None) -> dict | None:
    """Insumos de `evaluar_preparado` para el fold (conjunto común de predicciones,
    filtro de aglomerados, medias de train y peso de la lectura); `None` si falta algo."""
    tablas = {}
    for configuracion in configuraciones:
        ruta = ruta_fold(directorio_configuracion(raiz, configuracion, id_configuracion(configuracion)), anio_test)
        if not ruta.exists():
            return None
        prediccion = pd.read_parquet(ruta)
        va.verificar_predicciones_fuera_de_muestra(prediccion)
        prediccion = prediccion[(prediccion["fold"] == str(anio_test)) & (prediccion["conjunto_evaluacion"] == conjunto)]
        if prediccion.empty:
            return None
        tablas[configuracion.nombre] = prediccion.set_index(CLAVE_PAR)
    comunes = sorted(set.intersection(*(set(t.index) for t in tablas.values())))
    if not comunes:
        return None
    indice = pd.MultiIndex.from_tuples(comunes, names=CLAVE_PAR)
    aglomerado = pares.set_index(CLAVE_PAR).loc[indice, "AGLOMERADO_t"].astype(int).to_numpy()
    if aglomerados is not None:
        indice = indice[np.isin(aglomerado, list(aglomerados))]
    elif excluir_aglomerados is not None:
        indice = indice[~np.isin(aglomerado, list(excluir_aglomerados))]
    if not len(indice):
        return None
    primera = configuraciones[0]
    y = next(iter(tablas.values())).loc[indice, "y_observado"].astype(float).reset_index(drop=True)
    predicciones = {n: t.loc[indice, "prediccion"].astype(float).reset_index(drop=True) for n, t in tablas.items()}
    datos = pares.set_index(CLAVE_PAR).loc[indice, ["AGLOMERADO_t", "peso_normalizado_trimestre"]].reset_index()
    medias = {}
    for configuracion in configuraciones:
        train, _ = datos_fold_configuracion(pares, anio_test, configuracion)
        medias[configuracion.nombre] = float(_y(train, primera.objetivo_modelado, primera.variable).mean())
    if ponderacion == "sin_ponderar":
        peso = None
    elif ponderacion == "ponderado_trimestre":
        peso = datos["peso_normalizado_trimestre"].astype(float)
    elif ponderacion == cte.PONDERACION_PROPENSION:
        if peso_test is None:
            raise ValueError("la lectura ponderada por propensión exige `peso_test`")
        peso = pd.Series(peso_test.reindex(indice).to_numpy(dtype=float))
        if peso.isna().all():
            return None
    else:
        raise ValueError(f"ponderación desconocida: {ponderacion}")
    return {"predicciones": predicciones, "y": y, "objetivo_modelado": primera.objetivo_modelado, "medias": medias,
            "vivienda": datos[va.CLAVE_VIVIENDA], "peso": peso,
            "etiquetas": {"objetivo_modelado": primera.objetivo_modelado, "variable": primera.variable, "h": primera.h,
                          "anio_test": anio_test, "conjunto_evaluacion": conjunto, "ponderacion": ponderacion,
                          "n_pares": len(indice), "n_pares_con_peso": len(indice) if peso is None else int(peso.notna().sum())}}


def evaluar_preparado(preparado: dict | None, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """`evaluar_fold_comparado` sobre los insumos de `preparar_evaluacion_fold` (vacío si `None`)."""
    if preparado is None:
        return pd.DataFrame()
    tabla = evaluar_fold_comparado(preparado["predicciones"], preparado["y"], preparado["objetivo_modelado"],
                                   preparado["medias"], preparado["vivienda"], preparado["peso"], n_boot, seed)
    return tabla.assign(**preparado["etiquetas"])


@va.solo_train
def evaluar_configuraciones_fold(pares: pd.DataFrame, raiz: Path, configuraciones: list[Configuracion], anio_test: int,
                                 conjunto: str = "limpio", ponderacion: str = "sin_ponderar", n_boot: int = 200,
                                 seed: int = 42, aglomerados: tuple | None = None,
                                 excluir_aglomerados: tuple | None = None,
                                 peso_test: pd.Series | None = None) -> pd.DataFrame:
    """`evaluar_fold_comparado` sobre las predicciones escritas del fold (conjunto común,
    opcionalmente solo `aglomerados` o sin `excluir_aglomerados`); `peso_test` (índice
    `CLAVE_PAR`) para `ponderado_propension`. Vacío si falta algún fold."""
    preparado = preparar_evaluacion_fold(pares, raiz, configuraciones, anio_test, conjunto, ponderacion, aglomerados,
                                         excluir_aglomerados, peso_test)
    return evaluar_preparado(preparado, n_boot, seed)
