"""Líneas base de la variación de distancia `t`->`t+h`: persistencia (0),
media global de train y regresión a la media por decil de `ratio_t`, con sus
métricas de holdout."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ajustar_regresion_media_decil(x_train: pd.Series, y_train: pd.Series, n_cuantiles: int = 10) -> dict:
    """Cortes de cuantil de `x_train` (extremos abiertos) y media de `y_train`
    por cuantil; filas con algún `NA` se ignoran."""
    valido = x_train.notna() & y_train.notna()
    x, y = x_train[valido].astype(float), y_train[valido].astype(float)
    cortes = np.unique(np.quantile(x, np.linspace(0, 1, n_cuantiles + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    grupo = pd.cut(x, cortes, labels=False, include_lowest=True)
    return {"cortes": cortes, "medias": y.groupby(grupo).mean(), "media_global": float(y.mean())}


def predecir_regresion_media_decil(modelo: dict, x: pd.Series) -> pd.Series:
    """Media de train del cuantil de cada `x`; cuantil sin datos o `x` nulo ->
    media global de train."""
    grupo = pd.cut(x.astype(float), modelo["cortes"], labels=False, include_lowest=True)
    prediccion = grupo.map(modelo["medias"])
    return prediccion.fillna(modelo["media_global"]).astype(float)


def metricas_prediccion(y: pd.Series, prediccion: pd.Series, media_train: float,
                        peso: pd.Series | None = None) -> dict:
    """MAE, RMSE y R² (contra predecir `media_train`), ponderados si hay `peso`;
    filas con `y`, `prediccion` o `peso` nulos se ignoran."""
    valido = y.notna() & prediccion.notna()
    if peso is not None:
        valido &= peso.notna()
    yv, pv = y[valido].astype(float).to_numpy(), prediccion[valido].astype(float).to_numpy()
    w = np.ones(len(yv)) if peso is None else peso[valido].astype(float).to_numpy()
    if len(yv) == 0 or w.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2_vs_media_train": np.nan}
    error = yv - pv
    sse_media = np.sum(w * (yv - media_train) ** 2)
    return {"n": int(len(yv)), "mae": float(np.sum(w * np.abs(error)) / w.sum()),
            "rmse": float(np.sqrt(np.sum(w * error ** 2) / w.sum())),
            "r2_vs_media_train": float(1 - np.sum(w * error ** 2) / sse_media) if sse_media > 0 else np.nan}
