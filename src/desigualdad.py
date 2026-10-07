"""Coeficiente de Gini ponderado, normalizacion por media ponderada e IC95% por
bootstrap de cluster -- insumos de la seccion de desigualdad del notebook 05 (D17).
"""
# Funciones puras (reciben arrays/DataFrames, no tocan `data/`), mismo criterio que
# panel.py, trayectorias.py y bootstrap.py -- testeables con fixtures sinteticas.
from __future__ import annotations

import numpy as np
import pandas as pd


def _a_float(valores) -> np.ndarray:
    # `to_numpy(na_value=...)` y no `astype`: tolera los dtypes nullable de la capa
    # nucleo (Int32/boolean) y los restos de texto de la capa armonizada sin romper.
    return pd.to_numeric(pd.Series(valores), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _valores_y_pesos(valores, pesos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(x, w, mascara_valida)` en float64. Filas con valor/peso no finito o
    peso <= 0 quedan fuera de la mascara; un negativo entre las validas
    levanta `ValueError`."""
    x = _a_float(valores)
    w = np.ones_like(x) if pesos is None else _a_float(pesos)
    if w.shape != x.shape:
        raise ValueError(f"valores y pesos tienen largo distinto: {x.shape} vs {w.shape}")
    valida = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if np.any(x[valida] < 0):
        raise ValueError("hay valores negativos: el Gini no esta definido sobre ellos")
    return x, w, valida


def _gini_ordenado(x_ordenado: np.ndarray, w_ordenado: np.ndarray) -> float:
    peso_total = w_ordenado.sum()
    suma_ponderada = float((w_ordenado * x_ordenado).sum())
    if peso_total <= 0 or suma_ponderada <= 0:
        return float("nan")
    acumulado = np.cumsum(w_ordenado)
    numerador = float((w_ordenado * (2 * acumulado - w_ordenado) * x_ordenado).sum())
    return numerador / (peso_total * suma_ponderada) - 1


def gini_ponderado(valores, pesos=None) -> float:
    """Gini de `valores` con `pesos` (default: todos 1). Los ceros cuentan en
    la distribucion; pesos <= 0 y valores/pesos no finitos se excluyen.
    `nan` si no queda masa; `ValueError` con negativos."""
    # Estimador exacto del area de Lorenz para datos ponderados (no la version con
    # correccion de sesgo n/(n-1)): es el que reproduce la serie publicada por INDEC y
    # el unico invariante a replicacion -- partir un peso en dos no cambia el resultado,
    # que es lo que permite mezclar pesos de hogar y de persona sin sesgo.
    x, w, valida = _valores_y_pesos(valores, pesos)
    x, w = x[valida], w[valida]
    if x.size == 0:
        return float("nan")
    orden = np.argsort(x, kind="mergesort")
    return _gini_ordenado(x[orden], w[orden])


def normalizar_por_media_ponderada(valores, pesos=None) -> np.ndarray:
    """`valores / media_ponderada(valores, pesos)`, con `nan` donde el valor
    o el peso no es valido, o donde la media ponderada no es positiva."""
    # Permite juntar trimestres de distinto nivel de precios en una misma
    # ventana sin que la inflacion entre trimestres se cuele como dispersion.
    x, w, valida = _valores_y_pesos(valores, pesos)
    peso_total = w[valida].sum()
    media = float((w[valida] * x[valida]).sum() / peso_total) if peso_total > 0 else float("nan")
    salida = np.full_like(x, np.nan)
    if np.isfinite(media) and media > 0:
        salida[valida] = x[valida] / media
    return salida


def gini_ic_bootstrap_cluster(
    tabla: pd.DataFrame, columna_valor: str, columna_peso: str, columna_cluster: str,
    n_boot: int = 400, seed: int = 42, batch: int = 100,
) -> dict:
    """Gini ponderado de `columna_valor`/`columna_peso` mas IC95% por
    bootstrap de cluster (`columna_cluster`, `n_boot` replicas). Mismas
    exclusiones que `gini_ponderado`."""
    # Mismo mecanismo Poisson(1) por cluster que src/bootstrap.py, reimplementado y no
    # importado: las internals de ese modulo agregan conteos N/S de un evento binario por
    # cluster, y el Gini necesita el vector de valores completo -- no se puede expresar
    # con esa API.
    vacio = {"gini": float("nan"), "gini_ic95_low": float("nan"), "gini_ic95_high": float("nan"),
             "n_filas": 0, "n_clusters": 0, "peso_total": 0.0}
    if tabla.empty:
        return vacio
    x, w, valida = _valores_y_pesos(tabla[columna_valor], tabla[columna_peso])
    cluster = tabla[columna_cluster].to_numpy()[valida]
    x, w = x[valida], w[valida]
    if x.size == 0:
        return vacio

    # El orden por valor no depende de los pesos: se ordena una sola vez y cada replica
    # solo recalcula los acumulados, en vez de reordenar n_boot veces.
    orden = np.argsort(x, kind="mergesort")
    x, w, cluster = x[orden], w[orden], cluster[orden]
    codigos, _ = pd.factorize(cluster)
    n_clusters = int(codigos.max()) + 1 if codigos.size else 0

    observado = _gini_ordenado(x, w)
    rng = np.random.default_rng(seed)
    replicas = np.empty(n_boot)
    hecho = 0
    while hecho < n_boot:
        b = min(batch, n_boot - hecho)
        pesos_cluster = rng.poisson(1.0, size=(b, n_clusters)).astype("float64")
        w_b = w * pesos_cluster[:, codigos]
        acumulado = np.cumsum(w_b, axis=1)
        peso_total_b = acumulado[:, -1]
        suma_ponderada_b = (w_b * x).sum(axis=1)
        numerador_b = (w_b * (2 * acumulado - w_b) * x).sum(axis=1)
        denominador_b = peso_total_b * suma_ponderada_b
        with np.errstate(invalid="ignore", divide="ignore"):
            replicas[hecho:hecho + b] = np.where(denominador_b > 0, numerador_b / np.where(denominador_b > 0, denominador_b, 1) - 1, np.nan)
        hecho += b
    replicas = replicas[np.isfinite(replicas)]
    ic_low, ic_high = ((float(np.percentile(replicas, 2.5)), float(np.percentile(replicas, 97.5)))
                       if replicas.size else (float("nan"), float("nan")))

    return {
        "gini": round(observado, 4) if np.isfinite(observado) else float("nan"),
        "gini_ic95_low": round(ic_low, 4) if np.isfinite(ic_low) else float("nan"),
        "gini_ic95_high": round(ic_high, 4) if np.isfinite(ic_high) else float("nan"),
        "n_filas": int(x.size), "n_clusters": n_clusters, "peso_total": float(w.sum()),
    }
