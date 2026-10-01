"""Bootstrap de cluster (pesos Poisson(1) i.i.d.) para comparar una tasa de evento
binario entre 2 grupos -- mismo mecanismo que `resumen_estandarizado` de
`notebooks/03_exploracion_descriptiva.ipynb` (celda 37), generalizado y parametrizado
para reusarse desde otros notebooks sin duplicar la lógica de inferencia estadística
(a diferencia del resto de las reglas de negocio del repo, que sí se duplican por
notebook -- ver notebooks/06_poblacion_riesgo.ipynb).
"""
# Funciones puras (reciben DataFrames, no tocan `data/`), mismo criterio que panel.py
# y trayectorias.py -- permite testearlas con fixtures sintéticas (tests/test_bootstrap.py).
from __future__ import annotations

import numpy as np
import pandas as pd


def _tabla_cluster_dos_grupos(tabla: pd.DataFrame, columna_cluster: str, columna_evento: str,
                               columna_grupo: str, grupo_a: object, grupo_b: object) -> tuple[np.ndarray, np.ndarray, int]:
    d = tabla.dropna(subset=[columna_evento, columna_grupo]).copy()
    d[columna_evento] = d[columna_evento].astype(bool)
    d = d[d[columna_grupo].isin([grupo_a, grupo_b])]
    grp = d.groupby([columna_cluster, columna_grupo])[columna_evento].agg(n="size", s="sum").reset_index()
    piv_n = grp.pivot_table(index=columna_cluster, columns=columna_grupo, values="n", fill_value=0)
    piv_s = grp.pivot_table(index=columna_cluster, columns=columna_grupo, values="s", fill_value=0)
    for g in (grupo_a, grupo_b):
        if g not in piv_n.columns:
            piv_n[g], piv_s[g] = 0.0, 0.0
    # .reindex (no __getitem__ con lista) -- con grupo_a/grupo_b booleanos (p.ej. True/False),
    # `piv_n[[grupo_a, grupo_b]]` dispara la detección de "lista de bool = máscara" de pandas
    # y rompe con columnas reales llamadas True/False (mismo bug que ya se vio en el notebook 05).
    N = piv_n.reindex(columns=[grupo_a, grupo_b]).to_numpy(dtype="float64")
    S = piv_s.reindex(columns=[grupo_a, grupo_b]).to_numpy(dtype="float64")
    return N, S, N.shape[0]


def diferencia_dos_grupos_bootstrap_cluster(
    tabla: pd.DataFrame, columna_cluster: str, columna_evento: str, columna_grupo: str,
    grupo_a: object, grupo_b: object, n_boot: int = 400, seed: int = 42, batch: int = 100,
) -> dict:
    """Tasa de `columna_evento` (bool) en `grupo_a` vs. `grupo_b` de `columna_grupo`,
    con IC95% de la diferencia (a - b) por bootstrap de cluster (`columna_cluster`,
    pesos Poisson(1) i.i.d., `n_boot` réplicas). Filas con `columna_evento`/`columna_grupo`
    nulos, o con `columna_grupo` fuera de `{grupo_a, grupo_b}`, se excluyen antes de
    agregar por cluster.

    Nota de implementación -- el bug que motivó extraer esta función: `N`/`S` agregados
    por cluster (`W @ N`) ya no tienen un eje de cluster para reducir; cada tasa de
    réplica es una división elemento a elemento sobre la columna del grupo, **sin**
    ningún `.sum()` extra -- sumar de nuevo ahí colapsa el eje de réplicas en vez del de
    clusters (ya no existe) y da un IC artificialmente angosto."""
    N, S, n_clusters = _tabla_cluster_dos_grupos(tabla, columna_cluster, columna_evento, columna_grupo, grupo_a, grupo_b)
    vacio = {"tasa_a": np.nan, "tasa_b": np.nan, "diferencia": np.nan,
             "diferencia_ic95_low": np.nan, "diferencia_ic95_high": np.nan,
             "n_a": 0, "n_b": 0, "n_clusters": 0}
    if n_clusters == 0:
        return vacio

    def _tasa(Nb: np.ndarray, Sb: np.ndarray, col: int) -> np.ndarray:
        n_col = Nb[:, col]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_col > 0, Sb[:, col] / np.where(n_col > 0, n_col, 1) * 100, np.nan)

    N_tot, S_tot = N.sum(axis=0, keepdims=True), S.sum(axis=0, keepdims=True)
    tasa_a_obs, tasa_b_obs = float(_tasa(N_tot, S_tot, 0)[0]), float(_tasa(N_tot, S_tot, 1)[0])
    diferencia = tasa_a_obs - tasa_b_obs

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    hecho = 0
    while hecho < n_boot:
        b = min(batch, n_boot - hecho)
        W = rng.poisson(1.0, size=(b, n_clusters)).astype("float64")
        Nb, Sb = W @ N, W @ S  # (b, 2) -- ya agregado por cluster, sin eje de cluster restante
        diffs[hecho:hecho + b] = _tasa(Nb, Sb, 0) - _tasa(Nb, Sb, 1)
        hecho += b
    diffs = diffs[~np.isnan(diffs)]
    ic_low, ic_high = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) else (np.nan, np.nan)

    return {
        "tasa_a": round(tasa_a_obs, 2), "tasa_b": round(tasa_b_obs, 2), "diferencia": round(diferencia, 2),
        "diferencia_ic95_low": round(ic_low, 2) if pd.notna(ic_low) else np.nan,
        "diferencia_ic95_high": round(ic_high, 2) if pd.notna(ic_high) else np.nan,
        "n_a": int(N[:, 0].sum()), "n_b": int(N[:, 1].sum()), "n_clusters": n_clusters,
    }


def estandarizar_diferencia_bootstrap(
    tabla: pd.DataFrame, columna_cluster: str, columna_evento: str, columna_grupo: str,
    grupo_con: object, grupo_sin: object, columna_estrato: str,
    n_boot: int = 400, seed: int = 42, batch: int = 100,
) -> dict:
    """Estandarización directa (tasa observada en `grupo_con` vs. la tasa que tendría si
    tuviera la composición de `columna_estrato` del propio `grupo_con` pero las tasas por
    estrato de `grupo_sin`) + IC95% de la diferencia por bootstrap de cluster -- puerto
    generalizado (un solo `columna_estrato` en vez de quintil×educación fijo) de
    `_tabla_cluster`/`_estandarizar`/`resumen_estandarizado` de
    `notebooks/03_exploracion_descriptiva.ipynb` (celda 37). A diferencia de
    `diferencia_dos_grupos_bootstrap_cluster`, acá sí hay un eje real de estratos para
    reducir con `.sum(axis=-1)` dentro de cada réplica -- no tiene el bug de esa función."""
    d = tabla.dropna(subset=[columna_evento, columna_grupo, columna_estrato]).copy()
    d[columna_evento] = d[columna_evento].astype(bool)
    d = d[d[columna_grupo].isin([grupo_con, grupo_sin])]
    d["_grupo_lbl"] = np.where(d[columna_grupo] == grupo_con, "con", "sin")
    grp = d.groupby([columna_cluster, "_grupo_lbl", columna_estrato])[columna_evento].agg(n="size", s="sum").reset_index()
    piv_n = grp.pivot_table(index=columna_cluster, columns=["_grupo_lbl", columna_estrato], values="n", fill_value=0, aggfunc="sum")
    piv_s = grp.pivot_table(index=columna_cluster, columns=["_grupo_lbl", columna_estrato], values="s", fill_value=0, aggfunc="sum")
    cols = list(piv_n.columns)
    N, S = piv_n.to_numpy(dtype="float64"), piv_s.to_numpy(dtype="float64")
    n_clusters = N.shape[0]
    vacio = {"pct_observado_con": np.nan, "pct_estandarizado_sin": np.nan, "diferencia": np.nan,
             "diferencia_ic95_low": np.nan, "diferencia_ic95_high": np.nan, "n_clusters": 0, "n_con": 0}
    if n_clusters == 0 or not cols:
        return vacio

    idx_con = [i for i, (g, _) in enumerate(cols) if g == "con"]
    estratos_con = {e: i for i, (g, e) in enumerate(cols) if g == "con"}
    estratos_sin = {e: i for i, (g, e) in enumerate(cols) if g == "sin"}
    comunes = sorted(set(estratos_con) & set(estratos_sin))
    idx_con_e = np.array([estratos_con[e] for e in comunes], dtype=int)
    idx_sin_e = np.array([estratos_sin[e] for e in comunes], dtype=int)

    def _estandarizar(Nb: np.ndarray, Sb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_con_tot = Nb[..., idx_con].sum(axis=-1)
        s_con_tot = Sb[..., idx_con].sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_obs = np.where(n_con_tot > 0, s_con_tot / np.where(n_con_tot > 0, n_con_tot, 1) * 100, np.nan)
        if len(comunes) == 0:
            return pct_obs, np.full_like(pct_obs, np.nan), n_con_tot
        n_con_e, n_sin_e, s_sin_e = Nb[..., idx_con_e], Nb[..., idx_sin_e], Sb[..., idx_sin_e]
        tasa_sin_e = np.divide(s_sin_e, n_sin_e, out=np.full_like(s_sin_e, np.nan), where=n_sin_e > 0)
        peso_e = np.where(np.isnan(tasa_sin_e), 0.0, n_con_e)
        suma_peso = peso_e.sum(axis=-1)
        pct_std = np.where(suma_peso > 0,
                            np.nansum(peso_e * np.nan_to_num(tasa_sin_e), axis=-1) / np.where(suma_peso > 0, suma_peso, 1) * 100,
                            np.nan)
        return pct_obs, pct_std, n_con_tot

    N_tot, S_tot = N.sum(axis=0, keepdims=True), S.sum(axis=0, keepdims=True)
    pct_obs_arr, pct_std_arr, n_con_arr = _estandarizar(N_tot, S_tot)
    pct_obs, pct_std, n_con = float(pct_obs_arr[0]), float(pct_std_arr[0]), float(n_con_arr[0])
    diferencia = pct_obs - pct_std

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    hecho = 0
    while hecho < n_boot:
        b = min(batch, n_boot - hecho)
        W = rng.poisson(1.0, size=(b, n_clusters)).astype("float64")
        Nb, Sb = W @ N, W @ S
        pct_obs_b, pct_std_b, _ = _estandarizar(Nb, Sb)
        diffs[hecho:hecho + b] = pct_obs_b - pct_std_b
        hecho += b
    diffs = diffs[~np.isnan(diffs)]
    ic_low, ic_high = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) else (np.nan, np.nan)

    return {
        "pct_observado_con": round(pct_obs, 2), "pct_estandarizado_sin": round(pct_std, 2),
        "diferencia": round(diferencia, 2),
        "diferencia_ic95_low": round(ic_low, 2) if pd.notna(ic_low) else np.nan,
        "diferencia_ic95_high": round(ic_high, 2) if pd.notna(ic_high) else np.nan,
        "n_clusters": n_clusters, "n_con": int(n_con),
    }
