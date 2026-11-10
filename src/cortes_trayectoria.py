"""Marcas de cortes conocidos sobre pares de trayectoria `t`->`t+h` (ambas
eras) y peso de hogar unificado entre eras. Solo marcan: nunca excluyen."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mismas definiciones que `04_trayectorias.ipynb` (ruptura de CODUSU de las
# cohortes que entran en 2013T1-T3 y vuelven en 2014T1-T3).
RUPTURA_CODUSU_ORIGEN = {(2013, 1), (2013, 2), (2013, 3)}
RUPTURA_CODUSU_DESTINO = {(2014, 1), (2014, 2), (2014, 3)}
# Mismo criterio que `armonizacion` (`periodo_intervenido = 2007 <= anio <= 2015`).
INTERVENCION_DESDE, INTERVENCION_HASTA = (2007, 1), (2015, 4)
TRIMESTRE_NO_PUBLICADO_2007 = (2007, 3)
TRIMESTRE_TELEFONICO = (2020, 2)
# 2% queda cerca del menor trimestre marcado con los datos actuales (2004T1,
# 2,1%): quien corre registra el conjunto resultante, no solo el umbral.
UMBRAL_ITF0_PCT = 2.0

FLAGS_CORTE = ["par_periodo_intervenido", "par_cruza_ruptura_codusu", "previa_cruza_ruptura_codusu",
               "par_atraviesa_2007t3", "previa_no_observable_2007t3", "par_itf0_elevado", "par_toca_2020t2"]
FLAGS_PUNTUALES = ["par_cruza_ruptura_codusu", "previa_cruza_ruptura_codusu", "par_atraviesa_2007t3",
                   "previa_no_observable_2007t3", "par_itf0_elevado", "par_toca_2020t2"]


def _indice(anio: pd.Series, trimestre: pd.Series) -> np.ndarray:
    return anio.astype(int).to_numpy() * 4 + trimestre.astype(int).to_numpy() - 1


def _indices(trimestres) -> np.ndarray:
    return np.array([a * 4 + t - 1 for a, t in trimestres], dtype=int)


def pct_itf_cero_declarado(distancia: pd.DataFrame) -> float:
    """% de hogares con `ITF == 0` entre los que no tienen
    `ingreso_no_declarado == True` (no respuesta confirmada, solo regular)."""
    # No usa la columna `itf_cero_declarado` de 02b: en la era histórica vale NA
    # justo cuando ITF==0 (`ingreso_no_declarado` es NA ahí), y descartarla
    # daría 0% en toda la era.
    no_respuesta = (distancia["ingreso_no_declarado"] == True).fillna(False).astype(bool)  # noqa: E712
    itf = pd.to_numeric(distancia.loc[~no_respuesta, "ITF"], errors="coerce").dropna()
    return float(100 * (itf == 0).mean()) if len(itf) else np.nan


def trimestres_itf0_elevado(pct_por_trimestre: dict[tuple[int, int], float],
                            umbral_pct: float = UMBRAL_ITF0_PCT) -> list[tuple[int, int]]:
    """Trimestres (ordenados) cuyo % de `itf_cero_declarado` supera `umbral_pct`."""
    return sorted(q for q, pct in pct_por_trimestre.items() if pd.notna(pct) and pct > umbral_pct)


def marcar_cortes_par(pares: pd.DataFrame, trimestres_itf0: list[tuple[int, int]] | set) -> pd.DataFrame:
    """Agrega los 7 `FLAGS_CORTE` (booleanos) a cada par según `t`, `t+h` y
    `gap_previo`; no excluye ninguna fila."""
    resultado = pares.copy()
    t = _indice(pares["ANO4_t"], pares["TRIMESTRE_t"])
    th = _indice(pares["ANO4_th"], pares["TRIMESTRE_th"])
    gap = pares["gap_previo"].astype("Int8").to_numpy(dtype=float, na_value=np.nan) if "gap_previo" in pares else np.full(len(pares), np.nan)

    desde, hasta = _indices([INTERVENCION_DESDE, INTERVENCION_HASTA])
    origen_ruptura, destino_ruptura = _indices(RUPTURA_CODUSU_ORIGEN), _indices(RUPTURA_CODUSU_DESTINO)
    hueco_2007 = _indices([TRIMESTRE_NO_PUBLICADO_2007])[0]
    telefonico = _indices([TRIMESTRE_TELEFONICO])[0]

    resultado["par_periodo_intervenido"] = ((t >= desde) & (t <= hasta)) | ((th >= desde) & (th <= hasta))
    resultado["par_cruza_ruptura_codusu"] = (np.isin(t, origen_ruptura) & np.isin(th, destino_ruptura))
    resultado["previa_cruza_ruptura_codusu"] = (gap == 3) & np.isin(t - 4, origen_ruptura)
    resultado["par_atraviesa_2007t3"] = (t < hueco_2007) & (th > hueco_2007)
    resultado["previa_no_observable_2007t3"] = (t - 1 == hueco_2007) | (t - 4 == hueco_2007)
    itf0 = _indices(trimestres_itf0) if len(trimestres_itf0) else np.array([], dtype=int)
    resultado["par_itf0_elevado"] = np.isin(t, itf0) | np.isin(th, itf0)
    resultado["par_toca_2020t2"] = (t == telefonico) | (th == telefonico)
    return resultado


def peso_hogar_era(pares: pd.DataFrame, lado: str) -> pd.Series:
    """`PONDIH_{lado}` en pares `regional_2016`, `peso_hogar_{lado}` en
    `historica` (`Float64`). Error si una fila trae el peso de la otra era."""
    era = pares[f"era_{lado}"]
    pondih = pares.get(f"PONDIH_{lado}", pd.Series(pd.NA, index=pares.index)).astype("Float64")
    peso_hist = pares.get(f"peso_hogar_{lado}", pd.Series(pd.NA, index=pares.index)).astype("Float64")
    es_regular, es_historica = (era == "regional_2016").to_numpy(), (era == "historica").to_numpy()
    if (es_regular & peso_hist.notna().to_numpy()).any() or (es_historica & pondih.notna().to_numpy()).any():
        raise ValueError("par con peso de la otra era -- revisar antes de unificar")
    resultado = pd.Series(pd.NA, index=pares.index, dtype="Float64")
    resultado[es_regular] = pondih[es_regular]
    resultado[es_historica] = peso_hist[es_historica]
    return resultado


def peso_normalizado_trimestre(pares: pd.DataFrame, columna_peso: str) -> pd.Series:
    """`columna_peso` / su suma dentro de `(ANO4_t, TRIMESTRE_t)`: cada
    trimestre de origen pesa 1 en total, sin importar la era."""
    w = pares[columna_peso].astype(float)
    total = w.groupby([pares["ANO4_t"], pares["TRIMESTRE_t"]]).transform("sum")
    return w / total.replace(0, np.nan)
