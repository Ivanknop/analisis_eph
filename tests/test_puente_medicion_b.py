"""Puente entre la medición A (línea de pobreza) y B (posición relativa sin
línea de pobreza) en el solape 2016T2-2025T4. Requiere datos reales
procesados -- se salta si las capas no están en disco."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TRAYECTORIA_A = REPO / "data" / "05_trayectoria_distancia"
TRAYECTORIA_B = REPO / "data" / "06_trayectoria_relativa"
DISTANCIA_LP_A = REPO / "data" / "02b_distancia_lp"
POSICION_B = REPO / "data" / "02c_posicion_relativa"

REQUIERE_DATOS_REALES = pytest.mark.skipif(
    not (TRAYECTORIA_A.exists() and TRAYECTORIA_B.exists() and DISTANCIA_LP_A.exists() and POSICION_B.exists()),
    reason="requiere data/05_trayectoria_distancia/, data/06_trayectoria_relativa/, "
           "data/02b_distancia_lp/ y data/02c_posicion_relativa/ reales")

CLAVE_PAR = ["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "ANO4_t", "TRIMESTRE_t"]


def _particiones(base: Path, tipo: str) -> list[tuple[int, int]]:
    particiones = []
    for p_anio in sorted((base / tipo).glob("ANO4=*")):
        anio = int(p_anio.name.split("=")[1])
        for p_trim in sorted(p_anio.glob("TRIMESTRE=*")):
            if (p_trim / "data.parquet").exists():
                particiones.append((anio, int(p_trim.name.split("=")[1])))
    return particiones


def _cargar_particion(base: Path, tipo: str, anio: int, trimestre: int, columnas: list[str]) -> pd.DataFrame:
    p = base / tipo / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet"
    df = pd.read_parquet(p)
    return df[[c for c in columnas if c in df.columns]]


def _cargar_capa_completa(base: Path, tipo: str, columnas: list[str]) -> pd.DataFrame:
    partes = [_cargar_particion(base, tipo, anio, trimestre, columnas) for anio, trimestre in _particiones(base, tipo)]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=columnas)


@REQUIERE_DATOS_REALES
@pytest.mark.parametrize("h", [1, 4])
def test_variacion_propia_b_igual_a_variacion_propia_aglomerado_log(h):
    """Identidad algebraica (ver `.claude/decisiones.md`): en el solape
    2016+, `variacion_propia_b` debe coincidir exacto con
    `variacion_propia_aglomerado_log` de la medición A."""
    a = _cargar_capa_completa(TRAYECTORIA_A, f"pares_h{h}", CLAVE_PAR + ["variacion_propia_aglomerado_log"])
    b = _cargar_capa_completa(TRAYECTORIA_B, f"pares_h{h}", CLAVE_PAR + ["variacion_propia_b"])
    m = a.merge(b, on=CLAVE_PAR, how="inner")
    assert len(m) == len(a), f"h={h}: {len(a) - len(m)} pares de A (2016+) sin contraparte en B tras el merge"

    va = m["variacion_propia_aglomerado_log"].astype(float).to_numpy()
    vb = m["variacion_propia_b"].astype(float).to_numpy()
    ok = np.isclose(va, vb, rtol=1e-6, atol=1e-6, equal_nan=True)
    if not ok.all():
        peor = (m.loc[~ok].assign(diferencia=np.abs(va[~ok] - vb[~ok]))
                 .sort_values("diferencia", ascending=False, na_position="first"))
        pytest.fail(
            f"h={h}: {len(peor)}/{len(m)} pares con variacion_propia_b != "
            f"variacion_propia_aglomerado_log (tolerancia 1e-6).\nPeores 15 filas:\n{peor.head(15).to_string()}"
        )


@REQUIERE_DATOS_REALES
@pytest.mark.parametrize("h", [1, 4])
def test_variacion_propia_b_correlaciona_con_variante_simetrica_de_a(h):
    """`variacion_propia_b` vs. `variacion_propia_aglomerado_simetrica` (A):
    no se espera igualdad (normalizan por objetos distintos), sí
    correlación alta de Spearman (colas largas)."""
    a = _cargar_capa_completa(TRAYECTORIA_A, f"pares_h{h}", CLAVE_PAR + ["variacion_propia_aglomerado_simetrica"])
    b = _cargar_capa_completa(TRAYECTORIA_B, f"pares_h{h}", CLAVE_PAR + ["variacion_propia_b"])
    m = a.merge(b, on=CLAVE_PAR, how="inner").dropna(
        subset=["variacion_propia_aglomerado_simetrica", "variacion_propia_b"])

    correlacion = m["variacion_propia_b"].corr(m["variacion_propia_aglomerado_simetrica"], method="spearman")

    m = m.assign(decil_b=pd.qcut(m["variacion_propia_b"], 10, labels=False, duplicates="drop"))
    diferencia_por_decil = m.groupby("decil_b").apply(
        lambda g: (g["variacion_propia_aglomerado_simetrica"] - g["variacion_propia_b"]).median(),
        include_groups=False)
    print(f"h={h}: correlación de Spearman={correlacion:.4f}")
    print(f"diferencia mediana por decil de variacion_propia_b:\n{diferencia_por_decil}")

    assert correlacion > 0.9, f"h={h}: correlación de Spearman {correlacion:.3f} <= 0.9"


@REQUIERE_DATOS_REALES
def test_ranking_posicion_relativa_vs_ratio_lp_por_trimestre():
    """Correlación de ranking (Spearman, por trimestre) entre
    `posicion_relativa_aglomerado` (B, Capa 1) y `ratio_lp` (A, Capa 1) en
    `t` -- objetos de nivel distinto, se espera que ordenen igual."""
    columnas_a = ["CODUSU", "NRO_HOGAR", "ratio_lp"]
    columnas_b = ["CODUSU", "NRO_HOGAR", "posicion_relativa_aglomerado"]
    correlaciones = {}
    for anio, trimestre in _particiones(DISTANCIA_LP_A, "hogar"):
        a = _cargar_particion(DISTANCIA_LP_A, "hogar", anio, trimestre, columnas_a)
        b = _cargar_particion(POSICION_B, "hogar", anio, trimestre, columnas_b)
        m = a.merge(b, on=["CODUSU", "NRO_HOGAR"], how="inner").dropna()
        if len(m) < 30:
            continue
        correlaciones[(anio, trimestre)] = m["ratio_lp"].corr(m["posicion_relativa_aglomerado"], method="spearman")

    serie = pd.Series(correlaciones)
    mediana = serie.median()
    print(f"mediana de correlación de ranking por trimestre: {mediana:.4f}")
    print(f"peores 5 trimestres:\n{serie.sort_values().head(5)}")
    assert mediana > 0.9, f"mediana de correlación de ranking por trimestre = {mediana:.3f} <= 0.9"
