"""Test de regresión (D33): `calcular_distancia_lp` + `canasta_regional`
deben seguir reproduciendo la tasa oficial de INDEC. Requiere datos reales
no versionados -- se salta si no están."""
from pathlib import Path

import pandas as pd
import pytest

import canasta_regional
import linea_pobreza

REPO = Path(__file__).resolve().parents[1]
NUCLEO = REPO / "data" / "02_nucleo"
ARMONIZADO = REPO / "data" / "01_armonizado"
CANASTAS = REPO / "data" / "canastas_indec"

REQUIERE_DATOS_REALES = pytest.mark.skipif(
    not (NUCLEO.exists() and (CANASTAS / "cba_regional_crudo.csv").exists()),
    reason="requiere data/02_nucleo/ y data/canastas_indec/ reales")

OFICIAL = {
    (2016, 2): {"pobreza_hogares": 21.5, "pobreza_personas": 30.3,
                "indigencia_hogares": 4.5, "indigencia_personas": 6.1},
    (2024, 2): {"pobreza_hogares": 28.6, "pobreza_personas": 38.1,
                "indigencia_hogares": 6.4, "indigencia_personas": 8.2},
    (2025, 1): {"pobreza_hogares": 24.1, "pobreza_personas": 31.6,
                "indigencia_hogares": 5.6, "indigencia_personas": 6.9},
}
SEMESTRES = [(2016, 2), (2024, 2), (2025, 1)]
UMBRAL_OK, UMBRAL_REVISAR = 0.3, 1.0
COLUMNAS_HOGAR = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO", "ITF", "PONDIH", "ingreso_no_declarado"]
COLUMNAS_INDIVIDUAL = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH04", "CH06"]


def _cargar_trimestre(anio: int, trimestre: int, canasta_trimestral: pd.DataFrame) -> pd.DataFrame:
    hogar = pd.read_parquet(NUCLEO / "hogar" / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet",
                             columns=COLUMNAS_HOGAR)
    region = pd.read_parquet(ARMONIZADO / "hogar" / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet",
                              columns=["CODUSU", "NRO_HOGAR", "REGION"])
    region["REGION"] = pd.to_numeric(region["REGION"], errors="coerce").astype("Int8")
    hogar = hogar.merge(region, on=["CODUSU", "NRO_HOGAR"], how="left")
    individual = pd.read_parquet(NUCLEO / "individual" / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet",
                                  columns=COLUMNAS_INDIVIDUAL)
    canasta_t = canasta_trimestral[(canasta_trimestral["ANO4"] == anio)
                                    & (canasta_trimestral["TRIMESTRE"] == trimestre)]
    distancia = linea_pobreza.calcular_distancia_lp(hogar, individual, canasta_t[["REGION", "cba_ae", "cbt_ae"]])
    n_miembros = individual.groupby(["CODUSU", "NRO_HOGAR"]).size().rename("n_miembros")
    distancia = distancia.merge(hogar[["CODUSU", "NRO_HOGAR", "PONDIH"]], on=["CODUSU", "NRO_HOGAR"], how="left")
    distancia = distancia.merge(n_miembros, on=["CODUSU", "NRO_HOGAR"], how="left")
    return distancia


def _tasa_ponderada(df: pd.DataFrame, columna: str, peso: pd.Series) -> float:
    valido = df[columna].notna() & peso.notna()
    num = (df.loc[valido, columna].astype(float) * peso[valido]).sum()
    den = peso[valido].sum()
    return 100 * num / den


def _clasificar(desvio: float) -> str:
    if abs(desvio) <= UMBRAL_OK:
        return "aceptable"
    if abs(desvio) <= UMBRAL_REVISAR:
        return "revisar"
    return "frenar"


@REQUIERE_DATOS_REALES
def test_reproduce_la_distribucion_de_desvios_ya_validada():
    # Igual que la sección 4 del notebook 08: las dos ventanas aportan a la
    # distribución 19/4/1 (D25/D26) -- (a) sola da 12/12 aceptable, el
    # desvío viene todo de (b).
    canasta_por_metodo = {
        metodo: canasta_regional.construir_canasta_trimestral(CANASTAS, metodo)
        for metodo in ("a_mismo_trimestre", "b_un_mes_atras")
    }
    clasificaciones = []
    for metodo, canasta_trimestral in canasta_por_metodo.items():
        for anio, semestre in SEMESTRES:
            trimestres = (1, 2) if semestre == 1 else (3, 4)
            partes = [_cargar_trimestre(anio, t, canasta_trimestral) for t in trimestres]
            df = pd.concat(partes, ignore_index=True)
            peso_personas = df["PONDIH"] * df["n_miembros"]
            calculado = {
                "pobreza_hogares": _tasa_ponderada(df, "pobre", df["PONDIH"]),
                "pobreza_personas": _tasa_ponderada(df, "pobre", peso_personas),
                "indigencia_hogares": _tasa_ponderada(df, "indigente", df["PONDIH"]),
                "indigencia_personas": _tasa_ponderada(df, "indigente", peso_personas),
            }
            oficial = OFICIAL[(anio, semestre)]
            for indicador, valor in calculado.items():
                clasificaciones.append(_clasificar(valor - oficial[indicador]))

    assert clasificaciones.count("aceptable") == 19
    assert clasificaciones.count("revisar") == 4
    assert clasificaciones.count("frenar") == 1
