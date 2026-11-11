"""Cotas de ingreso para hogares con no respuesta: ingreso efectivamente
declarado (regular) o no imputado (histórica) contra la línea, clase de cota
por hogar y pares `t`->`t+h` recuperables para entrada/permanencia."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from imputacion_ingresos import limpiar_monto
from tipado_nucleo import parsear_monetario

CLAVE_HOGAR = ["CODUSU", "NRO_HOGAR"]
COMPONENTES_OCUPACION_PRINCIPAL = ("PP08D1", "PP08F1", "PP08F2")
MONTOS_CUENTA_PROPIA = ("PP06C", "PP06D")
COMPONENTES_FUERA_DE_P21 = ("PP08J1", "PP08J2", "PP08J3")
VARIABLES_CON_MENOS9 = ("P21", "TOT_P12", "T_VI", *COMPONENTES_FUERA_DE_P21)
CLASES_COTA = ("declarado", "seguro_no_pobre", "indeterminado", "sin_ingreso_declarado", "sin_cota")


def columnas_no_laborales(columnas) -> list[str]:
    """Montos no laborales por persona presentes (`V*_M`, `V19_AM`)."""
    return [c for c in columnas if c.startswith("V") and (c.endswith("_M") or c == "V19_AM")]


def _monto(individual: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in individual:
        return pd.Series(np.nan, index=individual.index, dtype=float)
    return limpiar_monto(individual[columna]).astype(float)


def _es_menos9(individual: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in individual:
        return pd.Series(False, index=individual.index)
    return (parsear_monetario(individual[columna]) == -9).fillna(False).astype(bool)


def _suma_no_negativa(individual: pd.DataFrame, columnas) -> pd.Series:
    montos = [(_monto(individual, c).clip(lower=0)) for c in columnas]
    if not montos:
        return pd.Series(0.0, index=individual.index)
    return pd.concat(montos, axis=1).sum(axis=1, min_count=1).fillna(0.0)


def ingreso_declarado_persona(individual: pd.DataFrame) -> pd.DataFrame:
    """Por persona: `cota_persona` (suma de lo declarado, `-9` cuenta 0),
    `algun_menos9`, `p47t_menos9` y `p47t_positivo_con_menos9` (P47T parcial)."""
    # P21 en -9: los subcomponentes declarados (D1/F1/F2, o PP06C/PP06D de cuenta propia)
    # son partes de P21, así que su máximo sigue siendo cota inferior. Ídem T_VI con sus V*_M.
    p21 = _monto(individual, "P21")
    respaldo_p21 = pd.concat([_suma_no_negativa(individual, COMPONENTES_OCUPACION_PRINCIPAL),
                              *[_monto(individual, c).clip(lower=0).fillna(0.0) for c in MONTOS_CUENTA_PROPIA]],
                             axis=1).max(axis=1)
    ocupacion_principal = p21.clip(lower=0).where(p21.notna(), respaldo_p21)
    t_vi = _monto(individual, "T_VI")
    respaldo_t_vi = _suma_no_negativa(individual, columnas_no_laborales(individual.columns))
    no_laboral = t_vi.clip(lower=0).where(t_vi.notna(), respaldo_t_vi)
    cota = (ocupacion_principal.fillna(0.0) + _suma_no_negativa(individual, COMPONENTES_FUERA_DE_P21)
            + _monto(individual, "TOT_P12").clip(lower=0).fillna(0.0) + no_laboral.fillna(0.0))
    algun_menos9 = pd.concat([_es_menos9(individual, c) for c in VARIABLES_CON_MENOS9], axis=1).any(axis=1)
    p47t_menos9 = _es_menos9(individual, "P47T")
    p47t = _monto(individual, "P47T")
    return pd.DataFrame({
        "cota_persona": cota, "algun_menos9": algun_menos9, "p47t_menos9": p47t_menos9,
        "p47t_positivo_con_menos9": algun_menos9 & (p47t > 0).fillna(False),
    }, index=individual.index)


def cota_ingreso_hogar(individual: pd.DataFrame) -> pd.DataFrame:
    """Por hogar (`CLAVE_HOGAR`): `cota_ingreso` (Σ `cota_persona`), `n_personas_menos9`,
    `n_p47t_menos9` y `n_p47t_parcial`."""
    personas = ingreso_declarado_persona(individual)
    agrupado = personas.assign(**{c: individual[c].to_numpy() for c in CLAVE_HOGAR}).groupby(CLAVE_HOGAR)
    return agrupado.agg(cota_ingreso=("cota_persona", "sum"), n_personas_menos9=("algun_menos9", "sum"),
                        n_p47t_menos9=("p47t_menos9", "sum"),
                        n_p47t_parcial=("p47t_positivo_con_menos9", "sum")).reset_index()


def cota_historica(itf: pd.Series, monto_imputado_max: pd.Series) -> pd.Series:
    """Ingreso no imputado = `ITF - monto_imputado_max`, recortado a `>= 0`; `NA`
    si el monto máximo imputado es `NA`."""
    # `monto_imputado_max` es cota superior de lo imputado (05d): su resta es cota
    # inferior de lo declarado, sin depender de que el hot-deck de INDEC sea correcto.
    return (itf.astype(float) - monto_imputado_max.astype(float)).clip(lower=0)


def clasificar_cota(ratio_cota: pd.Series, con_no_respuesta: pd.Series) -> pd.Series:
    """Clase de `CLASES_COTA` por hogar: `declarado` sin no respuesta; si no,
    según `ratio_cota` (cota / línea): `>= 1`, `(0, 1)`, `0` o `NA`."""
    r = ratio_cota.astype(float)
    nr = con_no_respuesta.astype("boolean").fillna(False).astype(bool).to_numpy()
    clase = np.select([~nr, r >= 1, r > 0, r == 0],
                      ["declarado", "seguro_no_pobre", "indeterminado", "sin_ingreso_declarado"], default="sin_cota")
    return pd.Series(clase, index=ratio_cota.index, dtype="object")


def pobre_segun_cota(clase: pd.Series, pobre_declarado: pd.Series) -> pd.Series:
    """`pobre` conocido: el declarado si `declarado`, `False` si `seguro_no_pobre`, `NA` si no."""
    resultado = pd.Series(pd.NA, index=clase.index, dtype="boolean")
    declarado = (clase == "declarado").to_numpy()
    resultado[declarado] = pobre_declarado.astype("boolean")[declarado]
    resultado[(clase == "seguro_no_pobre").to_numpy()] = False
    return resultado


def recuperacion_pares(pares: pd.DataFrame, clase_t: str = "clase_t", clase_th: str = "clase_th",
                       valido_actual: str = "valido_actual") -> pd.DataFrame:
    """Por par con `identidad_confirmada`: `pobre_t_cota`, `pobre_th_cota`,
    `recupera_entrada`, `recupera_permanencia` y `via` (`t`, `th`, `ambas`)."""
    # Solo se recupera lo que hoy queda afuera por ingreso: al menos una punta con no
    # respuesta, la otra conocida, y el par todavía no válido para el criterio actual.
    pobre_t = pobre_segun_cota(pares[clase_t], pares["pobre_t"])
    pobre_th = pobre_segun_cota(pares[clase_th], pares["pobre_th"])
    nr_t, nr_th = (pares[clase_t] != "declarado"), (pares[clase_th] != "declarado")
    candidato = (pares["identidad_confirmada"].astype("boolean").fillna(False).astype(bool)
                 & ~pares[valido_actual].astype(bool) & (nr_t | nr_th) & pobre_t.notna() & pobre_th.notna())
    via = pd.Series(np.select([nr_t & nr_th, nr_t, nr_th], ["ambas", "t", "th"], default=""), index=pares.index)
    return pd.DataFrame({
        "pobre_t_cota": pobre_t, "pobre_th_cota": pobre_th,
        "recupera_entrada": candidato & (pobre_t == False).fillna(False).astype(bool),  # noqa: E712
        "recupera_permanencia": candidato & (pobre_t == True).fillna(False).astype(bool),  # noqa: E712
        "via": via.where(candidato, ""),
    }, index=pares.index)


COLUMNAS_02B = ["CODUSU", "NRO_HOGAR", "AGLOMERADO", "ITF", "linea_pobreza", "ratio_linea_pobreza", "pobre"]


def _particion(raiz: Path, anio: int, trimestre: int) -> Path:
    return Path(raiz) / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet"


def _leer(ruta: Path, columnas) -> pd.DataFrame:
    disponibles = pq.read_schema(ruta).names
    return pd.read_parquet(ruta, columns=[c for c in dict.fromkeys(columnas) if c in disponibles])


def _con_clase(hogares: pd.DataFrame, anio: int, trimestre: int) -> pd.DataFrame:
    ratio = hogares["cota_ingreso"].astype(float) / hogares["linea_pobreza"].astype(float)
    return hogares.assign(ANO4=anio, TRIMESTRE=trimestre, ratio_cota=ratio,
                          clase=clasificar_cota(ratio, hogares["con_no_respuesta"]))


def clases_cota_regular(anio: int, trimestre: int, ruta_armonizado: Path, ruta_distancia: Path) -> tuple[pd.DataFrame, dict]:
    """Hogares de la era regular con `cota_ingreso`, `ratio_cota` y `clase`
    (no respuesta = `ingreso_no_declarado`), más chequeos de `P47T` del trimestre."""
    hogares = _leer(_particion(ruta_distancia, anio, trimestre), [*COLUMNAS_02B, "ingreso_no_declarado"])
    ruta_ind = _particion(Path(ruta_armonizado) / "individual", anio, trimestre)
    columnas_ind = [*CLAVE_HOGAR, *VARIABLES_CON_MENOS9, "P47T", *COMPONENTES_OCUPACION_PRINCIPAL,
                    *MONTOS_CUENTA_PROPIA, *columnas_no_laborales(pq.read_schema(ruta_ind).names)]
    individual = _leer(ruta_ind, columnas_ind)
    personas = ingreso_declarado_persona(individual)
    cotas = cota_ingreso_hogar(individual)
    hogares = hogares.merge(cotas, on=CLAVE_HOGAR, how="left", validate="one_to_one")
    hogares["con_no_respuesta"] = hogares["ingreso_no_declarado"].astype("boolean").fillna(False).astype(bool)
    hogares = _con_clase(hogares, anio, trimestre)
    declarante = ~hogares["con_no_respuesta"]
    nr_personas = individual[CLAVE_HOGAR].merge(hogares[[*CLAVE_HOGAR, "con_no_respuesta"]], on=CLAVE_HOGAR,
                                                how="left")["con_no_respuesta"].fillna(False).to_numpy(dtype=bool)
    chequeos = {
        "ANO4": anio, "TRIMESTRE": trimestre, "n_hogares": len(hogares), "n_no_respuesta": int((~declarante).sum()),
        "n_personas": len(individual), "n_personas_menos9": int(personas["algun_menos9"].sum()),
        "n_personas_menos9_en_declarantes": int((personas["algun_menos9"].to_numpy() & ~nr_personas).sum()),
        "n_p47t_menos9": int(personas["p47t_menos9"].sum()),
        "n_p47t_parcial": int(personas["p47t_positivo_con_menos9"].sum()),
        "n_p47t_menos9_sin_componente_menos9": int((personas["p47t_menos9"] & ~personas["algun_menos9"]).sum()),
        "n_declarantes_cota_distinta_itf": int((declarante & ~np.isclose(
            hogares["cota_ingreso"].astype(float), hogares["ITF"].astype(float), atol=2)).sum()),
        "n_no_respuesta_sin_menos9": int((~declarante & hogares["n_personas_menos9"].eq(0)).sum()),
    }
    return hogares, chequeos


def clases_cota_historica(anio: int, trimestre: int, ruta_armonizado: Path, ruta_distancia: Path) -> tuple[pd.DataFrame, dict]:
    """Hogares de la era histórica con cota = ITF no imputado (`cota_historica`),
    `ratio_cota` y `clase` (no respuesta = `IDIMPH` con alguna imputación)."""
    import imputacion_ingresos as ii

    hogar = ii.leer_marcas_imputacion(anio, trimestre, Path(ruta_armonizado), "hogar")
    individual = ii.leer_marcas_imputacion(anio, trimestre, Path(ruta_armonizado), "individual")
    fraccion = ii.fraccion_itf_imputada(hogar, individual)
    hogares = _leer(_particion(ruta_distancia, anio, trimestre), COLUMNAS_02B).merge(
        fraccion[[*CLAVE_HOGAR, "monto_imputado_max", "fraccion_max", "alguna_imputacion"]],
        on=CLAVE_HOGAR, how="left", validate="one_to_one")
    hogares["con_no_respuesta"] = hogares["alguna_imputacion"].astype("boolean").fillna(False).astype(bool)
    hogares["cota_ingreso"] = cota_historica(hogares["ITF"], hogares["monto_imputado_max"])
    hogares = _con_clase(hogares, anio, trimestre)
    seguro = hogares["clase"] == "seguro_no_pobre"
    chequeos = {
        "ANO4": anio, "TRIMESTRE": trimestre, "n_hogares": len(hogares),
        "n_no_respuesta": int(hogares["con_no_respuesta"].sum()),
        "n_fraccion_max_na": int((hogares["con_no_respuesta"] & hogares["fraccion_max"].isna()).sum()),
        "n_fraccion_max_1": int((hogares["fraccion_max"].astype(float) >= 1 - 1e-9).sum()),
        "n_monto_max_mayor_itf": int((hogares["monto_imputado_max"].astype(float) > hogares["ITF"].astype(float) + 1).sum()),
        "n_seguro_no_pobre_pobre_indec": int((seguro & hogares["pobre"].astype("boolean").fillna(False)).sum()),
    }
    return hogares, chequeos
