"""Fase 1 de la retropolación 2003-2015 (D25, sección 5): backcast de
CBA/CBT vía un índice de precios con ancla abril-2016 (D30), y los dos
mecanismos de pertinencia por tramo (CEDLAS, UCA).
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

import canasta_regional

ANCLA = (2016, 4)

# D37: INDEC, "Valorización mensual de la Canasta Básica de Alimentos (CBA)
# y de la Canasta Básica Total (CBT) para el adulto equivalente en el
# aglomerado Gran Buenos Aires, desde septiembre de 2000 en adelante"
# (sitioanterior.indec.gob.ar) -- serie real en pesos, no un backcast.
URL_CBA_CBT_GBA_HISTORICO = "https://sitioanterior.indec.gob.ar/ftp/nuevaweb/cuadros/74/sh-cba2.xls"
# Serie histórica de tasas de pobreza/indigencia por región, citada en
# `pob_tot_2sem12.pdf` (ya en el repo) -- permite validar por región, no
# solo nacional.
URL_POBREZA_REGIONAL_HISTORICO = "https://sitioanterior.indec.gob.ar/ftp/nuevaweb/cuadros/74/sh_pobrezaeindigencia_continua.xls"

_MESES_ABREVIADOS = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
                      "Jul": 7, "Ago": 8, "Sep": 9, "Set": 9, "Oct": 10, "Nov": 11, "Dic": 12}
NOMBRES_REGION_EPH = {1: "Gran Buenos Aires", 43: "Pampeana", 40: "Noroeste",
                       41: "Noreste", 42: "Cuyo", 44: "Patagonia"}


def descargar_xls_indec(url: str, destino_archivo: Path) -> bytes:
    """Descarga un `.xls` de `sitioanterior.indec.gob.ar` -- devuelve 403 sin
    `User-Agent` (D25: filtro anti-bot, no una restricción real de acceso)."""
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(pedido, timeout=30) as respuesta:
        contenido = respuesta.read()
    destino_archivo.parent.mkdir(parents=True, exist_ok=True)
    destino_archivo.write_bytes(contenido)
    return contenido


def cargar_cba_cbt_gba_historico(ruta_xls: Path) -> pd.DataFrame:
    """Serie mensual real de CBA/CBT de GBA publicada por INDEC (D37),
    indexada por `(ANO4, MES)`, columnas `cba_ae, cbt_ae`."""
    # Normaliza nombres de mes ("Set"/"Sep") y separador decimal
    # (coma/punto, mezclados en el archivo real).
    df = pd.read_excel(ruta_xls, sheet_name=0, header=None)
    patron_mes = re.compile(r"^[A-Za-z]{3}-\d{4}$")
    filas = []
    for i in range(len(df)):
        mes_str = df.iloc[i, 0]
        if not isinstance(mes_str, str):
            continue
        mes_str = mes_str.strip()  # el archivo real trae espacios al final en algunas filas (ej. "Sep-2000 ")
        if not patron_mes.match(mes_str):
            continue
        abreviado, anio_str = mes_str.split("-")
        if abreviado not in _MESES_ABREVIADOS:
            continue
        cba = float(str(df.iloc[i, 1]).replace(",", "."))
        cbt = float(str(df.iloc[i, 3]).replace(",", "."))
        filas.append({"ANO4": int(anio_str), "MES": _MESES_ABREVIADOS[abreviado], "cba_ae": cba, "cbt_ae": cbt})
    return pd.DataFrame(filas).set_index(["ANO4", "MES"])


def cargar_ppcc_regional(ruta_csv: Path) -> pd.DataFrame:
    """Valores de CBA/CBT por región a abril de 2001 (D37, `pob_tmay01.pdf`,
    INDEC 19/2/2002) -- columnas `REGION, cba_abr2001, cbt_abr2001`."""
    return pd.read_csv(ruta_csv)


def construir_canasta_regional_ppcc(cba_cbt_gba: pd.DataFrame, ppcc_regional: pd.DataFrame,
                                     inicio: tuple[int, int], fin: tuple[int, int]) -> pd.DataFrame:
    """CBA/CBT mensual por región en `[inicio, fin]`, multiplicando el
    valor real de GBA por el ratio región/GBA de abril de 2001 (PPCC, D37),
    asumido fijo en el tiempo."""
    gba = ppcc_regional[ppcc_regional["REGION"] == 1].iloc[0]
    filas = []
    for anio, mes in _meses_entre(inicio, fin):
        gba_cba, gba_cbt = cba_cbt_gba.loc[(anio, mes), ["cba_ae", "cbt_ae"]]
        for _, fila in ppcc_regional.iterrows():
            ratio_cba = fila["cba_abr2001"] / gba["cba_abr2001"]
            ratio_cbt = fila["cbt_abr2001"] / gba["cbt_abr2001"]
            filas.append({"REGION": fila["REGION"], "ANO4": anio, "MES": mes,
                           "cba_ae": gba_cba * ratio_cba, "cbt_ae": gba_cbt * ratio_cbt})
    return pd.DataFrame(filas)


def cargar_tasas_regionales_historico(ruta_xls: Path) -> pd.DataFrame:
    """Tasas oficiales de pobreza/indigencia (hogares y personas) por
    región EPH y semestre desde 1S2003 (D37). Formato largo: `REGION,
    semestre_label, indicador, valor`."""
    df = pd.read_excel(ruta_xls, sheet_name=0, header=None)
    row2, row3, row4 = df.iloc[2], df.iloc[3], df.iloc[4]
    starts = [c for c in range(3, df.shape[1]) if pd.notna(row2[c])] + [df.shape[1]]

    def _parsear_fila(data_row: pd.Series) -> dict:
        resultado = {}
        for i in range(len(starts) - 1):
            c0, c1 = starts[i], starts[i + 1]
            etiqueta_sem = row2[c0]
            indicador_actual = None
            for c in range(c0, c1):
                if pd.notna(row3[c]):
                    indicador_actual = "indigencia" if "indigencia" in row3[c] else "pobreza"
                if row4[c] in ("Hogares", "Personas") and indicador_actual:
                    unidad = "hogares" if row4[c] == "Hogares" else "personas"
                    resultado[(etiqueta_sem, f"{indicador_actual}_{unidad}")] = data_row[c]
        return resultado

    nombre_a_region = {v: k for k, v in NOMBRES_REGION_EPH.items()}
    filas = []
    for idx in range(len(df)):
        nombre = df.iloc[idx, 1]
        if nombre not in nombre_a_region:
            continue
        for (etiqueta_sem, indicador), valor in _parsear_fila(df.iloc[idx]).items():
            filas.append({"REGION": nombre_a_region[nombre], "semestre_label": etiqueta_sem,
                           "indicador": indicador, "valor": valor})
    return pd.DataFrame(filas)


def cargar_indice_precios(ruta_csv: Path, columna_valor: str) -> pd.Series:
    """Serie mensual de un índice de precios genérico (CSV
    `fecha,<columna_valor>`), indexada por `(ANO4, MES)` (D30)."""
    df = pd.read_csv(ruta_csv, parse_dates=["fecha"]).dropna(subset=[columna_valor])
    df["ANO4"], df["MES"] = df["fecha"].dt.year, df["fecha"].dt.month
    return df.set_index(["ANO4", "MES"])[columna_valor]


def cargar_facpce(ruta_csv: Path) -> pd.Series:
    """Índice FACPCE (IPC Nacional empalme IPIM 2003-2016, D29) -- wrapper de
    `cargar_indice_precios` con la columna `"ipc_facpce"` de
    `data/indice-FACPCE.csv`."""
    return cargar_indice_precios(ruta_csv, "ipc_facpce")


def _meses_entre(inicio: tuple[int, int], fin: tuple[int, int]):
    anio, mes = inicio
    while (anio, mes) <= fin:
        yield anio, mes
        mes += 1
        if mes > 12:
            mes, anio = 1, anio + 1


def backcast_canasta_larga(canastas_dir: Path, indice: pd.Series,
                            inicio: tuple[int, int], fin: tuple[int, int]) -> pd.DataFrame:
    """CBA/CBT mensual por región en `[inicio, fin]`, deflactando la
    canasta de `ANCLA` por la razón de `indice` (brecha interregional
    asumida constante, D25)."""
    indice_ancla = indice.loc[ANCLA]

    cba_ancla = canasta_regional.cargar_canasta_larga(canastas_dir / "cba_regional_crudo.csv", "cba_ae")
    cba_ancla = cba_ancla[(cba_ancla["ANO4"] == ANCLA[0]) & (cba_ancla["MES"] == ANCLA[1])].set_index("REGION")["cba_ae"]
    cbt_ancla = canasta_regional.cargar_canasta_larga(canastas_dir / "cbt_regional_crudo.csv", "cbt_ae")
    cbt_ancla = cbt_ancla[(cbt_ancla["ANO4"] == ANCLA[0]) & (cbt_ancla["MES"] == ANCLA[1])].set_index("REGION")["cbt_ae"]

    filas = []
    for anio, mes in _meses_entre(inicio, fin):
        ratio = indice.loc[(anio, mes)] / indice_ancla
        for region in cba_ancla.index:
            filas.append({"REGION": region, "ANO4": anio, "MES": mes,
                           "cba_ae": cba_ancla[region] * ratio, "cbt_ae": cbt_ancla[region] * ratio})
    return pd.DataFrame(filas)


def construir_canasta_backcast_trimestral(canastas_dir: Path, indice: pd.Series, metodo_ventana: str,
                                           inicio: tuple[int, int], fin: tuple[int, int]) -> pd.DataFrame:
    """Backcast trimestral por región, mismo shape que
    `canasta_regional.construir_canasta_trimestral`. `indice` es cualquier
    serie de precios ya cargada (D30)."""
    largo = backcast_canasta_larga(canastas_dir, indice, inicio, fin)
    cba = canasta_regional.agregar_trimestral(largo[["REGION", "ANO4", "MES", "cba_ae"]], "cba_ae", metodo_ventana)
    cbt = canasta_regional.agregar_trimestral(largo[["REGION", "ANO4", "MES", "cbt_ae"]], "cbt_ae", metodo_ventana)
    combinado = cba.merge(cbt, on=["ANO4", "TRIMESTRE", "REGION"], suffixes=("", "_cbt"))
    assert (combinado["canasta_ventana_parcial"] == combinado["canasta_ventana_parcial_cbt"]).all()
    return combinado.drop(columns="canasta_ventana_parcial_cbt")


def peso_hogar_historico(individual: pd.DataFrame) -> pd.Series:
    """Ponderador de hogar para la era histórica (sin `PONDIH`): `PONDERA`
    del hogar, indexado por `(CODUSU, NRO_HOGAR)`."""
    # PONDERA es el factor de expansión del hogar (confirmado constante entre
    # sus integrantes), no de la persona; para ponderar personas se multiplica
    # por la cantidad de integrantes (mismo patrón que PONDIH x n_miembros
    # en la era regular, D17), no se suma PONDERA directamente.
    por_hogar = individual.groupby(["CODUSU", "NRO_HOGAR"])["PONDERA"]
    if not (por_hogar.nunique() == 1).all():
        raise ValueError("PONDERA no es constante dentro de algún hogar -- revisar antes de ponderar")
    return por_hogar.first()


def clasificar_dentro_de_banda(valor_propio: float, valores_banda: list[float]) -> bool:
    """`True` si `valor_propio` cae dentro de `[min(valores_banda), max(valores_banda)]`."""
    return min(valores_banda) <= valor_propio <= max(valores_banda)


def clasificar_tramo_cedlas(fuera_por_semestre: list[bool]) -> str:
    """Pertinencia de un tramo por el mecanismo CEDLAS (D25, 5.c): 0 semestres
    fuera de la banda -> `"pertinente"`; 1 -> `"pertinente_con_reservas"`; 2+ ->
    `"no_pertinente"`."""
    n_fuera = sum(fuera_por_semestre)
    if n_fuera == 0:
        return "pertinente"
    if n_fuera == 1:
        return "pertinente_con_reservas"
    return "no_pertinente"


def clasificar_tramo_uca(z_por_anio: dict[int, float]) -> str:
    """Pertinencia de un tramo por el mecanismo UCA (D25, 5.c): `|z|<=1`
    en todos los años -> `"pertinente"`; 2+ años con `|z|>2` ->
    `"no_pertinente"`; resto -> `"pertinente_con_reservas"`."""
    valores = list(z_por_anio.values())
    n_fuera_2 = sum(abs(z) > 2 for z in valores)
    n_entre_1_2 = sum(1 < abs(z) <= 2 for z in valores)
    if n_fuera_2 >= 2:
        return "no_pertinente"
    if n_fuera_2 == 1 or n_entre_1_2 >= 1:
        return "pertinente_con_reservas"
    return "pertinente"


def brecha_puente_uca(serie_propia_anual: dict[int, float], serie_uca_anual: dict[int, float],
                       anios_puente: list[int], excluir_2020: bool = False) -> dict:
    """Media y desvío estándar de la brecha (propia - UCA) en
    `anios_puente` (D25, 5.c). `excluir_2020=True` saca 2020 del cálculo,
    no de la serie de brechas devuelta."""
    anios_calculo = [a for a in anios_puente if not (excluir_2020 and a == 2020)]
    brechas = {a: serie_propia_anual[a] - serie_uca_anual[a] for a in anios_puente}
    valores_calculo = pd.Series({a: brechas[a] for a in anios_calculo})
    return {"brechas": brechas, "media": valores_calculo.mean(), "std": valores_calculo.std(ddof=1)}
