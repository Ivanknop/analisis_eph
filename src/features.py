"""Matriz de predictores por par `t`->`t+h`: catálogo de features, armado por
trimestre de origen con información de `t` o anterior, lista blanca de X,
patrones prohibidos (leakage) y reporte de faltantes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import categorias_hogar as ch
import cortes_trayectoria as ct
from contexto_aglomerado import componente_comun_rezagado
from eventos_hogar import calcular_alta_activo_menor, calcular_se_jubila, calcular_tipologia_perceptores

CLAVE_PAR = ["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "ANO4_t", "TRIMESTRE_t"]
COLUMNAS_PASO = CLAVE_PAR + [
    "ANO4_th", "TRIMESTRE_th", "AGLOMERADO_t", "REGION_t", "h", "identidad_confirmada", "par_valido",
    "par_valido_simetrica", "par_valido_log", "par_valido_c", "par_valido_log_c", *ct.FLAGS_CORTE,
    "peso_hogar_era_t", "peso_normalizado_trimestre",
]

AGLOMERADOS_INCORPORADOS_2006T3 = (38, 91, 93)
# Imputación de INDEC (histórica) y no respuesta sin imputar (regular) son el
# mismo fenómeno con distinto tratamiento: separados funcionarían como indicador
# de era. En X van juntos; el detalle queda en `motivo_previa_na_detalle_t`.
MOTIVOS_PREVIA = ["primera_visita", "sin_identidad", "ingreso_no_declarado", "itf_cero_ambas_puntas", "disponible"]
MOTIVOS_PREVIA_DETALLE = {"imputada_indec": "ingreso_no_declarado", "no_respuesta_ingreso": "ingreso_no_declarado"}
TRIMESTRE_TELEFONICO = ct.TRIMESTRE_TELEFONICO

# Nada de `t+h` puede entrar en X: el resultado y sus derivados, los filtros y
# marcas que miran el destino, los eventos contemporáneos del par y el contexto
# político. `variacion_(comun|propia)` solo se admite en su versión rezagada.
PATRONES_PROHIBIDOS_EN_X = [
    r"_th$", r"_th_", r"^variacion_(simetrica|log)$", r"^variacion_(comun|propia)_(?!.*_rezagada)",
    r"^(aporte_|direccion$|transicion$|diferencia_absoluta|par_valido|identidad_confirmada|cambio_composicion$"
    r"|n_altas$|n_bajas$|mandato_|regimen_ingreso|mismo_sexo_jefe|cambio_jefatura|diferencia_edad_jefe)",
    r"^par_", r"^previa_(cruza|no_observable)", r"^peso_", r"^imputado", r"^fraccion_",
    # Vale 0 en todo train y 1 en todo el holdout: es un sello de fecha, no un predictor.
    r"^cuestionario_2023t4",
]


@dataclass(frozen=True)
class Feature:
    nombre: str
    bloque: str
    fuente: str
    definicion: str
    rol: str = "principal"
    rezago_min: int = 0
    rezago_max: int = 0
    aplica: str | None = None
    eras: str = "ambas"

    @property
    def en_x(self) -> bool:
        return self.rol in ("principal", "componente", "calendario")


def _f(nombre, bloque, fuente, definicion, **kw) -> Feature:
    return Feature(nombre, bloque, fuente, definicion, **kw)


# `rol`: principal/componente/calendario entran en X por default; alternativa,
# marca y auxiliar se calculan pero quedan fuera (decisiones del 2026-11-10:
# V2 de vivienda y la tasa de altas/bajas como alternativas; la previa sin
# filtrar no está limpia bajo el criterio C).
CATALOGO_FEATURES: list[Feature] = [
    _f("ratio_linea_pobreza_t", "posicion", "05", "ITF / (unidades ae * CBT por ae)"),
    _f("log_ratio_linea_pobreza_t", "posicion", "05", "log del ratio; NA con ITF=0", aplica="itf_positivo"),
    _f("ratio_linea_indigencia_t", "posicion", "02b", "ITF / (unidades ae * CBA por ae)"),
    _f("pobre_t", "posicion", "02b", "ratio_linea_pobreza < 1"),
    _f("indigente_t", "posicion", "02b", "ratio_linea_indigencia < 1"),
    _f("decil_ratio_t", "posicion", "02b", "decil ponderado (peso_hogar_era) del ratio dentro del trimestre"),
    _f("deccfr_t", "posicion", "02_nucleo", "decil INDEC de IPCF, 1-10 (0 y 12 en NA)"),
    _f("n_miembros_t", "composicion", "02_nucleo individual", "miembros del hogar"),
    _f("n_menores_14_t", "composicion", "02_nucleo individual", "CH06 < 14 (incluye -1, menor de 1 año)"),
    _f("n_menores_18_t", "composicion", "02_nucleo individual", "CH06 < 18"),
    _f("n_mayores_65_t", "composicion", "02_nucleo individual", "CH06 >= 65"),
    _f("unidades_adulto_equivalente_t", "composicion", "05", "unidades de adulto equivalente"),
    _f("share_dependientes_t", "composicion", "02_nucleo individual", "(menores de 15 + mayores de 65) / miembros"),
    _f("sexo_jefe_t", "composicion", "02_nucleo individual", "CH04 del jefe; NA si jefe_ambiguo"),
    _f("edad_jefe_t", "composicion", "02_nucleo individual", "CH06 del jefe; NA si jefe_ambiguo"),
    _f("nivel_ed_jefe_t", "composicion", "01_armonizado", "NIVEL_ED del jefe (1-7); NA si jefe_ambiguo"),
    _f("nivel_ed_jefe_bucket_t", "composicion", "01_armonizado", "bucket_nivel_ed del jefe"),
    _f("jefe_ambiguo_t", "composicion", "02_nucleo individual", "0 o más de 1 persona con CH03==1", rol="marca"),
    _f("hay_menor_no_asiste_t", "composicion", "01_armonizado", "algún miembro de 6-17 no asiste y no completó secundario"),
    _f("situacion_laboral_t", "laboral", "02_nucleo individual", "calcular_situacion_laboral (4 niveles)"),
    _f("condicion_actividad_jefe_t", "laboral", "02_nucleo individual", "ocupado/desocupado/inactivo del jefe"),
    _f("n_ocupados_t", "laboral", "02_nucleo individual", "ESTADO == 1"),
    _f("n_desocupados_t", "laboral", "02_nucleo individual", "ESTADO == 2"),
    _f("n_activos_t", "laboral", "02_nucleo individual", "ESTADO en {1, 2}"),
    _f("sin_activos_t", "laboral", "02_nucleo individual", "n_activos == 0"),
    _f("share_ocupados_t", "laboral", "02_nucleo individual", "ocupados / miembros"),
    _f("share_precario_A_t", "laboral", "02_nucleo individual", "(asalariados informales + desocupados) / activos clasificables", aplica="con_activos"),
    _f("share_precario_B_t", "laboral", "02_nucleo individual", "A + cuenta propia", aplica="con_activos"),
    _f("share_precario_C_t", "laboral", "02_nucleo individual", "B + patrón", aplica="con_activos"),
    _f("share_precario_D_t", "laboral", "01_armonizado PP04D_COD", "A + cuenta propia de calificación operativa o no calificada", aplica="con_activos_d"),
    _f("n_activos_clasificables_t", "laboral", "02_nucleo individual", "denominador de A-C", rol="auxiliar"),
    _f("tipologia_perceptores_t", "laboral", "02_nucleo", "calcular_tipologia_perceptores (V2 de hogar)"),
    _f("share_laboral_t", "ingresos", "01_armonizado", "(P21+TOT_P12+PP08J1-3) / ITF", aplica="itf_positivo"),
    _f("share_jubilacion_t", "ingresos", "01_armonizado", "V2_M+V21_M (≤2023T3) o suma de V2/V21/V22_0x_M (≥2023T4), / ITF", rol="alternativa", aplica="itf_positivo"),
    _f("share_transferencias_t", "ingresos", "01_armonizado", "V5_M+V11_M o suma de V5/V11_0x_M, / ITF", aplica="itf_positivo"),
    _f("share_otros_no_laborales_t", "ingresos", "01_armonizado", "V3,V4,V8,V9,V10,V12,V18,V19_AM / ITF", aplica="itf_positivo"),
    _f("share_residuo_no_laboral_t", "ingresos", "01_armonizado", "(T_VI - no laborales listados) / ITF", rol="alternativa", aplica="itf_positivo"),
    # Hasta 2023T3 los bonos previsionales quedan en el residuo y desde 2023T4
    # entran en jubilación: la suma es la serie continua a través del corte.
    _f("share_jubilacion_continua_t", "ingresos", "01_armonizado", "share_jubilacion + share_residuo_no_laboral", aplica="itf_positivo"),
    _f("vivienda_precaria_v1_t", "vivienda", "02_nucleo", "ancla_con_deficit: hacinamiento NBI, materiales o sanitarias"),
    _f("vivienda_precaria_v2_t", "vivienda", "02_nucleo", "V1 + techo, agua fuera del terreno, hoyo, villa", rol="alternativa"),
    *[_f(f"{c}_t", "vivienda", "02_nucleo", d, rol="componente") for c, d in [
        ("hacinamiento_nbi", "más de 3 personas por cuarto"), ("calidad_materiales_deficitaria", "IV1 en {3,4,5} o IV3 == 3"),
        ("condiciones_sanitarias_deficitarias", "IV8 == 2 o IV10 == 3"), ("techo_precario", "IV4 en {6, 7}"),
        ("agua_fuera_terreno", "IV6 == 3"), ("desague_a_hoyo", "IV11 == 4"), ("villa", "IV12_3 == 1"),
        ("entorno_basural_o_inundable", "IV12_1 == 1 o IV12_2 == 1"), ("tenencia_precaria", "II7 en {6, 7}"),
        ("inquilino", "II7 == 3"), ("cocina_lena_kerosene", "II8 == 3")]],
    # V13-V17 cambian con el cuestionario 2023T4 (`decisiones.md`); quedan fuera
    # de X hasta verificar la continuidad.
    _f("recurre_desahorro_t", "estrategias", "02_nucleo", "V13, V14 o V17 == 1", rol="alternativa"),
    _f("recurre_endeudamiento_formal_t", "estrategias", "02_nucleo", "V15 o V16 == 1", rol="alternativa"),
    _f("gap_previo", "previa", "05", "0 si la visita previa es t-1, 3 si es t-3; NA en 1ª visita", rezago_min=-4, aplica="tiene_previa"),
    _f("variacion_simetrica_previa", "previa", "05", "variación simétrica del par previo (t-1->t o t-4->t-3), sin filtrar", rol="alternativa", rezago_min=-4, aplica="tiene_previa"),
    _f("motivo_previa_na_t", "previa", "05", "por qué falta la previa limpia: " + ", ".join(MOTIVOS_PREVIA), rezago_min=-4),
    _f("motivo_previa_na_detalle_t", "previa", "05", "motivo con ingreso_no_declarado abierto en imputada_indec (histórica) / no_respuesta_ingreso (regular)", rol="alternativa", rezago_min=-4),
    _f("variacion_simetrica_previa_c", "previa", "05", "la previa, NA si el par previo no es par_valido_c", rezago_min=-4, aplica="tiene_previa"),
    _f("ratio_linea_pobreza_previo", "previa", "05", "ratio en la visita previa (t-1 o t-3); NA sin identidad", rezago_min=-3, aplica="tiene_previa"),
    _f("variacion_simetrica_hasta_t", "previa", "05", "variación simétrica del par que termina en t, sin filtrar", rol="alternativa", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("variacion_log_hasta_t", "previa", "05", "variación log del par que termina en t, sin filtrar", rol="alternativa", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("variacion_simetrica_hasta_t_c", "previa", "05", "la anterior, NA si el par no es par_valido_c", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("variacion_log_hasta_t_c", "previa", "05", "variación log, NA si el par no es par_valido_log_c", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("cambio_ae_previo_t", "previa", "05", "|Δ unidades ae| / ae del origen, par que termina en t; NA sin identidad", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("tasa_altas_bajas_previa_t", "previa", "03_panel", "(n_altas + n_bajas) / miembros del origen, par que termina en t", rol="alternativa", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("n_se_jubila_previo_t", "previa", "03_panel + 01_armonizado", "calcular_se_jubila sobre el par que termina en t", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("n_alta_activo_menor_previo_t", "previa", "03_panel + 01_armonizado", "calcular_alta_activo_menor sobre el par que termina en t", rezago_min=-4, aplica="tiene_par_hasta_t"),
    _f("tasa_pobreza_aglo_t", "contexto", "02b", "año móvil t-3..t, variante A, peso_hogar_era, mínimo 400 hogares", rezago_min=-3, aplica="con_pares_terminan_en_t"),
    _f("tasa_indigencia_aglo_t", "contexto", "02b", "ídem, indigencia", rezago_min=-3, aplica="con_pares_terminan_en_t"),
    _f("tasa_informalidad_aglo_t", "contexto", "02_nucleo individual", "asalariados con PP07H==2 / asalariados, PONDERA, año móvil", rezago_min=-3, aplica="con_pares_terminan_en_t"),
    _f("tasa_desocupacion_aglo_t", "contexto", "02_nucleo individual", "ESTADO==2 / ESTADO en {1,2}, PONDERA, año móvil", rezago_min=-3, aplica="con_pares_terminan_en_t"),
    _f("gini_ipcf_aglo_t", "contexto", "02_nucleo", "Gini del IPCF relativo, peso de personas, año móvil, mínimo 400 hogares", rezago_min=-3, aplica="con_pares_terminan_en_t"),
    _f("n_hogares_ventana_aglo_t", "contexto", "02b", "hogares con ratio en la ventana", rol="auxiliar", rezago_min=-3),
    _f("n_pares_terminan_en_t", "contexto", "05", "pares del mismo h que terminan en t (0 al comienzo de cada era o tras la ruptura de CODUSU)", rol="auxiliar", rezago_min=-4),
    *[_f(f"variacion_comun_{ambito}_{tipo}_rezagada_t", "contexto", "05",
         f"mediana ponderada de variacion_{tipo} de los pares h que terminan en t ({ambito}), universo C", rezago_min=-4,
         aplica="con_pares_terminan_en_t")
      for ambito in ("trimestre", "aglomerado") for tipo in ("simetrica", "log")],
    _f("era", "calendario", "05", "historica / regional_2016", rol="calendario"),
    _f("encuesta_telefonica_t", "calendario", "calendario", "t == 2020T2", rol="calendario"),
    _f("periodo_intervenido_t", "calendario", "calendario", "2007 <= año de t <= 2015", rol="calendario"),
    _f("cuestionario_2023t4_t", "calendario", "calendario", "t >= 2023T4 (fuentes V* abiertas en componentes); sello de fecha, nunca en X", rol="marca"),
    _f("aglomerado_incorporado_2006t3", "calendario", "calendario", "AGLOMERADO en {38, 91, 93}", rol="calendario"),
]


def catalogo_como_tabla(catalogo: list[Feature] = CATALOGO_FEATURES) -> pd.DataFrame:
    """Catálogo como DataFrame (una fila por feature, con `en_x`)."""
    return pd.DataFrame([{**f.__dict__, "en_x": f.en_x} for f in catalogo])


def nombres_prohibidos(columnas) -> list[str]:
    """Columnas que matchean algún patrón de `PATRONES_PROHIBIDOS_EN_X`."""
    return [c for c in columnas if any(re.search(p, c) for p in PATRONES_PROHIBIDOS_EN_X)]


def construir_matriz_X(features: pd.DataFrame, catalogo: list[Feature] = CATALOGO_FEATURES,
                       incluir_no_x: bool = False) -> pd.DataFrame:
    """Columnas de `features` en la lista blanca (`en_x`, o todo el catálogo
    con `incluir_no_x`). Error si alguna matchea un patrón prohibido."""
    columnas = [f.nombre for f in catalogo if (f.en_x or incluir_no_x) and f.nombre in features]
    prohibidas = nombres_prohibidos(columnas)
    if prohibidas:
        raise ValueError(f"columnas prohibidas en X: {prohibidas}")
    return features[columnas]


def armonizar_pares(pares: pd.DataFrame, h: int, trimestres_itf0) -> pd.DataFrame:
    """Completa en los pares de 05 lo que falta en la era regular (`era`,
    `par_valido_c = par_valido`, marcas de corte), ausente o en NULL, y
    agrega `h` y `peso_hogar_era_t`. No descarta filas."""
    resultado = pares.copy()
    resultado["h"] = h
    # Leída entera con `union_by_name`, la 05 trae estas columnas presentes pero en NULL
    # en la era regular: se completa por fila regular nula, no solo por columna ausente.
    regular = (resultado["era_t"] == "regional_2016").to_numpy()
    if "era" not in resultado:
        resultado["era"] = resultado["era_t"]
    elif resultado["era"].isna().any():
        resultado["era"] = resultado["era"].fillna(resultado["era_t"])
    for columna_c, columna in (("par_valido_c", "par_valido"), ("par_valido_log_c", "par_valido_log")):
        if columna_c not in resultado:
            resultado[columna_c] = resultado[columna]
            continue
        nulo_regular = regular & resultado[columna_c].isna().to_numpy()
        if nulo_regular.any():
            resultado[columna_c] = resultado[columna_c].astype("boolean")
            resultado.loc[nulo_regular, columna_c] = resultado.loc[nulo_regular, columna].astype("boolean")
    if any(flag not in resultado for flag in ct.FLAGS_CORTE):
        resultado = ct.marcar_cortes_par(resultado, trimestres_itf0)
    elif any(resultado.loc[regular, flag].isna().any() for flag in ct.FLAGS_CORTE):
        marcados = ct.marcar_cortes_par(resultado, trimestres_itf0)
        for flag in ct.FLAGS_CORTE:
            resultado[flag] = resultado[flag].astype("boolean").fillna(pd.Series(marcados[flag], dtype="boolean"))
    resultado["peso_hogar_era_t"] = ct.peso_hogar_era(resultado, "t")
    return resultado


def decil_ponderado(valores: pd.Series, pesos: pd.Series) -> pd.Series:
    """Decil 1-10 por peso acumulado (ordenando por valor). `NA` sin valor o
    sin peso positivo."""
    valido = valores.notna() & pesos.notna() & (pesos.astype(float) > 0)
    resultado = pd.Series(pd.NA, index=valores.index, dtype="Int8")
    if not valido.any():
        return resultado
    orden = valores[valido].astype(float).sort_values(kind="mergesort").index
    acumulado = pesos.loc[orden].astype(float).cumsum()
    # Punto medio del peso de cada hogar: el decil no depende del orden entre empates.
    centro = (acumulado - pesos.loc[orden].astype(float) / 2) / acumulado.iloc[-1]
    resultado.loc[orden] = np.clip(np.floor(centro * 10).astype(int) + 1, 1, 10).to_numpy()
    return resultado


@dataclass
class InsumosTrimestre:
    """Todo lo que usa `construir_features_trimestre` para el origen `t`:
    nada de `t+h` salvo la clave del par actual."""
    anio: int
    trimestre: int
    h: int
    pares: pd.DataFrame
    distancia: pd.DataFrame
    hogar: pd.DataFrame
    individual: pd.DataFrame
    pares_previos: pd.DataFrame
    panel_previos: pd.DataFrame
    individual_previo: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)
    contexto_aglomerado: pd.DataFrame | None = None


def _por_hogar_t(tabla: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    renombrar = {c: f"{c}_t" for c in columnas}
    return tabla[["CODUSU", "NRO_HOGAR", *columnas]].rename(columns={"NRO_HOGAR": "NRO_HOGAR_t", **renombrar})


def _foto_en_t(ins: InsumosTrimestre) -> list[pd.DataFrame]:
    hogar = ins.hogar.set_index(ch.CLAVE_HOGAR)
    individual = ins.individual
    composicion = ch.composicion_hogar(individual)
    n_miembros = composicion.set_index(ch.CLAVE_HOGAR)["n_miembros"]
    laboral = ch.laboral_hogar(individual)
    precariedad = ch.precariedad_hogar(individual)
    fuentes = ch.fuentes_ingreso_hogar(individual, ch.fuentes_por_trimestre(ins.anio, ins.trimestre), hogar["ITF"])
    vivienda = ch.vivienda_hogar(hogar, n_miembros)
    estrategias = ch.calcular_estrategias(hogar).reset_index()
    menor = ch.hay_menor_no_asiste(individual).reset_index()
    tipologia = calcular_tipologia_perceptores(ins.hogar, individual)

    distancia = ins.distancia.copy()
    peso = distancia["PONDIH"] if "PONDIH" in distancia else distancia["peso_hogar"]
    distancia["decil_ratio"] = decil_ponderado(distancia["ratio_linea_pobreza"], pd.to_numeric(peso, errors="coerce"))
    hogar_reset = ins.hogar.copy()
    hogar_reset["deccfr"] = pd.to_numeric(hogar_reset["DECCFR"], errors="coerce").where(lambda s: s.between(1, 10)).astype("Int8")

    columnas_composicion = ["n_miembros", "n_menores_14", "n_menores_18", "n_mayores_65", "share_dependientes",
                            "sexo_jefe", "edad_jefe", "nivel_ed_jefe", "nivel_ed_jefe_bucket", "jefe_ambiguo",
                            "condicion_actividad_jefe"]
    return [
        _por_hogar_t(distancia, ["ratio_linea_indigencia", "pobre", "indigente", "decil_ratio"]),
        _por_hogar_t(hogar_reset, ["deccfr"]),
        _por_hogar_t(composicion, columnas_composicion),
        _por_hogar_t(menor, ["hay_menor_no_asiste"]),
        _por_hogar_t(laboral, ["situacion_laboral", "n_ocupados", "n_desocupados", "n_activos", "sin_activos", "share_ocupados"]),
        _por_hogar_t(precariedad, ["share_precario_A", "share_precario_B", "share_precario_C", "share_precario_D",
                                   "n_activos_clasificables", "n_activos_clasificables_D"]),
        _por_hogar_t(tipologia, ["tipologia_perceptores"]),
        _por_hogar_t(fuentes, [c for c in fuentes.columns if c.startswith("share_")]),
        _por_hogar_t(vivienda, [c for c in vivienda.columns if c not in ch.CLAVE_HOGAR]),
        _por_hogar_t(estrategias, ["recurre_desahorro", "recurre_endeudamiento_formal"]),
    ]


def _termina_en_t(pares: pd.DataFrame, anio: int, trimestre: int) -> pd.DataFrame:
    # Sin pares (o sin columnas, como devuelve el notebook al principio de cada
    # era) se devuelve vacío; con columnas se conservan, aunque no haya filas.
    if pares is None or "ANO4_th" not in pares:
        return pd.DataFrame()
    return pares[(pares["ANO4_th"] == anio) & (pares["TRIMESTRE_th"] == trimestre)]


def _previa_limpia(ins: InsumosTrimestre, base: pd.DataFrame) -> pd.DataFrame:
    """`variacion_simetrica_previa_c`, `ratio_linea_pobreza_previo` y
    `motivo_previa_na_t` desde el par previo de cada hogar (h=1 desde `t-1`,
    o h=1 de `t-4` a `t-3`), buscado sin depender de `gap_previo`."""
    # `gap_previo` de 05 queda en NA también cuando el par previo existe pero su
    # variación es NA (ingreso no declarado, 0/0): no distingue esos casos de la
    # 1ª visita. Por eso el par previo se busca de nuevo acá.
    claves = ["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t"]
    salidas = ["variacion_simetrica_previa_c", "ratio_linea_pobreza_previo", "motivo_previa_na_t", "motivo_previa_na_detalle_t"]
    previos = ins.pares_previos
    if previos is None or previos.empty:
        return _con_motivo(base[claves].assign(_gap=pd.NA, variacion_simetrica_previa_c=np.nan,
                                               ratio_linea_pobreza_previo=np.nan))[claves + salidas]
    previos = previos.copy()
    for lado in ("t", "th"):
        if f"imputado_{lado}" not in previos:
            previos[f"imputado_{lado}"] = False
    previos["_imputada"] = (previos["imputado_t"] == True) | (previos["imputado_th"] == True)  # noqa: E712
    previos["_ratio_na"] = previos["ratio_linea_pobreza_t"].isna() | previos["ratio_linea_pobreza_th"].isna()
    columnas = ["variacion_simetrica", "par_valido_c", "identidad_confirmada", "_imputada", "_ratio_na"]
    # gap 0: par h=1 de t-1 a t; la visita previa es su origen.
    gap0 = _termina_en_t(previos[previos["h"] == 1], ins.anio, ins.trimestre)
    gap0 = gap0[["CODUSU", "NRO_HOGAR_th", "AGLOMERADO_th", *columnas, "ratio_linea_pobreza_t"]].rename(
        columns={"NRO_HOGAR_th": "NRO_HOGAR_t", "AGLOMERADO_th": "AGLOMERADO_t", "ratio_linea_pobreza_t": "_ratio"})
    # gap 3: par h=1 de t-4 a t-3 (mismo emparejamiento por origen que 05).
    anio_4, trimestre_4 = _restar(ins.anio, ins.trimestre, 4)
    gap3 = previos[(previos["h"] == 1) & (previos["ANO4_t"] == anio_4) & (previos["TRIMESTRE_t"] == trimestre_4)]
    gap3 = gap3[["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t", *columnas, "ratio_linea_pobreza_th"]].rename(
        columns={"ratio_linea_pobreza_th": "_ratio"})
    candidatos = pd.concat([gap0.assign(_gap=0), gap3.assign(_gap=3)], ignore_index=True)
    candidatos = candidatos.merge(base[claves].drop_duplicates(), on=claves, how="inner")
    resultado = base[claves + ["gap_previo"]].merge(candidatos, on=claves, how="left", validate="one_to_one")
    gap_05 = resultado["gap_previo"].astype("Float64")
    distinto = gap_05.notna() & (gap_05 != resultado["_gap"].astype("Float64")).fillna(True)
    if distinto.any():
        raise ValueError(f"{ins.anio}T{ins.trimestre}: {int(distinto.sum())} pares con gap_previo distinto del par previo hallado")
    limpio = (resultado["par_valido_c"] == True).fillna(False).to_numpy()  # noqa: E712
    con_identidad = (resultado["identidad_confirmada"] == True).fillna(False).to_numpy()  # noqa: E712
    resultado["variacion_simetrica_previa_c"] = resultado["variacion_simetrica"].astype(float).where(limpio)
    resultado["ratio_linea_pobreza_previo"] = resultado["_ratio"].astype(float).where(con_identidad)
    return _con_motivo(resultado)[claves + salidas]


def _con_motivo(previa: pd.DataFrame) -> pd.DataFrame:
    """Agrega `motivo_previa_na_detalle_t` y `motivo_previa_na_t` (`MOTIVOS_PREVIA`),
    coherentes con `variacion_simetrica_previa_c`: `disponible` si y solo si no es NA."""
    # El orden importa: cada par cae en el primer motivo que cumple.
    resultado = previa.copy()
    for c in ("identidad_confirmada", "_imputada", "_ratio_na"):
        if c not in resultado:
            resultado[c] = pd.NA
    motivo = pd.Series(pd.NA, index=resultado.index, dtype="object")
    encontrado = resultado["identidad_confirmada"].notna()
    disponible = resultado["variacion_simetrica_previa_c"].notna()
    condiciones = [
        ("disponible", disponible),
        ("primera_visita", resultado["_gap"].isna()),
        ("sin_identidad", encontrado & (resultado["identidad_confirmada"] != True)),  # noqa: E712
        ("imputada_indec", encontrado & (resultado["_imputada"] == True)),  # noqa: E712
        ("no_respuesta_ingreso", encontrado & (resultado["_ratio_na"] == True)),  # noqa: E712
        ("itf_cero_ambas_puntas", encontrado),
    ]
    for nombre, condicion in condiciones:
        motivo[motivo.isna() & condicion.fillna(False).astype(bool).to_numpy()] = nombre
    resultado["motivo_previa_na_detalle_t"] = motivo
    resultado["motivo_previa_na_t"] = motivo.replace(MOTIVOS_PREVIA_DETALLE)
    return resultado


def _restar(anio: int, trimestre: int, n: int) -> tuple[int, int]:
    indice = anio * 4 + trimestre - 1 - n
    return indice // 4, indice % 4 + 1


def _hasta_t(ins: InsumosTrimestre, base: pd.DataFrame) -> pd.DataFrame:
    """Features del par que termina en `t` (h=1 desde `t-1` o h=4 desde
    `t-4`): variaciones, cambio de adulto equivalente y eventos de composición."""
    claves = ["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t"]
    salida = base[claves].copy()
    columnas = ["variacion_simetrica_hasta_t", "variacion_log_hasta_t", "variacion_simetrica_hasta_t_c",
                "variacion_log_hasta_t_c", "cambio_ae_previo_t", "tasa_altas_bajas_previa_t",
                "n_se_jubila_previo_t", "n_alta_activo_menor_previo_t"]
    previos = _termina_en_t(ins.pares_previos, ins.anio, ins.trimestre)
    if previos.empty:
        return salida.assign(**{c: np.nan for c in columnas}, tiene_par_hasta_t=False)

    p = previos.rename(columns={"NRO_HOGAR_t": "NRO_HOGAR_origen", "NRO_HOGAR_th": "NRO_HOGAR_t",
                                "AGLOMERADO_t": "AGLOMERADO_origen", "AGLOMERADO_th": "AGLOMERADO_t"})
    # Un hogar en su 4ª visita tiene dos pares que terminan en t (h=1 desde t-1 y
    # h=4 desde t-4), pero no origina pares en t: se descarta antes de validar
    # que cada hogar del origen tenga un solo par previo.
    p = p.merge(salida[claves].drop_duplicates(), on=claves, how="inner")
    identidad = (p["identidad_confirmada"] == True).fillna(False)  # noqa: E712
    ae_origen = p["unidades_adulto_equivalente_t"].astype(float)
    p = p.assign(
        variacion_simetrica_hasta_t=p["variacion_simetrica"].astype(float),
        variacion_log_hasta_t=p["variacion_log"].astype(float),
        variacion_simetrica_hasta_t_c=p["variacion_simetrica"].astype(float).where((p["par_valido_c"] == True).fillna(False)),  # noqa: E712
        variacion_log_hasta_t_c=p["variacion_log"].astype(float).where((p["par_valido_log_c"] == True).fillna(False)),  # noqa: E712
        cambio_ae_previo_t=((p["unidades_adulto_equivalente_th"].astype(float) - ae_origen).abs()
                            / ae_origen.replace(0, np.nan)).where(identidad),
    )
    eventos = _eventos_previos(ins)
    p = p.merge(eventos, on=["CODUSU", "NRO_HOGAR_origen", "NRO_HOGAR_t", "h"], how="left")
    identidad = (p["identidad_confirmada"] == True).fillna(False)  # noqa: E712
    for c in ("tasa_altas_bajas_previa_t", "n_se_jubila_previo_t", "n_alta_activo_menor_previo_t"):
        p[c] = p[c].astype(float).where(identidad)
    p["tiene_par_hasta_t"] = True
    salida = salida.merge(p[claves + columnas + ["tiene_par_hasta_t"]], on=claves, how="left", validate="one_to_one")
    salida["tiene_par_hasta_t"] = salida["tiene_par_hasta_t"].fillna(False).astype(bool)
    return salida


def _eventos_previos(ins: InsumosTrimestre) -> pd.DataFrame:
    """Altas/bajas (de 03_panel), jubilaciones y altas de activos menores
    sobre los pares de 03_panel que terminan en `t`, por horizonte previo."""
    vacio = pd.DataFrame(columns=["CODUSU", "NRO_HOGAR_origen", "NRO_HOGAR_t", "h", "tasa_altas_bajas_previa_t",
                                  "n_se_jubila_previo_t", "n_alta_activo_menor_previo_t"])
    panel = _termina_en_t(ins.panel_previos, ins.anio, ins.trimestre)
    if panel.empty:
        return vacio
    partes = []
    for (anio_o, trimestre_o, h_previo), pares_hogar in panel.groupby(["ANO4_t", "TRIMESTRE_t", "h"]):
        individual_origen = ins.individual_previo.get((int(anio_o), int(trimestre_o)))
        if individual_origen is None:
            continue
        n_miembros = individual_origen.groupby(["CODUSU", "NRO_HOGAR"]).size().rename("n_miembros_origen").reset_index()
        tabla = pares_hogar[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "n_altas", "n_bajas"]].merge(
            n_miembros.rename(columns={"NRO_HOGAR": "NRO_HOGAR_t"}), on=["CODUSU", "NRO_HOGAR_t"], how="left")
        tabla["tasa_altas_bajas_previa_t"] = (tabla["n_altas"] + tabla["n_bajas"]) / tabla["n_miembros_origen"]
        jubila = calcular_se_jubila(individual_origen, ins.individual, pares_hogar, int(h_previo))
        menor = calcular_alta_activo_menor(individual_origen, ins.individual, pares_hogar, int(h_previo))
        tabla = tabla.merge(jubila, on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"], how="left").merge(
            menor[["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "n_alta_activo_menor"]],
            on=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th"], how="left")
        partes.append(tabla.rename(columns={
            "NRO_HOGAR_t": "NRO_HOGAR_origen", "NRO_HOGAR_th": "NRO_HOGAR_t",
            "n_se_jubila": "n_se_jubila_previo_t", "n_alta_activo_menor": "n_alta_activo_menor_previo_t"}).assign(h=int(h_previo)))
    if not partes:
        return vacio
    return pd.concat(partes, ignore_index=True)[vacio.columns]


def _contexto(ins: InsumosTrimestre, base: pd.DataFrame) -> pd.DataFrame:
    salida = base[["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t"]].copy()
    columnas = ["tasa_pobreza_aglo", "tasa_indigencia_aglo", "tasa_informalidad_aglo", "tasa_desocupacion_aglo",
                "gini_ipcf_aglo", "n_hogares_ventana_aglo"]
    if ins.contexto_aglomerado is None or ins.contexto_aglomerado.empty:
        salida = salida.assign(**{f"{c}_t": np.nan for c in columnas})
    else:
        contexto = ins.contexto_aglomerado.rename(columns={"AGLOMERADO": "AGLOMERADO_t"})
        contexto = contexto[["AGLOMERADO_t", *[c for c in columnas if c in contexto]]]
        contexto = contexto.rename(columns={c: f"{c}_t" for c in columnas})
        contexto["AGLOMERADO_t"] = contexto["AGLOMERADO_t"].astype(salida["AGLOMERADO_t"].dtype)
        salida = salida.merge(contexto, on="AGLOMERADO_t", how="left", validate="many_to_one")

    # Mismo horizonte que el objetivo: para h, los pares h que terminan en t.
    previos = _termina_en_t(ins.pares_previos, ins.anio, ins.trimestre)
    previos = previos[previos["h"] == ins.h] if not previos.empty else previos
    salida["n_pares_terminan_en_t"] = len(previos)
    if previos.empty:
        for ambito in ("trimestre", "aglomerado"):
            for tipo in ("simetrica", "log"):
                salida[f"variacion_comun_{ambito}_{tipo}_rezagada_t"] = np.nan
        return salida
    escalares, por_aglomerado = componente_comun_rezagado(previos)
    for nombre, valor in escalares.items():
        salida[f"{nombre}_t"] = valor
    por_aglomerado = por_aglomerado.rename(columns={"AGLOMERADO": "AGLOMERADO_t"})
    por_aglomerado = por_aglomerado.rename(columns={c: f"{c}_t" for c in por_aglomerado.columns if c != "AGLOMERADO_t"})
    por_aglomerado["AGLOMERADO_t"] = por_aglomerado["AGLOMERADO_t"].astype(salida["AGLOMERADO_t"].dtype)
    return salida.merge(por_aglomerado, on="AGLOMERADO_t", how="left", validate="many_to_one")


def construir_features_trimestre(ins: InsumosTrimestre) -> pd.DataFrame:
    """Una fila por par de `ins.pares` (sin filtrar, mismo orden), con las
    columnas de paso (clave, filtros, pesos) y todas las del catálogo."""
    pares = ins.pares.copy()
    pares["peso_normalizado_trimestre"] = ct.peso_normalizado_trimestre(pares, "peso_hogar_era_t")
    base = pares[[c for c in COLUMNAS_PASO if c in pares] + [
        "era", "ratio_linea_pobreza_t", "log_ratio_linea_pobreza_t", "unidades_adulto_equivalente_t", "ITF_t",
        "variacion_simetrica_previa", "gap_previo"]].copy()
    n_original = len(base)

    for tabla in _foto_en_t(ins):
        base = base.merge(tabla, on=["CODUSU", "NRO_HOGAR_t"], how="left", validate="many_to_one")
    claves = ["CODUSU", "NRO_HOGAR_t", "AGLOMERADO_t"]
    for tabla in (_previa_limpia(ins, base), _hasta_t(ins, base), _contexto(ins, base)):
        base = base.merge(tabla, on=claves, how="left", validate="one_to_one")

    base["variacion_simetrica_previa"] = base["variacion_simetrica_previa"].astype(float)
    base["share_jubilacion_continua_t"] = base["share_jubilacion_t"] + base["share_residuo_no_laboral_t"]
    base["encuesta_telefonica_t"] = (ins.anio, ins.trimestre) == TRIMESTRE_TELEFONICO
    base["periodo_intervenido_t"] = ct.INTERVENCION_DESDE <= (ins.anio, ins.trimestre) <= ct.INTERVENCION_HASTA
    base["cuestionario_2023t4_t"] = (ins.anio, ins.trimestre) >= ch.TRIMESTRE_CUESTIONARIO_NUEVO
    base["aglomerado_incorporado_2006t3"] = base["AGLOMERADO_t"].isin(AGLOMERADOS_INCORPORADOS_2006T3)
    if len(base) != n_original:
        raise ValueError(f"{ins.anio}T{ins.trimestre} h={ins.h}: el armado cambió la cantidad de filas")
    return base


MASCARAS_APLICA = {
    "itf_positivo": lambda df: df["ITF_t"].astype(float) > 0,
    "con_activos": lambda df: df["n_activos_clasificables_t"].fillna(0) > 0,
    "con_activos_d": lambda df: df["n_activos_clasificables_D_t"].fillna(0) > 0,
    "con_pares_terminan_en_t": lambda df: df["n_pares_terminan_en_t"] > 0,
    "tiene_previa": lambda df: df["motivo_previa_na_t"].notna() & (df["motivo_previa_na_t"] != "primera_visita"),
    "tiene_par_hasta_t": lambda df: df["tiene_par_hasta_t"].fillna(False).astype(bool),
}


def reporte_na(features: pd.DataFrame, catalogo: list[Feature] = CATALOGO_FEATURES) -> pd.DataFrame:
    """Por feature: % de NA en todos los pares y en `par_valido_c`, separando
    el NA estructural (filas donde la feature no aplica) del resto."""
    filas = []
    universos = {"todos": pd.Series(True, index=features.index),
                 "par_valido_c": (features["par_valido_c"] == True).fillna(False).astype(bool)}  # noqa: E712
    for f in catalogo:
        if f.nombre not in features:
            continue
        es_na = features[f.nombre].isna()
        aplica = MASCARAS_APLICA[f.aplica](features) if f.aplica else pd.Series(True, index=features.index)
        fila = {"feature": f.nombre, "bloque": f.bloque, "rol": f.rol, "en_x": f.en_x, "aplica": f.aplica}
        for nombre, universo in universos.items():
            n = int(universo.sum())
            fila[f"n_{nombre}"] = n
            fila[f"pct_na_{nombre}"] = 100 * es_na[universo].mean() if n else np.nan
            fila[f"pct_na_estructural_{nombre}"] = 100 * (es_na & ~aplica)[universo].mean() if n else np.nan
            fila[f"pct_na_dato_{nombre}"] = 100 * es_na[universo & aplica].mean() if (universo & aplica).any() else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)
