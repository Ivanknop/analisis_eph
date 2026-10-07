"""Tests de `src/cascada.py` con DataFrames sintéticos chicos, salvo el
test marcado que lee `data/02_nucleo/` real."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascada import (
    activos_sin_empleo_formal,
    aglomerados_excluidos_d1,
    calcular_persistencia,
    clasificar_fold,
    clasificar_trabajo_lexicografico,
    clasificar_vinculacion_trimestre,
    construir_fila_cascada,
    frecuencias_vivienda,
    tiene_destino_valido,
)
from intensificacion import calcular_share_precario, emparejar_activos

REPO = Path(__file__).resolve().parents[1]


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH04", "CH06", "ESTADO", "CAT_OCUP", "PP07H"]
    return pd.DataFrame(filas, columns=columnas)


def _pares(codusu="V1", nro_hogar=1):
    return pd.DataFrame({"CODUSU": [codusu], "NRO_HOGAR_t": [nro_hogar], "NRO_HOGAR_th": [nro_hogar]})


class TestTieneDestinoValido:
    def test_hueco_de_publicacion(self):
        assert tiene_destino_valido(2015, 2, 1) is False  # destino 2015T3, en el hueco

    def test_borde_final_de_la_serie(self):
        assert tiene_destino_valido(2025, 4, 1) is False  # destino 2026T1, no publicado

    def test_cruza_el_corte_de_codusu_d7(self):
        assert tiene_destino_valido(2015, 2, 4) is False  # destino 2016T2, cruza histórico/regular

    def test_origen_normal(self):
        assert tiene_destino_valido(2018, 2, 1) is True


class TestClasificarVinculacionTrimestre:
    def test_atricion_neta_resta_la_rotacion_teorica(self):
        hogar_t = pd.DataFrame({"CODUSU": ["V1", "V2", "V3", "V4"], "NRO_HOGAR": [1, 1, 1, 1],
                                 "AGLOMERADO": [1, 1, 1, 1]})
        hogar_th = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1]})
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        r = clasificar_vinculacion_trimestre(hogar_t, hogar_th, pares, pct_rotacion_teorica=50.0)
        assert r["n_viviendas_t"] == 4
        assert r["n_vinculada"] == 1
        assert r["n_vivienda_sin_par"] == 3
        assert r["pct_vivienda_sin_par_crudo"] == pytest.approx(75.0)
        assert r["pct_atricion_neta_rotacion"] == pytest.approx(25.0)

    def test_desglose_a_nivel_hogar_con_vivienda_multihogar(self):
        # V1 tiene 2 hogares en t (multi-hogar, D9): solo el hogar 1 matchea en th,
        # el hogar 2 queda "sin_match_multihogar" aunque su vivienda esté vinculada.
        # V2 (1 hogar) matchea. V3 no está más en th (vivienda_sin_par). V4 sigue en
        # th pero con otro NRO_HOGAR (hogar_reemplazado). 4 viviendas, 5 hogares.
        hogar_t = pd.DataFrame({
            "CODUSU": ["V1", "V1", "V2", "V3", "V4"], "NRO_HOGAR": [1, 2, 1, 1, 1],
            "AGLOMERADO": [1, 1, 1, 1, 1],
        })
        hogar_th = pd.DataFrame({"CODUSU": ["V1", "V2", "V4"], "NRO_HOGAR": [1, 1, 2]})
        pares = pd.DataFrame({"CODUSU": ["V1", "V2"], "NRO_HOGAR_t": [1, 1], "NRO_HOGAR_th": [1, 1]})

        r = clasificar_vinculacion_trimestre(hogar_t, hogar_th, pares)
        assert r["n_viviendas_t"] == 4
        assert r["n_vinculada"] == 2  # V1, V2
        assert r["n_hogar_reemplazado"] == 1  # V4
        assert r["n_vivienda_sin_par"] == 1  # V3

        assert r["n_hogares_matcheados"] == 2  # (V1,1) y (V2,1)
        assert r["n_hogares_sin_match_multihogar"] == 1  # (V1,2)
        assert r["n_hogares_en_vivienda_reemplazada"] == 1  # (V4,1)
        assert r["n_hogares_en_vivienda_sin_par"] == 1  # (V3,1)
        n_hogares_en_t = len(hogar_t[["CODUSU", "NRO_HOGAR"]].drop_duplicates())
        assert n_hogares_en_t == 5
        assert (r["n_hogares_matcheados"] + r["n_hogares_sin_match_multihogar"]
                + r["n_hogares_en_vivienda_reemplazada"] + r["n_hogares_en_vivienda_sin_par"]) == n_hogares_en_t

    def test_excluye_aglomerados_d1_del_pct_pero_no_del_conteo_crudo(self):
        hogar_t = pd.DataFrame({"CODUSU": ["V1", "V2", "V3"], "NRO_HOGAR": [1, 1, 1],
                                 "AGLOMERADO": [1, 8, 8]})
        hogar_th = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1]})
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})

        sin_excluir = clasificar_vinculacion_trimestre(hogar_t, hogar_th, pares)
        assert sin_excluir["n_vivienda_sin_par"] == 2  # V2, V3 -- conteo crudo intacto
        assert sin_excluir["pct_vivienda_sin_par_crudo"] == pytest.approx(100 * 2 / 3)

        con_exclusion = clasificar_vinculacion_trimestre(
            hogar_t, hogar_th, pares, aglomerados_excluidos_atricion=aglomerados_excluidos_d1(2019, 3))
        assert con_exclusion["n_vivienda_sin_par"] == 2  # el conteo crudo no cambia
        assert con_exclusion["pct_vivienda_sin_par_crudo"] == pytest.approx(0.0)  # V1 (aglomerado 1) está vinculada


class TestAglomeradosExcluidosD1:
    def test_aglomerado_8_en_2019t3(self):
        assert aglomerados_excluidos_d1(2019, 3) == frozenset({8})

    def test_aglomerado_31_en_2020t3(self):
        assert aglomerados_excluidos_d1(2020, 3) == frozenset({31})

    def test_vacio_en_cualquier_otro_destino(self):
        assert aglomerados_excluidos_d1(2021, 1) == frozenset()


class TestClasificarTrabajoLexicografico:
    def _share_y_emparejados(self, individual_t, individual_th, h=1):
        emparejados = emparejar_activos(individual_t, individual_th, _pares(), h)
        share = calcular_share_precario(emparejados)
        return emparejados, share

    def test_sube_share_es_siempre_empeora(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 1]])  # formal
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 2]])  # informal
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "empeora"

    def test_baja_share_es_siempre_mejora(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])  # informal
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 1]])  # formal
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "mejora"

    def test_desocupado_a_informal_es_mejora_aunque_el_share_no_cambie(self):
        # share_precario cuenta informal_asalariado y desocupado por igual -- D16
        # tal cual daría "no cambia"; la lexicográfica desempata por el share de
        # desocupados, que baja (D24).
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 2, pd.NA, pd.NA],   # desocupado
            ["V1", 1, 2, 2, 40, 1, 3, 1],           # formal, estable, de relleno
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 3, 2],           # pasó a informal_asalariado
            ["V1", 1, 2, 2, 41, 1, 3, 1],
        ])
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert share.iloc[0]["share_precario_t"] == pytest.approx(share.iloc[0]["share_precario_th"])
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "mejora"

    def test_informal_a_desocupado_en_el_techo_es_empeora(self):
        # share_t == 1.0 (techo, D20) -- D16 tal cual nunca podría marcar "empeora"
        # acá; la lexicográfica sí, vía el share de desocupados que sube.
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])   # informal_asalariado
        individual_th = _individual([["V1", 1, 1, 1, 31, 2, pd.NA, pd.NA]])  # desocupado
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert share.iloc[0]["share_precario_t"] == pytest.approx(1.0)
        assert share.iloc[0]["share_precario_th"] == pytest.approx(1.0)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "empeora"

    def test_sin_cambio_real_es_igual(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 1]])
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "igual"

    def test_sin_activos_estables_se_mantiene_aparte(self):
        individual_t = _individual([["V1", 1, 1, 1, 60, 1, 3, 2]])
        individual_th = _individual([["V1", 1, 1, 1, 61, 3, pd.NA, pd.NA]])  # se jubila, sale del denominador
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "sin_activos_estables"

    def test_prioridad_el_nivel_1_pisa_al_nivel_2(self):
        # A: desocupado -> informal_asalariado (bajaría la desocupación, nivel 2
        # diría "mejora" si se mirara solo); B: formal -> informal_asalariado (sube
        # el share). El share sube en conjunto -- tiene que ganar "empeora", el
        # nivel 2 nunca debe pisar a un nivel 1 ya decidido.
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 2, pd.NA, pd.NA],   # A desocupado
            ["V1", 1, 2, 2, 40, 1, 3, 1],           # B formal
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 3, 2],           # A' informal_asalariado
            ["V1", 1, 2, 2, 41, 1, 3, 2],           # B' informal_asalariado
        ])
        emparejados, share = self._share_y_emparejados(individual_t, individual_th)
        assert share.iloc[0]["share_precario_t"] == pytest.approx(0.5)
        assert share.iloc[0]["share_precario_th"] == pytest.approx(1.0)
        assert clasificar_trabajo_lexicografico(emparejados, share).iloc[0] == "empeora"


class TestActivosSinEmpleoFormal:
    def test_poblacion_b(self):
        individual = _individual([
            ["V1", 1, 1, 1, 30, 1, 3, 2],   # informal_asalariado -- activo sin empleo formal
            ["V2", 1, 1, 1, 30, 1, 3, 1],   # formal -- no entra en la población
            ["V3", 1, 1, 1, 30, 3, pd.NA, pd.NA],  # inactivo -- no entra
        ])
        idx = pd.MultiIndex.from_frame(pd.DataFrame({"CODUSU": ["V1", "V2", "V3"], "NRO_HOGAR": [1, 1, 1]}))
        r = activos_sin_empleo_formal(individual, idx)
        assert r.tolist() == [True, False, False]


class TestCalcularPersistencia:
    def test_sale_por_empleo_no_persiste(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])   # activo sin empleo formal
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 1]])  # pasa a formal -- sale
        r = calcular_persistencia(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["activos_sin_empleo_formal_t"] == True  # noqa: E712
        assert r["estado_th"] == "sale_por_empleo"
        assert r["persiste_principal"] == False  # noqa: E712
        assert r["persiste_robustez"] == False  # noqa: E712

    def test_permanece_es_positivo_en_las_dos_variantes(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 2]])  # sigue sin empleo formal
        r = calcular_persistencia(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["estado_th"] == "permanece"
        assert r["persiste_principal"] == True  # noqa: E712
        assert r["persiste_robustez"] == True  # noqa: E712

    def test_pasa_a_sin_activos_difiere_entre_principal_y_robustez(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])
        individual_th = _individual([["V1", 1, 1, 1, 31, 3, pd.NA, pd.NA]])  # inactivo
        r = calcular_persistencia(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["estado_th"] == "pasa_a_sin_activos"
        assert r["persiste_principal"] == True  # noqa: E712 -- cuenta como "no sale"
        assert pd.isna(r["persiste_robustez"])  # excluida en la variante de robustez

    def test_fuera_de_la_poblacion_queda_na(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 1]])   # formal -- no es población b
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 1]])
        r = calcular_persistencia(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["activos_sin_empleo_formal_t"] == False  # noqa: E712
        assert pd.isna(r["persiste_principal"])
        assert pd.isna(r["persiste_robustez"])

    @pytest.mark.skipif(not (REPO / "data" / "02_nucleo").exists(), reason="requiere data/02_nucleo/ real")
    @pytest.mark.parametrize("anio, trimestre", [
        (2010, 2),  # era histórica
        (2018, 2),  # era regular, cuestionario pre-2023T4
        (2023, 4),  # era regular, cuestionario posrediseño (D2)
    ])
    def test_poblacion_coincide_con_riesgo_poblacion_trimestre(self, anio, trimestre):
        # Regresión: duplicar activos_sin_empleo_formal fuera de la notebook 06 no
        # puede introducir una diferencia contra el número ya publicado y
        # confirmado en D14 (data/meta/riesgo_poblacion_trimestre.csv) -- en
        # ninguna de las eras/cuestionarios (D2) que cubre esa tabla.
        publicado = pd.read_csv(REPO / "data" / "meta" / "riesgo_poblacion_trimestre.csv")
        fila_publicada = publicado[(publicado["anio"] == anio) & (publicado["trimestre"] == trimestre)].iloc[0]

        hogar = pd.read_parquet(REPO / "data" / "02_nucleo" / "hogar" / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet")
        individual = pd.read_parquet(
            REPO / "data" / "02_nucleo" / "individual" / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet")
        idx = pd.MultiIndex.from_frame(hogar[["CODUSU", "NRO_HOGAR"]])
        n_precario_b = int((activos_sin_empleo_formal(individual, idx) == True).sum())  # noqa: E712
        assert n_precario_b == int(fila_publicada["n_precario_b"])


class TestClasificarFold:
    def test_embargo_excluye_pares_cuyo_destino_solapa_la_ventana_de_test(self):
        pares = pd.DataFrame({
            "ANO4_t": [2019, 2020, 2021], "TRIMESTRE_t": [4, 4, 2],
            "ANO4_th": [2021, 2021, 2021], "TRIMESTRE_th": [1, 2, 3],
        })
        # test = origen 2021T2; límite de embargo = 2021T2 - 1 = 2021T1.
        r = clasificar_fold(pares, anio_test=2021, trimestre_test=2, embargo=1)
        # fila 0: destino 2021T1 <= límite -> train.
        # fila 1: destino 2021T2 > límite (solapa la ventana de test) y el origen
        # (2020T4) no es el trimestre de test -> descartado, no entrena ni evalúa.
        # fila 2: origen == trimestre de test -> test.
        assert r.tolist() == ["train", "descartado", "test"]


class TestConstruirFilaCascada:
    def test_sin_destino_publicado_solo_tiene_hogares_en_t(self):
        fila = construir_fila_cascada(1234, "sin_destino_publicado", None, None)
        assert fila == {"hogares_en_t": 1234, "estado_destino": "sin_destino_publicado"}

    def test_invariante_de_suma_por_fila(self):
        # Viviendas != hogares a propósito (D24): 10 viviendas (nivel vivienda) y,
        # por separado, 10 hogares repartidos en 4 categorías a nivel hogar (de los
        # cuales 5 matchean -> son los 5 pares de tabla_pares).
        vinculacion = {"n_viviendas_t": 10, "n_vinculada": 6, "n_hogar_reemplazado": 1,
                       "n_vivienda_sin_par": 3, "pct_vivienda_sin_par_crudo": 30.0,
                       "pct_atricion_neta_rotacion": -20.0,
                       "n_hogares_en_vivienda_sin_par": 3, "n_hogares_en_vivienda_reemplazada": 1,
                       "n_hogares_sin_match_multihogar": 1, "n_hogares_matcheados": 5}
        tabla_pares = pd.DataFrame({
            "identidad_confirmada": [True, True, True, True, False],
            # fila 3: ingreso_no_declarado_t en NA (era histórica, D3) -- no excluye.
            "ingreso_no_declarado_t": pd.array([False, False, False, pd.NA, False], dtype="boolean"),
            # fila 1: ingreso no declarado en t+h (era regular) -- sí excluye.
            "ingreso_no_declarado_th": pd.array([False, True, False, False, False], dtype="boolean"),
            "n_activos_estables_t": [1, 1, 0, 1, 1],
            "n_activos_estables_th": [1, 1, 1, 0, 1],
            "trabajo_d16": ["empeora", "igual", "igual", "igual", "igual"],
            "trabajo_lexicografico": ["empeora", "mejora", "sin_activos_estables", "sin_activos_estables", "igual"],
            "persiste_principal": [True, False, pd.NA, pd.NA, True],
            "persiste_robustez": [True, False, pd.NA, pd.NA, True],
        })
        fila = construir_fila_cascada(10, "valido", vinculacion, tabla_pares)

        assert fila["n_pares"] == 5
        assert fila["n_identidad_confirmada"] + fila["n_identidad_no_confirmada"] == fila["n_pares"]
        assert fila["n_identidad_confirmada"] == 4

        # fila 1 queda excluida por ingreso no declarado en t+h; fila 3 (NA, era
        # histórica) NO se excluye -- D3.
        assert fila["n_excluido_ingreso_no_declarado"] == 1
        assert fila["n_elegibles_post_ingreso"] == 3
        assert fila["n_excluido_ingreso_no_declarado"] + fila["n_elegibles_post_ingreso"] == fila["n_identidad_confirmada"]

        # Y1/Y2 sin sufijo = NO excluyen ingreso no declarado por default (pedido de
        # Iván 2026-10-05): población = confirmados (filas 0-3) con activos estables
        # en las dos puntas -- filas 0 y 1 (la 2 y la 3 no tienen activos estables en
        # alguna punta; la 1 ya NO se excluye acá, a diferencia de antes).
        assert fila["n_poblacion_y1"] == 2
        assert fila["n_y1_empeora_lexicografica"] + fila["n_y1_igual_lexicografica"] + \
            fila["n_y1_mejora_lexicografica"] == fila["n_poblacion_y1"]
        assert fila["n_y1_empeora_lexicografica"] == 1  # fila 0
        assert fila["n_y1_mejora_lexicografica"] == 1  # fila 1 (desocupado->informal, D24)
        assert fila["n_y2_poblacion_principal"] == 2  # filas 0 y 1 tienen persiste_principal no nulo

        # Y2 invertido (positivo = sale_por_empleo): complemento exacto de persiste,
        # mismo denominador.
        assert fila["n_y2_sale_principal"] == fila["n_y2_poblacion_principal"] - fila["n_y2_persiste_principal"]
        assert fila["tasa_y2_sale_principal"] == pytest.approx(100 - fila["tasa_y2_principal"])

        # ..._excl_ingreso repite el cálculo excluyendo la fila 1 (ingreso no
        # declarado en t+h) -- población Y1 queda en la sola fila 0.
        assert fila["n_poblacion_y1_excl_ingreso"] == 1
        assert fila["n_y1_empeora_lexicografica_excl_ingreso"] == 1
        assert fila["n_y2_poblacion_principal_excl_ingreso"] == 1
        assert fila["n_y2_sale_principal_excl_ingreso"] == 0


class TestFrecuenciasVivienda:
    def test_cuenta_y_porcentaje_por_trimestre(self):
        hogar = pd.DataFrame({
            "ANO4": [2020, 2020, 2020, 2021], "TRIMESTRE": [1, 1, 1, 1],
            "AGLOMERADO": [1, 2, 1, 1], "IV1": [1, 1, 2, 2],
        })
        r = frecuencias_vivienda(hogar, ["IV1"])
        fila_valor_1 = r[(r["ANO4"] == 2020) & (r["valor"] == 1)].iloc[0]
        fila_valor_2 = r[(r["ANO4"] == 2020) & (r["valor"] == 2)].iloc[0]
        assert fila_valor_1["n"] == 2
        assert fila_valor_1["pct"] == pytest.approx(200 / 3)
        assert fila_valor_2["n"] == 1
        assert fila_valor_1["pct"] + fila_valor_2["pct"] == pytest.approx(100.0)
