"""Tests de `src/panel.py`. Todo con DataFrames sintéticos chicos armados a
mano -- nada acá toca `data/` real."""
import pandas as pd
import pytest

from panel import (
    clasificar_viviendas_no_vinculadas,
    clasificar_vinculacion_trimestre,
    construir_pares,
    contar_altas_bajas,
    contar_nacimientos,
    emparejar_componentes,
    hash_esquema_panel,
    identidad_nucleo,
    mandato_presidencial,
    particion_train_holdout,
    regimen_ingreso_par,
    sumar_trimestres,
    tasa_coincidencia_componente,
    tasa_vinculacion_viviendas,
    tiene_destino_valido,
    validar_jefe,
    vincular_hogares,
)


class TestSumarTrimestres:
    def test_dentro_del_mismo_anio(self):
        assert sumar_trimestres(2020, 1, 1) == (2020, 2)

    def test_cruza_anio(self):
        assert sumar_trimestres(2020, 3, 4) == (2021, 3)

    def test_h_cero(self):
        assert sumar_trimestres(2020, 2, 0) == (2020, 2)

    def test_desde_q4(self):
        assert sumar_trimestres(2020, 4, 1) == (2021, 1)


class TestMandatoPresidencial:
    def test_dentro_de_un_mandato(self):
        assert mandato_presidencial(2010, 2) == "CFK I"

    def test_primer_trimestre_de_transicion_queda_con_el_saliente(self):
        # Macri asume 2015-12-10 -- convención: 2015T4 completo queda con CFK II.
        assert mandato_presidencial(2015, 4) == "CFK II"

    def test_trimestre_siguiente_a_la_transicion(self):
        assert mandato_presidencial(2016, 1) == "Macri"

    def test_ultimo_mandato_de_la_tabla(self):
        assert mandato_presidencial(2025, 4) == "Milei"


class TestRegimenIngresoPar:
    def test_ambos_historicos(self):
        assert regimen_ingreso_par(True, True) == "historico"

    def test_ambos_regulares(self):
        assert regimen_ingreso_par(False, False) == "regular"

    def test_mixto(self):
        assert regimen_ingreso_par(True, False) == "mixto"
        assert regimen_ingreso_par(False, True) == "mixto"


class TestTasaVinculacionViviendas:
    def test_vinculacion_parcial(self):
        r = tasa_vinculacion_viviendas({"A", "B", "C", "D"}, {"A", "B", "Z"})
        assert r == {"n_viviendas_t": 4, "n_vinculadas": 2, "tasa": 0.5}

    def test_sin_viviendas_en_t(self):
        r = tasa_vinculacion_viviendas(set(), {"A"})
        assert r["n_viviendas_t"] == 0
        assert r["tasa"] is None

    def test_vinculacion_total(self):
        r = tasa_vinculacion_viviendas({"A", "B"}, {"A", "B", "C"})
        assert r["tasa"] == 1.0


def _hogar(filas):
    return pd.DataFrame(filas, columns=["CODUSU", "NRO_HOGAR", "AGLOMERADO"])


class TestVincularHogares:
    def test_match_exacto(self):
        hogar_t = _hogar([["V1", 1, 10], ["V2", 1, 10]])
        hogar_th = _hogar([["V1", 1, 10], ["V3", 1, 10]])
        pares = vincular_hogares(hogar_t, hogar_th)
        assert list(pares["CODUSU"]) == ["V1"]
        assert pares.iloc[0]["NRO_HOGAR_t"] == pares.iloc[0]["NRO_HOGAR_th"] == 1

    def test_nro_hogar_distinto_no_matchea(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V1", 2, 10]])
        assert vincular_hogares(hogar_t, hogar_th).empty

    def test_aglomerado_distinto_no_matchea(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V1", 1, 20]])
        assert vincular_hogares(hogar_t, hogar_th).empty

    def test_sin_vivienda_en_comun(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V2", 1, 10]])
        assert vincular_hogares(hogar_t, hogar_th).empty

    def test_vivienda_multi_hogar_no_arma_producto_cruzado(self):
        hogar_t = _hogar([["V1", 1, 10], ["V1", 2, 10]])
        hogar_th = _hogar([["V1", 1, 10], ["V1", 3, 10]])
        pares = vincular_hogares(hogar_t, hogar_th)
        assert len(pares) == 1
        assert pares.iloc[0]["NRO_HOGAR_t"] == 1


class TestClasificarViviendasNoVinculadas:
    def test_vinculada(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V1", 1, 10]])
        pares = vincular_hogares(hogar_t, hogar_th)
        r = clasificar_viviendas_no_vinculadas(hogar_t, hogar_th, pares)
        assert r.iloc[0]["estado_vinculacion"] == "vinculada"

    def test_hogar_reemplazado(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V1", 2, 10]])
        pares = vincular_hogares(hogar_t, hogar_th)
        r = clasificar_viviendas_no_vinculadas(hogar_t, hogar_th, pares)
        assert r.iloc[0]["estado_vinculacion"] == "hogar_reemplazado"

    def test_vivienda_sin_par(self):
        hogar_t = _hogar([["V1", 1, 10]])
        hogar_th = _hogar([["V2", 1, 10]])
        pares = vincular_hogares(hogar_t, hogar_th)
        r = clasificar_viviendas_no_vinculadas(hogar_t, hogar_th, pares)
        assert r.iloc[0]["estado_vinculacion"] == "vivienda_sin_par"


def _individual(filas):
    return pd.DataFrame(filas, columns=["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH03", "CH04", "CH06", "ESTADO"])


class TestValidarJefe:
    def test_jefe_consistente(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])
        r = validar_jefe(individual_t, individual_th, pares).iloc[0]
        assert not r["jefe_ambiguo"]
        assert r["mismo_sexo_jefe"]
        assert r["diferencia_edad_jefe"] == 1
        assert not r["cambio_jefatura"]

    def test_cambio_de_sexo_del_jefe_no_es_ambiguo_pero_marca_cambio_jefatura(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 2, 1, 2, 38, 1]])
        r = validar_jefe(individual_t, individual_th, pares).iloc[0]
        assert not r["jefe_ambiguo"]
        assert not r["mismo_sexo_jefe"]
        assert r["cambio_jefatura"]

    def test_sin_jefe_es_ambiguo(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 2, 1, 40, 1]])  # nadie con CH03==1
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])
        r = validar_jefe(individual_t, individual_th, pares).iloc[0]
        assert r["jefe_ambiguo"]
        assert r["cambio_jefatura"]  # ambiguo cuenta como cambio de jefatura

    def test_dos_jefes_es_ambiguo(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 1, 2, 38, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])
        r = validar_jefe(individual_t, individual_th, pares).iloc[0]
        assert r["jefe_ambiguo"]


class TestEmparejarComponentesYTasaCoincidencia:
    def test_empareja_por_numero_de_componente(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 2, 2, 35, 0]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 2, 1, 36, 0]])
        emp = emparejar_componentes(individual_t, individual_th, pares)
        assert len(emp) == 2
        fila_1 = emp[emp["COMPONENTE"] == 1].iloc[0]
        assert fila_1["mismo_sexo"] and fila_1["diferencia_edad"] == 1
        fila_2 = emp[emp["COMPONENTE"] == 2].iloc[0]
        assert not fila_2["mismo_sexo"]  # CH04 cambió de 2 a 1

    def test_componente_que_no_matchea_no_aparece(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 3, 3, 2, 10, 0]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])  # el componente 3 se fue
        emp = emparejar_componentes(individual_t, individual_th, pares)
        assert len(emp) == 1

    def test_tasa_coincidencia_agrega_correctamente(self):
        emp = pd.DataFrame({"CODUSU": ["V1", "V1"], "NRO_HOGAR_t": [1, 1], "NRO_HOGAR_th": [1, 1],
                             "COMPONENTE": [1, 2], "mismo_sexo": [True, False], "diferencia_edad": [1, 5]})
        r = tasa_coincidencia_componente(emp)
        assert r["n_componentes_matcheados"] == 2
        assert r["tasa_mismo_sexo"] == 0.5
        assert r["distribucion_diferencia_edad"] == [1, 5]

    def test_tasa_coincidencia_vacia(self):
        emp = pd.DataFrame(columns=["CODUSU", "NRO_HOGAR_t", "NRO_HOGAR_th", "COMPONENTE", "mismo_sexo", "diferencia_edad"])
        r = tasa_coincidencia_componente(emp)
        assert r["n_componentes_matcheados"] == 0
        assert r["tasa_mismo_sexo"] is None


class TestIdentidadNucleo:
    def test_jefe_persiste_como_jefe(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["identidad_confirmada"]

    def test_jefe_pasa_a_conyuge_intercambio_de_rol(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 2, 2, 1, 41, 1], ["V1", 1, 1, 1, 2, 38, 1]])
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["identidad_confirmada"]

    def test_jefe_distinto_no_confirmada(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 2, 35, 1]])
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert not r["identidad_confirmada"]

    def test_edad_fuera_de_ventana_no_confirmada(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 45, 1]])  # +5 en h=1
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert not r["identidad_confirmada"]

    def test_jefe_ambiguo_en_t_da_na(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 1, 2, 38, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert pd.isna(r["identidad_confirmada"])

    def test_sin_nucleo_en_th_no_confirmada(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 3, 1, 10, 1]])  # nadie con CH03 1 o 2
        r = identidad_nucleo(individual_t, individual_th, pares, h=1).iloc[0]
        assert not r["identidad_confirmada"]

    def test_ventana_de_h4_centrada_en_uno(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1]])  # diferencia_edad=1
        r = identidad_nucleo(individual_t, individual_th, pares, h=4).iloc[0]
        assert r["identidad_confirmada"]


class TestContarAltasBajas:
    def test_sin_cambios(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 2, 2, 35, 0]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 2, 2, 36, 0]])
        r = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_altas"] == 0 and r["n_bajas"] == 0
        assert not r["cambio_composicion"]

    def test_una_alta_y_una_baja(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 3, 2, 10, 0]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 3, 2, 25, 0]])
        r = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_altas"] == 1 and r["n_bajas"] == 1
        assert r["cambio_composicion"]

    def test_no_matchea_por_componente_sino_por_sexo_y_edad(self):
        # el COMPONENTE cambia (2->3) pero sexo+edad en ventana matchean igual
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 2, 2, 2, 35, 0]])
        individual_th = _individual([["V1", 1, 3, 2, 2, 36, 0]])
        r = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_altas"] == 0 and r["n_bajas"] == 0

    def test_hogar_sin_datos_en_th(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 2, 2, 35, 0]])
        individual_th = _individual([["V2", 1, 1, 1, 1, 40, 1]])
        r = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_bajas"] == 2 and r["n_altas"] == 0


class TestContarNacimientos:
    def test_alta_con_ch06_menos_uno_cuenta_como_nacimiento_h1(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 3, 2, -1, 0]])
        r = contar_nacimientos(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_nacimientos"] == 1

    def test_alta_con_otra_edad_no_cuenta_h1(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 3, 2, 10, 0]])
        r = contar_nacimientos(individual_t, individual_th, pares, h=1).iloc[0]
        assert r["n_nacimientos"] == 0

    def test_alta_con_ch06_uno_cuenta_como_nacimiento_h4_pero_no_h1(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th_h4 = _individual([["V1", 1, 1, 1, 1, 44, 1], ["V1", 1, 2, 3, 2, 1, 0]])
        r_h4 = contar_nacimientos(individual_t, individual_th_h4, pares, h=4).iloc[0]
        assert r_h4["n_nacimientos"] == 1

        individual_th_h1 = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 3, 2, 1, 0]])
        r_h1 = contar_nacimientos(individual_t, individual_th_h1, pares, h=1).iloc[0]
        assert r_h1["n_nacimientos"] == 0

    def test_no_altera_el_resultado_de_contar_altas_bajas(self):
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1], ["V1", 1, 2, 3, 2, 10, 0]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 1], ["V1", 1, 2, 3, 2, -1, 0]])
        antes = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        contar_nacimientos(individual_t, individual_th, pares, h=1)
        despues = contar_altas_bajas(individual_t, individual_th, pares, h=1).iloc[0]
        assert antes["n_altas"] == despues["n_altas"]
        assert antes["n_bajas"] == despues["n_bajas"]
        assert antes["cambio_composicion"] == despues["cambio_composicion"]


class TestConstruirPares:
    def test_columnas_y_valores_basicos(self):
        hogar_t = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1], "AGLOMERADO": [10], "ingreso_no_declarado": [False]})
        hogar_th = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1], "AGLOMERADO": [10], "ingreso_no_declarado": [True]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 1, 41, 0]])

        r = construir_pares(hogar_t, hogar_th, individual_t, individual_th,
                             2020, 1, 4, historico_t=False, historico_th=False)
        assert len(r) == 1
        fila = r.iloc[0]
        assert fila["h"] == 4
        assert (fila["ANO4_t"], fila["TRIMESTRE_t"]) == (2020, 1)
        assert (fila["ANO4_th"], fila["TRIMESTRE_th"]) == (2021, 1)
        assert fila["ingreso_declarado_t"] == True  # noqa: E712 -- ingreso_no_declarado_t era False
        assert fila["ingreso_declarado_th"] == False  # noqa: E712 -- ingreso_no_declarado_th era True
        assert fila["hay_ocupado_t"] and not fila["hay_ocupado_th"]
        assert fila["regimen_ingreso"] == "regular"
        assert fila["identidad_confirmada"]  # mismo jefe, diferencia_edad=1 dentro de la ventana de h=4
        assert fila["n_altas"] == 0 and fila["n_bajas"] == 0
        assert not fila["cambio_composicion"]

    def test_sin_pares_devuelve_vacio_con_columnas(self):
        hogar_t = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1], "AGLOMERADO": [10], "ingreso_no_declarado": [False]})
        hogar_th = pd.DataFrame({"CODUSU": ["V2"], "NRO_HOGAR": [1], "AGLOMERADO": [10], "ingreso_no_declarado": [False]})
        individual_t = _individual([["V1", 1, 1, 1, 1, 40, 1]])
        individual_th = _individual([["V2", 1, 1, 1, 1, 41, 1]])
        r = construir_pares(hogar_t, hogar_th, individual_t, individual_th,
                             2020, 1, 1, historico_t=False, historico_th=False)
        assert r.empty
        assert "identidad_confirmada" in r.columns


def test_hash_esquema_panel_es_estable():
    assert hash_esquema_panel() == hash_esquema_panel()


class TestParticionTrainHoldout:
    def test_clasifica_train_embargo_holdout(self):
        pares = pd.DataFrame({
            "ANO4_t": [2022, 2022, 2023], "TRIMESTRE_t": [3, 4, 1],
            "ANO4_th": [2022, 2023, 2023], "TRIMESTRE_th": [4, 1, 2],
        })
        r = particion_train_holdout(pares, holdout_desde=(2023, 1))
        assert list(r) == ["train", "embargo", "holdout"]


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

    def test_excluye_aglomerados_dados_del_pct_pero_no_del_conteo_crudo(self):
        hogar_t = pd.DataFrame({"CODUSU": ["V1", "V2", "V3"], "NRO_HOGAR": [1, 1, 1],
                                 "AGLOMERADO": [1, 8, 8]})
        hogar_th = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1]})
        pares = pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR_t": [1], "NRO_HOGAR_th": [1]})

        sin_excluir = clasificar_vinculacion_trimestre(hogar_t, hogar_th, pares)
        assert sin_excluir["n_vivienda_sin_par"] == 2  # V2, V3 -- conteo crudo intacto
        assert sin_excluir["pct_vivienda_sin_par_crudo"] == pytest.approx(100 * 2 / 3)

        con_exclusion = clasificar_vinculacion_trimestre(
            hogar_t, hogar_th, pares, aglomerados_excluidos_atricion=frozenset({8}))
        assert con_exclusion["n_vivienda_sin_par"] == 2  # el conteo crudo no cambia
        assert con_exclusion["pct_vivienda_sin_par_crudo"] == pytest.approx(0.0)  # V1 (aglomerado 1) está vinculada
