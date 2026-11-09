"""Tests de `src/canasta_retropolada.py` -- fixtures sintéticas chicas con la
forma real de los insumos ya verificados (`dic_2006_gba_metodologia22.csv`,
canastas regionales oficiales). Nada de red."""
import pandas as pd
import pytest

from canasta_retropolada import (
    ANCLA_ABR_2016,
    ANCLA_DIC_2006,
    REGION_GBA,
    agregar_trimestral_por_origen,
    backcast_simple,
    blend_dos_anclas,
    cargar_ancla_abr_2016,
    cargar_ancla_dic_2006,
    construir_canasta_gba_mensual,
    encadenar_indice,
    extender_regiones,
    peso_lineal,
)


def _serie_mensual(valores: dict[str, float]) -> pd.Series:
    return pd.Series({pd.Timestamp(k): v for k, v in valores.items()})


class TestCargarAnclaDic2006:
    def test_lee_cba_y_cbt_de_la_columna_nueva(self, tmp_path):
        csv = tmp_path / "dic_2006.csv"
        csv.write_text(
            "concepto,valor,unidad,region,periodo,pagina,cuadro,fuente\n"
            "cba_adulto_equivalente_metodologia_nueva,149.13,x,GBA,2006-12,15,Cuadro 5,f\n"
            "cbt_adulto_equivalente_metodologia_nueva,389.28,x,GBA,2006-12,15,Cuadro 5,f\n"
            "ice_metodologia_nueva,2.61,x,GBA,2006-12,15,Cuadro 5,f\n"
            "cba_adulto_equivalente_metodologia_anterior,134.14,x,GBA,2006-12,15,Cuadro 5,f\n"
            "cbt_adulto_equivalente_metodologia_anterior,291.08,x,GBA,2006-12,15,Cuadro 5,f\n"
        )
        ancla = cargar_ancla_dic_2006(csv)
        assert ancla == {"cba_adulto_equivalente": pytest.approx(149.13), "cbt_adulto_equivalente": pytest.approx(389.28)}


class TestCargarAnclaAbr2016:
    def test_une_cba_y_cbt_de_abril_2016_por_region(self, tmp_path):
        (tmp_path / "cba_regional_crudo.csv").write_text(
            "indice_tiempo,gran_buenos_aires,cuyo\n2016-03-01,1400.0,1300.0\n2016-04-01,1514.53,1358.29\n")
        (tmp_path / "cbt_regional_crudo.csv").write_text(
            "indice_tiempo,gran_buenos_aires,cuyo\n2016-03-01,3500.0,3400.0\n2016-04-01,3665.17,3504.40\n")
        ancla = cargar_ancla_abr_2016(tmp_path)
        assert set(ancla["REGION"]) == {1, 42}
        fila_gba = ancla[ancla["REGION"] == REGION_GBA].iloc[0]
        assert fila_gba["cba_adulto_equivalente"] == pytest.approx(1514.53)
        assert fila_gba["cbt_adulto_equivalente"] == pytest.approx(3665.17)


class TestEncadenarIndice:
    def test_empalma_exacto_en_el_mes_de_empalme(self):
        indec = _serie_mensual({"2006-11-01": 184.86, "2006-12-01": 186.67, "2007-01-01": 190.0})
        cifra = _serie_mensual({"2007-01-01": 22.03, "2007-02-01": 22.28})
        indice, variacion = encadenar_indice(indec, cifra)
        assert indice[pd.Timestamp("2007-01-01")] == pytest.approx(190.0)
        assert variacion == pytest.approx(190.0 / 186.67 - 1)

    def test_preserva_el_crecimiento_propio_de_cifra_despues_del_empalme(self):
        indec = _serie_mensual({"2006-12-01": 100.0, "2007-01-01": 110.0})
        cifra = _serie_mensual({"2007-01-01": 50.0, "2007-02-01": 55.0})
        indice, _ = encadenar_indice(indec, cifra)
        crecimiento_cifra = indice[pd.Timestamp("2007-02-01")] / indice[pd.Timestamp("2007-01-01")] - 1
        assert crecimiento_cifra == pytest.approx(0.10)

    def test_mantiene_el_tramo_indec_sin_tocar(self):
        indec = _serie_mensual({"2003-01-01": 139.38, "2006-12-01": 186.67, "2007-01-01": 190.0})
        cifra = _serie_mensual({"2007-01-01": 22.03})
        indice, _ = encadenar_indice(indec, cifra)
        assert indice[pd.Timestamp("2003-01-01")] == pytest.approx(139.38)
        assert indice[pd.Timestamp("2006-12-01")] == pytest.approx(186.67)


class TestPesoLineal:
    def test_uno_en_el_inicio(self):
        assert peso_lineal(ANCLA_DIC_2006) == pytest.approx(1.0)

    def test_cero_en_el_fin(self):
        assert peso_lineal(ANCLA_ABR_2016) == pytest.approx(0.0)

    def test_lineal_en_un_punto_medio(self):
        # dic-2006 a abr-2016 son 112 meses; a 56 meses de dic-2006 (ago-2011) w=0.5.
        assert peso_lineal(pd.Timestamp("2011-08-01")) == pytest.approx(0.5)


class TestBackcastSimple:
    def test_devuelve_el_ancla_exacto_en_su_propio_mes(self):
        indice = _serie_mensual({"2006-11-01": 180.0, "2006-12-01": 186.67, "2007-01-01": 190.0})
        c = backcast_simple(149.13, indice, ANCLA_DIC_2006)
        assert c[ANCLA_DIC_2006] == pytest.approx(149.13)

    def test_escala_proporcional_al_indice(self):
        indice = _serie_mensual({"2006-06-01": 93.335, "2006-12-01": 186.67})
        c = backcast_simple(149.13, indice, ANCLA_DIC_2006)
        assert c[pd.Timestamp("2006-06-01")] == pytest.approx(149.13 / 2)


class TestBlendDosAnclas:
    def test_pesos_uno_y_cero_devuelven_cada_ancla_pura(self):
        c_b = _serie_mensual({"2007-01-01": 200.0})
        c_a = _serie_mensual({"2007-01-01": 300.0})
        assert blend_dos_anclas(c_b, c_a, _serie_mensual({"2007-01-01": 1.0}))[pd.Timestamp("2007-01-01")] == pytest.approx(200.0)
        assert blend_dos_anclas(c_b, c_a, _serie_mensual({"2007-01-01": 0.0}))[pd.Timestamp("2007-01-01")] == pytest.approx(300.0)

    def test_promedio_a_medio_camino(self):
        c_b = _serie_mensual({"2007-01-01": 200.0})
        c_a = _serie_mensual({"2007-01-01": 300.0})
        r = blend_dos_anclas(c_b, c_a, _serie_mensual({"2007-01-01": 0.5}))
        assert r[pd.Timestamp("2007-01-01")] == pytest.approx(250.0)


class TestConstruirCanastaGbaMensual:
    def _indice_sintetico(self) -> pd.Series:
        # Índice constante -> la canasta retropolada debe quedar constante en todo
        # el tramo, sea cual sea el peso -- aísla el comportamiento de las anclas.
        meses = pd.date_range("2003-01-01", "2016-12-01", freq="MS")
        return pd.Series(100.0, index=meses)

    def test_continuidad_en_ambos_extremos(self):
        indice = self._indice_sintetico()
        r = construir_canasta_gba_mensual(149.13, 1514.53, indice, "cba_adulto_equivalente")
        fila_dic2006 = r[(r["ANO4"] == 2006) & (r["MES"] == 12)].iloc[0]
        assert fila_dic2006["cba_adulto_equivalente"] == pytest.approx(149.13)
        fila_mar2016 = r[(r["ANO4"] == 2016) & (r["MES"] == 3) & (r["origen"] == "retropolada_2_anclas")].iloc[0]
        # w(marzo-2016)=1/112, casi todo peso en el ancla de abril-2016 -- con
        # índice constante, backcast_simple devuelve el ancla sin importar el mes.
        assert fila_mar2016["cba_adulto_equivalente"] == pytest.approx(1 / 112 * 149.13 + 111 / 112 * 1514.53)

    def test_incluye_la_version_de_un_solo_ancla_para_sensibilidad(self):
        indice = self._indice_sintetico()
        r = construir_canasta_gba_mensual(149.13, 1514.53, indice, "cba_adulto_equivalente")
        una_ancla = r[r["origen"] == "retropolada_1_ancla"]
        assert set(zip(una_ancla["ANO4"], una_ancla["MES"])) == {
            (a, m) for a in range(2007, 2017) for m in range(1, 13) if (a, m) <= (2016, 3)
        }
        # Con índice constante, la versión de un solo ancla es el ancla de abril-2016 en todo el tramo.
        assert una_ancla["cba_adulto_equivalente"].to_numpy() == pytest.approx(1514.53)

    def test_no_incluye_meses_fuera_de_2003_2016(self):
        indice = self._indice_sintetico()
        r = construir_canasta_gba_mensual(149.13, 1514.53, indice, "cba_adulto_equivalente")
        assert r["ANO4"].min() == 2003
        assert (r["ANO4"] < 2003).sum() == 0


class TestExtenderRegiones:
    def test_proporcion_regional_constante_en_todo_el_tramo(self):
        canasta_gba = pd.DataFrame({
            "REGION": [REGION_GBA, REGION_GBA], "ANO4": [2003, 2010], "MES": [1, 6],
            "cba_adulto_equivalente": [100.0, 150.0], "origen": ["retropolada_2_anclas", "retropolada_2_anclas"],
        })
        ancla_abr_2016 = pd.DataFrame({"REGION": [REGION_GBA, 42], "cba_adulto_equivalente": [1514.53, 1358.29]})
        r = extender_regiones(canasta_gba, ancla_abr_2016, "cba_adulto_equivalente")
        ratio_oficial = 1358.29 / 1514.53
        cuyo = r[r["REGION"] == 42].set_index("ANO4")["cba_adulto_equivalente"]
        assert cuyo.loc[2003] == pytest.approx(100.0 * ratio_oficial)
        assert cuyo.loc[2010] == pytest.approx(150.0 * ratio_oficial)

    def test_gba_queda_sin_cambios(self):
        canasta_gba = pd.DataFrame({
            "REGION": [REGION_GBA], "ANO4": [2003], "MES": [1], "cba_adulto_equivalente": [100.0], "origen": ["retropolada_2_anclas"],
        })
        ancla_abr_2016 = pd.DataFrame({"REGION": [REGION_GBA, 42], "cba_adulto_equivalente": [1514.53, 1358.29]})
        r = extender_regiones(canasta_gba, ancla_abr_2016, "cba_adulto_equivalente")
        assert r[r["REGION"] == REGION_GBA]["cba_adulto_equivalente"].iloc[0] == pytest.approx(100.0)


class TestAgregarTrimestralPorOrigen:
    def test_agrega_cada_origen_por_separado(self):
        mensual = pd.DataFrame({
            "REGION": [1] * 6,
            "ANO4": [2010] * 6,
            "MES": [1, 2, 3, 1, 2, 3],
            "cba_adulto_equivalente": [100.0, 110.0, 120.0, 200.0, 220.0, 240.0],
            "origen": ["retropolada_2_anclas"] * 3 + ["retropolada_1_ancla"] * 3,
        })
        r = agregar_trimestral_por_origen(mensual, "cba_adulto_equivalente")
        dos_anclas = r[r["origen"] == "retropolada_2_anclas"].iloc[0]
        una_ancla = r[r["origen"] == "retropolada_1_ancla"].iloc[0]
        assert dos_anclas["cba_adulto_equivalente"] == pytest.approx(110.0)
        assert una_ancla["cba_adulto_equivalente"] == pytest.approx(220.0)
