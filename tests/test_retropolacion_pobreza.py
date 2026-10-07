"""Tests de `src/retropolacion_pobreza.py`. Fixtures sintéticas chicas -- mismo
criterio que `tests/test_canasta_regional.py`."""
import pandas as pd
import pytest

from retropolacion_pobreza import (
    backcast_canasta_larga,
    brecha_puente_uca,
    cargar_cba_cbt_gba_historico,
    cargar_facpce,
    cargar_indice_precios,
    cargar_ppcc_regional,
    cargar_tasas_regionales_historico,
    clasificar_dentro_de_banda,
    clasificar_tramo_cedlas,
    clasificar_tramo_uca,
    construir_canasta_backcast_trimestral,
    construir_canasta_regional_ppcc,
    peso_hogar_historico,
)


def _facpce_csv(tmp_path, filas):
    csv = tmp_path / "facpce.csv"
    df = pd.DataFrame(filas, columns=["fecha", "ipc_facpce"])
    df.to_csv(csv, index=False)
    return csv


def _canasta_cruda_csv(tmp_path, nombre, filas):
    csv = tmp_path / f"{nombre}_regional_crudo.csv"
    df = pd.DataFrame(filas, columns=["indice_tiempo", "gran_buenos_aires"])
    df.to_csv(csv, index=False)
    return csv


class TestCargarFacpce:
    def test_indexado_por_anio_mes_sin_nulos(self, tmp_path):
        _facpce_csv(tmp_path, [["2016-03-01", 80.0], ["2016-04-01", ""], ["2016-05-01", 90.0]])
        r = cargar_facpce(tmp_path / "facpce.csv")
        assert list(r.index) == [(2016, 3), (2016, 5)]


class TestCargarIndicePrecios:
    def test_columna_arbitraria(self, tmp_path):
        csv = tmp_path / "otro_indice.csv"
        pd.DataFrame([["2016-03-01", 10.0], ["2016-04-01", 20.0]], columns=["fecha", "ipc_otro"]).to_csv(csv, index=False)
        r = cargar_indice_precios(csv, "ipc_otro")
        assert r.loc[(2016, 4)] == pytest.approx(20.0)


class TestBackcastCanastaLarga:
    def test_deflacta_por_razon_del_indice_respecto_de_la_ancla(self, tmp_path):
        facpce_csv = _facpce_csv(tmp_path, [["2016-02-01", 44.0], ["2016-03-01", 88.0], ["2016-04-01", 88.0]])
        indice = cargar_facpce(facpce_csv)
        _canasta_cruda_csv(tmp_path, "cba", [["2016-04-01", 1000.0]])
        _canasta_cruda_csv(tmp_path, "cbt", [["2016-04-01", 2000.0]])

        r = backcast_canasta_larga(tmp_path, indice, inicio=(2016, 2), fin=(2016, 3))
        fila_feb = r[r["MES"] == 2].iloc[0]
        fila_mar = r[r["MES"] == 3].iloc[0]
        assert fila_feb["cba_ae"] == pytest.approx(500.0)  # 44/88 = 0.5
        assert fila_feb["cbt_ae"] == pytest.approx(1000.0)
        assert fila_mar["cba_ae"] == pytest.approx(1000.0)  # 88/88 = 1.0, mismo mes que la ancla


class TestConstruirCanastaBackcastTrimestral:
    def test_shape_igual_a_canasta_regional(self, tmp_path):
        facpce_csv = _facpce_csv(tmp_path, [["2016-01-01", 44.0], ["2016-02-01", 44.0], ["2016-03-01", 44.0],
                                             ["2016-04-01", 88.0]])
        indice = cargar_facpce(facpce_csv)
        _canasta_cruda_csv(tmp_path, "cba", [["2016-04-01", 1000.0]])
        _canasta_cruda_csv(tmp_path, "cbt", [["2016-04-01", 2000.0]])

        r = construir_canasta_backcast_trimestral(tmp_path, indice, "a_mismo_trimestre",
                                                    inicio=(2016, 1), fin=(2016, 3))
        fila = r[(r["ANO4"] == 2016) & (r["TRIMESTRE"] == 1)].iloc[0]
        assert fila["cba_ae"] == pytest.approx(500.0)
        assert fila["cbt_ae"] == pytest.approx(1000.0)
        assert fila["canasta_ventana_parcial"] == False  # noqa: E712


class TestPesoHogarHistorico:
    def test_pondera_constante_entre_integrantes(self):
        individual = pd.DataFrame({
            "CODUSU": ["V1", "V1", "V2"], "NRO_HOGAR": [1, 1, 1], "PONDERA": [100, 100, 200],
        })
        r = peso_hogar_historico(individual)
        assert r.loc[("V1", 1)] == 100
        assert r.loc[("V2", 1)] == 200

    def test_pondera_no_constante_levanta_error(self):
        individual = pd.DataFrame({
            "CODUSU": ["V1", "V1"], "NRO_HOGAR": [1, 1], "PONDERA": [100, 150],
        })
        with pytest.raises(ValueError):
            peso_hogar_historico(individual)


class TestClasificarDentroDeBanda:
    def test_dentro(self):
        assert clasificar_dentro_de_banda(5.0, [4.0, 6.0, 5.5]) is True

    def test_fuera(self):
        assert clasificar_dentro_de_banda(7.0, [4.0, 6.0, 5.5]) is False


class TestClasificarTramoCedlas:
    def test_pertinente_sin_ningun_semestre_fuera(self):
        assert clasificar_tramo_cedlas([False, False, False]) == "pertinente"

    def test_con_reservas_un_semestre_aislado(self):
        assert clasificar_tramo_cedlas([False, True, False]) == "pertinente_con_reservas"

    def test_no_pertinente_dos_o_mas(self):
        assert clasificar_tramo_cedlas([True, True, False]) == "no_pertinente"


class TestClasificarTramoUca:
    def test_pertinente_todos_dentro_de_un_desvio(self):
        assert clasificar_tramo_uca({2010: 0.5, 2011: -0.8, 2012: 0.9}) == "pertinente"

    def test_con_reservas_un_anio_entre_uno_y_dos(self):
        assert clasificar_tramo_uca({2010: 0.5, 2011: 1.5, 2012: 0.2}) == "pertinente_con_reservas"

    def test_con_reservas_exactamente_un_anio_fuera_de_dos(self):
        assert clasificar_tramo_uca({2010: 0.5, 2011: 2.5, 2012: 0.2}) == "pertinente_con_reservas"

    def test_no_pertinente_dos_anios_fuera_de_dos(self):
        assert clasificar_tramo_uca({2010: 2.5, 2011: -2.2, 2012: 0.2}) == "no_pertinente"


class TestBrechaPuenteUca:
    def test_media_y_std_sobre_el_puente(self):
        propia = {2016: 30.0, 2017: 28.0, 2018: 32.0}
        uca = {2016: 28.0, 2017: 27.0, 2018: 29.0}
        r = brecha_puente_uca(propia, uca, [2016, 2017, 2018])
        assert r["brechas"] == {2016: 2.0, 2017: 1.0, 2018: 3.0}
        assert r["media"] == pytest.approx(2.0)

    def test_excluir_2020_no_pesa_en_media_pero_queda_en_brechas(self):
        propia = {2019: 30.0, 2020: 50.0, 2021: 31.0}
        uca = {2019: 29.0, 2020: 35.0, 2021: 30.0}
        r = brecha_puente_uca(propia, uca, [2019, 2020, 2021], excluir_2020=True)
        assert set(r["brechas"]) == {2019, 2020, 2021}
        assert r["media"] == pytest.approx(1.0)  # promedio de 2019 (1.0) y 2021 (1.0), sin 2020 (15.0)


def _xlsx_cba_cbt_gba(tmp_path, filas_mes):
    # Mismo shape que `sh-cba2.xls` real (D37): encabezado en varias filas,
    # col0=mes "Abr-2001", col1=CBA, col2=inversa Engel (no usada), col3=CBT.
    filas = [[None]*4]*8 + filas_mes
    df = pd.DataFrame(filas, columns=["mes", "cba", "ice", "cbt"])
    ruta = tmp_path / "sh-cba2.xlsx"
    df.to_excel(ruta, header=False, index=False)
    return ruta


class TestCargarCbaCbtGbaHistorico:
    def test_parsea_mes_y_normaliza_coma_decimal(self, tmp_path):
        ruta = _xlsx_cba_cbt_gba(tmp_path, [
            ["Abr-2001", "63,24", "2,44", "154,30"],
            ["Set-2005", 120.14, 2.16, 259.49],
        ])
        r = cargar_cba_cbt_gba_historico(ruta)
        assert r.loc[(2001, 4), "cba_ae"] == pytest.approx(63.24)
        assert r.loc[(2001, 4), "cbt_ae"] == pytest.approx(154.30)
        # "Set" (no "Sep") en 2005 -- mismo mes 9, variante real del archivo.
        assert r.loc[(2005, 9), "cbt_ae"] == pytest.approx(259.49)

    def test_tolera_espacios_al_final_del_mes(self, tmp_path):
        # el archivo real trae "Sep-2000 "/"Abr-2001 " con espacio final en algunas filas.
        ruta = _xlsx_cba_cbt_gba(tmp_path, [["Abr-2001 ", 63.24, 2.44, 154.30]])
        r = cargar_cba_cbt_gba_historico(ruta)
        assert r.loc[(2001, 4), "cba_ae"] == pytest.approx(63.24)

    def test_ignora_filas_sin_mes(self, tmp_path):
        ruta = _xlsx_cba_cbt_gba(tmp_path, [
            ["Abr-2001", 63.24, 2.44, 154.30],
            ["Nota: metodología...", None, None, None],
        ])
        r = cargar_cba_cbt_gba_historico(ruta)
        assert len(r) == 1


class TestCargarPpccRegional:
    def test_columnas_region_cba_cbt(self, tmp_path):
        ruta = tmp_path / "ppcc.csv"
        pd.DataFrame({"REGION": [1, 44], "cba_abr2001": [63.24, 65.46],
                      "cbt_abr2001": [154.30, 146.93]}).to_csv(ruta, index=False)
        r = cargar_ppcc_regional(ruta)
        assert r[r["REGION"] == 44]["cbt_abr2001"].iloc[0] == pytest.approx(146.93)


class TestConstruirCanastaRegionalPpcc:
    def test_escala_por_el_ratio_region_gba(self, tmp_path):
        cba_cbt_gba = pd.DataFrame({"ANO4": [2003], "MES": [1], "cba_ae": [100.0], "cbt_ae": [200.0]}
                                    ).set_index(["ANO4", "MES"])
        ppcc = pd.DataFrame({"REGION": [1, 44], "cba_abr2001": [63.24, 65.46], "cbt_abr2001": [154.30, 146.93]})
        r = construir_canasta_regional_ppcc(cba_cbt_gba, ppcc, (2003, 1), (2003, 1))
        fila_gba = r[r["REGION"] == 1].iloc[0]
        fila_patagonia = r[r["REGION"] == 44].iloc[0]
        assert fila_gba["cba_ae"] == pytest.approx(100.0)  # ratio 1.0 contra sí mismo
        assert fila_patagonia["cba_ae"] == pytest.approx(100.0 * 65.46 / 63.24)
        assert fila_patagonia["cbt_ae"] == pytest.approx(200.0 * 146.93 / 154.30)


class TestCargarTasasRegionalesHistorico:
    def test_parsea_gba_por_semestre(self, tmp_path):
        filas = [[None] * 9 for _ in range(2)]
        filas.append(["", "", "", "Primer semestre de 2003", None, None, None, None, None])
        filas.append(["", "", "", "Bajo la línea de indigencia", None, None, "Bajo la línea de pobreza", None, None])
        filas.append(["", "", "", "Hogares", "Personas", None, "Hogares", "Personas", None])
        filas.append(["Total aglomerados urbanos", None, None, 20.4, 27.7, None, 42.7, 54.0, None])
        filas.append([None, "Gran Buenos Aires", None, 19.5, 26.5, None, 41.2, 52.3, None])
        df = pd.DataFrame(filas)
        ruta = tmp_path / "pobreza_regional.xlsx"
        df.to_excel(ruta, header=False, index=False)

        r = cargar_tasas_regionales_historico(ruta)
        fila = r[(r["REGION"] == 1) & (r["indicador"] == "pobreza_hogares")].iloc[0]
        assert fila["valor"] == pytest.approx(41.2)
