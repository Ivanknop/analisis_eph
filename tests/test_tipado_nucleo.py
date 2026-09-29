"""Tests de `src/tipado_nucleo.py`. Todo con datos sintéticos -- nada acá lee
`data/eph_cache/` ni `data/processed/`."""
import pandas as pd
import pytest

from tipado_nucleo import (
    ESQUEMA_NUCLEO,
    detectar_formato_decimal,
    ingreso_no_declarado,
    parsear_entero_con_blancos,
    parsear_entero_directo,
    parsear_monetario,
    reporte_no_parseables,
    tipar_variable,
)


class TestParsearMonetario:
    def test_coma_decimal(self):
        serie = pd.Series(["9033,33", "22666,67", "0"])
        resultado = parsear_monetario(serie)
        assert resultado.tolist() == pytest.approx([9033.33, 22666.67, 0.0])

    def test_punto_decimal(self):
        # 2021T1/T3: el trimestre entero usa punto, no coma (ver diagnóstico §1).
        serie = pd.Series(["23333.33", "29666.67", "0"])
        resultado = parsear_monetario(serie)
        assert resultado.tolist() == pytest.approx([23333.33, 29666.67, 0.0])

    def test_ya_numerico(self):
        serie = pd.Series([100.0, 200.5])
        resultado = parsear_monetario(serie)
        assert resultado.tolist() == [100.0, 200.5]


class TestDetectarFormatoDecimal:
    def test_solo_coma(self):
        r = detectar_formato_decimal(pd.Series(["9033,33", "0", "22666,67"]))
        assert r == {"tiene_coma": True, "tiene_punto": False, "mezcla": False}

    def test_solo_punto(self):
        r = detectar_formato_decimal(pd.Series(["23333.33", "0", "29666.67"]))
        assert r == {"tiene_coma": False, "tiene_punto": True, "mezcla": False}

    def test_mezcla_es_la_anomalia_real(self):
        r = detectar_formato_decimal(pd.Series(["9033,33", "23333.33"]))
        assert r["mezcla"] is True

    def test_enteros_sin_separador_no_marcan_nada(self):
        r = detectar_formato_decimal(pd.Series(["0", "100", "-9"]))
        assert r == {"tiene_coma": False, "tiene_punto": False, "mezcla": False}

    def test_serie_ya_numerica_sin_ambiguedad(self):
        r = detectar_formato_decimal(pd.Series([100.0, 200.5]))
        assert r == {"tiene_coma": False, "tiene_punto": False, "mezcla": False}

    def test_valor_no_parseable_queda_nulo(self):
        serie = pd.Series(["1234,56", "ver nota"])
        resultado = parsear_monetario(serie)
        assert resultado.iloc[0] == pytest.approx(1234.56)
        assert pd.isna(resultado.iloc[1])


class TestParsearEnteroConBlancos:
    def test_blanco_literal_es_nulo(self):
        serie = pd.Series(["01", "  ", "12"])
        resultado = parsear_entero_con_blancos(serie)
        assert resultado.tolist()[0] == 1
        assert pd.isna(resultado.tolist()[1])
        assert resultado.tolist()[2] == 12

    def test_zero_padding_no_afecta_valor(self):
        serie = pd.Series(["01", "1", "00", "0"])
        resultado = parsear_entero_con_blancos(serie)
        assert resultado.tolist() == [1, 1, 0, 0]

    def test_vacio_es_nulo(self):
        serie = pd.Series(["", "5"])
        resultado = parsear_entero_con_blancos(serie)
        assert pd.isna(resultado.tolist()[0])
        assert resultado.tolist()[1] == 5


class TestParsearEnteroDirecto:
    def test_cast_simple(self):
        serie = pd.Series([1, 2, 3])
        resultado = parsear_entero_directo(serie, "Int8")
        assert str(resultado.dtype) == "Int8"
        assert resultado.tolist() == [1, 2, 3]


class TestReporteNoParseables:
    def test_valor_no_numerico_se_reporta_no_se_descarta(self):
        original = pd.Series(["1234,56", "ver nota", "500"])
        tipada = parsear_monetario(original)
        reporte = reporte_no_parseables(original, tipada, 2020, 1, "hogar", "IPCF")
        assert len(reporte) == 1
        assert reporte[0]["valor_original"] == "ver nota"
        assert reporte[0]["cantidad"] == 1
        assert reporte[0]["variable"] == "IPCF"

    def test_sin_fallos_devuelve_vacio(self):
        original = pd.Series(["100,00", "200,00"])
        tipada = parsear_monetario(original)
        assert reporte_no_parseables(original, tipada, 2020, 1, "hogar", "IPCF") == []

    def test_nulo_legitimo_no_se_reporta(self):
        original = pd.Series(["100,00", None])
        tipada = parsear_monetario(original)
        assert reporte_no_parseables(original, tipada, 2020, 1, "hogar", "IPCF") == []


class TestTiparVariable:
    def test_dispatch_monetario(self):
        df = pd.DataFrame({"IPCF": ["100,50"]})
        assert tipar_variable(df, "IPCF").iloc[0] == pytest.approx(100.50)

    def test_dispatch_entero_con_blancos(self):
        df = pd.DataFrame({"DECCFR": ["01"]})
        assert tipar_variable(df, "DECCFR").iloc[0] == 1

    def test_variable_desconocida_levanta_keyerror(self):
        with pytest.raises(KeyError):
            tipar_variable(pd.DataFrame({"X": [1]}), "X")


class TestNivelEd:
    def test_en_esquema_como_individual_entero_directo(self):
        assert ESQUEMA_NUCLEO["NIVEL_ED"]["base"] == "individual"
        assert ESQUEMA_NUCLEO["NIVEL_ED"]["regla_parseo"] == "entero_directo"

    def test_tipar_variable_dispatch(self):
        df = pd.DataFrame({"NIVEL_ED": [4]})
        resultado = tipar_variable(df, "NIVEL_ED")
        assert str(resultado.dtype) == "Int8"
        assert resultado.iloc[0] == 4


class TestIngresoNoDeclarado:
    def test_deccfr_12_da_true(self):
        hogar = pd.DataFrame({"DECCFR": ["12"], "PONDIH": [0]})
        resultado = ingreso_no_declarado(hogar, es_historico=False)
        assert bool(resultado.iloc[0]) is True

    def test_pondih_cero_con_deccfr_nulo_da_true(self):
        hogar = pd.DataFrame({"DECCFR": [None], "PONDIH": [0]})
        resultado = ingreso_no_declarado(hogar, es_historico=False)
        assert bool(resultado.iloc[0]) is True

    def test_hogar_normal_da_false(self):
        hogar = pd.DataFrame({"DECCFR": ["05"], "PONDIH": [850]})
        resultado = ingreso_no_declarado(hogar, es_historico=False)
        assert bool(resultado.iloc[0]) is False

    def test_historico_da_na(self):
        hogar = pd.DataFrame({"CODUSU": ["1"], "NRO_HOGAR": [1]})  # sin PONDIH
        resultado = ingreso_no_declarado(hogar, es_historico=True)
        assert pd.isna(resultado.iloc[0])

    def test_assert_falla_si_deccfr_12_no_tiene_pondih_cero(self):
        hogar = pd.DataFrame({"DECCFR": ["12"], "PONDIH": [500]})
        with pytest.raises(AssertionError):
            ingreso_no_declarado(hogar, es_historico=False)
