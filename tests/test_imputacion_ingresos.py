import numpy as np
import pandas as pd
import pytest

from imputacion_ingresos import (
    VARIABLES_MONTO, clasificar_estado_imputacion_par, contar_codigos_especiales, descomponer_codigo_imputacion,
    fraccion_itf_imputada, limpiar_monto, monto_imputado_persona, pendiente_regresion_media,
    reponderar_dentro_de_celda, resumen_dispersion, tramo_fraccion, verificar_idimph_contra_idimpp,
    verificar_identidad_ocupacion_principal, verificar_identidades_ingreso, verificar_monto_marcado,
)


def _persona(idimpp="00000", codusu="A", nro_hogar=1, **montos):
    fila = {v: 0.0 for v in VARIABLES_MONTO}
    fila.update(montos)
    fila.update({"CODUSU": codusu, "NRO_HOGAR": nro_hogar, "COMPONENTE": 1, "IDIMPP": idimpp})
    return fila


def _individual(*filas, **columnas_extra):
    df = pd.DataFrame(list(filas))
    for nombre, valores in columnas_extra.items():
        df[nombre] = valores
    return df


def _asalariado_i1a(idimpp, d1=1000.0, d4=200.0, f1=0.0, j1=0.0):
    # I1a: D4 ya está dentro de D1 y el aguinaldo J1 va por fuera de P21 (entra a P47T).
    return _persona(idimpp, P21=d1 + f1, PP08D1=d1, PP08D4=d4, PP08F1=f1, PP08J1=j1, P47T=d1 + f1 + j1)


def _asalariado_i1b(idimpp, d1=1000.0, d4=200.0):
    return _persona(idimpp, P21=d1 + d4, PP08D1=d1, PP08D4=d4, P47T=d1 + d4)


def test_limpiar_monto_codigos_especiales():
    resultado = limpiar_monto(pd.Series(["-7", "-8", "-9", "1500,5", None]))
    assert resultado.iloc[:2].tolist() == [0.0, 0.0]
    assert pd.isna(resultado.iloc[2]) and pd.isna(resultado.iloc[4])
    assert resultado.iloc[3] == 1500.5


def test_contar_codigos_especiales():
    df = pd.DataFrame({"PP06C": ["-7", "-8", "-8", "-9", "10"]})
    fila = contar_codigos_especiales(df, ["PP06C", "AUSENTE"]).iloc[0]
    assert (fila["n_menos7"], fila["n_menos8"], fila["n_menos9"]) == (1, 2, 1)


def test_descomponer_codigo_valido_e_invalido():
    resultado = descomponer_codigo_imputacion(pd.Series(["00000", "30071", "60000", "1234", "00080"]))
    assert resultado["codigo_invalido"].tolist() == [False, False, True, True, True]
    assert resultado.loc[1, ["pos_1", "pos_4", "pos_5"]].tolist() == [3, 7, 1]
    assert resultado["alguna_imputacion"].iloc[0] == False  # noqa: E712
    assert resultado["alguna_imputacion"].iloc[1] == True  # noqa: E712
    assert pd.isna(resultado["alguna_imputacion"].iloc[2])


def test_identidad_i1a_i1b_independiente():
    individual = _individual(
        _asalariado_i1a("00000"), _asalariado_i1b("00000", d1=1000, d4=200),
        _persona(P21=500.0, PP06C=500.0), _persona(P21=900.0, PP08D1=100.0))
    identidad = verificar_identidad_ocupacion_principal(individual)
    assert identidad["variante_i1"].iloc[:3].tolist() == ["a", "b", "independiente"]
    assert pd.isna(identidad["variante_i1"].iloc[3])
    assert identidad["cumple_i1"].tolist() == [True, True, False, False]


def test_i1a_digito_2_suma_d4():
    persona = monto_imputado_persona(_individual(_asalariado_i1a("20000", d1=1000, d4=200, f1=50)))
    # D1 no está marcado: D4 entra aunque esté contenido en D1.
    assert persona["monto_imputado_min"].iloc[0] == persona["monto_imputado_max"].iloc[0] == 250.0


def test_i1a_digito_3_no_cuenta_d4_dos_veces():
    persona = monto_imputado_persona(_individual(_asalariado_i1a("30000", d1=1000, d4=200, f1=50)))
    assert persona["monto_imputado_min"].iloc[0] == persona["monto_imputado_max"].iloc[0] == 1050.0


def test_i1b_digito_3_suma_todos():
    persona = monto_imputado_persona(_individual(_asalariado_i1b("30000", d1=1000, d4=200)))
    assert persona["monto_imputado_min"].iloc[0] == persona["monto_imputado_max"].iloc[0] == 1200.0


def test_respaldo_p21_suma_pp08j_en_max_y_en_min():
    # P21 no cumple I1 (componentes no suman) -> pos. 1 cae al respaldo P21; J1 está fuera de P21.
    individual = _individual(_persona("11000", P21=2000.0, PP08D1=1000.0, PP08J1=300.0, P47T=2300.0))
    persona = monto_imputado_persona(individual)
    assert persona["usa_respaldo_p21"].iloc[0]
    assert persona["monto_imputado_max"].iloc[0] == 2300.0
    assert persona["monto_imputado_min"].iloc[0] == 300.0


def test_i1a_se_cumple_con_aguinaldo_fuera_de_p21():
    individual = _individual(_asalariado_i1a("11000", d1=1000, d4=0, j1=500))
    assert verificar_identidad_ocupacion_principal(individual)["variante_i1"].iloc[0] == "a"
    persona = monto_imputado_persona(individual)
    assert persona["monto_imputado_min"].iloc[0] == persona["monto_imputado_max"].iloc[0] == 1500.0


def test_respaldo_p21_suma_bonificaciones_j2_j3():
    individual = _individual(_persona("12000", P21=900.0, PP08D1=100.0, PP08J2=50.0, PP08J3=25.0, P47T=975.0))
    persona = monto_imputado_persona(individual)
    assert persona["monto_imputado_max"].iloc[0] == 975.0
    assert persona["monto_imputado_min"].iloc[0] == 75.0


def test_pos5_a_p21_sin_doble_conteo():
    individual = _individual(_asalariado_i1a("10001", d1=1000, d4=0), PJ1_1=[1], PJ2_1=[0], PJ3_1=[0])
    persona = monto_imputado_persona(individual)
    assert persona["monto_imputado_max"].iloc[0] == 1000.0
    assert persona["monto_imputado_min"].iloc[0] == 1000.0


def test_pos5_sin_pj_es_na_en_max():
    sin_columnas = monto_imputado_persona(_individual(_persona("00001", P21=800.0)))
    con_cero = monto_imputado_persona(_individual(_persona("00001", P21=800.0), PJ1_1=[0], PJ2_1=[0], PJ3_1=[0]))
    for persona in (sin_columnas, con_cero):
        assert pd.isna(persona["monto_imputado_max"].iloc[0])
        assert persona["max_na_plan_sin_pj"].iloc[0]
        assert persona["monto_imputado_min"].iloc[0] == 0.0
    assert verificar_monto_marcado(_individual(_persona("00001", P21=800.0)))["pos5_sin_pj"].iloc[0]


def test_pos4_residuo_y_grupo_amplio():
    con_residuo = _persona("00040", T_VI=500.0, V2_M=300.0, P47T=500.0)
    sin_residuo = _persona("00040", T_VI=300.0, V2_M=300.0, P47T=300.0)
    grupo = _persona("00030", T_VI=700.0, V2_M=300.0, V5_M=150.0, V12_M=250.0, P47T=700.0)
    individual = _individual(con_residuo, sin_residuo, grupo)
    persona = monto_imputado_persona(individual)
    assert persona["monto_imputado_max"].tolist() == [200.0, 0.0, 700.0]
    assert persona["monto_imputado_min"].tolist() == [0.0, 0.0, 300.0]
    assert verificar_monto_marcado(individual)["pos4_marcado_sin_monto"].tolist() == [False, True, False]


def test_menos9_es_na_y_causa():
    persona = monto_imputado_persona(_individual(_persona("00100", TOT_P12=-9.0)))
    assert pd.isna(persona["monto_imputado_max"].iloc[0]) and pd.isna(persona["monto_imputado_min"].iloc[0])
    assert persona["max_na_menos9"].iloc[0]


def test_fraccion_hogar_min_max_e_itf_cero():
    individual = _individual(
        _persona("00100", codusu="A", TOT_P12=400.0, P47T=400.0),
        _persona("00020", codusu="A", T_VI=600.0, V5_M=600.0, P47T=600.0),
        _persona("00000", codusu="B"))
    hogar = pd.DataFrame({"CODUSU": ["A", "B"], "NRO_HOGAR": [1, 1], "ITF": [1000.0, 0.0],
                          "IDIMPH": ["00110", "00000"]})
    fraccion = fraccion_itf_imputada(hogar, individual)
    assert fraccion["fraccion_min"].iloc[0] == pytest.approx(0.4)
    assert fraccion["fraccion_max"].iloc[0] == pytest.approx(1.0)
    assert fraccion["fraccion_min"].iloc[0] <= fraccion["fraccion_max"].iloc[0]
    assert pd.isna(fraccion["fraccion_min"].iloc[1]) and fraccion["itf_cero"].iloc[1]


def test_idimph_contra_idimpp():
    individual = _individual(_persona("10000", codusu="A"), _persona("00000", codusu="B"))
    hogar = pd.DataFrame({"CODUSU": ["A", "B"], "NRO_HOGAR": [1, 1], "IDIMPH": ["10000", "00010"]})
    resultado = verificar_idimph_contra_idimpp(hogar, individual)
    assert resultado["discrepa_pos_1"].tolist() == [False, False]
    assert resultado["discrepa_pos_4"].tolist() == [False, True]


def test_identidades_i2_i3():
    individual = _individual(_persona(codusu="A", P21=100.0, T_VI=50.0, P47T=150.0),
                             _persona(codusu="A", P21=10.0, P47T=99.0),
                             _persona(codusu="A", P21=100.0, PP08J1=50.0, PP08J2=20.0, PP08J3=5.0, P47T=175.0),
                             _persona(codusu="A", P21=100.0, PP08J1=50.0, P47T=100.0))
    hogar = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "ITF": [524.0]})
    i2, i3 = verificar_identidades_ingreso(hogar, individual)
    assert i2.tolist() == [True, False, True, False]
    assert i3.tolist() == [True]


def test_identidad_i2_na_si_falta_un_componente():
    individual = _individual(_persona(codusu="A", P21=100.0, PP08J1=np.nan, P47T=100.0))
    hogar = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "ITF": [100.0]})
    i2, _ = verificar_identidades_ingreso(hogar, individual)
    assert pd.isna(i2.iloc[0])


def test_tramo_fraccion():
    tramos = tramo_fraccion(pd.Series([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.3, np.nan]))
    assert tramos.iloc[:7].tolist() == ["0", "(0,0.1]", "(0,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1]", ">1"]
    assert pd.isna(tramos.iloc[7])


def test_estado_par():
    estado = clasificar_estado_imputacion_par(pd.Series([False, True, False, True, pd.NA], dtype="boolean"),
                                              pd.Series([False, False, True, True, True], dtype="boolean"))
    assert estado.iloc[:4].tolist() == ["ninguna", "solo_t", "solo_th", "ambas"]
    assert pd.isna(estado.iloc[4])


def test_resumen_dispersion_y_pendiente():
    resumen = resumen_dispersion(pd.Series([1.0, 2.0, 3.0, 4.0, np.nan]))
    assert resumen["n"] == 4 and resumen["iqr"] == pytest.approx(1.5)
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert pendiente_regresion_media(x, -0.5 * x + 2) == pytest.approx(-0.5)
    assert np.isnan(pendiente_regresion_media(pd.Series([1.0, 1.0]), pd.Series([1.0, 2.0])))


def test_reponderar_conserva_total_por_celda():
    df = pd.DataFrame({"celda": ["a", "a", "a", "b", "b"], "w": [1.0, 2.0, 3.0, 4.0, 4.0]})
    excluir = pd.Series([True, False, False, False, True])
    nuevo = reponderar_dentro_de_celda(df, "w", excluir, ["celda"])
    assert nuevo.groupby(df["celda"]).sum().tolist() == pytest.approx([6.0, 8.0])
    assert nuevo[excluir].eq(0).all()
