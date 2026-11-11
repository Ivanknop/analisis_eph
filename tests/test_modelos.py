"""Tests de `modelos`: X del bloque A, validación interna, reajuste, determinismo,
criterios de imputación, escritura y reanudación de `07_predicciones`."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import constantes as cte
import features as ft
import modelos as mo
import validacion as va
from features import CLAVE_PAR

COLUMNAS = ("ratio_linea_pobreza_t", "share_precario_A_t", "situacion_laboral_t", "era", "hay_menor_no_asiste_t")
HIPER = (("feature_fraction", 0.9), ("min_data_in_leaf", 20), ("num_leaves", 7))


def _pares(h: int, desde=(2003, 3), hasta=(2025, 3), por_trimestre: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    filas = []
    for indice in range(desde[0] * 4 + desde[1] - 1, hasta[0] * 4 + hasta[1]):
        anio, trimestre = divmod(indice, 4)
        destino = indice + h
        for k in range(por_trimestre):
            historica = anio <= 2015
            imputado = historica and k % 4 == 0
            ratio = float(rng.lognormal(0.3, 0.6))
            precario = float(rng.uniform()) if k % 9 else np.nan
            filas.append({
                "CODUSU": f"V{(indice // 8) * 1000 + k}", "AGLOMERADO_t": 1 + k % 3, "NRO_HOGAR_t": 1, "NRO_HOGAR_th": 1,
                "ANO4_t": anio, "TRIMESTRE_t": trimestre + 1, "ANO4_th": destino // 4, "TRIMESTRE_th": destino % 4 + 1,
                "h": h, "era": "historica" if historica else "regional_2016",
                "par_valido_simetrica": True, "par_valido_log": True, "par_valido_c": not imputado,
                "par_valido_log_c": not imputado, "imputado_t": imputado, "imputado_th": False,
                "fraccion_max_t": 0.3 if imputado else 0.0, "fraccion_max_th": 0.0,
                "ratio_linea_pobreza_t": ratio, "pobre_t": ratio < 1, "share_precario_A_t": precario,
                "situacion_laboral_t": cte.ORDEN_SITUACION_LABORAL[k % 4],
                "hay_menor_no_asiste_t": pd.NA if k % 11 == 0 else bool(k % 2),
                "pobre_th": bool(rng.uniform() < (0.75 if ratio < 1 else 0.08 + 0.2 * (0 if np.isnan(precario) else precario))),
                "variacion_simetrica": float(rng.normal(0.4 * (1 - ratio) + 0.2 * (k % 4 == 3), 0.3)),
                "variacion_log": float(rng.normal(0.4 * (1 - ratio), 0.3)),
                "peso_normalizado_trimestre": 1.0 / por_trimestre,
            })
    pares = pd.DataFrame(filas)
    pares["hay_menor_no_asiste_t"] = pares["hay_menor_no_asiste_t"].astype("boolean")
    pares["particion"] = va.panel.particion_train_holdout(pares, va.HOLDOUT_DESDE)
    pares["fraccion_max_par"] = va.fraccion_max_par(pares)
    return pares[pares["particion"] == "train"].reset_index(drop=True)


@pytest.fixture(scope="module")
def pares_h1():
    return _pares(1)


def _configuracion(tecnica="gradient_boosting", criterio="imputacion_excluye", objetivo="trayectoria", h=1,
                   penalizacion=None):
    hiper = HIPER if tecnica == "gradient_boosting" else ()
    if tecnica == "lineal_regularizado" and penalizacion is None:
        penalizacion = "l2" if va.es_clasificacion(objetivo) else "ridge"
    return mo.Configuracion(objetivo, "simetrica", h, tecnica, criterio, COLUMNAS, hiper, penalizacion)


def test_columnas_x_bloque_a():
    for h in (1, 4):
        columnas = mo.columnas_x(h)
        assert not ft.nombres_prohibidos(columnas)
        assert "share_precario_A_t" in columnas and "era" in columnas
        assert not set(mo.PRECARIEDAD_FUERA_DE_X) & set(columnas)
        assert set(columnas) <= {f.nombre for f in ft.CATALOGO_FEATURES if f.en_x}
    assert "variacion_simetrica_hasta_t_c" in mo.columnas_x(1)
    assert "variacion_simetrica_hasta_t_c" not in mo.columnas_x(4)


def test_preparar_x_gb_niveles_fijos_y_rechaza_imprevistos(pares_h1):
    x = mo.preparar_x_gb(pares_h1, list(COLUMNAS))
    assert list(x["era"].cat.categories) == cte.ERAS and x["ratio_linea_pobreza_t"].dtype == float
    with pytest.raises(ValueError):
        mo.preparar_x_gb(pares_h1.assign(era="otra"), list(COLUMNAS))


@pytest.mark.parametrize("h", [1, 4])
def test_validacion_interna_del_ultimo_anio_sin_tocar_el_test(h):
    pares = _pares(h)
    anio_test = 2019
    train, test = mo.datos_fold(pares, anio_test, "trayectoria", "simetrica", "imputacion_conserva")
    mascara_es, mascara_val, anio_validacion = mo.particion_validacion_interna(train)
    assert anio_validacion == int(train["ANO4_t"].max()) == (anio_test - 1 if h == 1 else anio_test - 2)
    origen = va.indice_trimestre(train["ANO4_t"], train["TRIMESTRE_t"])
    destino = va.indice_trimestre(train["ANO4_th"], train["TRIMESTRE_th"])
    assert destino[mascara_es].max() < origen[mascara_val].min()
    embargo = ~mascara_es & ~mascara_val & (train["ANO4_t"] != anio_validacion)
    assert origen[embargo].nunique() == h
    assert train.loc[mascara_val, "par_valido_c"].all()
    assert train.loc[(train["ANO4_t"] == anio_validacion) & ~train["par_valido_c"]].index.isin(train.index[~mascara_val]).all()
    claves_val = set(map(tuple, train.loc[mascara_val, CLAVE_PAR].to_numpy()))
    assert not claves_val & set(map(tuple, test[CLAVE_PAR].to_numpy()))
    assert destino[mascara_val].max() < va.indice_trimestre(test["ANO4_t"], test["TRIMESTRE_t"]).min()


def test_iteraciones_reajuste():
    assert mo.iteraciones_reajuste(100, 1000, 1070) == 107
    assert mo.iteraciones_reajuste(100, 1000, 3000) == 150
    assert mo.iteraciones_reajuste(1, 1000, 1000) == 1


def test_gb_reajusta_con_todo_el_train_y_es_determinista(pares_h1):
    train, test = mo.datos_fold(pares_h1, 2018, "trayectoria", "simetrica", "imputacion_excluye")
    a = mo.entrenar_fold(train, _configuracion())
    b = mo.entrenar_fold(train, _configuracion())
    assert a.info["n_final"] == len(train) > a.info["n_es"] + a.info["n_validacion"]
    assert a.info["n_iteraciones_final"] == mo.iteraciones_reajuste(a.info["mejor_iteracion"], a.info["n_es"], len(train))
    assert a.estimador.current_iteration() == a.info["n_iteraciones_final"]
    pd.testing.assert_series_equal(mo.predecir(a, test), mo.predecir(b, test))


def test_conserva_reusa_hiperparametros_y_reelige_iteraciones(pares_h1):
    excluye, _ = mo.datos_fold(pares_h1, 2012, "trayectoria", "simetrica", "imputacion_excluye")
    conserva, _ = mo.datos_fold(pares_h1, 2012, "trayectoria", "simetrica", "imputacion_conserva")
    assert len(conserva) > len(excluye)
    m_excluye = mo.entrenar_fold(excluye, _configuracion())
    m_conserva = mo.entrenar_fold(conserva, _configuracion(criterio="imputacion_conserva"))
    for parametro, valor in HIPER:
        assert m_excluye.estimador.params[parametro] == m_conserva.estimador.params[parametro] == valor
    mascara_es, mascara_val, _ = mo.particion_validacion_interna(conserva)
    assert m_conserva.info["n_validacion"] == int(mascara_val.sum()) and conserva.loc[mascara_val, "par_valido_c"].all()
    assert m_conserva.info["n_final"] == len(conserva)


@pytest.mark.parametrize("objetivo", ["trayectoria", "entrada"])
def test_lineal_elige_fuerza_en_validacion_y_reusa_la_dada(pares_h1, objetivo):
    train, test = mo.datos_fold(pares_h1, 2018, objetivo, "simetrica", "imputacion_excluye")
    elegido = mo.entrenar_fold(train, _configuracion("lineal_regularizado", objetivo=objetivo))
    assert elegido.info["fuerza"] in mo.FUERZAS_LINEAL and len(elegido.info["perdidas_validacion"]) == len(mo.FUERZAS_LINEAL)
    dado = mo.entrenar_fold(train, _configuracion("lineal_regularizado", "imputacion_conserva", objetivo), fuerza=100.0)
    assert dado.info["fuerza"] == 100.0 and "perdidas_validacion" not in dado.info
    prediccion = mo.predecir(elegido, test)
    assert prediccion.notna().all()
    if objetivo == "entrada":
        assert prediccion.between(0, 1).all()


def test_sin_validacion_interna_utilizable_devuelve_none():
    pares = _pares(4)
    train, _ = mo.datos_fold(pares, 2006, "trayectoria", "simetrica", "imputacion_excluye")
    assert mo.entrenar_fold(train, _configuracion(h=4)) is None


def _correr(pares, raiz, folds, **kw):
    return mo.correr_configuracion(pares, folds, _configuracion(**kw), raiz, Path("."), "test")


def test_correr_configuracion_escribe_verificado_y_reanuda_igual(pares_h1, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "_salida_git", lambda repo, *args: "")
    folds = [2010, 2016, 2021]
    completo = _correr(pares_h1, tmp_path / "completo", folds)
    _correr(pares_h1, tmp_path / "reanudado", folds[:1])
    reanudado = _correr(pares_h1, tmp_path / "reanudado", folds)
    assert reanudado["reanudado"].tolist() == [True, False, False]
    configuracion = _configuracion()
    a = mo.leer_predicciones(tmp_path / "completo", configuracion)
    b = mo.leer_predicciones(tmp_path / "reanudado", configuracion)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["fold"]) == {str(f) for f in folds}
    assert set(a.loc[a["fold"] == "2010", "conjunto_evaluacion"]) == {"limpio", "imputados"}
    assert set(a.loc[a["fold"] == "2021", "conjunto_evaluacion"]) == {"limpio"}
    columnas_ajuste = ["n_es", "n_validacion", "n_final", "mejor_iteracion", "n_iteraciones_final"]
    pd.testing.assert_frame_equal(completo[columnas_ajuste], reanudado[columnas_ajuste])


def test_rehacer_reescribe_y_criterio_cambia_el_id(pares_h1, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "_salida_git", lambda repo, *args: "")
    _correr(pares_h1, tmp_path, [2016])
    rehecho = mo.correr_configuracion(pares_h1, [2016], _configuracion(), tmp_path, Path("."), "test", rehacer=True)
    assert not rehecho["reanudado"].any()
    assert mo.id_configuracion(_configuracion()) != mo.id_configuracion(_configuracion(criterio="imputacion_conserva"))


def test_fold_corrupto_no_cuenta_como_completo(pares_h1, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "_salida_git", lambda repo, *args: "")
    _correr(pares_h1, tmp_path, [2016])
    directorio = mo.directorio_configuracion(tmp_path, _configuracion(), mo.id_configuracion(_configuracion()))
    ruta = mo.ruta_fold(directorio, 2016)
    pd.read_parquet(ruta).assign(ultimo_destino_train=2030 * 4).to_parquet(ruta)
    assert not mo.fold_completo(directorio, 2016)


def test_fold_que_falla_se_registra_y_sigue_pero_la_fuga_corta(pares_h1, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "_salida_git", lambda repo, *args: "")
    original = mo.entrenar_fold

    def falla_en_2016(train, configuracion, fuerza=None):
        if train["ANO4_t"].max() == 2015:
            raise RuntimeError("falla simulada")
        return original(train, configuracion, fuerza)

    monkeypatch.setattr(mo, "entrenar_fold", falla_en_2016)
    registro = _correr(pares_h1, tmp_path, [2016, 2021])
    assert registro["error"].notna().tolist() == [True, False]
    assert not mo.fold_completo(mo.directorio_configuracion(tmp_path, _configuracion(), mo.id_configuracion(_configuracion())), 2016)

    def fuga(*args, **kwargs):
        raise va.FugaHoldoutError("fuga simulada")

    monkeypatch.setattr(mo, "entrenar_fold", fuga)
    with pytest.raises(va.FugaHoldoutError):
        _correr(pares_h1, tmp_path / "fuga", [2016])


def test_evaluar_fold_comparado_diferencias_pareadas():
    rng = np.random.default_rng(4)
    n = 600
    y = pd.Series((rng.uniform(size=n) < 0.3).astype(float))
    predicciones = {f"{t}|{c}": pd.Series(np.clip(0.3 + rng.normal(0, s, n), 0.01, 0.99))
                    for t, s in (("decil_ratio", 0.05), ("gradient_boosting", 0.1)) for c in mo.CRITERIOS_BLOQUE_A}
    vivienda = pd.DataFrame({"CODUSU": np.repeat(np.arange(n // 3), 3), "AGLOMERADO_t": 1})
    tabla = mo.evaluar_fold_comparado(predicciones, y, "entrada", {c: 0.3 for c in mo.CRITERIOS_BLOQUE_A}, vivienda,
                                      n_boot=20).set_index(["configuracion", "metrica"])["valor"]
    diferencia = tabla[("gradient_boosting|conserva_menos_excluye", "log_loss")]
    assert diferencia == pytest.approx(tabla[("gradient_boosting|imputacion_conserva", "log_loss")]
                                       - tabla[("gradient_boosting|imputacion_excluye", "log_loss")])
    assert ("gradient_boosting|imputacion_excluye", "auc_roc__menos_decil") in tabla.index


def test_resumir_grilla_marca_borde():
    tabla = pd.DataFrame([{"objetivo_modelado": "entrada", "variable": "simetrica", "h": 1, "anio_test": a,
                           "combinacion": i, **g, "mejor_iteracion": 100, "perdida_validacion": 1.0 + i / 100}
                          for i, g in enumerate(mo.GRILLA_GB) for a in (2020, 2021)])
    resumen = mo.resumir_grilla_gb(tabla)
    elegida = resumen[resumen["elegida"]].iloc[0]
    assert elegida["combinacion"] == 0 and elegida["en_borde"]


@pytest.mark.parametrize("nombre", sorted(va.FUNCIONES_SOLO_TRAIN))
def test_funciones_solo_train_de_modelos_rechazan_holdout(nombre):
    pares = _pares(1, hasta=(2025, 3))
    completos = pd.concat([pares, pares.assign(ANO4_t=2024, ANO4_th=2024)], ignore_index=True)
    with pytest.raises(va.FugaHoldoutError):
        va.FUNCIONES_SOLO_TRAIN[nombre](completos[completos["ANO4_t"] == 2024])


def test_trayectoria_log_lleva_decil_de_referencia():
    tecnicas = {c.tecnica for c in mo.configuraciones_bloque_a("trayectoria", "log", 1)}
    assert tecnicas == {"decil_ratio", "gradient_boosting"}
