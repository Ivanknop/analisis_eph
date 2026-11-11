"""Tests de `validacion`: candado del holdout, folds walk-forward, criterios de
imputación, métricas, bootstrap y verificación de predicciones. Los de datos
reales se saltan si las capas no están."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import constantes as cte
import universo_entrenamiento as ue
import validacion as va
from features import CLAVE_PAR

REPO = Path(__file__).resolve().parents[1]
TRAYECTORIA = REPO / "data" / "05_trayectoria_distancia"
FEATURES = REPO / "data" / "06_features"
REQUIERE_CAPAS = pytest.mark.skipif(
    not (ue.particiones(TRAYECTORIA, 1) and ue.particiones(FEATURES, 1)),
    reason="requiere data/05_trayectoria_distancia/ y data/06_features/ reales")


def _pares(h: int, desde=(2003, 3), hasta=(2025, 3), por_trimestre: int = 6, seed: int = 0) -> pd.DataFrame:
    """Pares sintéticos: `por_trimestre` por origen; la vivienda se repite a 4
    trimestres de distancia (rotación 2-2-2); históricos con 1 de cada 3 imputado."""
    rng = np.random.default_rng(seed)
    filas = []
    for indice in range(desde[0] * 4 + desde[1] - 1, hasta[0] * 4 + hasta[1]):
        anio, trimestre = divmod(indice, 4)
        destino = indice + h
        for k in range(por_trimestre):
            historica = anio <= 2015
            imputado = historica and k % 3 == 0
            ratio = float(rng.uniform(0.3, 3.0))
            filas.append({
                "CODUSU": f"V{(indice // 8) * 100 + k}", "AGLOMERADO_t": 1 + k % 2,
                "NRO_HOGAR_t": 1, "NRO_HOGAR_th": 1, "ANO4_t": anio, "TRIMESTRE_t": trimestre + 1,
                "ANO4_th": destino // 4, "TRIMESTRE_th": destino % 4 + 1, "h": h,
                "era": "historica" if historica else "regional_2016",
                "par_valido_simetrica": True, "par_valido_log": True,
                "par_valido_c": not imputado, "par_valido_log_c": not imputado,
                "imputado_t": imputado, "imputado_th": False,
                "fraccion_max_t": 0.8 if (imputado and k % 2 == 0) else (0.2 if imputado else 0.0), "fraccion_max_th": 0.0,
                "ratio_linea_pobreza_t": ratio, "pobre_t": ratio < 1,
                "pobre_th": bool(rng.uniform() < (0.7 if ratio < 1 else 0.1)),
                "variacion_simetrica": float(rng.normal(0.3 * (1 - ratio), 0.3)), "variacion_log": float(rng.normal()),
                "peso_normalizado_trimestre": 1.0 / por_trimestre + 0.01 * k,
            })
    pares = pd.DataFrame(filas)
    pares["particion"] = va.panel.particion_train_holdout(pares, va.HOLDOUT_DESDE)
    pares["fraccion_max_par"] = va.fraccion_max_par(pares)
    return pares


def _train(h: int, **kwargs) -> pd.DataFrame:
    pares = _pares(h, **kwargs)
    return pares[pares["particion"] == "train"].reset_index(drop=True)


def _parcial(pares: pd.DataFrame, particion: str) -> pd.DataFrame:
    return pares[pares["particion"] == particion].reset_index(drop=True)



def test_exigir_sin_holdout_rechaza_holdout_y_embargo_y_acepta_train():
    pares = _pares(1)
    va.exigir_sin_holdout(_parcial(pares, "train"))
    for particion in ("holdout", "embargo"):
        with pytest.raises(va.FugaHoldoutError):
            va.exigir_sin_holdout(_parcial(pares, particion))
    with pytest.raises(TypeError):
        va.exigir_sin_holdout(pd.DataFrame({"ANO4_t": [2010]}))


@pytest.mark.parametrize("nombre", sorted(va.FUNCIONES_SOLO_TRAIN))
@pytest.mark.parametrize("particion", ["holdout", "embargo"])
def test_funciones_solo_train_rechazan_holdout_y_embargo(nombre, particion):
    fuera = _parcial(_pares(4), particion)
    with pytest.raises(va.FugaHoldoutError):
        va.FUNCIONES_SOLO_TRAIN[nombre](fuera)


def test_funciones_solo_train_incluyen_ajuste_y_seleccion():
    esperadas = {"conjunto_train", "generar_folds", "asignar_fold", "ajustar_linea_base",
                 "ajustar_preprocesamiento_lineal", "evaluar_lineas_base", "perdida_agrupada_folds"}
    assert esperadas <= set(va.FUNCIONES_SOLO_TRAIN)


def test_cargar_pares_modelado_devuelve_solo_train_limpio(monkeypatch):
    monkeypatch.setattr(va, "_leer_pares", lambda *a, **k: _pares(1))
    cargados = va.cargar_pares_modelado(Path("x"), Path("y"), 1)
    assert (cargados["particion"] == "train").all() and cargados["par_valido_c"].all()
    va.exigir_sin_holdout(cargados)


def test_criterio_por_defecto_es_imputacion_excluye():
    for funcion in (va.cargar_pares_modelado, va.conjunto_train, va.mascara_train, va.prevalencias):
        assert va.parametros_por_defecto(funcion)["criterio_imputacion"] == "imputacion_excluye"
    assert cte.CRITERIO_IMPUTACION_PRINCIPAL == "imputacion_excluye"
    assert cte.CRITERIOS_IMPUTACION_EXPERIMENTO[0] == "imputacion_excluye"
    train, _ = va.conjunto_train(_train(1))
    assert train["par_valido_c"].all() and not train["imputado_t"].any()


def test_abrir_holdout_exige_notebook_motivo_y_registra(monkeypatch, tmp_path):
    monkeypatch.setattr(va, "_leer_pares", lambda *a, **k: _pares(1))
    corridas = tmp_path / "corridas.csv"
    argumentos = dict(archivo_corridas=corridas, repo=tmp_path)
    with pytest.raises(PermissionError):
        va.abrir_holdout(Path("x"), Path("y"), 1, notebook="09_modelos_trayectoria", motivo="m", **argumentos)
    with pytest.raises(ValueError):
        va.abrir_holdout(Path("x"), Path("y"), 1, notebook=va.NOTEBOOK_HOLDOUT, motivo="", **argumentos)
    assert not corridas.exists()
    holdout = va.abrir_holdout(Path("x"), Path("y"), 1, notebook=va.NOTEBOOK_HOLDOUT, motivo="final", **argumentos)
    assert (holdout["particion"] == "holdout").all() and len(holdout) > 0
    registro = pd.read_csv(corridas, dtype=str)
    assert len(registro) == 1 and json.loads(registro.loc[0, "parametros_json"])["evento"] == va.EVENTO_APERTURA
    with pytest.raises(PermissionError):
        va.abrir_holdout(Path("x"), Path("y"), 1, notebook=va.NOTEBOOK_HOLDOUT, motivo="otra", **argumentos)
    va.abrir_holdout(Path("x"), Path("y"), 1, notebook=va.NOTEBOOK_HOLDOUT, motivo="otra", reapertura="bug", **argumentos)
    va.abrir_holdout(Path("x"), Path("y"), 4, notebook=va.NOTEBOOK_HOLDOUT, motivo="h4", **argumentos)
    assert len(pd.read_csv(corridas, dtype=str)) == 3


def test_notebooks_previos_al_14_no_abren_el_holdout():
    notebooks = [p for p in (REPO / "notebooks").glob("*.ipynb")
                 if p.name[:2].isdigit() and 8 <= int(p.name[:2]) <= 13]
    for ruta in notebooks:
        assert "abrir_holdout" not in ruta.read_text(encoding="utf-8"), ruta.name



@pytest.mark.parametrize("h", [1, 4])
def test_folds_sin_superposicion_ni_fuga_y_embargo_de_h_trimestres(h):
    train = _train(h)
    folds = va.generar_folds(train, h)
    assert len(folds) > 5
    origen = va.indice_trimestre(train["ANO4_t"], train["TRIMESTRE_t"])
    destino = va.indice_trimestre(train["ANO4_th"], train["TRIMESTRE_th"])
    for anio in folds["anio_test"]:
        rol = va.asignar_fold(train, int(anio))
        claves_train = set(map(tuple, train.loc[rol == "train", CLAVE_PAR].to_numpy()))
        claves_test = set(map(tuple, train.loc[rol == "test", CLAVE_PAR].to_numpy()))
        assert not claves_train & claves_test
        assert destino[rol == "train"].max() < origen[rol == "test"].min()
        assert origen[rol == "embargo"].nunique() == min(h, origen[(origen < anio * 4)].nunique())
        assert (destino[rol == "test"] <= va.INDICE_ULTIMO_DESTINO_TRAIN).all()
    assert (folds["n_trimestres_origen_train"] >= va.MIN_TRIMESTRES_TRAIN).all()
    assert folds["anio_test"].max() == (2022 if h == 1 else 2021)


def test_folds_respetan_minimo_y_universo_2016_es_subconjunto():
    train = _train(4)
    amplio = va.generar_folds(train, 4)
    regular = va.generar_folds(train[ue.en_universo(train, "universo_2016_2025")], 4)
    assert set(regular["anio_test"]) <= set(amplio["anio_test"])
    assert regular["anio_test"].min() == 2018
    assert va.generar_folds(train, 4, min_trimestres_train=1000).empty


def test_folds_marcan_contexto():
    folds = va.generar_folds(_train(4), 4).set_index("anio_test")
    assert folds.loc[2020, "contexto_fold"] == "pandemia"
    assert folds.loc[2019, "destino_en_contexto"] == "pandemia"
    assert folds.loc[2010, "periodo_intervenido"] and not folds.loc[2017, "periodo_intervenido"]


def test_sin_viviendas_de_train():
    train = pd.DataFrame({"CODUSU": ["A", "B"], "AGLOMERADO_t": [1, 1]})
    test = pd.DataFrame({"CODUSU": ["A", "A", "C"], "AGLOMERADO_t": [1, 2, 1]})
    assert va.sin_viviendas_de_train(test, train).tolist() == [False, True, True]



def test_mascara_train_por_criterio():
    pares = _train(1)
    conserva = va.mascara_train(pares, criterio_imputacion="imputacion_conserva")
    excluye = va.mascara_train(pares, criterio_imputacion="imputacion_excluye")
    fraccion = va.mascara_train(pares, criterio_imputacion="imputacion_excluye_fraccion_max_0.5")
    assert excluye.sum() < fraccion.sum() < conserva.sum()
    assert (pares.loc[fraccion & ~excluye, "fraccion_max_par"] <= 0.5).all()
    with pytest.raises(ValueError):
        va.mascara_train(pares, criterio_imputacion="C_total")


def test_fraccion_max_par_regla_05d():
    pares = pd.DataFrame({"imputado_t": [False, True, True, False], "imputado_th": [False, False, True, False],
                          "fraccion_max_t": [np.nan, 0.3, 0.3, 0.0], "fraccion_max_th": [0.0, 0.0, np.nan, 0.0]})
    resultado = va.fraccion_max_par(pares)
    assert resultado.iloc[:2].tolist() == [0.0, 0.3] and np.isnan(resultado.iloc[2])


@pytest.mark.parametrize("ponderacion", cte.PONDERACIONES)
def test_reponderado_conserva_el_total_por_celda(ponderacion):
    pares = _train(1)
    train, peso = va.conjunto_train(pares, criterio_imputacion="imputacion_excluye_reponderado", ponderacion=ponderacion)
    assert train["par_valido_c"].all() and peso.index.equals(train.index)
    universo = pares[va.mascara_train(pares, criterio_imputacion="imputacion_conserva")]
    base = 1.0 if ponderacion == "sin_ponderar" else universo["peso_normalizado_trimestre"]
    total = universo.assign(_w=base).groupby(va.CELDAS_REPONDERACION)["_w"].sum()
    retenido = train.assign(_w=peso).groupby(va.CELDAS_REPONDERACION)["_w"].sum()
    pd.testing.assert_series_equal(retenido, total.loc[retenido.index], check_names=False)


def test_mascara_evaluacion_limpio_e_imputados_son_disjuntos():
    pares = _train(1)
    limpio, imputados = va.mascara_evaluacion(pares), va.mascara_evaluacion(pares, conjunto="imputados")
    assert not (limpio & imputados).any() and (pares.loc[imputados, "era"] == "historica").all()



def test_metricas_regresion_a_mano():
    y, p, base = pd.Series([0.0, 1.0, 2.0]), pd.Series([0.0, 1.0, 1.0]), pd.Series([1.0, 1.0, 1.0])
    m = va.metricas_regresion(y, p, media_train=1.0, prediccion_base=base)
    assert m["mae"] == pytest.approx(1 / 3) and m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["r2_vs_media_train"] == pytest.approx(0.5) and m["skill_vs_base"] == pytest.approx(0.5)


def test_metricas_clasificacion_contra_sklearn():
    rng = np.random.default_rng(1)
    p = pd.Series(rng.uniform(0.05, 0.95, 2000))
    y = pd.Series(rng.uniform(size=2000) < p).astype(float)
    m = va.metricas_clasificacion(y, p, probabilidad_base=pd.Series(0.5, index=p.index), tasa_train=0.5)
    assert m["log_loss"] == pytest.approx(log_loss(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["auc_roc"] == pytest.approx(roc_auc_score(y, p))
    assert m["brier_skill_vs_base"] == pytest.approx(m["brier_skill_vs_tasa_train"])
    assert abs(m["intercepto_calibracion"]) < 0.2 and abs(m["pendiente_calibracion"] - 1) < 0.2
    assert m["ece"] < 0.05 and abs(m["diferencia_media_predicha"]) < 0.03


def test_calibracion_detecta_pendiente_y_nivel():
    rng = np.random.default_rng(2)
    logit = rng.normal(0, 1.5, 20000)
    y = pd.Series(rng.uniform(size=20000) < 1 / (1 + np.exp(-logit))).astype(float)
    exagerada = pd.Series(1 / (1 + np.exp(-2 * logit)))
    desplazada = pd.Series(1 / (1 + np.exp(-(logit + 1))))
    assert va.metricas_clasificacion(y, exagerada)["pendiente_calibracion"] == pytest.approx(0.5, abs=0.05)
    nivel = va.metricas_clasificacion(y, desplazada)
    assert nivel["intercepto_calibracion"] == pytest.approx(-1, abs=0.1) and nivel["diferencia_media_predicha"] > 0.1


def test_metricas_clasificacion_persistencia_degenerada():
    y, p = pd.Series([0.0, 1.0, 0.0, 0.0]), pd.Series([0.0, 0.0, 0.0, 0.0])
    m = va.metricas_clasificacion(y, p)
    assert np.isnan(m["log_loss"]) and np.isnan(m["auc_roc"]) and np.isnan(m["pendiente_calibracion"])
    assert m["brier"] == pytest.approx(0.25) and m["prevalencia"] == pytest.approx(0.25)


def test_metricas_ponderadas_igual_a_repetir_filas():
    y, p = pd.Series([0.0, 1.0, 1.0]), pd.Series([0.2, 0.6, 0.9])
    ponderada = va.metricas_clasificacion(y, p, peso=pd.Series([2.0, 1.0, 1.0]))
    repetida = va.metricas_clasificacion(pd.Series([0.0, 0.0, 1.0, 1.0]), pd.Series([0.2, 0.2, 0.6, 0.9]))
    for metrica in ("log_loss", "brier", "auc_roc", "auc_pr", "media_predicha"):
        assert ponderada[metrica] == pytest.approx(repetida[metrica])


def test_curva_confiabilidad_un_bin_por_valor_y_ece():
    curva = va.curva_confiabilidad(pd.Series([0.0, 1.0, 1.0, 1.0]), pd.Series([0.5, 0.5, 0.9, 0.9]))
    assert len(curva) == 2 and curva["tasa_observada"].tolist() == [0.5, 1.0]
    assert va.ece(curva) == pytest.approx(0.05)



def test_bootstrap_metricas_cluster_reproducible_y_contiene_observado():
    datos = pd.DataFrame({"cluster": np.repeat(np.arange(200), 3), "x": np.random.default_rng(3).normal(size=600)})

    def media(d, w):
        return {"media": float(np.sum(w * d["x"]) / w.sum())}
    a = va.bootstrap_metricas_cluster(datos, "cluster", media, n_boot=100, seed=5)
    b = va.bootstrap_metricas_cluster(datos, "cluster", media, n_boot=100, seed=5)
    pd.testing.assert_frame_equal(a, b)
    fila = a.iloc[0]
    assert fila["ic95_low"] < fila["valor"] < fila["ic95_high"] and fila["n_clusters"] == 200
    assert fila["valor"] == pytest.approx(datos["x"].mean())



def test_lineas_base_persistencia_y_decil():
    train = _train(1)
    entrada = train[va.poblacion_objetivo(train, "entrada")]
    persistencia = va.ajustar_linea_base(entrada, "entrada", "persistencia")
    assert (va.predecir_linea_base(persistencia, entrada) == 0).all()
    trayectoria = va.ajustar_linea_base(train, "trayectoria", "persistencia")
    assert (va.predecir_linea_base(trayectoria, train) == 0).all()
    decil = va.ajustar_linea_base(train, "trayectoria", "decil_ratio")
    assert va.predecir_linea_base(decil, train).nunique() == 10
    with pytest.raises(ValueError):
        va.ajustar_linea_base(train, "trayectoria", "otra")


@pytest.mark.parametrize("objetivo", cte.OBJETIVOS_MODELADO)
def test_evaluar_lineas_base_cubre_folds_y_lecturas(objetivo):
    train = _train(1, desde=(2016, 2), por_trimestre=40)
    folds = va.generar_folds(train, 1)
    tabla = va.evaluar_lineas_base(train, folds, objetivo, n_boot=10)
    assert set(tabla["anio_test"]) == set(folds["anio_test"])
    assert set(tabla["linea_base"]) == set(va.LINEAS_BASE) and set(tabla["ponderacion"]) == set(cte.PONDERACIONES)
    skill = "brier_skill_vs_base" if objetivo != "trayectoria" else "skill_vs_base"
    propia = tabla[(tabla["linea_base"] == "decil_ratio") & (tabla["metrica"] == skill)]
    assert np.allclose(propia["valor"].astype(float), 0.0)
    assert tabla.loc[tabla["linea_base"] == "decil_ratio", "ic95_low"].notna().any()


def _con_columnas_lineales(pares: pd.DataFrame) -> pd.DataFrame:
    n = len(pares)
    return pares.assign(
        share_x_t=np.where(np.arange(n) % 5 == 0, np.nan, np.linspace(0, 1, n)),
        cont_t=np.where(np.arange(n) % 7 == 0, np.nan, np.arange(n, dtype=float) ** 2),
        bin_t=pd.array([None if i % 4 == 0 else bool(i % 2) for i in range(n)], dtype="boolean"),
        cat_t=np.where(np.arange(n) % 3 == 0, "a", np.where(np.arange(n) % 3 == 1, "b", None)),
        constante_t=1.0)


def test_preprocesamiento_lineal_estimado_solo_en_train():
    pares = _con_columnas_lineales(_train(1))
    train, test = pares.iloc[: len(pares) // 2], pares.iloc[len(pares) // 2:]
    prep = va.ajustar_preprocesamiento_lineal(train, ["share_x_t"], ["cont_t", "constante_t"], ["bin_t"], ["cat_t"])
    cont = pd.to_numeric(train["cont_t"])
    assert prep["numericas"]["cont_t"]["neutro"] == pytest.approx(np.nanmedian(cont))
    assert prep["numericas"]["cont_t"]["limites"][1] == pytest.approx(np.nanquantile(cont, 0.995))
    assert prep["numericas"]["share_x_t"]["neutro"] == 0.0 and prep["numericas"]["bin_t"]["neutro"] == 0.0
    assert "constante_t" in prep["descartadas"]
    matriz_train = va.aplicar_preprocesamiento_lineal(prep, train)
    assert np.allclose(matriz_train.mean(), 0) and not matriz_train.isna().any().any()
    matriz_test = va.aplicar_preprocesamiento_lineal(prep, test)
    assert list(matriz_test.columns) == list(matriz_train.columns)
    assert {"share_x_t__na", "cont_t__na", "bin_t__na"} <= set(matriz_test.columns)
    assert sum(c.startswith("cat_t==") for c in matriz_test.columns) == 2
    crudo = va._matriz_lineal(prep, test)
    assert crudo["cont_t"].max() <= prep["numericas"]["cont_t"]["limites"][1]
    assert (crudo.loc[test["cont_t"].isna(), "cont_t"] == prep["numericas"]["cont_t"]["neutro"]).all()


def test_preprocesamiento_lineal_sin_na_en_train_no_crea_indicador():
    pares = _con_columnas_lineales(_train(1)).assign(cont_t=1.0 + np.arange(len(_train(1))))
    prep = va.ajustar_preprocesamiento_lineal(pares, [], ["cont_t"])
    nuevo = pares.head(3).assign(cont_t=[np.nan, 1.0, 2.0])
    matriz = va.aplicar_preprocesamiento_lineal(prep, nuevo)
    assert "cont_t__na" not in matriz.columns and not matriz.isna().any().any()


def test_leer_pares_no_duplica_columnas_del_universo(monkeypatch):
    pedidas = {}

    def falso_leer_universo(raiz, h, columnas_extra=()):
        pedidas["extra"] = columnas_extra
        return _pares(1).drop(columns=["particion", "fraccion_max_par"])
    monkeypatch.setattr(va.ue, "leer_universo", falso_leer_universo)
    monkeypatch.setattr(va.ue, "unir_objetivo", lambda pares, *a, **k: pares)
    monkeypatch.setattr(va.ue, "unir_marcas_imputacion", lambda pares, *a, **k: pares)
    pares = va._leer_pares(Path("x"), Path("y"), 1, ("era", "pobre_t", "AGLOMERADO_t", "edad_jefe_t"))
    assert not set(pedidas["extra"]) & set(ue.COLUMNAS_UNIVERSO)
    assert "edad_jefe_t" in pedidas["extra"] and "pobre_t" in pedidas["extra"]
    assert not pares.columns.duplicated().any()



def _predicciones(h: int = 4) -> pd.DataFrame:
    train = _train(h)
    rol = va.asignar_fold(train, 2012)
    test = train[rol == "test"]
    return test[CLAVE_PAR].assign(h=h, fold="2012", ultimo_destino_train=2012 * 4 - 1, evaluacion_agrupada_excluido=False,
                                  conjunto_evaluacion="limpio", prediccion=0.1, y_observado=0.0)


def test_verificar_predicciones_acepta_fuera_de_muestra():
    va.verificar_predicciones_fuera_de_muestra(_predicciones())


def test_verificar_predicciones_detecta_par_del_train():
    predicciones = _predicciones()
    train = _train(4)
    del_train = train[va.asignar_fold(train, 2012) == "train"].head(3)[CLAVE_PAR]
    filtrada = pd.concat([predicciones, del_train.assign(**predicciones.iloc[0][
        ["h", "fold", "ultimo_destino_train", "evaluacion_agrupada_excluido", "conjunto_evaluacion",
         "prediccion", "y_observado"]].to_dict())], ignore_index=True)
    with pytest.raises(ValueError, match="par_en_train_del_fold"):
        va.verificar_predicciones_fuera_de_muestra(filtrada)


def test_verificar_predicciones_holdout_y_claves_repetidas():
    predicciones = _predicciones()
    with pytest.raises(ValueError, match="clave_repetida"):
        va.verificar_predicciones_fuera_de_muestra(pd.concat([predicciones, predicciones.head(1)]))
    holdout = predicciones.assign(fold="holdout", ultimo_destino_train=va.INDICE_ULTIMO_DESTINO_TRAIN)
    with pytest.raises(ValueError, match="holdout_con_origen_en_train"):
        va.verificar_predicciones_fuera_de_muestra(holdout)


def test_manifest_configuracion_estado_git(monkeypatch, tmp_path):
    salidas = {"status": " M src/x.py\n", "diff": "diff --git a/src/x.py\n"}
    monkeypatch.setattr(va, "_salida_git", lambda repo, *args: salidas[args[0]])
    a = va.manifest_configuracion({"tecnica": "gradient_boosting", "h": 1}, tmp_path, "09_modelos_trayectoria")
    b = va.manifest_configuracion({"h": 1, "tecnica": "gradient_boosting"}, tmp_path, "09_modelos_trayectoria")
    c = va.manifest_configuracion({"h": 4, "tecnica": "gradient_boosting"}, tmp_path, "09_modelos_trayectoria")
    assert a["configuracion_id"] == b["configuracion_id"] != c["configuracion_id"]
    assert a["arbol_sucio"] is True and len(a["hash_diff"]) == 16
    salidas.update(status="", diff="")
    limpio = va.manifest_configuracion({"h": 1}, tmp_path, "09")
    assert limpio["arbol_sucio"] is False and limpio["hash_diff"] == ""
    assert va.ruta_predicciones(tmp_path, "entrada", 1, "abc").parts[-3:] == ("h=1", "configuracion=abc", "data.parquet")



@REQUIERE_CAPAS
@pytest.mark.parametrize("h,esperado_amplio,esperado_2016", [
    (1, list(range(2005, 2023)), list(range(2018, 2023))),
    (4, [*range(2006, 2015), *range(2016, 2022)], list(range(2019, 2022))),
])
def test_folds_reales(h, esperado_amplio, esperado_2016):
    pares = va.cargar_pares_modelado(FEATURES, TRAYECTORIA, h)
    va.exigir_sin_holdout(pares)
    amplio = va.generar_folds(pares, h)
    regular = va.generar_folds(pares[ue.en_universo(pares, "universo_2016_2025")], h)
    assert amplio["anio_test"].tolist() == esperado_amplio
    assert regular["anio_test"].tolist() == esperado_2016
