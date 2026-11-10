import numpy as np
import pandas as pd
import pytest

from linea_base_trayectoria import ajustar_regresion_media_decil, metricas_prediccion, predecir_regresion_media_decil


def test_regresion_media_decil_recupera_medias_por_grupo():
    x = pd.Series(np.repeat([1.0, 2.0], 50))
    y = pd.Series(np.repeat([0.5, -0.5], 50))
    modelo = ajustar_regresion_media_decil(x, y, n_cuantiles=2)
    prediccion = predecir_regresion_media_decil(modelo, pd.Series([0.0, 1.0, 2.0, 99.0, np.nan]))
    assert prediccion.iloc[:4].tolist() == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert prediccion.iloc[4] == pytest.approx(modelo["media_global"])


def test_metricas_r2_cero_para_media_train_y_uno_perfecto():
    y = pd.Series([1.0, 2.0, 3.0])
    media = metricas_prediccion(y, pd.Series([2.0, 2.0, 2.0]), media_train=2.0)
    perfecta = metricas_prediccion(y, y, media_train=2.0)
    assert media["r2_vs_media_train"] == pytest.approx(0.0)
    assert perfecta["r2_vs_media_train"] == pytest.approx(1.0) and perfecta["mae"] == 0.0


def test_metricas_ponderadas_e_ignora_nulos():
    y = pd.Series([0.0, 1.0, np.nan])
    resultado = metricas_prediccion(y, pd.Series([0.0, 0.0, 0.0]), media_train=0.0, peso=pd.Series([1.0, 3.0, 1.0]))
    assert resultado["n"] == 2
    assert resultado["mae"] == pytest.approx(0.75)
    assert resultado["rmse"] == pytest.approx(np.sqrt(0.75))
