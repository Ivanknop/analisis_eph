"""Tests de `src/trayectoria_relativa.py`. DataFrames sintéticos chicos --
mismo criterio que `tests/test_trayectoria_distancia.py`."""
import numpy as np
import pandas as pd
import pytest

from trayectoria_distancia import agregar_componente_comun_y_propio, construir_pares_distancia
from trayectoria_relativa import construir_pares_relativa, contar_exclusiones_y_ae, diferencia_log_y_ae


def _pares_hogar(codusus):
    return pd.DataFrame({
        "CODUSU": codusus, "NRO_HOGAR_t": [1] * len(codusus), "NRO_HOGAR_th": [1] * len(codusus),
        "identidad_confirmada": [True] * len(codusus),
    })


def _lado_a(codusus, anio, trimestre, region, aglomerado, itf, unidades_ae, cbt_ae):
    ratio_lp = pd.Series(itf) / (pd.Series(cbt_ae) * pd.Series(unidades_ae))
    return pd.DataFrame({
        "CODUSU": codusus, "NRO_HOGAR": [1] * len(codusus), "ANO4": anio, "TRIMESTRE": trimestre,
        "REGION": region, "AGLOMERADO": aglomerado, "era": "regular",
        "ITF": itf, "PONDIH": [100.0] * len(codusus), "unidades_ae": unidades_ae, "cbt_ae": cbt_ae,
        "ratio_lp": ratio_lp, "log_ratio_lp": np.log(ratio_lp), "pobre": ratio_lp < 1,
    })


def _lado_b(codusus, anio, trimestre, aglomerado, itf, unidades_ae):
    y_ae = pd.Series(itf) / pd.Series(unidades_ae)
    return pd.DataFrame({
        "CODUSU": codusus, "NRO_HOGAR": [1] * len(codusus), "ANO4": anio, "TRIMESTRE": trimestre,
        "AGLOMERADO": aglomerado, "era": "regular", "ITF": itf, "unidades_ae": unidades_ae,
        "y_ae": y_ae, "log_y_ae": np.log(y_ae), "peso_hogar": [100.0] * len(codusus),
        # No hace falta que sean el valor "real" para este test -- solo se
        # usan variacion_log_b y peso_hogar_t/AGLOMERADO_t/ANO4_t/TRIMESTRE_t
        # para la identidad algebraica bajo prueba.
        "posicion_relativa_nacional": y_ae, "posicion_relativa_aglomerado": y_ae,
        "ingreso_no_declarado": [False] * len(codusus), "hot_deck_disponible": [True] * len(codusus),
    })


class TestIdentidadAlgebraicaConLineaDePobreza:
    def test_variacion_propia_b_coincide_con_variacion_propia_aglomerado_log(self):
        # 2 aglomerados (cada uno con su propia región -> su propio cbt_ae,
        # y su propia inflación entre t y t+1) para probar que el término de
        # precio se cancela DENTRO de cada grupo aglomerado-trimestre de
        # origen, no por casualidad de un único cbt_ae global.
        codusus = ["V1", "V2", "V3", "V4", "V5"]
        aglomerado = [32, 32, 32, 33, 33]
        region = [1, 1, 1, 2, 2]
        itf_t = [1000.0, 2000.0, 4000.0, 1500.0, 3000.0]
        itf_th = [1100.0, 1800.0, 5000.0, 1600.0, 2800.0]
        unidades_ae = [1.0, 2.0, 1.5, 1.0, 2.5]

        cbt_ae_t = [2000.0 if a == 32 else 1500.0 for a in aglomerado]
        # 10% de inflación en el aglomerado 32 (región 1), 6% en el 33 (región 2)
        # entre t y t+1 -- magnitudes distintas por grupo, a propósito.
        cbt_ae_th = [2200.0 if a == 32 else 1590.0 for a in aglomerado]

        distancia_t = _lado_a(codusus, 2024, 1, region, aglomerado, itf_t, unidades_ae, cbt_ae_t)
        distancia_th = _lado_a(codusus, 2024, 2, region, aglomerado, itf_th, unidades_ae, cbt_ae_th)
        posicion_t = _lado_b(codusus, 2024, 1, aglomerado, itf_t, unidades_ae)
        posicion_th = _lado_b(codusus, 2024, 2, aglomerado, itf_th, unidades_ae)
        pares_hogar = _pares_hogar(codusus)

        pares_a = construir_pares_distancia(distancia_t, distancia_th, pares_hogar)
        pares_a = agregar_componente_comun_y_propio(
            pares_a, "variacion_log", "PONDIH_t", ["ANO4_t", "TRIMESTRE_t", "AGLOMERADO_t"], "aglomerado_log")

        pares_b = construir_pares_relativa(posicion_t, posicion_th, pares_hogar)

        np.testing.assert_allclose(
            pares_a["variacion_propia_aglomerado_log"].astype(float),
            pares_b["variacion_propia_b"].astype(float),
            rtol=1e-12, atol=1e-12,
        )
        # El componente común, en cambio, NO coincide -- A le resta el
        # cambio de precios (queda en términos reales), B no (queda en
        # pesos corrientes). Si coincidieran, la fabricación del test
        # estaría mal (cbt_ae constante), no estaría probando nada.
        assert not np.allclose(
            pares_a["variacion_comun_aglomerado_log"].astype(float),
            pares_b["variacion_comun_b"].astype(float),
        )


class TestDiferenciaLogYAe:
    def test_ambos_positivos(self):
        r = diferencia_log_y_ae(pd.Series([100.0]), pd.Series([150.0]))
        assert r.iloc[0] == pytest.approx(np.log(1.5))

    def test_un_lado_no_positivo_da_na(self):
        r = diferencia_log_y_ae(pd.Series([0.0]), pd.Series([150.0]))
        assert pd.isna(r.iloc[0])


class TestContarExclusionesYAe:
    def test_desagrega_nulo_vs_no_positivo(self):
        pares = pd.DataFrame({
            "variacion_log_b": [np.nan, np.nan, 0.1],
            "y_ae_t": [np.nan, 0.0, 5.0],
            "y_ae_th": [10.0, 5.0, 6.0],
        })
        r = contar_exclusiones_y_ae(pares)
        assert r["n_pares"] == 3
        assert r["n_sin_variacion_log_b"] == 2
        assert r["n_y_ae_t_nulo"] == 1
        assert r["n_y_ae_t_no_positivo"] == 1
