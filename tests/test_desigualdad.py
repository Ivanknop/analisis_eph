"""Tests de `src/desigualdad.py`. Todo con arrays y DataFrames sinteticos chicos armados
a mano -- nada aca toca `data/` real (mismo criterio que tests/test_bootstrap.py)."""
import numpy as np
import pandas as pd
import pytest

from desigualdad import gini_ic_bootstrap_cluster, gini_ponderado, normalizar_por_media_ponderada


def _tabla_un_cluster_por_fila(valores: list[float], pesos: list[float] | None = None) -> pd.DataFrame:
    pesos = pesos if pesos is not None else [1.0] * len(valores)
    return pd.DataFrame({"ipcf": valores, "peso": pesos,
                          "cluster": [f"c{i}" for i in range(len(valores))]})


class TestGiniPonderado:
    def test_todos_iguales_da_cero(self):
        assert gini_ponderado([5, 5, 5, 5]) == pytest.approx(0.0)
        assert gini_ponderado([5, 5, 5, 5], [3, 1, 7, 2]) == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [2, 5, 10, 1000])
    def test_uno_concentra_todo_da_n_menos_1_sobre_n(self, n):
        valores = [0.0] * (n - 1) + [100.0]
        assert gini_ponderado(valores) == pytest.approx((n - 1) / n)

    def test_peso_dos_equivale_a_duplicar_la_fila(self):
        """Invariancia a replicacion: es la propiedad que permite usar el mismo estimador
        con pesos de hogar (PONDIH x integrantes) y de persona (suma de PONDERA)."""
        valores = [0.0, 10.0, 25.0, 100.0]
        con_peso = gini_ponderado(valores, [2, 2, 2, 2])
        duplicado = gini_ponderado([v for v in valores for _ in range(2)])
        assert con_peso == pytest.approx(duplicado)
        # Tambien con pesos desparejos, que es el caso real.
        assert gini_ponderado([0.0, 10.0, 100.0], [1, 2, 1]) == pytest.approx(
            gini_ponderado([0.0, 10.0, 10.0, 100.0]))

    def test_los_ceros_entran_en_la_distribucion(self):
        """D3: `IPCF == 0` no es "sin ingreso", no se descarta -- sacarlo baja el Gini."""
        con_ceros = gini_ponderado([0, 0, 10, 20, 30])
        sin_ceros = gini_ponderado([10, 20, 30])
        assert con_ceros > sin_ceros
        assert con_ceros == pytest.approx(gini_ponderado([0, 0, 10, 20, 30], [1, 1, 1, 1, 1]))

    def test_pesos_cero_se_ignoran(self):
        """Los no declarantes de la era regular llegan con PONDIH == 0 (D3): tienen que
        salir del calculo sin dejar rastro, no contar como ingreso cero."""
        con_pesos_cero = gini_ponderado([0, 0, 10, 20, 30], [0, 0, 1, 1, 1])
        assert con_pesos_cero == pytest.approx(gini_ponderado([10, 20, 30]))

    @pytest.mark.parametrize("k", [0.01, 3.0, 1e6])
    def test_invariante_a_escala(self, k):
        valores = [0.0, 10.0, 25.0, 100.0]
        assert gini_ponderado([v * k for v in valores]) == pytest.approx(gini_ponderado(valores))

    def test_valores_negativos_levantan_error(self):
        with pytest.raises(ValueError, match="negativos"):
            gini_ponderado([-1.0, 10.0, 20.0])

    def test_valor_negativo_con_peso_cero_no_molesta(self):
        assert gini_ponderado([-1.0, 10.0, 20.0, 30.0], [0, 1, 1, 1]) == pytest.approx(
            gini_ponderado([10, 20, 30]))

    def test_nulos_se_excluyen(self):
        valores = pd.Series(pd.array([10, None, 20, 30], dtype="Int32"))
        assert gini_ponderado(valores) == pytest.approx(gini_ponderado([10, 20, 30]))

    def test_sin_masa_da_nan(self):
        assert np.isnan(gini_ponderado([]))
        assert np.isnan(gini_ponderado([0.0, 0.0, 0.0]))
        assert np.isnan(gini_ponderado([10.0, 20.0], [0, 0]))

    def test_largo_distinto_levanta_error(self):
        with pytest.raises(ValueError, match="largo distinto"):
            gini_ponderado([1.0, 2.0], [1.0])


class TestNormalizarPorMediaPonderada:
    def test_divide_por_la_media(self):
        np.testing.assert_allclose(normalizar_por_media_ponderada([1, 2, 3]), [0.5, 1.0, 1.5])
        np.testing.assert_allclose(normalizar_por_media_ponderada([1, 3], [3, 1]), [2 / 3, 2.0])

    def test_ventana_de_dos_trimestres_con_inflacion(self):
        """El test del ano movil: dos copias de la misma distribucion, una escalada por
        `k` (inflacion entre trimestres), normalizada cada una por su propia media y
        concatenadas, dan exactamente el Gini de la distribucion original. Sin normalizar,
        el Gini de la ventana sale inflado."""
        base = np.array([0.0, 5.0, 10.0, 40.0, 100.0])
        pesos = np.array([2.0, 1.0, 3.0, 1.0, 1.0])
        k = 2.7
        gini_original = gini_ponderado(base, pesos)

        ventana_normalizada = np.concatenate([normalizar_por_media_ponderada(base, pesos),
                                              normalizar_por_media_ponderada(base * k, pesos)])
        ventana_cruda = np.concatenate([base, base * k])
        pesos_ventana = np.concatenate([pesos, pesos])

        assert gini_ponderado(ventana_normalizada, pesos_ventana) == pytest.approx(gini_original)
        assert gini_ponderado(ventana_cruda, pesos_ventana) > gini_original + 0.03

    def test_invalidos_quedan_nan(self):
        salida = normalizar_por_media_ponderada([10.0, np.nan, 30.0], [1.0, 1.0, 0.0])
        assert np.isnan(salida[1]) and np.isnan(salida[2])
        assert salida[0] == pytest.approx(1.0)

    def test_sin_masa_da_todo_nan(self):
        assert np.all(np.isnan(normalizar_por_media_ponderada([0.0, 0.0])))


class TestGiniIcBootstrapCluster:
    def test_punto_coincide_con_gini_ponderado(self):
        tabla = _tabla_un_cluster_por_fila([0.0, 10.0, 25.0, 100.0], [1, 2, 1, 1])
        resultado = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", n_boot=200)
        assert resultado["gini"] == pytest.approx(round(gini_ponderado(tabla["ipcf"], tabla["peso"]), 4))
        assert resultado["n_filas"] == 4 and resultado["n_clusters"] == 4
        assert resultado["peso_total"] == pytest.approx(5.0)

    def test_ic_cubre_el_gini_observado(self):
        rng = np.random.default_rng(0)
        tabla = _tabla_un_cluster_por_fila(list(rng.lognormal(mean=0.0, sigma=0.8, size=600)))
        resultado = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", n_boot=400)
        assert resultado["gini_ic95_low"] < resultado["gini"] < resultado["gini_ic95_high"]

    def test_el_ic_se_angosta_con_mas_clusters(self):
        rng = np.random.default_rng(1)
        anchos = []
        for n in (200, 2000):
            tabla = _tabla_un_cluster_por_fila(list(rng.lognormal(mean=0.0, sigma=0.8, size=n)))
            r = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", n_boot=400)
            anchos.append(r["gini_ic95_high"] - r["gini_ic95_low"])
        assert anchos[1] < anchos[0] / 2

    def test_clusters_grandes_dan_ic_mas_ancho_que_filas_sueltas(self):
        """Las personas de un hogar comparten el IPCF exacto: si el cluster no agrupa, el
        IC sale artificialmente angosto. Mismo n de filas, distinta cantidad de clusters."""
        rng = np.random.default_rng(2)
        valores = list(rng.lognormal(mean=0.0, sigma=0.8, size=100))
        filas = [v for v in valores for _ in range(5)]
        sueltas = pd.DataFrame({"ipcf": filas, "peso": 1.0,
                                 "cluster": [f"c{i}" for i in range(len(filas))]})
        agrupadas = pd.DataFrame({"ipcf": filas, "peso": 1.0,
                                   "cluster": [f"c{i // 5}" for i in range(len(filas))]})
        r_sueltas = gini_ic_bootstrap_cluster(sueltas, "ipcf", "peso", "cluster", n_boot=400)
        r_agrupadas = gini_ic_bootstrap_cluster(agrupadas, "ipcf", "peso", "cluster", n_boot=400)
        assert r_sueltas["gini"] == pytest.approx(r_agrupadas["gini"])
        ancho_sueltas = r_sueltas["gini_ic95_high"] - r_sueltas["gini_ic95_low"]
        ancho_agrupadas = r_agrupadas["gini_ic95_high"] - r_agrupadas["gini_ic95_low"]
        assert ancho_agrupadas > 1.5 * ancho_sueltas

    def test_reproducible_con_misma_seed(self):
        tabla = _tabla_un_cluster_por_fila([0.0, 10.0, 25.0, 100.0])
        r1 = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", seed=7)
        r2 = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", seed=7)
        assert r1 == r2

    def test_tabla_vacia_da_todo_nan(self):
        vacia = pd.DataFrame({"ipcf": [], "peso": [], "cluster": []})
        resultado = gini_ic_bootstrap_cluster(vacia, "ipcf", "peso", "cluster")
        assert np.isnan(resultado["gini"]) and resultado["n_filas"] == 0 and resultado["n_clusters"] == 0

    def test_solo_pesos_cero_da_todo_nan(self):
        tabla = _tabla_un_cluster_por_fila([10.0, 20.0], [0, 0])
        resultado = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster")
        assert np.isnan(resultado["gini"]) and resultado["n_filas"] == 0

    def test_batch_no_cambia_el_resultado(self):
        tabla = _tabla_un_cluster_por_fila([0.0, 10.0, 25.0, 100.0, 7.0, 3.0])
        r1 = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", n_boot=300, batch=300, seed=5)
        r2 = gini_ic_bootstrap_cluster(tabla, "ipcf", "peso", "cluster", n_boot=300, batch=50, seed=5)
        assert r1["gini"] == r2["gini"]
        assert abs(r1["gini_ic95_low"] - r2["gini_ic95_low"]) < 0.05
