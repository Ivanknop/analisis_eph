"""Tests de `src/bootstrap.py` con DataFrames sintéticos chicos, sin tocar
`data/` real."""
import numpy as np
import pandas as pd
import pytest

from bootstrap import diferencia_dos_grupos_bootstrap_cluster, estandarizar_diferencia_bootstrap


def _tabla_dos_grupos(n_a: int, p_a: float, n_b: float, p_b: float) -> pd.DataFrame:
    """Un cluster por fila, evento determinístico: exactamente `round(n*p)`
    eventos `True` en cada grupo -- la tasa observada queda exacta."""
    filas = []
    s_a = round(n_a * p_a)
    for i in range(n_a):
        filas.append({"cluster_id": f"a{i}", "grupo": "A", "evento": i < s_a})
    s_b = round(n_b * p_b)
    for i in range(n_b):
        filas.append({"cluster_id": f"b{i}", "grupo": "B", "evento": i < s_b})
    return pd.DataFrame(filas)


class TestDiferenciaDosGruposBootstrapCluster:
    def test_tasas_observadas_exactas(self):
        tabla = _tabla_dos_grupos(2000, 0.30, 2000, 0.50)
        resultado = diferencia_dos_grupos_bootstrap_cluster(
            tabla, "cluster_id", "evento", "grupo", "A", "B", n_boot=400, seed=42)
        assert resultado["tasa_a"] == pytest.approx(30.0)
        assert resultado["tasa_b"] == pytest.approx(50.0)
        assert resultado["diferencia"] == pytest.approx(-20.0)
        assert resultado["n_a"] == 2000 and resultado["n_b"] == 2000 and resultado["n_clusters"] == 4000

    def test_ancho_del_ic_del_orden_del_analitico(self):
        """Con cluster=1 fila, el ancho del IC converge al analítico de dos
        proporciones binomiales -- el bug original daba ~30x más chico."""
        n_a, p_a, n_b, p_b = 2000, 0.30, 2000, 0.50
        tabla = _tabla_dos_grupos(n_a, p_a, n_b, p_b)
        resultado = diferencia_dos_grupos_bootstrap_cluster(
            tabla, "cluster_id", "evento", "grupo", "A", "B", n_boot=400, seed=42)
        ancho = resultado["diferencia_ic95_high"] - resultado["diferencia_ic95_low"]
        se_analitico = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        ancho_analitico_pp = 2 * 1.96 * se_analitico * 100
        assert 0.3 * ancho_analitico_pp < ancho < 3 * ancho_analitico_pp
        # El IC debe cubrir la diferencia observada (-20.0), con margen.
        assert resultado["diferencia_ic95_low"] < -20.0 < resultado["diferencia_ic95_high"]

    def test_reproducible_con_misma_seed(self):
        tabla = _tabla_dos_grupos(500, 0.4, 500, 0.4)
        r1 = diferencia_dos_grupos_bootstrap_cluster(tabla, "cluster_id", "evento", "grupo", "A", "B", seed=7)
        r2 = diferencia_dos_grupos_bootstrap_cluster(tabla, "cluster_id", "evento", "grupo", "A", "B", seed=7)
        assert r1 == r2

    def test_grupo_b_ausente_da_nan(self):
        tabla = _tabla_dos_grupos(50, 0.5, 0, 0.0)  # sin filas de "B"
        resultado = diferencia_dos_grupos_bootstrap_cluster(tabla, "cluster_id", "evento", "grupo", "A", "B")
        assert resultado["n_b"] == 0
        assert np.isnan(resultado["tasa_b"])
        assert np.isnan(resultado["diferencia"])

    def test_tabla_vacia_da_todo_nan(self):
        tabla = pd.DataFrame(columns=["cluster_id", "evento", "grupo"])
        resultado = diferencia_dos_grupos_bootstrap_cluster(tabla, "cluster_id", "evento", "grupo", "A", "B")
        assert resultado["n_clusters"] == 0
        assert np.isnan(resultado["diferencia"])

    def test_grupo_booleano_true_false(self):
        """`grupo_a`/`grupo_b` booleanos -- regresión de un caso donde
        `df[[True, False]]` dispara la detección de máscara de pandas."""
        filas = []
        for i in range(300):
            filas.append({"cluster_id": f"c{i}", "con_deficit": True, "evento": i < 90})  # 30%
        for i in range(300):
            filas.append({"cluster_id": f"s{i}", "con_deficit": False, "evento": i < 150})  # 50%
        tabla = pd.DataFrame(filas)
        resultado = diferencia_dos_grupos_bootstrap_cluster(
            tabla, "cluster_id", "evento", "con_deficit", True, False, n_boot=100, seed=1)
        assert resultado["tasa_a"] == pytest.approx(30.0)
        assert resultado["tasa_b"] == pytest.approx(50.0)
        assert resultado["n_clusters"] == 600

    def test_evento_nulo_se_excluye(self):
        tabla = _tabla_dos_grupos(100, 0.5, 100, 0.5)
        tabla["evento"] = tabla["evento"].astype("boolean")  # nullable, como sale_binario en el notebook
        tabla.loc[tabla.index[:10], "evento"] = pd.NA
        resultado = diferencia_dos_grupos_bootstrap_cluster(tabla, "cluster_id", "evento", "grupo", "A", "B")
        assert resultado["n_a"] + resultado["n_b"] == 190


class TestEstandarizarDiferenciaBootstrap:
    def _tabla_estandarizacion(self) -> pd.DataFrame:
        """con: 50% obs. sin: 20%/80% por estrato. Estandarizado = 35,
        diferencia esperada = 15."""
        filas = []
        contador = 0

        def _agregar(grupo, estrato, n, s):
            nonlocal contador
            for i in range(n):
                filas.append({"cluster_id": f"c{contador}", "grupo": grupo, "estrato": estrato, "evento": i < s})
                contador += 1

        _agregar("con", "e1", 300, 150)
        _agregar("con", "e2", 100, 50)
        _agregar("sin", "e1", 200, 40)
        _agregar("sin", "e2", 200, 160)
        return pd.DataFrame(filas)

    def test_estandarizacion_exacta(self):
        tabla = self._tabla_estandarizacion()
        resultado = estandarizar_diferencia_bootstrap(
            tabla, "cluster_id", "evento", "grupo", "con", "sin", "estrato", n_boot=100, seed=1)
        assert resultado["pct_observado_con"] == pytest.approx(50.0)
        assert resultado["pct_estandarizado_sin"] == pytest.approx(35.0)
        assert resultado["diferencia"] == pytest.approx(15.0)
        assert resultado["n_con"] == 400 and resultado["n_clusters"] == 800

    def test_ic_cubre_la_diferencia_observada(self):
        tabla = self._tabla_estandarizacion()
        resultado = estandarizar_diferencia_bootstrap(
            tabla, "cluster_id", "evento", "grupo", "con", "sin", "estrato", n_boot=400, seed=1)
        assert resultado["diferencia_ic95_low"] <= 15.0 <= resultado["diferencia_ic95_high"]
        assert resultado["diferencia_ic95_high"] - resultado["diferencia_ic95_low"] > 0.5

    def test_sin_estratos_comunes_da_estandarizado_nan(self):
        tabla = self._tabla_estandarizacion()
        tabla.loc[tabla["grupo"] == "sin", "estrato"] = "e3"  # ningún estrato compartido con "con"
        resultado = estandarizar_diferencia_bootstrap(
            tabla, "cluster_id", "evento", "grupo", "con", "sin", "estrato", n_boot=50, seed=1)
        assert resultado["pct_observado_con"] == pytest.approx(50.0)
        assert np.isnan(resultado["pct_estandarizado_sin"])

    def test_tabla_vacia_da_todo_nan(self):
        tabla = pd.DataFrame(columns=["cluster_id", "evento", "grupo", "estrato"])
        resultado = estandarizar_diferencia_bootstrap(tabla, "cluster_id", "evento", "grupo", "con", "sin", "estrato")
        assert resultado["n_clusters"] == 0
        assert np.isnan(resultado["diferencia"])
