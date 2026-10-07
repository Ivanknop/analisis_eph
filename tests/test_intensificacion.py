"""Tests de `src/intensificacion.py`. Todo con DataFrames sintéticos chicos armados a
mano -- nada acá toca `data/` real (mismo criterio que tests/test_panel.py)."""
import pandas as pd
import pytest

from intensificacion import (
    calcular_intensificacion,
    calcular_share_precario,
    clasificar_precariedad_persona,
    emparejar_activos,
    matriz_transicion_persona,
)


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH04", "CH06", "ESTADO", "CAT_OCUP", "PP07H"]
    return pd.DataFrame(filas, columns=columnas)


def _pares(codusu="V1", nro_hogar=1):
    return pd.DataFrame({"CODUSU": [codusu], "NRO_HOGAR_t": [nro_hogar], "NRO_HOGAR_th": [nro_hogar]})


class TestClasificarPrecariedadPersona:
    def test_formal(self):
        ind = _individual([["V1", 1, 1, 1, 30, 1, 3, 1]])
        assert clasificar_precariedad_persona(ind).iloc[0] == "formal"

    def test_informal_asalariado(self):
        ind = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])
        assert clasificar_precariedad_persona(ind).iloc[0] == "informal_asalariado"

    def test_patron(self):
        ind = _individual([["V1", 1, 1, 1, 30, 1, 1, pd.NA]])
        assert clasificar_precariedad_persona(ind).iloc[0] == "patron"

    def test_cuenta_propia(self):
        ind = _individual([["V1", 1, 1, 1, 30, 1, 2, pd.NA]])
        assert clasificar_precariedad_persona(ind).iloc[0] == "cuenta_propia"

    def test_desocupado(self):
        ind = _individual([["V1", 1, 1, 1, 30, 2, pd.NA, pd.NA]])
        assert clasificar_precariedad_persona(ind).iloc[0] == "desocupado"


class TestAltaOBajaNoCambiaElShare:
    def test_share_estable_igual_con_alta_y_baja(self):
        # A (formal) y B (informal) siguen estables; C (formal) se va (baja);
        # D (informal) entra nueva (alta) -- ninguna debería mover el share de A/B.
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 1, 3, 1],   # A formal
            ["V1", 1, 2, 2, 28, 1, 3, 2],   # B informal_asalariado
            ["V1", 1, 3, 1, 50, 1, 3, 1],   # C formal, se va
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 3, 1],   # A' formal
            ["V1", 1, 2, 2, 29, 1, 3, 2],   # B' informal_asalariado
            ["V1", 1, 4, 2, 20, 1, 3, 2],   # D nueva, informal_asalariado
        ])
        r = calcular_share_precario(emparejar_activos(individual_t, individual_th, _pares(), h=1)).iloc[0]
        assert r["n_activos_estables_t"] == 2 and r["n_activos_estables_th"] == 2
        assert r["share_precario_t"] == pytest.approx(0.5)
        assert r["share_precario_th"] == pytest.approx(0.5)


class TestJubilacionSaleDelDenominador:
    def test_activo_que_pasa_a_inactivo_no_genera_intensificacion_espuria(self):
        individual_t = _individual([
            ["V1", 1, 1, 1, 60, 1, 3, 2],   # único activo, informal_asalariado
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 61, 3, pd.NA, pd.NA],  # inactivo (se jubiló) -- ESTADO=3
        ])
        r = calcular_intensificacion(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["estado"] == "sin_activos_estables"
        assert pd.isna(r["intensificacion"])


class TestHogarSinActivosEstablesNoDesaparece:
    def test_estado_explicito(self):
        individual_t = _individual([["V1", 1, 1, 1, 40, 1, 3, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 41, 3, pd.NA, pd.NA]])
        r = calcular_intensificacion(individual_t, individual_th, _pares(), h=1)
        assert len(r) == 1
        assert r.iloc[0]["estado"] == "sin_activos_estables"


class TestEmpateDeBucket:
    def test_empate_marcado_y_excluido_de_la_matriz(self):
        # 2 personas con mismo sexo+edad en t (empate), 2 en th también -- no se puede
        # asignar con certeza quién es quién.
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 1, 3, 1],   # formal
            ["V1", 1, 2, 1, 30, 1, 3, 2],   # informal_asalariado, mismo sexo+edad que el anterior
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 3, 1],
            ["V1", 1, 2, 1, 31, 1, 3, 2],
        ])
        emparejados = emparejar_activos(individual_t, individual_th, _pares(), h=1)
        assert emparejados["t"]["empate"].all()
        assert emparejados["th"]["empate"].all()
        matriz = matriz_transicion_persona(emparejados)
        assert matriz.empty

    def test_bucket_sin_empate_si_aparece_en_la_matriz(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 1]])
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 2]])
        emparejados = emparejar_activos(individual_t, individual_th, _pares(), h=1)
        matriz = matriz_transicion_persona(emparejados)
        assert len(matriz) == 1
        assert matriz.iloc[0]["categoria_t"] == "formal"
        assert matriz.iloc[0]["categoria_th"] == "informal_asalariado"


class TestTechoDeclarado:
    def test_share_t_uno_nunca_intensifica(self):
        individual_t = _individual([["V1", 1, 1, 1, 30, 1, 3, 2]])  # informal, share_t=1.0
        individual_th = _individual([["V1", 1, 1, 1, 31, 1, 3, 2]])  # sigue informal, share_th=1.0
        r = calcular_intensificacion(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["share_precario_t"] == pytest.approx(1.0)
        assert r["estado"] == "no_intensifica"
        assert r["intensificacion"] == False  # noqa: E712


class TestTratamientoIndependiente:
    def _fixture(self):
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 1, 3, 1],   # formal
            ["V1", 1, 2, 2, 28, 1, 1, pd.NA],  # patron
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 3, 1],
            ["V1", 1, 2, 2, 29, 1, 1, pd.NA],
        ])
        return individual_t, individual_th

    def test_informal_cuenta_independiente_como_precario(self):
        individual_t, individual_th = self._fixture()
        r = calcular_share_precario(
            emparejar_activos(individual_t, individual_th, _pares(), h=1), "informal").iloc[0]
        assert r["share_precario_t"] == pytest.approx(0.5)

    def test_categoria_propia_no_cuenta_independiente_como_precario(self):
        individual_t, individual_th = self._fixture()
        r = calcular_share_precario(
            emparejar_activos(individual_t, individual_th, _pares(), h=1), "categoria_propia").iloc[0]
        assert r["share_precario_t"] == pytest.approx(0.0)
        assert r["n_activos_estables_t"] == 2

    def test_excluir_saca_independiente_del_denominador(self):
        individual_t, individual_th = self._fixture()
        r = calcular_share_precario(
            emparejar_activos(individual_t, individual_th, _pares(), h=1), "excluir").iloc[0]
        assert r["n_activos_estables_t"] == 1
        assert r["share_precario_t"] == pytest.approx(0.0)

    def test_cuenta_propia_precario_distingue_de_patron(self):
        individual_t = _individual([
            ["V1", 1, 1, 1, 30, 1, 1, pd.NA],   # patron, no precario
            ["V1", 1, 2, 2, 28, 1, 2, pd.NA],   # cuenta propia, precario bajo este tratamiento
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 31, 1, 1, pd.NA],
            ["V1", 1, 2, 2, 29, 1, 2, pd.NA],
        ])
        r = calcular_share_precario(
            emparejar_activos(individual_t, individual_th, _pares(), h=1), "cuenta_propia_precario").iloc[0]
        assert r["n_activos_estables_t"] == 2
        assert r["share_precario_t"] == pytest.approx(0.5)
