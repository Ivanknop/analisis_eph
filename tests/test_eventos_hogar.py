"""Tests de `src/eventos_hogar.py`. Todo con DataFrames sintéticos chicos armados a
mano -- nada acá toca `data/` real (mismo criterio que tests/test_panel.py)."""
import pandas as pd
import pytest

from eventos_hogar import (
    calcular_alta_activo_menor,
    calcular_se_jubila,
    calcular_tipologia_perceptores,
)


def _individual(filas):
    columnas = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH04", "CH06", "ESTADO",
                "CAT_OCUP", "PP07H", "CAT_INAC", "CH10", "NIVEL_ED"]
    return pd.DataFrame(filas, columns=columnas)


def _pares(codusu="V1", nro_hogar=1):
    return pd.DataFrame({"CODUSU": [codusu], "NRO_HOGAR_t": [nro_hogar], "NRO_HOGAR_th": [nro_hogar]})


class TestCalcularSeJubila:
    def test_activo_que_pasa_a_cat_inac_uno_cuenta(self):
        individual_t = _individual([["V1", 1, 1, 1, 60, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([["V1", 1, 1, 1, 61, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA]])
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 1

    def test_cat_inac_distinto_de_uno_no_cuenta(self):
        individual_t = _individual([["V1", 1, 1, 1, 60, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([["V1", 1, 1, 1, 61, 3, pd.NA, pd.NA, 4, pd.NA, pd.NA]])  # ama de casa, no jubilado
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 0

    def test_fuera_de_la_ventana_de_edad_no_cuenta(self):
        individual_t = _individual([["V1", 1, 1, 1, 60, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([["V1", 1, 1, 1, 70, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA]])  # +10 años, fuera de ventana h=1
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 0

    def test_distinto_sexo_no_cuenta_no_es_la_misma_persona(self):
        individual_t = _individual([["V1", 1, 1, 1, 60, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([["V1", 1, 1, 2, 61, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA]])  # sexo distinto
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 0

    def test_jubilado_preexistente_no_se_reutiliza_para_otro_activo(self):
        # A ya era jubilado en t y sigue jubilado en t+h; B es un activo distinto,
        # del mismo sexo y edad que A en t, que no se jubila -- sin excluir a A
        # como candidato ya usado, B podría matchear por error contra A.
        individual_t = _individual([
            ["V1", 1, 1, 1, 65, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA],
            ["V1", 1, 2, 1, 65, 1, 3, 1, pd.NA, pd.NA, pd.NA],
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 66, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA],
            ["V1", 1, 2, 1, 66, 1, 3, 1, pd.NA, pd.NA, pd.NA],
        ])
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 0

    def test_quien_ya_era_jubilado_en_t_no_es_candidato(self):
        # confirmado empíricamente (CLAUDE.md/docs): CAT_INAC solo toma valor
        # cuando ESTADO no es 1/2 -- un jubilado en t nunca es "activo", así que
        # nunca entra a activos_t y no puede volver a contarse en t+h.
        individual_t = _individual([
            ["V1", 1, 1, 1, 65, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA],   # ya jubilado en t, inactivo
            ["V1", 1, 2, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA],       # activo real, se jubila después
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 66, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA],   # sigue jubilado
            ["V1", 1, 2, 1, 41, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA],   # se jubiló recién
        ])
        r = calcular_se_jubila(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_se_jubila"] == 1


class TestCalcularAltaActivoMenor:
    def test_menor_que_entra_a_trabajar_y_sigue_en_la_escuela(self):
        individual_t = _individual([["V1", 1, 1, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([
            ["V1", 1, 1, 1, 41, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 16, 1, 3, 2, pd.NA, 1, 3],  # alta, menor, asiste, no terminó
        ])
        r = calcular_alta_activo_menor(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_alta_activo_menor"] == 1
        assert r["n_alta_activo_menor_asiste_entra_activo"] == 1

    def test_menor_que_entra_a_trabajar_y_dejo_la_escuela(self):
        individual_t = _individual([["V1", 1, 1, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([
            ["V1", 1, 1, 1, 41, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 16, 1, 3, 2, pd.NA, 2, 3],  # alta, menor, no asiste (dejó)
        ])
        r = calcular_alta_activo_menor(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_alta_activo_menor"] == 1
        assert r["n_alta_activo_menor_no_asiste_no_completo_entra_activo"] == 1

    def test_menor_que_ya_termino_la_secundaria_no_es_inasistencia(self):
        individual_t = _individual([["V1", 1, 1, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([
            ["V1", 1, 1, 1, 41, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 17, 1, 3, 2, pd.NA, 2, 4],  # alta, menor, no asiste pero NIVEL_ED=4 (terminó)
        ])
        r = calcular_alta_activo_menor(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_alta_activo_menor_termino_entra_activo"] == 1

    def test_menor_que_ya_estaba_inactivo_en_el_hogar(self):
        individual_t = _individual([
            ["V1", 1, 1, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 16, 4, pd.NA, pd.NA, pd.NA, 1, 3],  # inactivo en t (menor de 10 / estudiante)
        ])
        individual_th = _individual([
            ["V1", 1, 1, 1, 41, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 17, 1, 3, 2, pd.NA, 1, 3],  # misma persona, ahora activa
        ])
        r = calcular_alta_activo_menor(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_alta_activo_menor_asiste_inactivo_a_activo"] == 1

    def test_alta_mayor_de_edad_no_cuenta(self):
        individual_t = _individual([["V1", 1, 1, 1, 40, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        individual_th = _individual([
            ["V1", 1, 1, 1, 41, 1, 3, 1, pd.NA, pd.NA, pd.NA],
            ["V1", 1, 2, 2, 25, 1, 3, 2, pd.NA, pd.NA, pd.NA],  # alta, mayor de 18
        ])
        r = calcular_alta_activo_menor(individual_t, individual_th, _pares(), h=1).iloc[0]
        assert r["n_alta_activo_menor"] == 0


class TestCalcularTipologiaPerceptores:
    def _hogar(self, v2):
        return pd.DataFrame({"CODUSU": ["V1"], "NRO_HOGAR": [1], "V2": [v2]})

    def test_solo_laborales(self):
        individual = _individual([["V1", 1, 1, 1, 30, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        r = calcular_tipologia_perceptores(self._hogar(2), individual).iloc[0]
        assert r["tipologia_perceptores"] == "solo_laborales"

    def test_solo_previsionales(self):
        individual = _individual([["V1", 1, 1, 1, 70, 3, pd.NA, pd.NA, 1, pd.NA, pd.NA]])
        r = calcular_tipologia_perceptores(self._hogar(1), individual).iloc[0]
        assert r["tipologia_perceptores"] == "solo_previsionales"

    def test_mixto(self):
        individual = _individual([["V1", 1, 1, 1, 30, 1, 3, 1, pd.NA, pd.NA, pd.NA]])
        r = calcular_tipologia_perceptores(self._hogar(1), individual).iloc[0]
        assert r["tipologia_perceptores"] == "mixto"

    def test_sin_perceptores(self):
        individual = _individual([["V1", 1, 1, 1, 30, 3, pd.NA, pd.NA, 4, pd.NA, pd.NA]])
        r = calcular_tipologia_perceptores(self._hogar(2), individual).iloc[0]
        assert r["tipologia_perceptores"] == "sin_perceptores"

    def test_v2_ns_nr_sin_ocupados_da_na(self):
        individual = _individual([["V1", 1, 1, 1, 30, 3, pd.NA, pd.NA, 4, pd.NA, pd.NA]])
        r = calcular_tipologia_perceptores(self._hogar(9), individual).iloc[0]
        assert pd.isna(r["tipologia_perceptores"])
