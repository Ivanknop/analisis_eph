"""Tests de `src/retropolacion_pobreza.py`. Fixtures sintéticas chicas -- mismo
criterio que `tests/test_canasta_regional.py`."""
import pandas as pd
import pytest

from retropolacion_pobreza import peso_hogar_historico


class TestPesoHogarHistorico:
    def test_pondera_constante_entre_integrantes(self):
        individual = pd.DataFrame({
            "CODUSU": ["V1", "V1", "V2"], "NRO_HOGAR": [1, 1, 1], "PONDERA": [100, 100, 200],
        })
        r = peso_hogar_historico(individual)
        assert r.loc[("V1", 1)] == 100
        assert r.loc[("V2", 1)] == 200

    def test_pondera_no_constante_levanta_error(self):
        individual = pd.DataFrame({
            "CODUSU": ["V1", "V1"], "NRO_HOGAR": [1, 1], "PONDERA": [100, 150],
        })
        with pytest.raises(ValueError):
            peso_hogar_historico(individual)
