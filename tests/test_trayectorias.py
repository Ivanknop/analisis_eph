"""Tests de `src/trayectorias.py`. Todo con DataFrames sintéticos chicos
armados a mano -- nada acá toca `data/` real."""
import pandas as pd
import pytest

from trayectorias import (
    claves_hogar,
    claves_vivienda,
    clasificar_patron_apariciones,
    clasificar_perdida_por_patron,
    cohorte_completable,
    contar_n_diseno,
    encadenar_dos_eslabones,
    encadenar_trayectoria,
    inferir_visita,
    verificar_contra_h4_t1,
)


def _vivienda(filas: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(filas, columns=["CODUSU", "AGLOMERADO"])


def _presencia_completa(offsets_presentes: set[int]) -> dict[int, pd.DataFrame | None]:
    """Vivienda `"H1"` presente en offset `0` y en cada offset de `offsets_presentes`,
    ausente (tabla vacía, no `None`) en el resto de `OFFSETS_VISITA`."""
    from trayectorias import OFFSETS_VISITA
    presencia = {0: _vivienda([("H1", 1)])}
    for offset in OFFSETS_VISITA:
        presencia[offset] = _vivienda([("H1", 1)]) if offset in offsets_presentes else _vivienda([])
    return presencia


class TestInferirVisita:
    def test_visita_1_limpia(self):
        resultado = inferir_visita(_presencia_completa({1, 4, 5}))
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 1
        assert fila["n_matches"] == 8 and fila["n_mismatches"] == 0
        assert not fila["ambigua"]

    def test_visita_2_limpia(self):
        resultado = inferir_visita(_presencia_completa({-1, 3, 4}))
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 2
        assert fila["n_matches"] == 8 and fila["n_mismatches"] == 0

    def test_visita_3_limpia(self):
        resultado = inferir_visita(_presencia_completa({-4, -3, 1}))
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 3
        assert fila["n_matches"] == 8 and fila["n_mismatches"] == 0

    def test_visita_4_limpia(self):
        resultado = inferir_visita(_presencia_completa({-5, -4, -1}))
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 4
        assert fila["n_matches"] == 8 and fila["n_mismatches"] == 0

    def test_no_respuesta_en_visita_intermedia(self):
        # Presente en t0, t0+1, t0+5 (el propio t, offset 0); ausente en t0+4
        # (offset -1 relativo a t=t0+5) por no respuesta puntual, y ausente
        # en todo lo demás (ya salió de la rotación). Debe ganar la
        # hipótesis 4 pese al hueco, con el mismatch marcado en offset -1.
        resultado = inferir_visita(_presencia_completa({-5, -4}))
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 4
        assert fila["n_matches"] == 7 and fila["n_mismatches"] == 1
        assert not fila["ambigua"]

    def test_reentrada_imposible_da_ambigua(self):
        # Presente en t-5, t (dado) y t+5 -- ventana de 10 trimestres, más
        # ancha que cualquier calendario de 4 visitas posible. Empatan las
        # hipótesis 1 y 4 (score +2 cada una).
        resultado = inferir_visita(_presencia_completa({-5, 5}))
        fila = resultado.iloc[0]
        assert fila["ambigua"]
        assert pd.isna(fila["visita_inferida"])
        assert fila["n_matches"] == 5 and fila["n_mismatches"] == 3

    def test_offset_no_determinable_no_cuenta_ni_a_favor_ni_en_contra(self):
        presencia = _presencia_completa({1, 4, 5})
        presencia[5] = None
        resultado = inferir_visita(presencia)
        fila = resultado.iloc[0]
        assert fila["visita_inferida"] == 1
        assert fila["n_determinables"] == 7
        assert fila["n_matches"] == 7 and fila["n_mismatches"] == 0


class TestClavesVivienda:
    def test_dedup(self):
        hogar = pd.DataFrame({"CODUSU": ["A", "A"], "NRO_HOGAR": [1, 2], "AGLOMERADO": [10, 10]})
        assert len(claves_vivienda(hogar)) == 1

    def test_claves_hogar_no_colapsa_multihogar(self):
        hogar = pd.DataFrame({"CODUSU": ["A", "A"], "NRO_HOGAR": [1, 2], "AGLOMERADO": [10, 10]})
        assert len(claves_hogar(hogar)) == 2


class TestClasificarPatronApariciones:
    def _cohorte(self):
        return pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})

    def test_trayectoria_completa(self):
        t1 = t4 = t5 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        resultado = clasificar_patron_apariciones(self._cohorte(), t1, t4, t5)
        assert resultado.iloc[0]["patron"] == "1111"

    def test_corte_temprano(self):
        t1 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        t4 = t5 = pd.DataFrame({"CODUSU": [], "NRO_HOGAR": [], "AGLOMERADO": []})
        resultado = clasificar_patron_apariciones(self._cohorte(), t1, t4, t5)
        assert resultado.iloc[0]["patron"] == "1100"

    def test_hueco_con_reaparicion(self):
        t1 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        t4 = pd.DataFrame({"CODUSU": [], "NRO_HOGAR": [], "AGLOMERADO": []})
        t5 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        resultado = clasificar_patron_apariciones(self._cohorte(), t1, t4, t5)
        assert resultado.iloc[0]["patron"] == "1101"

    def test_nro_hogar_distinto_no_matchea_como_el_mismo_hogar(self):
        t1 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        # En t0+4 la vivienda sigue pero con otro NRO_HOGAR -- no es la misma
        # clave exacta, cuenta como ausente ahí (mismo criterio que D9).
        t4 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [2], "AGLOMERADO": [10]})
        t5 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        resultado = clasificar_patron_apariciones(self._cohorte(), t1, t4, t5)
        assert resultado.iloc[0]["patron"] == "1101"

    def test_trimestre_no_determinable_se_marca_distinto_de_ausencia(self):
        t1 = pd.DataFrame({"CODUSU": ["A"], "NRO_HOGAR": [1], "AGLOMERADO": [10]})
        resultado = clasificar_patron_apariciones(self._cohorte(), t1, None, None)
        assert resultado.iloc[0]["patron"] == "11??"


def _pares(codusu: list[str], nro_hogar_t: list[int], aglomerado: list[int], identidad) -> pd.DataFrame:
    return pd.DataFrame({
        "CODUSU": codusu, "NRO_HOGAR_t": nro_hogar_t, "NRO_HOGAR_th": nro_hogar_t,
        "AGLOMERADO": aglomerado, "identidad_confirmada": pd.array(identidad, dtype="boolean"),
    })


class TestEncadenarTrayectoria:
    def test_encadena_solo_lo_presente_en_los_3_eslabones(self):
        pares_h1_t = _pares(["A", "B"], [1, 1], [10, 10], [True, False])
        pares_h4_t = _pares(["A", "B"], [1, 1], [10, 10], [True, True])
        # B falta en el eslabón t+4->t+5 (hueco intermedio) -- no debe encadenar.
        # C aparece recién acá -- tampoco debe encadenar (no estaba en t ni t+4).
        pares_h1_t4 = _pares(["A", "C"], [1, 1], [10, 10], [True, True])

        resultado = encadenar_trayectoria(pares_h1_t, pares_h4_t, pares_h1_t4)

        assert set(resultado["CODUSU"]) == {"A"}
        fila = resultado.iloc[0]
        assert fila["identidad_confirmada_1"] and fila["identidad_confirmada_h4"] and fila["identidad_confirmada_2"]
        assert fila["identidad_confirmada_trayectoria"]

    def test_and_kleene_con_na(self):
        # D: un eslabón NA (jefe ambiguo), los otros dos True -> trayectoria NA.
        # E: un eslabón False, otro NA -> trayectoria False (Kleene: False & NA = False).
        pares_h1_t = _pares(["D", "E"], [1, 1], [10, 10], [True, True])
        pares_h4_t = _pares(["D", "E"], [1, 1], [10, 10], [pd.NA, False])
        pares_h1_t4 = _pares(["D", "E"], [1, 1], [10, 10], [True, pd.NA])

        resultado = encadenar_trayectoria(pares_h1_t, pares_h4_t, pares_h1_t4).set_index("CODUSU")

        assert pd.isna(resultado.loc["D", "identidad_confirmada_trayectoria"])
        assert resultado.loc["E", "identidad_confirmada_trayectoria"] == False  # noqa: E712

    def test_quiebre_de_era_no_matchea_y_no_rompe(self):
        # CODUSU histórico (6 dígitos) de un lado, regular (29 caracteres) del
        # otro -- 0% de overlap por formato incompatible (D7), no por error.
        pares_h1_t = _pares(["393389"], [1], [10], [True])
        pares_h4_t = _pares(["TQXXXXXXXXXXXXXXXXXXX00000000"], [1], [10], [True])
        pares_h1_t4 = _pares(["TQXXXXXXXXXXXXXXXXXXX00000000"], [1], [10], [True])

        resultado = encadenar_trayectoria(pares_h1_t, pares_h4_t, pares_h1_t4)

        assert resultado.empty


class TestVerificarContraH4T1:
    def test_marca_presente_y_ausente_en_verificacion(self):
        trayectoria = pd.DataFrame({"CODUSU": ["A", "B"], "NRO_HOGAR": [1, 1], "AGLOMERADO": [10, 10]})
        pares_h4_t1 = _pares(["A"], [1], [10], [True])

        resultado = verificar_contra_h4_t1(trayectoria, pares_h4_t1).set_index("CODUSU")

        assert resultado.loc["A", "presente_en_verificacion"]
        assert resultado.loc["A", "identidad_confirmada_verificacion"]
        assert not resultado.loc["B", "presente_en_verificacion"]
        assert pd.isna(resultado.loc["B", "identidad_confirmada_verificacion"])


class TestCohorteCompletable:
    PUBLICADOS = {(2015, 1), (2015, 2), (2016, 2), (2016, 3),
                  (2020, 1), (2020, 2), (2020, 3), (2020, 4), (2021, 1), (2021, 2)}

    @staticmethod
    def _historico(anio, _trimestre):
        return anio <= 2015

    def test_completable(self):
        resultado = cohorte_completable(2020, 1, self.PUBLICADOS, self._historico)
        assert resultado == {"completable": True, "motivo": None}

    def test_trimestre_no_publicado(self):
        resultado = cohorte_completable(2015, 1, self.PUBLICADOS, self._historico)
        assert not resultado["completable"]
        assert "no publicado" in resultado["motivo"]

    def test_cruce_de_era_en_eslabon_h4(self):
        # t0=2015T2 (histórico), t0+4=2016T2 (regular) -- ambos publicados,
        # pero el eslabón h=4 cruza la frontera de CODUSU (D7).
        publicados = self.PUBLICADOS | {(2015, 3)}
        resultado = cohorte_completable(2015, 2, publicados, self._historico)
        assert not resultado["completable"]
        assert "frontera" in resultado["motivo"]


class TestEncadenarDosEslabones:
    def test_interseca_por_clave_de_hogar(self):
        # B falta en pares_2 -- no debe entrar al resultado (a diferencia de un diseño
        # de 3 eslabones, acá no hace falta el eslabón del medio para nada).
        pares_1 = _pares(["A", "B"], [1, 1], [10, 10], [True, True])
        pares_2 = _pares(["A", "C"], [1, 1], [10, 10], [False, True])

        resultado = encadenar_dos_eslabones(pares_1, pares_2).set_index("CODUSU")

        assert set(resultado.index) == {"A"}
        assert resultado.loc["A", "identidad_confirmada_1"]
        assert not resultado.loc["A", "identidad_confirmada_ambos"]  # True & False = False


class TestContarNDiseno:
    def test_cuenta_vinculada_e_identidad_por_origen(self):
        tabla = pd.DataFrame({
            "ANO4_t": [2010, 2010, 2011],
            "TRIMESTRE_t": [1, 1, 2],
            "identidad_confirmada_ambos": pd.array([True, False, pd.NA], dtype="boolean"),
        })

        resultado = contar_n_diseno(tabla, "identidad_confirmada_ambos").set_index(["ANO4_t", "TRIMESTRE_t"])

        assert resultado.loc[(2010, 1), "vinculada"] == 2
        assert resultado.loc[(2010, 1), "identidad_confirmada"] == 1
        assert resultado.loc[(2011, 2), "vinculada"] == 1
        assert resultado.loc[(2011, 2), "identidad_confirmada"] == 0  # NA no cuenta como confirmada


class TestClasificarPerdidaPorPatron:
    def test_completa(self):
        assert clasificar_perdida_por_patron(pd.Series(["1111"])).iloc[0] == "completa"

    def test_cortes_limpios(self):
        resultado = clasificar_perdida_por_patron(pd.Series(["1000", "1100", "1110"]))
        assert list(resultado) == ["se_pierde_en_t0_1", "se_pierde_en_t0_4", "se_pierde_en_t0_5"]

    def test_hueco_con_reaparicion(self):
        resultado = clasificar_perdida_por_patron(pd.Series(["1011", "1101", "1001", "1010"]))
        assert all(resultado == "hueco_con_reaparicion")

    def test_no_determinable(self):
        assert clasificar_perdida_por_patron(pd.Series(["11??"])).iloc[0] == "no_determinable"
