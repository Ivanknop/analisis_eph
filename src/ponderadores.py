"""Ponderador de hogar para la era histórica de la EPH (sin `PONDIH`),
usado por la capa de distancia histórica (`data/02b_distancia_lp/hogar/`,
era retropolada).
"""
from __future__ import annotations

import pandas as pd


def peso_hogar_historico(individual: pd.DataFrame) -> pd.Series:
    """Ponderador de hogar para la era histórica (sin `PONDIH`): `PONDERA`
    del hogar, indexado por `(CODUSU, NRO_HOGAR)`."""
    # PONDERA es el factor de expansión del hogar (confirmado constante entre
    # sus integrantes), no de la persona; para ponderar personas se multiplica
    # por la cantidad de integrantes (mismo patrón que PONDIH x n_miembros
    # en la era regular), no se suma PONDERA directamente.
    por_hogar = individual.groupby(["CODUSU", "NRO_HOGAR"])["PONDERA"]
    if not (por_hogar.nunique() == 1).all():
        raise ValueError("PONDERA no es constante dentro de algún hogar -- revisar antes de ponderar")
    return por_hogar.first()
