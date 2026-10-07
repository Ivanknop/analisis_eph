"""Utilidades compartidas por los notebooks del pipeline: registro de corridas,
resumen incremental por trimestre, y helpers de staleness de particiones para
el parámetro `REHACER`.
"""
# No importa nada de `eph_client`/`armonizacion`/`tipado_nucleo` -- es de más
# bajo nivel que esos tres, cualquier notebook del pipeline lo puede usar.
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def obtener_commit_git(repo: Path) -> str:
    """`git rev-parse --short HEAD`, o `"sin commit"` si falla o no hay
    commits todavía. Nunca propaga la excepción."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "sin commit"
    if resultado.returncode == 0 and resultado.stdout.strip():
        return resultado.stdout.strip()
    return "sin commit"


def ruta_particion(base_dir: Path, tipo: str, anio: int, trimestre: int) -> Path:
    return base_dir / tipo / f"ANO4={anio}" / f"TRIMESTRE={trimestre}" / "data.parquet"


def particion_existe(base_dir: Path, tipo: str, anio: int, trimestre: int) -> bool:
    return ruta_particion(base_dir, tipo, anio, trimestre).exists()


def particion_desactualizada(origen: Path, destino: Path) -> bool:
    """`True` si `destino` no existe, o si `origen` es más nuevo (mtime) que
    `destino`."""
    # Si falta `origen` no se marca desactualizado por esa sola ausencia
    # (no debería pasar en uso normal: `destino` depende de `origen`).
    if not destino.exists():
        return True
    if not origen.exists():
        return False
    return origen.stat().st_mtime > destino.stat().st_mtime


def registrar_corrida(archivo: Path, notebook: str, parametros: dict, n_trimestres: int,
                       n_filas: int, duracion_segundos: float, commit_git: str) -> None:
    """Agrega una fila a `archivo` -- append puro, cada corrida es un evento
    distinto del notebook, no hay upsert acá (a diferencia de
    `actualizar_resumen_trimestre`)."""
    fila = {
        "notebook": notebook,
        "fecha_hora": pd.Timestamp.now().isoformat(timespec="seconds"),
        "commit_git": commit_git,
        "parametros_json": json.dumps(parametros, ensure_ascii=False, default=str),
        "n_trimestres_procesados": n_trimestres,
        "n_filas_escritas": n_filas,
        "duracion_segundos": round(duracion_segundos, 1),
    }
    df_nueva = pd.DataFrame([fila])
    archivo.parent.mkdir(parents=True, exist_ok=True)
    if archivo.exists():
        # dtype=str al releer: sin esto, pandas infiere "01" como el entero 1
        # (pierde el cero a la izquierda del nombre del notebook) al concatenar.
        df_previo = pd.read_csv(archivo, dtype=str)
        df_final = pd.concat([df_previo, df_nueva], ignore_index=True)
    else:
        df_final = df_nueva
    df_final.to_csv(archivo, index=False)


def actualizar_resumen_trimestre(archivo: Path, filas_nuevas: list[dict],
                                  claves: tuple[str, ...] = ("anio", "trimestre", "tipo")) -> None:
    """Upsert por `claves`: reemplaza las filas existentes con esas claves y
    agrega las nuevas -- no duplica al re-correr el mismo trimestre."""
    # Pensado para escribirse apenas termina cada trimestre dentro del loop
    # principal, no en un solo batch al final -- así una corrida interrumpida
    # deja registro de hasta dónde llegó.
    if not filas_nuevas:
        return
    df_nuevas = pd.DataFrame(filas_nuevas)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    if archivo.exists():
        df_previo = pd.read_csv(archivo)
        # comparar como texto: tras el read_csv, "anio"/"trimestre" numéricos
        # vuelven int64, pero las claves de df_nuevas pueden llegar como
        # python int -- sin normalizar, la comparación de tuplas nunca matchea
        # y el upsert termina duplicando en vez de reemplazar.
        claves_nuevas = set(df_nuevas[list(claves)].astype(str).itertuples(index=False, name=None))
        claves_previas = df_previo[list(claves)].astype(str).apply(tuple, axis=1)
        mask_mantener = ~claves_previas.isin(claves_nuevas)
        df_final = pd.concat([df_previo[mask_mantener], df_nuevas], ignore_index=True)
    else:
        df_final = df_nuevas
    df_final = df_final.sort_values(list(claves)).reset_index(drop=True)
    df_final.to_csv(archivo, index=False)

def cargar_hogar(anio: int, trimestre: int, columnas: list[str], ruta_nucleo: Path) -> pd.DataFrame | None:
    """Lee la partición `hogar` de `ruta_nucleo` y se queda solo con las
    `columnas` pedidas que existan. `None` si la partición no existe."""
    p = ruta_particion(ruta_nucleo, "hogar", anio, trimestre)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df[[c for c in columnas if c in df.columns]].copy()


def cargar_individual(
    anio: int, trimestre: int, columnas: list[str],
    ruta_nucleo: Path, ruta_armonizado: Path | None = None,
) -> pd.DataFrame | None:
    """Igual que `cargar_hogar`, para `individual`. Si se pasa
    `ruta_armonizado`, además mergea `NIVEL_ED` desde esa capa."""
    p = ruta_particion(ruta_nucleo, "individual", anio, trimestre)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    ind = df[[c for c in columnas if c in df.columns]].copy()
    if ruta_armonizado is None:
        return ind
    pa = ruta_particion(ruta_armonizado, "individual", anio, trimestre)
    if pa.exists():
        arm = pd.read_parquet(pa, columns=["CODUSU", "NRO_HOGAR", "COMPONENTE", "NIVEL_ED"])
        arm["NIVEL_ED"] = pd.to_numeric(arm["NIVEL_ED"], errors="coerce").astype("Int8")
        ind = ind.merge(arm, on=["CODUSU", "NRO_HOGAR", "COMPONENTE"], how="left")
    else:
        ind["NIVEL_ED"] = pd.array([pd.NA] * len(ind), dtype="Int8")
    return ind