
ORDEN_SITUACION_LABORAL = ["algun_asalariado_formal", "algun_independiente_sin_asalariado_formal",
                            "solo_asalariados_informales", "sin_ocupados"]

ORDEN_QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]

ORDEN_GRUPOS_INGRESO = ORDEN_QUINTILES + ["decil_0", "ingreso_no_declarado"]

# Nombres para los outputs de las Fases C y D (`.claude/decisiones.md`, "Convención de
# nombres"). Las columnas existentes no se renombran: estas tablas las traducen.
ERAS = ["historica", "regional_2016"]

ANIO_DESDE_UNIVERSO = {"universo_2003_2025": 2003, "universo_2016_2025": 2016}
UNIVERSOS = list(ANIO_DESDE_UNIVERSO)

OBJETIVOS = ["simetrica", "log"]
CRITERIOS_IMPUTACION = ["imputacion_conserva", "imputacion_excluye"]
COLUMNA_VALIDEZ = {
    ("simetrica", "imputacion_conserva"): "par_valido_simetrica",
    ("simetrica", "imputacion_excluye"): "par_valido_c",
    ("log", "imputacion_conserva"): "par_valido_log",
    ("log", "imputacion_excluye"): "par_valido_log_c",
}

VARIANTES_PRECARIEDAD = {
    "informal_desocupado": "share_precario_A_t",
    "mas_cuenta_propia": "share_precario_B_t",
    "mas_cuenta_propia_y_patron": "share_precario_C_t",
    "mas_cuenta_propia_baja_calificacion": "share_precario_D_t",
}

INDICES_VIVIENDA = {"deficit_ancla_nbi": "vivienda_precaria_v1_t", "deficit_ampliado": "vivienda_precaria_v2_t"}

SENSIBILIDADES_CORTES = ["con_marcados", "sin_cortes_puntuales", "sin_cortes_ni_intervenido"]