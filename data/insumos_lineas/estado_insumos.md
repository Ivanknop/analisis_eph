# Estado de insumos — retropolación de líneas de pobreza (Tornarolli 2018 / CEDLAS DT 226)

Ronda de **conseguir, verificar y registrar insumos**. No se calculó ninguna
línea, backcast, distancia ni tasa. Fecha de esta ronda: 2026-10-19.

Insumos 1, 6 y 7 ya estaban en el repo de una ronda anterior; siguen siendo
válidos y no se redescargaron. Insumos 2, 3 y 4 son los nuevos de esta ronda, en
`data/insumos_lineas/` (manifiesto con URL/fecha/SHA-256 en
`data/insumos_lineas/manifest.json`).

## Tabla de estado

| # | Insumo | Conseguido | Fuente / URL | Cobertura | Huecos | Observaciones |
|---|---|---|---|---|---|---|
| 1 | Canastas regionales oficiales desde abril 2016 (CBA/CBT por adulto equivalente, 6 regiones) | **sí** (ya en el repo) | `data/canastas_indec/{cba,cbt}_regional_crudo.csv` — infra.datos.gob.ar, datasets sspm/445 y sspm/446 | abril 2016 – diciembre 2025, mensual, 6 regiones (GBA, Pampeana, Noroeste, Noreste, Cuyo, Patagonia) | ninguno detectado | `src/canasta_regional.py` ya lo descarga/parsea. No se re-bajó esta ronda. |
| 2 | CBA/CBT GBA diciembre 2006, metodología nueva (Metodología INDEC N°22) | **sí** (nuevo, esta ronda) | PDF: `https://www.indec.gob.ar/ftp/cuadros/sociedad/EPH_metodologia_22_pobreza.pdf` → cacheado en `data/insumos_lineas/metodologia_22_pobreza.pdf`. Valores extraídos a `data/insumos_lineas/dic_2006_gba_metodologia22.csv` | un solo mes (dic-2006), GBA | — | **Cuadro 5, página 15**, columna "Nuevo" (ENGHo 1996/97 validada con 2004/05): **CBA_ae = $149,13**, **CBT_ae = $389,28**, ICE = 2,61 (pesos corrientes/mes). La columna "Anterior" (ENGHo 1985/86: CBA=$134,14, CBT=$291,08) queda solo de contraste — no es el ancla, no usar. Chequeo de consistencia: 149,13×2,61=389,22 ≈ 389,28 (redondeo), ok. |
| 3 | IPC-GBA mensual 2003-2006, nivel general y alimentos | **sí** (nuevo, esta ronda) | `https://infra.datos.gob.ar/catalog/sspm/dataset/178/distribution/178.1/download/ipc-historico.csv` → `data/insumos_lineas/ipc_gba_historico_1943_2008_crudo.csv` | 1943-01 a 2008-04, mensual; **2003-2006 completo, 48/48 meses, nivel general y alimentos_bebidas sin huecos** | ninguno en la ventana pedida | Mismo índice que `sh_ipc_2008.xls` (sitioanterior.indec.gob.ar, candidato alternativo verificado pero no descargado — redundante) y que el dataset 96 (`indice-precios-al-consumidor-apertura-por-capitulos-base-2008-mensual.csv`, base abril-2008, 1993-2013, cacheado también como `ipc_gba_base2008_capitulos_crudo.csv` para cruce). Las tres fuentes son el mismo empalme oficial INDEC, por canales de publicación distintos — se cachearon dos (178 y 96) para contraste cruzado. |
| 4 | Índice alternativo 2007 – abril 2016, nivel general (y alimentos si existe) | **candidato principal conseguido**; resto relevado, ver comparación abajo | ver comparación de candidatos | — | ver comparación | Bevacqua **no se rebuscó** (decisión del usuario esta ronda: descartado, no se vuelve a buscar sin una fuente nueva y concreta). |
| 5 | Empalme en abril 2016 | **documentado, no calculado** | — | — | — | Ver sección "Empalme" abajo. |
| 6 | Tabla de adultos equivalentes (INDEC) | **sí** (ya en el repo, y verificado esta ronda contra la fuente primaria) | `src/adulto_equivalente.py` (`TABLA_ADULTO_EQUIVALENTE`, hardcodeada en el módulo) | todas las bandas de edad/sexo | ninguno | Comparé fila por fila contra la **Tabla de equivalencias oficial** (Metodología N°22, Anexo 7.1, p.21, ya cacheada en `metodologia_22_pobreza.pdf`) — coincide exactamente (ej. varón 30-45=1,00; mujer 61-75=0,67; "menor de 1 año"/0,35 ↔ banda 9-12 meses=0,35). No hacía falta volver a cargarla. |
| 7a | Serie vieja CBA/CBT GBA, sep-2000/dic-2013 (`sh-cba2.xls`) | **sí** (ya en el repo) | `data/canastas_indec/retropolacion/indec_pre_intervencion/sh-cba2.xls` | sep-2000 a dic-2013 | — | Solo contraste (metodología anterior), nunca para calcular. |
| 7b | Anexo B Tornarolli (tasas CEDLAS por semestre) | **sí** (ya en el repo) | `data/canastas_indec/retropolacion/cedlas/{tornarolli_2018_documento_trabajo_226.pdf, series_comparables_anexo_b.csv}` | 2003-II a 2017-I, semestral | — | Ya transcripto del PDF (p.26-27). |

## Comparación de candidatos para insumo 4 (índice alternativo 2007–abril 2016)

| Candidato | Disponible | Cobertura real | Alimentos | Base | Fuente |
|---|---|---|---|---|---|
| **a) Bevacqua** | **no se rebuscó** | — | — | — | `.claude/decisiones.md`: "no se consigue como dataset descargable pese a búsqueda exhaustiva. No volver a buscarlo sin una fuente nueva y concreta" — se respeta esa decisión esta ronda (confirmado por el usuario). Tornarolli/CEDLAS sí lo usa en su propio empalme (Anexo A del paper, ya en el repo), pero como descripción metodológica, no como serie numérica descargable. |
| **b) San Luis individual** | **no** (sitio actual) | El sitio actual (`estadistica.sanluis.gov.ar`, sección IPC San Luis) solo lista informes mensuales **desde enero 2016** — ~96 informes, todos en PDF suelto, ninguno de 2007-2012 | — | — | Verificado por scraping del listado de la sección; no hay serie consolidada descargable ni PDFs del tramo 2007-2012 en el sitio vigente. Contacto provisto por el sitio para pedido directo: `deyc@sanluis.gov.ar` (no contactado esta ronda — "manual" si se decide perseguirlo). |
| **c) CABA (IPCBA)** | **sí, descargable** | **Desde julio 2012** en adelante (`estadisticaciudad.gob.ar`, banco de datos) — no cubre 2007-jun2012 | no verificado en esta ronda (no se descargó el archivo, solo confirmada la cobertura) | jul-2011=100 (serie empalmada con bases posteriores) | `estadisticaciudad.gob.ar/eyc/categoria-banco-datos/indice-mensual/` — URL exacta del archivo descargable no confirmada todavía (requiere bajar desde el banco de datos, no se automatizó esta ronda). |
| **d) IPC Provincias (CIFRA-CTA / Fernández y González 2019)** | **sí, descargable — CONSEGUIDO** | **enero 2007 a diciembre 2018, mensual, sin huecos (144/144 meses)** — cubre de sobra el tramo 2007–abril 2016 | **no** (un solo nivel general; es un promedio ponderado de IPC provinciales que no desagregan capítulos de forma uniforme) | enero 2014=100 | `https://centrocifra.org.ar/wp-content/uploads/2023/08/IPC-Provincias-2007-2018.xlsx` → `data/insumos_lineas/ipc_provincias_cifra_2007_2018_crudo.xlsx`. Metodología documentada en `https://centrocifra.org.ar/wp-content/uploads/2023/08/Nota-metodologica-IPC-Provincias.pdf` (también cacheada para referencia, no en el manifiesto porque es solo nota metodológica, no insumo numérico). Citado explícitamente: extraído de **Fernández, A.L. y M. González (2019), "Cambios metodológicos en la medición de la pobreza en Argentina. Revisión y construcción de series homogéneas", DT N°16, CIFRA, abril** — el trabajo que el pedido original menciona. |

**Qué es el índice (d) — ya resuelve la combinación encadenada pedida**: el
"IPC Provincias" de CIFRA-CTA **ya es** la combinación "provincial 2007–2012 +
CABA desde julio 2012" que se pidió relevar, solo que construida y mantenida
por los propios autores (Fernández y González, CIFRA), no por mí esta ronda:

- 2007–2011: promedio ponderado de 7 provincias (Jujuy, Neuquén, Paraná/Entre
  Ríos, Rawson-Trelew/Chubut, Salta, Santa Rosa/La Pampa, Viedma/Río Negro),
  ponderado por gasto de consumo de cada provincia (ENGHo).
- 2012: se reevalúa — entran Posadas/Misiones, San Luis, Santa Fe/Rosario,
  Ushuaia/Tierra del Fuego; salen Paraná y Viedma (discontinuados) → "IPC 9
  provincias".
- **Desde julio 2012**: se incorpora el índice de la Ciudad de Buenos Aires
  (único de cobertura real, no solo un aglomerado).
- Discontinuaciones puntuales de provincias (ej. 5 de 9 en 1S2018) se
  imputan por la variación promedio de las que sí publican ese mes — límite
  metodológico documentado por los propios autores, no oculto.
- No construí yo este empalme ni evalué sus huecos internos por provincia —
  eso es trabajo de Fernández/González/CIFRA, ya hecho y citado; si se decide
  usar este índice, haría falta solo el empalme externo con el IPC oficial en
  abril 2016 (ver sección siguiente), no rearmar la combinación provincial.

**Otros provinciales relevados, no re-sourceados aparte** (ya son insumo de
(d), evaluarlos por separado sería redundante salvo que se descarte el
índice de CIFRA):
- **Neuquén, Chubut**: miembros directos del IPC Provincias de CIFRA.
- **Santa Fe (IPEC)**: cobertura propia 2003-2013 (base 2003=100), discontinuada
  cuando adoptó el índice nacional — ya incorporada a (d) desde 2012 (y
  "desde 2014" en la reevaluación de 2018 de CIFRA, por discontinuidades
  puntuales). No se descargó la serie provincial standalone esta ronda.
- **Mendoza (DEIE)**: no es miembro de CIFRA. El sitio vigente
  (`deie.mendoza.gov.ar`) solo expone informes PDF mensuales recientes (base
  2010=100) — no se confirmó una serie consolidada ni cobertura 2007-2012;
  **no verificado**, candidato débil si hiciera falta uno adicional.
- **Córdoba**: dataset oficial en datos.gob.ar
  (`indice-de-precios-al-consumidor-de-la-provincia-de-cordoba-base-2014-100`,
  Subsecretaría de Programación Macroeconómica) cubre **solo jul-2013 a
  ago-2017** — no llega a 2007, cobertura parcial e insuficiente por sí solo
  para el tramo pedido.

**Conclusión de esta comparación (no es una elección, es un insumo para que
decida el usuario)**: el candidato **(d) IPC Provincias/CIFRA** es, de lejos,
el más completo y verificado — único con cobertura mensual sin huecos de todo
2007–abril 2016, ya descargado, y ya resuelve la combinación provincial+CABA
que se pidió relevar. Su limitación principal es que no desagrega alimentos
por separado (solo nivel general).

## Empalme en abril 2016 (insumo 5) — documentado, sin calcular

El ancla oficial de abril 2016 corresponde a la serie **"IPC-GBA. Núcleo. Base
abr-2016=100"** que el propio INDEC publicó como puente hasta que arrancó el
IPC Nacional (base dic-2016=100) — confirmado vía datos.gob.ar, serie
`103.1_I2N_2016_M_15`. Es decir, el mes de empalme natural con **cualquier**
candidato de insumo 4 es **abril de 2016**, llevando el índice elegido a
coincidir en nivel (factor de reescala = 100 / índice_candidato en abril-2016)
con esa base.

Para el candidato (d) (CIFRA, único con datos efectivamente descargados):
abril de 2016 **está dentro** de su cobertura (enero 2007–diciembre 2018) —
el empalme es directo, sin necesidad de interpolar ni extrapolar: se
reescala toda la serie CIFRA por el factor `100 / valor_cifra(abril-2016)`
y a partir de ahí coincide, por construcción, con el nivel de la serie oficial
en ese mes exacto. Mismo razonamiento valdría para CABA (IPCBA), que también
cubre abril-2016 si se confirma su descarga.
