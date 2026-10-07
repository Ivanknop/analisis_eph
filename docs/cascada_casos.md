# Cascada de casos — informe breve

Resumen de `notebooks/07_cascada_targets.ipynb` (D24), corrida completa sobre
los 86 trimestres de origen × 2 horizontes (`REHACER=True`, recálculo
completo). Detalle metodológico completo en `docs/decisiones_metodologicas.md`,
entrada D24. Todos los números son [confirmado] sobre la corrida real, salvo
donde se marca lo contrario.

**Lectura primaria de los targets (confirmada por Iván, D24)**: **Y1 =
lexicográfica** (D16 tal cual queda de sensibilidad); **Y2 = `sale_por_empleo`,
variante principal** (`persiste_principal` queda de lectura descriptiva, es su
complemento exacto). El ingreso no declarado **no excluye** Y1/Y2 — queda
como `grupo_ingreso_t="no_declarado"`, visible en el desglose; la exclusión
se reserva para cuando se arme la grilla 3×3.

## Totales

- Pares hogar-a-hogar: 614.438 (`h=1`) / 508.568 (`h=4`) — exacto contra
  `panel_resumen_pares.csv`/D10.
- `identidad_confirmada`: 554.574 (`h=1`) / 450.251 (`h=4`).
- Población Y1 (≥1 activo estable en ambas puntas, tras excluir D13 en `h=4`;
  **sin** excluir ingreso no declarado): 241.033 (`h=1` histórica) / 188.131
  (`h=1` regular) / 184.587 (`h=4` histórica) / 152.419 (`h=4` regular) —
  exacto contra D16.
- 160 de los 172 orígenes×`h` posibles tienen destino publicado y comparable;
  12 quedan `sin_destino_publicado` (huecos de publicación, borde final de la
  serie, o cruce del corte de `CODUSU` de D7) y no entran en ningún cálculo
  de atrición ni de Y1/Y2 — quedan en la tabla solo con `hogares_en_t`.

## Tasas primarias, por `h` × era

Misma población que los totales de arriba (Y1 excluyendo los orígenes D13 en
`h=4`, sin excluir ingreso no declarado):

| `h` | era | Y1 lexicográfica | Y2 sale_por_empleo (principal) |
|---|---|---|---|
| 1 | histórica | 10,2% | 9,9% |
| 1 | regular | 9,8% | 9,0% |
| 4 | histórica | 11,1% | 15,0% |
| 4 | regular | 11,1% | 12,5% |

## Positivos por fold (walk-forward, un fold por año 2008-2022)

Año de test = primer trimestre del año con destino válido para `h=1` y `h=4`
a la vez (T1 en casi todos los años, T2 en 2016 — T1 no está publicado — y
**2015 se descarta entero**, ningún trimestre de ese año tiene destino
válido para los dos horizontes a la vez). Con este ajuste **no queda ningún
fold de desarrollo con N=0** (la primera versión de esta corrida tenía 3,
todos por el hueco de publicación 2015T3-2016T1).

| `h` | Y1 — mínimo | Y1 — mediana | Y1 — máximo | Y2 (`sale_por_empleo`) — mínimo | mediana | máximo |
|---|---|---|---|---|---|---|
| 1 | 419 | 521 | 653 | 140 | 232 | 281 |
| 4 | 232 | 527 | 615 | 167 | 297 | 375 |

`cascada_folds.csv`, columna `positivos_y2`, ya cuenta directamente
`sale_por_empleo` variante *principal* (la lectura primaria de Y2, D24 — no
`persiste`). **520 celdas** (`fold × h × partición × ancla_con_deficit ×
grupo_ingreso`, de 834) quedan `celda_chica` (menos de 100 positivos en Y1
o Y2) — sube de las 438 que daba la versión anterior (que contaba
`persiste`), porque `sale_por_empleo` es la clase minoritaria.

**Holdout** (origen desde 2023T1, de uso único): `h=1` — 52.035 pares Y1
(5.135 positivos lexicográficos, 9,9%), 24.015 pares Y2 (2.128
`sale_por_empleo`, 8,9%); `h=4` — 33.655 pares Y1 (3.822 positivos, 11,4%),
15.892 pares Y2 (1.892 `sale_por_empleo`, 11,9%).

## Atrición por período

`%` neto de la rotación teórica (2-2-2, D9), promedio de los orígenes con
destino válido:

| `h` | histórica | regular | período intervenido (2007-2015) | fuera del período intervenido |
|---|---|---|---|---|
| 1 | 4,7% | 6,7% | 4,8% | 6,1% |
| 4 | 8,3% | 9,3% | 9,4% | 8,4% |

La atrición real (neta de rotación) es sistemáticamente más alta en `h=4`
que en `h=1` (pasa más tiempo entre `t` y `t+h`, más oportunidad de mudanza
o rechazo) y algo más alta en la era regular que en la histórica — ninguna
de las dos diferencias es grande, del orden de 1-2 puntos porcentuales.

## Ingreso no declarado por período

Sobre pares con `identidad_confirmada==True`: **0% en la era histórica**
(`PONDIH` de hogar no existe antes de 2016, D3 — queda `NA`, no se puede
medir) y **~29,3% (`h=1`) / ~29,8% (`h=4`) en la era regular** (al menos una
punta, `t` o `t+h`, con `PONDIH==0`). **No se excluye de Y1/Y2** (decisión
de Iván, D24) — queda como categoría propia `grupo_ingreso_t="no_declarado"`
en el Parquet intermedio, y coincide al 100% con la era regular (ningún
caso antes de 2016, verificado sobre los 101.251 hogares-par marcados
`"no_declarado"`). Por eso cualquier modelo que use `grupo_ingreso_t` tiene
que incluir **`era` como variable explícita** — si no, "no declarado" se
confunde con un efecto de era (cambio de cuestionario/ponderación/cobertura)
en vez de leerse como una posición de ingreso faltante.

## Quiebre de cuestionario 2023T4 (D2) dentro del holdout

De los 114.343 pares con `identidad_confirmada==True` en el holdout, el
**88,6%** tiene al menos una punta (`t` o `t+h`) en el cuestionario
posrediseño (2023T4 en adelante) — **100% en `h=4`** (el origen más
temprano del holdout, 2023T1, ya tiene destino 2024T1, posterior a 2023T4
por construcción) y **81,1% en `h=1`** (los primeros trimestres de origen
del holdout, 2023T1-2023T3, todavía tienen destino pre-2023T4). El holdout
queda entonces dominado por el cuestionario nuevo, sobre todo en `h=4` —
cualquier variable que dependa del bloque reestructurado por D2 (las `V*`
de estrategias de subsistencia) hay que leerla con esa salvedad en el
holdout.

## Verificación de los códigos nuevos de 2016 contra el diseño de registro

`vivienda_frecuencias.csv` marca 6 variables (`II1, II2, II3_1, II5_1, IV2,
IV9`) con distinto conjunto de valores observados entre 2015T2 y 2016T2.
Contra el diseño de registro (codebook p.7-9): ninguna es una redefinición
real. `IV2=99` es el Ns/Nr general de un campo N(2); `IV9=3` ya es una
categoría válida del diseño ("...fuera del terreno"), solo ausente por azar
muestral en 2015T2; `II1`/`II2`/`II3_1`/`II5_1` son conteos sin tope fijo,
su rango varía por muestra. Solo `II1` e `II5_1` entran en
`perfil_hogar.calcular_componentes_ancla`, y `II5_1` únicamente alimenta
`hacinamiento_para_dormir`, que no integra `ancla_con_deficit` — ningún
`ancla_con_deficit` ya calculado cambia por este hallazgo. No se modificó
`perfil_hogar.py`.

## Otros eventos de composición (D19), totales sobre toda la serie

- `alta_activo_menor`: 9.840 hogares-par (ambos horizontes, 4 tratamientos
  sumados — no depende del tratamiento, mismo número en las 4 filas de cada
  origen).
- `se_jubila`: 32.134 hogares-par.
- `cambio_tamanio_hogar` (`n_miembros_t != n_miembros_th`, D11): 154.471
  hogares-par.

## Pendiente

- **Grilla 3×3 trabajo × ingreso**: no se construye esta ronda — bloqueada
  hasta que Iván valide el deflactor FACPCE (D24). El eje "trabajo" de 3
  niveles ya está calculado y guardado en el Parquet intermedio
  (`data/07_targets/pares_targets/`); la exclusión de ingreso no declarado
  (`..._excl_ingreso` en `cascada_trimestral.csv`) queda lista para esa
  grilla específicamente.
- **Tratamiento de independientes** (D16/D20, pendiente): las 4 variantes
  están calculadas en paralelo en `cascada_trimestral.csv`
  (`tratamiento_independiente`), sin elegir ninguna — la lectura primaria de
  Y1 de este informe usa `categoria_propia` (el default de
  `intensificacion.py`).
