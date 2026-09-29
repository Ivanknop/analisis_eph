# Decisiones metodológicas

Log secuencial de decisiones metodológicas del TFI, con ID único (`D1`, `D2`, ...)
asignado en el momento en que se toma la decisión. No se reutilizan IDs.

Cada entrada: fecha, decisión, motivo, alternativas descartadas si las hubo.

---

## D1 (2026-09-28) — Aglomerados faltantes puntuales, 2019T3 y 2020T3

Verificado sobre los microdatos cacheados que 2019T3 no releva el aglomerado 8 y 2020T3
no releva el aglomerado 31 (32 aglomerados en los trimestres inmediatamente antes y
después de ambos). No es un error de caché ni de parseo — está en los datos originales
de INDEC.

**Causa**: pendiente de verificar contra notas técnicas de INDEC (no investigado en esta
sesión).

**Impacto en el panel**: los hogares de esos dos aglomerados en esos trimestres pierden
las comparaciones de panel t+1/t+4 correspondientes (la rotación 2-2-2 no tiene con qué
emparejar esa porción de la muestra) — afecta tanto el cálculo de cualquier etiqueta de
deterioro basada en comparación intertemporal como las tasas de atrición del panel para
esos trimestres.

Documentado como limitación conocida, sin resolver.

## D2 (2026-09-28) — Rediseño del cuestionario EPH en 2023T4

Individual pasa de 177 a 235 columnas y hogar de 88 a 98 entre 2023T3 y 2023T4
(confirmado contra los datos y contra el codebook oficial
`EPH_tot_urbano_estructura_bases_2025.pdf`, que documenta explícitamente que estas
variables se incorporaron "a partir del cuarto trimestre de 2023"). Verificado en
detalle: el bloque de vivienda (`II*`/`IV*`, las 27 variables) no cambia de códigos —
el cambio de tipo detectado en el notebook 01 es artefacto de NA (2023T3 tiene ~0.1% de
faltantes que fuerzan `float64`), no una redefinición.

El cambio estructural real está en las variables `V` de estrategias de subsistencia, y
no es uniforme: `V2` (hogar) se mantiene como pregunta general y se le agregan
sub-ítems `V2_01/02/03` (por tipo de jubilación/pensión); `V5` y `V11` (hogar) se
reemplazan enteras por sub-ítems numerados sin dejar la raíz plana. `V21_0N`/`V22_0N`
(nuevas) no son una raíz propia ni continuación de las `V21`/`V22` viejas — son
"aguinaldo de"/"retroactivo de" cada categoría de `V2`, según el codebook; la similitud
de nombre con las `V21`/`V22` del cuestionario anterior (no cubierto por este codebook)
es coincidencia de numeración, no continuidad semántica confirmada. Las `V*_M` de
Personas son montos en pesos (confirmado por el codebook) y requieren la misma regla de
parseo que `IPCF`/`ITF`/`P21`. No hay equivalencia automática construida entre variables
viejas y nuevas — la única relación confirmada contra el codebook es la de `V2` con sus
propios sub-ítems; el resto queda pendiente de validar contra el diccionario completo de
INDEC si hace falta ir más atrás de 2023T4.

## D3 (2026-09-28) — Régimen de ingresos no declarados, distinto por era

**[confirmado, patrón empírico]** Desde 2016, los hogares con `DECCFR=12` tienen
`PONDIH=0` (con 6 casos puntuales adicionales de `DECCFR` nulo también con `PONDIH=0`,
ver notebook 01b); antes de 2016 no existe `PONDIH` para hogar, no existe código `12`, y
`P21` nunca vale `-9` en los 47 trimestres históricos verificados (queda en `0`, igual
que ingreso cero real; con `unar` instalado, los 47 -- no solo 42 -- están recuperados
sin trimestres saltados).

**[confirmado]** `P21` tampoco es nulo (`NaN`) entre ocupados en ninguno de los 86
trimestres de `data/02_nucleo/individual/` (histórico + regular) -- la no respuesta
post-2016 se expresa siempre como `-9`, nunca como valor faltante. Cierra el punto que
quedaba abierto más abajo: no hace falta revisar D3 de fondo por esta vía.

**[inferido, pendiente de documentación INDEC]**: desde 2016 la no respuesta de
ingresos se corregiría por reponderación (`PONDIH`/`PONDII`/`PONDIIO`), quedando los no
respondentes con código `12` y ponderador `0` — compatible con la frase del Anexo I del
codebook sobre corrección por no respuesta, pero no la confirma. Antes de 2016 los
ingresos se habrían imputado directamente en la base (sin marca visible), lo que
explicaría la ausencia de `-9`, de `12` y de `PONDIH` — hipótesis, no verificada.

**Consecuencia**: la etiqueta no usa `IPCF=0` como criterio de "sin ingreso". Post-2016,
los hogares con ingreso no declarado se separan con el flag `ingreso_no_declarado =
PONDIH == 0` (incluye `DECCFR = 12` y 6 casos con decil nulo); pre-2016, la variación de
`IPCF`/`ITF` puede incluir ruido de imputación no identificable con las variables
disponibles. El % de ocupados con `P21=0` no muestra escalón justo en 2016 (tendencia
decreciente continua, ~9%→2.5% en toda la serie) — no se usa como evidencia a favor de
ninguna hipótesis, solo se deja registrado. Detalle completo en el notebook 01b.

El `-9` (no respuesta) aparece en el 15.8% de los ocupados en la era regular (promedio
2016-2025); los asalariados muestran `P21=0` con una frecuencia algo menor a la de los
ocupados en general (3.2% pre-2016, 2.1% post-2016 vs. 4.2%/3.1% general), sin un
quiebre propio adicional en 2016.

## D4 (2026-09-29) — Momento de observación dentro del trimestre, no precisable

**[confirmado]** El codebook define `TRIMESTRE` (p.14) solo como "ventana de
observación" 1-4 por año. La condición de actividad se refiere a la "semana de
referencia" (p.20) y los ingresos de la ocupación principal a un "mes de referencia"
(p.29, Anexo I p.31) — ambos relativos a la fecha de entrevista de cada caso, no fijados
por trimestre.

**[no cubierto]** El codebook (`EPH_tot_urbano_estructura_bases_2025.pdf`) es un "Diseño
de registros", no el documento de diseño muestral operativo de INDEC — no dice cómo se
distribuye la muestra a lo largo de las ~13 semanas del trimestre. Tampoco existe
ninguna variable en los microdatos que identifique semana/mes de relevamiento del caso
(confirmado contra `data/meta/inventario_variables.csv`: las únicas variables con
"MES"/"SEM" son de preguntas de historia laboral puntuales, no la fecha de la entrevista
en sí).

**Consecuencia**: el momento de observación dentro del trimestre no se puede precisar
con lo disponible en este repo. Requeriría el documento de diseño muestral de INDEC
(fuera de este PDF) o asumir el punto medio del trimestre como aproximación
**[inferido, pendiente de decisión de la directora]**.

## D5 (2026-09-29) — Convención de mandato presidencial por trimestre

**[confirmado, hecho público]** Fechas reales de asunción: Néstor Kirchner 2003-05-25;
CFK I 2007-12-10; CFK II 2011-12-10; Macri 2015-12-10; Alberto Fernández 2019-12-10;
Milei 2023-12-10.

**Convención de este módulo, no un hallazgo empírico**: `panel.mandato_presidencial`
asigna cada `(año, trimestre)` al presidente en funciones el primer día de ese
trimestre. Todas las asunciones reales caen el 10/12 (dentro del Q4 del año de cambio de
mando) salvo Kirchner (25/5, Q2) — bajo esta regla, el primer trimestre COMPLETO de cada
mandato es el T1 del año siguiente a la asunción (el 10/12 es posterior al 1/10, así que
ese Q4 todavía empieza con el presidente saliente). Por eso el "desde" de la tabla
`MANDATO_PRESIDENCIAL` no coincide con el año de asunción, y el trimestre de transición
completo queda asignado al saliente, no se reparte.

## D6 — Ventana de edad del panel de hogares, PROVISORIA

`TOLERANCIA_EDAD_POR_HORIZONTE = {1: (-1, 1), 4: (0, 2)}` (ventana de
`diferencia_edad` aceptada como "misma persona" entre `t` y `t+h`, por horizonte, usada
tanto en `identidad_nucleo` como en `contar_altas_bajas` -- ver D10) es **provisoria,
pendiente de que la directora la fije en forma definitiva**. Se documenta su origen para
que esa revisión tenga con qué comparar, no como decisión cerrada: con estos valores, la
ventana cubre 92.92% (h=1) y 93.22% (h=4) de los componentes con mismo sexo en los datos
reales; centrada en 0 para h=1 y en +1 para h=4 porque en h=4 pasa un año completo (moda
de `diferencia_edad` observada en +1, no en 0).

## D7 (2026-09-29) — `CODUSU` cambia de formato entre la era histórica y la regular

**[confirmado empíricamente]** `CODUSU` tiene un formato completamente distinto entre
eras: en la histórica (DBF, 2003T3-2015T4) es un código puramente numérico de 6 dígitos
(ej. `"393389"`); en la regular (texto, 2016T1 en adelante) es un código alfanumérico de
29 caracteres (ej. `"TQXXXXXXXXXXXXXXXXXXX00000000"`). Son esquemas de identificación
incompatibles, no la misma clave en dos formatos — cualquier par de panel que cruce la
frontera histórico/regular tiene 0% de vinculación por construcción, sin importar si
INDEC mantuvo la muestra real o no. Verificado: la intersección de `CODUSU` entre 2015T2
(histórico) y 2016T2 (regular) es exactamente 0, mientras que 2016T2 vs. 2016T3 (ambos
regulares) comparten 8,378 códigos.

**Impacto en el panel**: afecta a un solo par en la serie actual —
`2015T2→2016T2` (h=4) es el único origen histórico cuyo destino h=4 cae en la era
regular y además existe (2015T1→2016T1 y 2015T3/T4→2016T2/T3 no existen por el hueco de
publicación 2015T3-2016T1). No es una limitación general del panel, es específica de
este único par.

## D8 (2026-09-29) — `CH03` agregada al esquema núcleo + detección de esquema desactualizado

**[confirmado]** `CH03` (relación de parentesco) faltaba en `tipado_nucleo.ESQUEMA_NUCLEO`
y se agregó (`base="individual"`, códigos 1-10 confirmados contra el codebook p.16 y
contra los 86 trimestres). El notebook 01b se re-corrió una vez con `REHACER=True` para
incorporarla (cambio de esquema/código, no de datos de origen — `particion_desactualizada`
no lo detecta por sí sola, porque solo compara mtimes de datos).

Se agregó `tipado_nucleo.hash_esquema()` + columna `esquema_hash` en
`resumen_02_nucleo.csv`, con un chequeo en `notebooks/02_panel_hogares.ipynb` que compara
el hash vigente contra el registrado por partición, para no repetir este tipo de
desfasaje en silencio la próxima vez que se agregue una variable al esquema. Cuando una
partición reusada no tiene hash registrado (falta la clave o es nulo — por ejemplo, si es
anterior a que existiera esta columna), el notebook 01b guarda la cadena `"desconocido"`
en vez de asumir que coincide con el esquema vigente; ese chequeo posterior ya trata
cualquier valor distinto al hash vigente (incluido `"desconocido"`) como desactualizado.

## D9 (2026-09-29) — Vinculación de hogares por `CODUSU+NRO_HOGAR+AGLOMERADO`

**[confirmado, bug corregido]** `panel.vincular_hogares` unía hogares de `t` y `t+h`
solo por `CODUSU` (vivienda), reportando `mismo_nro_hogar` en vez de exigirlo. Esto arma
producto cruzado cuando una vivienda tiene más de un hogar en `t` o en `t+h` — confirmado
contra `data/02_nucleo/hogar/`: 13,161 de 1,443,534 combinaciones vivienda-trimestre
(~0.9%) tienen más de un `NRO_HOGAR` bajo el mismo `CODUSU`.

**Diagnóstico previo a decidir el fix**: sobre viviendas con exactamente un hogar en `t`
y uno en `t+h` (aislando el caso de "un solo hogar que cambia de número" del bug de
producto cruzado), los casos con `NRO_HOGAR` distinto son 8,243 (h=1, 1.35% de 610,349) y
24,419 (h=4, 4.67% de 522,597). De esos, más del 96% no tiene composición coincidente
(sexo+edad por `COMPONENTE`, mediana de proporción = 0) — no parecen el mismo hogar
renumerado, parecen hogares distintos. Se pierden 323 (h=1) y 703 (h=4) casos que sí
tenían ≥90% de componentes coincidentes (<4% del total de casos con `NRO_HOGAR`
distinto en ambos horizontes).

**Decisión**: `vincular_hogares` pasa a exigir match exacto por
`CODUSU+NRO_HOGAR+AGLOMERADO` (`AGLOMERADO` como resguardo adicional — 0 casos de
`CODUSU` con más de un `AGLOMERADO` en el mismo trimestre en los datos actuales, no
cambia el resultado hoy pero previene falsos positivos). `mismo_nro_hogar` se elimina de
`clasificar_par` (con match exacto, siempre sería `True`, no aportaba información).

**Impacto**: hogares cuyo `NRO_HOGAR` cambia entre `t` y `t+h` (o que quedan en una
vivienda multi-hogar sin match exacto) ya no arman par automáticamente — quedan
reportados aparte por `panel.clasificar_viviendas_no_vinculadas` como
`hogar_reemplazado` o `vivienda_sin_par`.

- `hogar_reemplazado` (la vivienda sigue en `t+h`, ningún hogar matcheó exacto): dos
  lecturas según el denominador — 0.60% (h=1) / 1.89% (h=4) sobre **todas** las
  viviendas de origen; 1.35% (h=1) / 4.66% (h=4) sobre las viviendas que **sí** siguen
  existiendo en `t+h` (denominador comparable al diagnóstico que definió el join exacto,
  más arriba).
- `vivienda_sin_par` (la vivienda ya no está en `t+h`): 55.6% (h=1) / 59.5% (h=4). La
  rotación 2-2-2 del panel explica ~50 pp teóricos (ver el gráfico de tasa de
  vinculación de vivienda) — el excedente, 5.6 pp (h=1) y 9.4 pp (h=4), es **atrición
  real** (vivienda que debería seguir en el esquema y no aparece: mudanza, rechazo, no
  localización), no un artefacto de la rotación.

**Antes/después de `panel_resumen_pares.csv`, agregado por horizonte** (incluye también
el efecto de la tolerancia de edad de D6, corridos juntos en la misma re-ejecución):

| h | n_pares antes → después | confirmado | composición cambiada | identidad dudosa |
|---|---|---|---|---|
| 1 | 638,558 → 614,438 (-24,120, exactamente los `NRO_HOGAR` distintos de arriba) | 92.0% → 78.7% | 6.0% → 16.9% | 2.1% → 4.0% |
| 4 | 547,105 → 508,568 (-38,537) | 88.7% → 79.9% | 7.9% → 15.8% | 3.4% → 4.4% |

La caída de `confirmado` es mayor de lo que explica solo sacar los pares con
`NRO_HOGAR` distinto -- es sobre todo la tolerancia de edad (D6): antes la composición
solo miraba sexo; ahora un hogar necesita ≥90% de sus componentes con sexo **y** edad
dentro de ventana para seguir `confirmado`. Con ~93% de coincidencia por componente
(D6), la probabilidad de que **todos** los componentes de un hogar coincidan cae rápido
a medida que crece la cantidad de miembros -- efecto esperado de agregar un criterio más
estricto, no un error.

**Nota**: la clasificación `confirmado`/`composicion_cambiada`/`identidad_dudosa` de
esta sección quedó **reemplazada por D10** (`identidad_nucleo` + `cambio_composicion`),
justamente por el efecto de tamaño de hogar que se acaba de describir. Los números de
esta tabla son el estado antes de D10, quedan como registro histórico de por qué se
cambió el método.

## D10 (2026-09-29) — Identidad del panel por núcleo, no por composición completa

**[confirmado, diagnóstico que motivó el cambio]** La clasificación de D9
(`confirmado`/`composicion_cambiada`/`identidad_dudosa`, sobre proporción de
`COMPONENTE` coincidentes) caía monótonamente con la cantidad de miembros del hogar en
`t` (90.1%→62.5% de 1 a 6+ miembros en h=1; 89.2%→66.0% en h=4), sin relación con si el
hogar era "el mismo". Es **mecánico**: exigía ≥90% de componentes coincidentes, así que
un hogar de `n` miembros necesitaba que coincidieran casi todos; con ~93% de
coincidencia por persona (D6), la probabilidad de que coincidan los `n` cae como
≈0.93^n, aunque la tasa de coincidencia por persona no dependa del tamaño del hogar.

**[confirmado]** La reasignación de `COMPONENTE` entre visitas explica solo una fracción
chica de los componentes que no coinciden: 6.8% (h=1) / 6.9% (h=4) tienen, en el mismo
hogar y bajo otro número de `COMPONENTE`, a alguien de sexo y edad compatibles con la
persona de `t`. **Corrección sobre una lectura anterior de este mismo número**: el
93%/94% restante NO es "turnover real" sin más — mezcla cambio real de integrantes
(alguien se fue, alguien nuevo llegó) y error de declaración de edad (la misma persona,
pero `CH06` corrido más allá de la ventana de tolerancia). La EPH no tiene ninguna
variable que distinga ambos casos; se deja como límite del método, no se fuerza una
atribución.

**Decisión implementada — identidad por núcleo, regla (a)**:
`panel.identidad_nucleo(individual_t, individual_th, pares_hogar, h)` reemplaza la
clasificación de D9. `identidad_confirmada` es `True` si el/la jefe/a de `t` (`CH03==1`)
aparece en el núcleo de `t+h` (`CH03` 1 **o** 2 -- permite intercambio de rol
jefe/a↔cónyuge) con sexo y edad dentro de `TOLERANCIA_EDAD_POR_HORIZONTE[h]` (D6); `NA`
si el/la jefe/a de `t` es ambiguo/a. La composición completa queda aparte, en
`panel.contar_altas_bajas` (`n_altas`/`n_bajas`/`cambio_composicion`, emparejando por
sexo+edad, no por `COMPONENTE`) — no entra en la identidad, mismo criterio que
`cambio_jefatura` (tampoco entra). Se descartó la alternativa (b) — tolerar un
integrante fuera de rango en hogares de 3+ — por no eliminar el sesgo por tamaño (un
hogar de 6 con 2 miembros fuera de rango seguía cayendo) y por un corte arbitrario sin
base empírica.

**Verificación tras re-correr la 02**: `identidad_confirmada_pct` por tamaño de hogar
queda prácticamente plana, a diferencia de la caída de D9:

| tamaño | h=1 | h=4 |
|---|---|---|
| 1 | 90.1% | 89.2% |
| 2 | 90.8% | 88.8% |
| 3 | 90.5% | 88.7% |
| 4 | 90.9% | 89.4% |
| 5 | 90.1% | 88.1% |
| 6+ | 87.9% | 85.6% |

Queda un residuo chico en 6+ (~2-3 pp por debajo del resto) — no elimina el sesgo al
100%, pero es un orden de magnitud menor que la caída de ~28-30 pp de D9. `jefe_ambiguo`
es prácticamente nulo (4 casos en h=1, 1 en h=4, sobre ~610-620 mil pares).
`cambio_composicion` sí sube fuerte con el tamaño (13.4%→53.6% en h=1, 17.6%→61.1% en
h=4) — esperable, es un flag descriptivo, no la identidad.

**Unipersonales**: sin margen para "composición cambiada" (un solo miembro) — si no
coincide, es directamente un cambio de ocupante. 9.9% (h=1, 10,581 de 106,982) / 10.8%
(h=4, 9,051 de 83,624) de los hogares unipersonales en `t` tienen identidad NO
confirmada.

**Totales `panel_resumen_pares.csv`** (nuevo esquema, reemplaza el de D9): h=1 —
614,438 pares, 554,574 (90.3%) `identidad_confirmada`; h=4 — 508,568 pares, 450,251
(88.5%) `identidad_confirmada`.

## D11 (2026-09-29) — `cambio_composicion` sobreestimado por tamaño de hogar, PENDIENTE

**[confirmado]** `cambio_composicion` (D10) es 28.3% (h=1) / 35.6% (h=4) -- muy por
encima de cuántos hogares cambian de **tamaño** entre `t` y `t+h`
(`n_miembros_t != n_miembros_th`, sin mirar sexo/edad de nadie): 9.1% (h=1) / 19.4%
(h=4). La brecha crece con el tamaño del hogar en ambos horizontes (9.1pp en hogares de
1 miembro hasta 34.3pp en 6+, h=1; 9.3pp a 23.5pp en h=4) -- mismo patrón mecánico que
D10: con ~93% de coincidencia por persona (D6), la chance de que **al menos uno** de
`n` miembros caiga fuera de la ventana de edad sube con `n`, aunque nadie se haya ido
realmente. Ensanchar la ventana de edad solo para este flag (±2 en vez de ±1 para h=1,
-1..+3 en vez de 0..+2 para h=4) baja el total a 22.3%/31.5% pero la brecha por tamaño
persiste -- atenúa, no resuelve.

**Dos opciones para redefinir el flag, ninguna implementada, pendiente de decisión**:
- Por cambio de cantidad de integrantes (`n_miembros_t != n_miembros_th`): robusto al
  sesgo por tamaño, pero no detecta un reemplazo de integrante que mantiene el mismo
  tamaño (alguien se va, otra persona distinta entra).
- Ventana de edad más amplia solo para este flag (no para `identidad_nucleo`, que ya
  funciona bien con la ventana de D6): atenúa la brecha sin descartar el matcheo por
  persona, pero no la elimina.

Sin recomendación cerrada -- `identidad_nucleo` (D10) no se ve afectada por este
problema (se apoya solo en el/la jefe/a, no en toda la composición).

**`data/meta/panel_resumen_final.csv` ampliado**: se agregaron las mismas métricas de
composición/jefatura/ejes restringidas a la subpoblación con `identidad_confirmada`
(prefijo `confirmada_`), y `n_ingreso_presente_historico_ambas_puntas` -- pares de la
era histórica con `IPCF` presente en ambas puntas, sin pasar por el flag
`ingreso_declarado` (`NA` en histórico, D3). Resultado: 335,958 (h=1) / 273,875 (h=4)
pares históricos con `IPCF` en ambas puntas -- **más** que los pares de la era regular
con `ingreso_declarado` en ambas puntas (195,634 / 163,590) -- dimensiona una expansión
grande si se decide usar `IPCF` crudo en vez del flag para extender el eje de ingreso a
toda la serie.
