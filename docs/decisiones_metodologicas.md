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

**Nota agregada (2026-10-01) — atrición real recalculada excluyendo 2013T1-T3**: la
ruptura de `CODUSU` documentada en D13 reduce o anula el eslabón `h=4` saliente de
`2013T1, 2013T2, 2013T3` (3,560 / 0 / 3,743 pares, contra ~7,000-7,900 normal). Recorriendo
`panel.clasificar_viviendas_no_vinculadas` sobre los 78 orígenes `h=4` reales: con los 3
orígenes, `vivienda_sin_par = 59.445%` (776,077/1,305,532, igual al 59.5% ya citado arriba,
atrición real ~9.4pp); excluyéndolos, `vivienda_sin_par = 58.395%` (732,428/1,254,268 sobre
75 orígenes), atrición real **~8.4pp**. Baja ~1pp -- perceptible pero chico, son solo 3 de
78 orígenes. Cifra aparte que sí cambia de forma notable: `panel_resumen_final.csv`, fila
`h=4, mandato=CFK II` (`n_pares_vinculados=58,749`) incluye apenas 7,303 pares de estos 3
orígenes (12.4% del total de CFK II) contra los ~21,000 que darían tres trimestres
normales -- no están mal contados (son pares reales, D13), pero el total de CFK II está
sesgado a la baja por la ruptura, no por mayor atrición real de ese mandato. No se modifica
`panel.py` ni se re-corre el notebook 02 por esto -- ver D13 para el detalle completo.

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

## D12 (2026-10-01) — Trayectorias de 4 apariciones (t, t+1, t+4, t+5), PENDIENTE de confirmación de Iván

Diagnóstico de factibilidad para seguir cada hogar en sus 4 apariciones dentro de una
cohorte de rotación "overlapping" (`src/trayectorias.py`,
`notebooks/04_trayectorias.ipynb`) -- no construye ninguna etiqueta de deterioro. Entrada
pendiente de que la revise y la confirme Iván; se deja con números reales de una corrida
completa (86 trimestres) para que esa revisión tenga con qué comparar.

**Dos claves, dos preguntas distintas**: la rotación 2-2-2 es propiedad de la *vivienda*
(`CODUSU+AGLOMERADO`) -- INDEC vuelve a visitarla según su cronograma, viva ahí quien
viva. La identidad de *hogar* (`CODUSU+NRO_HOGAR+AGLOMERADO`, D9) es otra cosa -- si es la
misma familia. `inferir_visita` usa la primera; `encadenar_trayectoria` y el resto de la
selectividad usan la segunda.

**Inferencia de visita, método de hipótesis simétrico** [inferido, sin variable directa en
los datos -- confirmado que no existe en `data/meta/inventario_variables.csv` ni en el
codebook]: para una vivienda presente en `t`, se evalúan 4 hipótesis sobre qué visita (1-4)
es `t` dentro de su propia cohorte de entrada `t0` (`t0=t`, `t0=t-1`, `t0=t-4`, `t0=t-5`).
Cada hipótesis predice presencia o ausencia en el mismo conjunto simétrico de 8 offsets
relativos a `t`: `{-5,-4,-3,-1,+1,+3,+4,+5}` (el `±3` hace falta para que las 4 hipótesis
prediccionen sobre el mismo número de offsets -- sin él, las hipótesis de visita 2 y 3
quedan con menos chequeos que las de visita 1 y 4, y el puntaje deja de ser comparable).
Puntaje = `n_matches - n_mismatches` sobre los offsets determinables; gana la hipótesis de
mayor puntaje; empate -> `ambigua=True`, `visita_inferida=NA` (no se fuerza una elección).
Un offset no es determinable si el trimestre no está en `trimestres_publicados()` o cae en
otra era que `t` (`es_historico` distinto, D7) -- mismo tratamiento para ambos casos, sin
nota de anomalía aparte.

El método resuelve el caso que motivó el rediseño (un método ingenuo que solo mira
`t-1`/`t-4` clasifica mal cuando hay no respuesta en una visita intermedia): con presencia
en `t0, t0+1, t0+5` y ausencia en `t0+4`, evaluado en `t=t0+5`, la hipótesis "visita 4" gana
con 7 matches y 1 mismatch (justo en el offset que cae en `t0+4`), en vez de perder contra
"visita 3" como pasaba con el método viejo.

**Resultado real, 86 trimestres (2003T3-2025T4), 1,443,534 viviendas-trimestre
clasificadas**: visita 1: 320,379; visita 2: 327,625; visita 3: 317,760; visita 4: 320,534;
`ambigua`/no clasificable: 157,236 (10.9%). `n_determinables` (de 8 posibles): mediana 8,
media 7.31, mínimo 4 -- más bajo cerca de los bordes de la serie (2003T3-2003T4, 2025T4),
de los huecos de publicación y del cruce de era 2016T2-2016T3, como se esperaba del método
(ver `notebooks/04_trayectorias.ipynb`, Sección 1). El 10.9% de `ambigua` en una serie con
huecos/bordes/cruces de era reales es mayor a lo que daría una serie sin esos problemas --
no se intentó separar cuánto de ese 10.9% es censura estructural vs. mudanza/no respuesta
real genuina, queda como trabajo pendiente si se necesita precisar.

**Dos variantes de "trayectoria encontrada", en paralelo, sin elegir una** -- `vinculada`
(clave exacta `CODUSU+NRO_HOGAR+AGLOMERADO` presente en los 3 eslabones t->t+1, t->t+4,
t+4->t+5, vía `tr.encadenar_trayectoria`, reusando `data/03_panel/pares_h1`/`pares_h4` sin
reconstruir desde `02_nucleo/`) e `identidad_confirmada_trayectoria` (AND Kleene de
`identidad_confirmada` en los 3 eslabones). Resultado real: 204,741 trayectorias
`vinculada`, 163,914 (80.1%) con `identidad_confirmada_trayectoria=True`.

**Verificación cruzada contra `pares_h4(t+1)`** (t+1->t+5, mismo horizonte real que la
trayectoria completa, D6): **0 de 204,741 trayectorias** (0.0%) quedan sin presencia en esa
verificación -- consistencia total entre el camino encadenado y el camino redundante, en
toda la serie. No se construyó una identidad directa de un solo salto t->t+5 (no haría
falta ventana de edad nueva para `h=5` en `TOLERANCIA_EDAD_POR_HORIZONTE`, el camino
siempre pasa por `h=1`/`h=4` ya definidos).

**Cohortes completables en teoría** (`tr.cohorte_completable`, chequea existencia de
`t0,t0+1,t0+4,t0+5` en `trimestres_publicados()` y cruce histórico/regular en el eslabón
`h=4`, D7): 73 de 86 cohortes de origen son completables; las 13 no completables son todas
por hueco de publicación (2007T3 o 2015T3-2016T1 cayendo en alguno de los 4 puntos) --
**el branch de "cruza la frontera histórico/regular" nunca se activa en los datos reales**,
porque cualquier cohorte cuyo eslabón `h=4` cruzaría la frontera (D7: el único caso real es
2015T2->2016T2) ya falla antes por tener `t0+1=2015T3`, que no está publicado. Tasa de
completitud observada sobre las 73 cohortes completables: 79.9% `vinculada`, 64.0%
`identidad_confirmada_trayectoria` (sobre el tamaño de la cohorte de visita 1 en `t0`).

**N por diseño dinámico** (`tr.n_diseno_dinamico`, sobre 71 trimestres de origen con datos
para los 3 diseños -- 2 menos que las 73 cohortes con trayectoria, por orígenes con 0 filas
tras el AND Kleene): (a) t->t+1 predice t+1->t+5: 204,741; (b) t->t+4 predice t+4->t+5:
204,739 (2 menos, los únicos `jefe_ambiguo` de la serie, D10); (c) trayectoria completa:
204,741. N mínimo por trimestre de origen: 1,265 (2020T2, coincide con el operativo
telefónico por pandemia) -- **todos** los 71 trimestres superan los 3 umbrales evaluados
(30/100/500, marcados `[inferido]`, no son criterio INDEC). **Una vez exista una etiqueta
de deterioro, el umbral relevante no va a ser el N total por trimestre sino el N de la
clase minoritaria** -- estos umbrales genéricos solo dan una cota superior optimista.

**Ilustración de dos pasos** (Sección 6, sobre 135,423 hogares con `identidad_confirmada_
trayectoria=True` y quintil observable en `t0,t0+1,t0+5`): el quintil de `t0` agrega
información más allá del quintil de `t0+1` para predecir `t0+5` -- ej. partiendo de
`q_t0_1=Q1`, la chance de volver a `Q1` en `t0+5` es 76.8% si `q_t0=Q1` pero cae a 12.5% si
`q_t0=Q5`. Es solo ilustrativo (no la etiqueta, no causal) pero es una señal a favor de que
valga la pena un panel dinámico con dependencia de estado más allá de un paso.

## D13 (2026-10-01) — Ruptura de `CODUSU` en las cohortes de entrada 2013T1/2013T2, PENDIENTE de confirmación de Iván

Revisión de la corrida real de D12 detectó dos bugs (`n_diseno_dinamico` medía casi lo mismo
en los 3 diseños porque operaba sobre la trayectoria ya completa; `_clasificar_perdida` del
notebook clasificaba por el último eslabón alcanzado en vez del patrón real de presencia,
confundiendo huecos con reaparición con atrición hasta `t0+5`) -- corregidos en
`src/trayectorias.py` (`encadenar_dos_eslabones`, `contar_n_diseno`,
`clasificar_perdida_por_patron` reemplazan `n_diseno_dinamico`/`_clasificar_perdida`) y en
`notebooks/04_trayectorias.ipynb`.

Al agregar el chequeo cruzado pedido en la primera revisión (¿las trayectorias nacen de
viviendas `visita_inferida==1`? Sí, 0.00% de las 204,741 no lo son) apareció una anomalía de
datos real, no relacionada con ningún bug del código. Una segunda revisión encontró que esa
anomalía **contaminaba también `inferir_visita`** para 2012-2015 (no solo los diseños de panel
dinámico) y pidió acotar el mecanismo antes de cerrar esta entrada. Esta versión reemplaza por
completo el borrador anterior (mismo criterio que D9: nunca se cerró, sigue pendiente).

**[confirmado] Matriz de solapamiento de `CODUSU`, 2012T4-2014T4** (% de intersección de cada
origen contra cada destino, ambas direcciones): la ruptura es **direccional entre dos
conjuntos chicos**, no un bloque. Cualquier par `(origen, destino)` con un lado en
`{2013T1,2013T2,2013T3}` y el otro en `{2014T1,2014T2,2014T3}` da 0% o la mitad de lo
esperado, en cualquier distancia de offset -- mientras que `2013T4` compara normal contra los
4 trimestres de 2014 (44.8%/23.4%/45.7% en gaps 1/3/4), `2014T1-T3` comparan normal entre sí
(gap1: 46.1%, 45.0%) y hacia adelante (`2014T1→2015T1` normal), y `2012T4-2013T4` comparan
normal entre sí en todos los gaps. No es un problema de `NRO_HOGAR`/`AGLOMERADO` (la
intersección solo por `CODUSU` da los mismos números) ni el mismo caso que D7 (formato
histórico/regular -- acá los 8 trimestres de 2013-2014 son histórico/DBF). Un barrido con
mediana móvil sobre toda la serie de `pares_h4` (`resumen_03_panel.csv`) no encontró ningún
otro trimestre con una caída así, fuera de este rango y del caso ya conocido de D7 (2015T2).

**[confirmado] Mecanismo -- es un problema de cohorte de entrada, no de trimestre**: en
cualquier `t0` conviven el grupo en 1.ª visita (entra en `t0`, vincula `h=1` hacia `t0+1` y
`h=4` hacia `t0+4`) y el grupo en 2.ª visita (entró en `t0-1`, vincula `h=4` hacia `t0+4` pero
nunca `h=1` desde `t0`, por construcción) -- de ahí que el solapamiento `h1∩h4` normal sea
~45% del `h4` (la porción que aporta el grupo de 1.ª visita, el único que estructuralmente
pertenece a los dos). Construyendo cada cohorte de entrada `E` (presentes en `E` y `E+1`,
ausentes en `E-1`, 2012T3-2013T4) y midiendo su presencia en `E+4`/`E+5`:

| entrada `E` | n cohorte | % presente en `E+4` | % presente en `E+5` |
|---|---|---|---|
| 2012T3 | 7,655 | 44.2% | 43.6% |
| 2012T4 | 7,533 | 44.1% | 43.5% |
| **2013T1** | 7,703 | **0.0%** | **0.0%** |
| **2013T2** | 7,760 | **0.0%** | **0.0%** |
| 2013T3 | 7,825 | 46.8% | 46.6% |
| 2013T4 | 7,763 | 46.7% | 45.8% |

Exactamente dos cohortes de entrada rotas (2013T1, 2013T2 -- 0.0% limpio en ambas), el resto
normal (43-47%). Esto explica mecánicamente, sin nada más, el patrón de pares `h=4` por
`CODUSU` de esos 4 orígenes (3,560 / 0 / 3,743 / 7,622 para 2013T1-T4, contra ~7,000-7,900
normal): **2013T1** sobrevive solo con su grupo de 2.ª visita (entrada 2012T4, sano, nunca
solapa `h1`) -- mitad de volumen, 0% de solapamiento con `h1`; **2013T2** pierde los dos
grupos que la alimentan (entrada 2013T1 y 2013T2, ambos rotos) -- 0 pares; **2013T3**
sobrevive solo con su grupo de 1.ª visita (entrada 2013T3, sano, sí solapa `h1`) -- mitad de
volumen, ~94% de solapamiento con `h1`; **2013T4** tiene los dos grupos sanos (entrada 2013T3
y 2013T4) -- normal en volumen y en solapamiento. Verificado directo sobre los pares reales
(`panel.vincular_hogares`/`panel.identidad_nucleo`):

| origen (`h=4`) | n pares | `identidad_confirmada` | `h1(t0) ∩ h4` (diseño c) | `h1(t0+4) ∩ h4` (diseño b) |
|---|---|---|---|---|
| 2012T1 (normal) | 7,074 | 88.7% | 46.4% | 45.3% |
| **2013T1** | 3,560 | 88.7% | **0.0%** | **0.0%** |
| 2013T3 | 3,743 | 88.1% | 94.5% | 93.7% |

**Son pares reales, no coincidencias de `CODUSU`** -- si los 3,560/3,743 pares sobrevivientes
fueran matches espurios por choque de código, `identidad_confirmada` (que exige sexo+edad del
jefe/a dentro de la ventana D6) daría una tasa cercana al azar, no 88.1-88.7%, prácticamente
idéntica a la normal (88.7%). Esto descarta la hipótesis de "coincidencia" para los pares que
sí existen -- la pregunta de por qué 2013T1 segrega sus poblaciones de `h1`/`h4` (0% en ambas
puntas) también queda resuelta por el mecanismo de arriba: es exactamente el grupo de 2.ª
visita, que nunca solapa con `h1` haya ruptura o no.

**Impacto en `inferir_visita` (D12)**: `_presencia_offsets` ahora trata como no determinable
cualquier offset cuyo `(origen, destino)` cruce `{2013T1,2013T2,2013T3}` ↔
`{2014T1,2014T2,2014T3}` (mismo criterio que D7). Antes/después, distribución de visita
2012-2015 (sin ponderar):

| trimestre | antes (visita1/2/3/4, `ambigua`) | después (visita1/2/3/4, `ambigua`) |
|---|---|---|
| 2013T1 | NA/23.1/23.5/23.4, 30.0% | 27.5/25.2/20.3/23.1, 3.9% |
| 2013T2 | NA/NA/23.3/23.2, 53.5% | 23.3/23.0/20.6/20.1, 13.0% |
| 2013T3 | 23.8/NA/22.5/22.9, 30.8% | 25.9/28.2/22.3/20.1, 3.4% |
| 2014T1 | 23.3/24.0/NA/21.9, 30.8% | 20.6/23.8/28.4/23.7, 3.5% |
| 2014T2 | 53.6/23.5/NA/NA, 22.9% | 22.0/20.9/24.8/22.9, 9.5% |
| 2014T3 | 30.4/46.8/19.4/NA, 3.3% | 25.1/22.4/21.5/24.4, 6.6% |

`2013T4` y `2014T4` no cambian (ya eran normales antes del fix, confirmado). Nota sobre
`2014T3`: antes del fix ya tenía huecos (visita4 en `NA`, visita2 inflada a 46.8%) aunque
menos severos que 2014T1/T2 -- consistente con el mecanismo (recibe el hueco de la 4.ª visita
del grupo 2013T2 nada más, no dos huecos superpuestos como 2014T2).

**Alcance real (por cohorte de entrada) vs. alcance de la regla de código (por trimestre,
conservador)**: mecánicamente, lo único roto son las cohortes de entrada 2013T1 y 2013T2.
2014T1 recibe el hueco completo de la 3.ª visita de 2013T1; 2014T2 recibe el hueco completo de
la 4.ª visita de 2013T1 **y** de la 3.ª visita de 2013T2 (los dos huecos juntos, por eso es el
trimestre más afectado); 2014T3 recibe solo el hueco parcial de la 4.ª visita de 2013T2. 2013T3
está mecánicamente sano (su propia cohorte de entrada reaparece normal) -- su degradación de
volumen en `h=4` es consecuencia de que la mitad de *su* población (el grupo de 2.ª visita,
heredado de la cohorte 2013T2 rota) no está, no de un problema propio. La regla de
`_presencia_offsets` y la exclusión de la Sección 4 (`VISITA_AFECTADA_D13`), en cambio, tratan
las 6 combinaciones origen×destino entre `{2013T1,2013T2,2013T3}` y `{2014T1,2014T2,2014T3}`
como no determinables parejo -- es **conservador por trimestre** (en rigor, solo
2013T1/2013T2 como origen de la ruptura, y 2014T1/2014T2 como destinos con hueco completo,
necesitan excluirse; 2013T3 y 2014T3 están mezclados con la ruptura pero no rotos en sí
mismos -- 2014T3 con un hueco parcial, no limpio). Se mantiene así por simplicidad, verificado
que no introduce falsos negativos nuevos.

**[inferido, recalibrado contra una base normal] Matching por composición, cohorte de entrada
contra cohorte de entrada**: ¿las viviendas que entraron en 2013T1 (presentes en 2013T1 y
2013T2) tienen candidatas plausibles entre las que aparecen en 2014T1 con la misma estructura
de entrada (presentes en 2014T1 y 2014T2, ausentes en todo 2013)? Heurístico de bucket
(`AGLOMERADO + n_miembros + sexos` de todos los integrantes, sexo del jefe/a y edad dentro de
la ventana `h=4` de D6), comparado con la misma construcción en un año sin ruptura conocida:

| | cohorte de entrada | candidatos (año siguiente, ausentes todo el año anterior) | evaluables | con ≥1 candidato | tasa |
|---|---|---|---|---|---|
| **Anomalía**: entrada 2013T1 → candidatos 2014T1 | 7,703 | 8,134 | 7,772 | 3,698 | **47.6%** |
| **Base normal**: entrada 2011T1 → candidatos 2012T1 | 7,928 | 3,965 | 8,007 | 2,559 | **32.0%** |

47.6% vs. 32.0% (+15.6pp, ~49% relativo) -- la base normal ya es alta por el bucket grueso,
pero la anomalía está claramente por encima, más que en una calibración preliminar sin el
chequeo de sexo del jefe/a (que daba 57.6%/41.1% -- la versión final del notebook exige
además que el sexo del jefe/a coincida exactamente entre candidata y cohorte, no solo la
composición agregada de sexos, un criterio más estricto y más correcto; la brecha relativa se
mantiene igual de clara con el criterio bueno). Se inclina con más confianza hacia
**recodificación de `CODUSU`** (el mismo hogar sigue en el panel pero el código cambió, y el
método lo ve como entrante nuevo) por sobre una renovación de muestra completa (que
predeciría una tasa cercana a la base, no claramente por encima) -- sigue sin ser un linkage
definitivo. Nota aparte: el pool de candidatos 2014T1 (8,134) es más del doble que el de
2012T1 (3,965) -- hay más viviendas "ausentes todo 2013" que "ausentes todo 2011", consistente
con que 2013 tiene su propia degradación de continuidad interna; se deja anotado, no se
investiga más en esta entrada.

**[inferido, pendiente de verificar contra notas técnicas de INDEC]**: la causa de fondo --
por qué específicamente las cohortes de entrada 2013T1 y 2013T2 no reaparecen con el mismo
`CODUSU` -- queda sin confirmar. Una pista parcial sin cerrar: los 4 trimestres de 2014 se
capturaron de Wayback en 2017 (`eph_client.py`), mientras 2003-2013 se capturaron en 2014 --
un lote de publicación distinto, que no alcanza por sí solo a explicar por qué 2014T4 recibe
con normalidad a la cohorte 2013T3/2013T4 (ambas sanas) pero 2014T1-T3 no reciben con
normalidad a 2013T1/2013T2. No se corrige nada en `panel.py` ni en el notebook 02.

**Convención adoptada, pendiente de confirmación**: `2013T1, 2013T2, 2013T3` quedan marcados
`corte_estructural_d13=True` en `trayectorias_completitud_cohorte.csv` (completables según el
criterio teórico de `tr.cohorte_completable`, pero empíricamente rotos en el eslabón `h=4`) y
se excluyen de la tasa de completitud observada de la Sección 2, reportada con y sin ellos --
ahora sí difieren, con numeradores/denominadores explícitos: **con** los 3 (73 orígenes),
vinculada=77.3% (204,741/264,754), identidad_confirmada=61.9% (163,914/264,754); **sin** ellos
(70 orígenes), vinculada=80.0% (201,401/251,647), identidad_confirmada=64.1%
(161,292/251,647). (Antes del fix de `_presencia_offsets` esta comparación salía idéntica
porque 2013T1/2013T2 ya tenían `n_vivienda_visita1=0` por la contaminación -- quedaban fuera
del denominador antes de llegar al filtro de D13; corregido.) La Sección 4 (selectividad)
excluye el conjunto más amplio `VISITA_AFECTADA_D13 = {2013T1,2013T2,2013T3,2014T1,2014T2,2014T3}`
(las 6 cuya propia `inferir_visita` toca la ruptura, ver arriba) -- 250,928 hogares acumulados
para selectividad tras esa exclusión.

**N por diseño dinámico** (`src/trayectorias.py`, Sección 5 -- estos totales **no cambian**
con el fix de `_presencia_offsets`, porque dependen solo de `pares_h1`/`pares_h4` por clave
exacta, nunca de `visita_inferida`): sobre 72 orígenes con datos (a/b) y 71 (c) -- (a) 219,458
`vinculada` / 183,557 `identidad_confirmada`; (b) 218,506 / 182,458; (c) 204,741 / 163,914. Los
3 diseños dan N del mismo orden de magnitud (comparten el mismo universo subyacente de
`pares_h1`/`pares_h4`) pero ya no idénticos -- (a) y (b) no exigen la trayectoria completa y
por eso dan más N que (c). Orígenes afectados por diseño: (a) (`pares_h1(t)` ∩
`pares_h4(t+1)`) pierde todo su N en `t0 ∈ {2013T1, 2013T2}` (el eslabón roto cae en `t+1` de
esos orígenes); (b)/(c) (`pares_h4(t)` directo) pierden todo su N en `t0 ∈ {2013T1, 2013T2}`
también, directo. `2013T3` como origen de cualquier diseño SÍ supera N≥500 en los 3 (sano a
nivel de cohorte de entrada, ver mecanismo). Ningún otro origen de los 73 completables queda
bajo el umbral más estricto (N≥500) en ningún diseño -- la anomalía es acotada a estos 2
orígenes, no se propaga al resto de la serie.

**Nota agregada a D9**: la atrición real de D9 (~9.4pp) recalculada excluyendo 2013T1-T3 da
~8.4pp -- ver el párrafo agregado al final de D9.

## D14 (2026-10-01, reescrita tras revisión) — Definición de "hogar en riesgo" (ingreso + empleo), PENDIENTE de confirmación de Iván

`notebooks/06_poblacion_riesgo.ipynb` evalúa 3 candidatas en paralelo para la población de
la que se va a medir "salida de la precariedad" -- **no elige ninguna**, esta entrada deja
los números reales (corrida completa, 86 trimestres) para que la decisión la tome Iván.
**Reescrita** tras una revisión de Iván sobre la primera versión, que encontró un bug real
de cálculo y pidió 3 mejoras metodológicas -- el borrador anterior quedaba reemplazado
por completo, mismo criterio que D9/D13.

**[confirmado, bug corregido]** El bootstrap de la primera versión (comparación por
`ancla_con_deficit`) tenía un error de implementación: dentro del loop de réplicas, un
`.sum()` de más colapsaba el eje de réplicas en vez del de clusters (que el `einsum`/`W @
N` ya había reducido), dando un IC95% ~30 veces más angosto que el real. Se extrajo el
mecanismo a `src/bootstrap.py` (dos funciones: comparación simple de 2 grupos y
estandarización por estrato, puerto generalizado de `resumen_estandarizado` del notebook
03) con tests (`tests/test_bootstrap.py`, 11 casos, incluido uno que reproduce
exactamente el bug para que no vuelva a pasar inadvertido). Es el único módulo de este
repo que un notebook importa en vez de duplicar -- duplicar un mecanismo de inferencia
estadística con un bug ya encontrado una vez es más riesgoso que duplicar una regla de
negocio.

**[confirmado, corrección de diseño]** La "salida" en `t+h` se redefinió en positivo y
**distinta por definición** (antes, pasar a no tener ningún miembro activo contaba como
"salida" en las 3 -- inflaba la tasa):
- **a** (`DECCFR` Q1-Q2 + `activos_sin_empleo_formal`): `sale_por_empleo` (prioridad
  máxima) > `sale_por_ingreso_con_activos`/`sale_por_ingreso_sin_activos` (decil≥5 en
  `t+h`, esta última cuenta aparte cuando sube de decil sin nadie activo -- ingresos no
  laborales) > `pasa_a_sin_activos` > `permanece`.
- **b** (`activos_sin_empleo_formal` solo): `sale_por_empleo` > `pasa_a_sin_activos` >
  `permanece` -- no hay condición de ingreso que abandonar.
- **c** (`DECCFR` Q1 solo): `sale` = decil≥3 en `t+h`, sin condición de actividad (la
  precariedad de c no depende de tener activos, así que acá no existe
  `pasa_a_sin_activos` como categoría aparte -- es sub-conteo descriptivo de `permanece`).

Dos variantes de `sale_binario`, **ambas reportadas siempre**: **principal** (
`pasa_a_sin_activos` cuenta como "no sale" -- la lectura recomendada por default, excluirla
sesga la tasa hacia arriba) y **robustez** (se excluye del denominador). Para **c** son
idénticas por construcción.

| definición | % en riesgo (hist./reg.) | salida h=1 principal (hist./reg.) | salida h=1 robustez (hist./reg.) | salida h=4 principal (hist./reg.) | dif. por `ancla_con_deficit` (h=1, IC95%) |
|---|---|---|---|---|---|
| a | 21.8% / 16.9% | 20.8% / 19.7% | 21.6% / 20.7% | 26.0% / 24.4% | -11.8pp [-12.3, -11.2] |
| b | 34.7% / 33.7% | 9.9% / 9.0% | 10.5% / 9.7% | 15.0% / 12.5% | -2.9pp [-3.3, -2.5] |
| c | 22.2% / 17.1% | 30.4% / 29.7% | = principal | 33.6% / 33.7% | -18.1pp [-18.8, -17.5] |

Tasas bastante más bajas que en el borrador anterior (ej. a, h=1 regular: 19.7% vs. el
32.7% viejo) -- la diferencia es casi toda el sesgo que introducía contar
`pasa_a_sin_activos` como salida.

**[confirmado]** Profundidad (`data/meta/riesgo_salida_ancla_por_decil.csv`/
`_estandarizada.csv`): la brecha por `ancla_con_deficit` **sobrevive dentro de cada
decil**, no es solo composición -- la diferencia estandarizada (neta de la distribución
de decil) es 58% (a), 62% (b) y 78% (c) de la diferencia pooled de la sección 2b. Dicho
de otro modo: incluso comparando hogares del mismo decil, el déficit habitacional sigue
asociado a una salida más lenta.

**[confirmado]** La variante estricta (sale y se mantiene fuera en la visita siguiente
real del hogar -- `t0+4` para `h=1`, `t0+5` para `h=4` -- sobre
`data/04_trayectorias/trayectorias/`, excluyendo `corte_estructural_d13`) da tasas
~8-10pp más bajas que la simple (principal) en las 3 definiciones
(`data/meta/riesgo_salida_estricta.csv`) -- una parte de las "salidas" simples son
inestables (sale y vuelve a entrar antes de la visita siguiente).

**[confirmado]** `n_no_determinable` por `ancla_con_deficit_t`, era regular (`decil_th`
falta cuando el hogar no declara ingreso en `t+h`, D3): en a y c, los hogares **sin**
déficit habitacional en `t` tienen *más* missingness de `decil_th` (11.1%/11.6%) que los
que sí lo tienen (8.3%/8.7%) -- diferencia de ~3pp, no enorme pero tampoco pareja; se deja
registrado como límite del IC de la sección 2b, no se corrige (no hay con qué imputar sin
violar la regla de no-imputación).

**[confirmado]** Exclusión por D3 (`data/meta/riesgo_exclusion_d3.csv`, sin cambios en
esta revisión): en la era regular, los hogares sin quintil válido (~20% del total) no
difieren mucho en `ancla_con_deficit` (2.2% vs. 3.4%) pero sí algo más en
`activos_sin_empleo_formal` (35.4% vs. 33.3%).

**[confirmado, corrección de encuadre]** El inventario de predictores (sección 5) ya no
trata la familia de deciles de `t` (`DECCFR` y variantes) como fuga de datos -- la fuga es
cualquier variable medida en `t+h` usada para predecir el resultado en `t+h`, no la
posición de ingreso en `t`. Pasan a un bloque propio `ingresos_posicion_t` (42 variables),
separado de `identificadores_pesos_excluidos` (pesos/identificadores/metadata, esos sí
quedan fuera).

**Pendiente de decisión de Iván**: cuál de las 3 (o una combinación) define la población
en riesgo para el resto del TFI, y qué variante de `sale_binario` (principal/robustez) usar
como etiqueta. No se recomienda ninguna desde acá.
