# Diccionario de variables en lenguaje llano

> Estado: borrador. Las descripciones siguen el cuestionario de la EPH continua. Las marcadas con
> **(a confirmar)** hay que verificarlas contra el diseño de registros del INDEC, porque el codebook
> disponible solo cubre la estructura vigente desde 2023T4.

---

## 1. Identificación

| Código | Qué es |
|---|---|
| CODUSU | Código de la vivienda. Permite seguirla entre trimestres |
| NRO_HOGAR | Número de hogar dentro de la vivienda (una vivienda puede tener más de un hogar; 51 = servicio doméstico, 71 = pensionistas, son hogares de otro tipo dentro de la misma vivienda) |
| COMPONENTE | Número de persona dentro del hogar |
| ANO4, TRIMESTRE | Año y trimestre del relevamiento |
| AGLOMERADO | Ciudad o conglomerado urbano (31 en total; CABA y Partidos del GBA van por separado) |

## 2. Quiénes forman el hogar

| Código | Pregunta | Respuestas |
|---|---|---|
| CH03 | ¿Qué relación tiene con el jefe o jefa del hogar? | jefe/a, cónyuge o pareja, hijo/a o hijastro/a, yerno o nuera, nieto/a, madre o padre, suegro/a, hermano/a, otros familiares, no familiares |
| CH04 | Sexo | varón, mujer |
| CH06 | ¿Cuántos años cumplidos tiene? | edad en años |

## 3. Trabajo

| Código | Pregunta | Respuestas |
|---|---|---|
| ESTADO | Condición de actividad | ocupado, desocupado, inactivo, menor de 10 años (0 = entrevista individual no realizada) |
| CAT_OCUP | En su trabajo, ¿es… ? | patrón, cuenta propia, obrero o empleado, trabajador familiar sin remuneración |
| PP07H | Por ese trabajo, ¿tiene descuento jubilatorio? (solo asalariados) | sí, no |

Los cuentapropistas y patrones no responden PP07H. Queda por definir cómo se clasifica su formalidad.

## 4. Ingresos de cada persona

| Código | Qué es |
|---|---|
| P21 | Cuánto cobró el mes pasado por su ocupación principal (−9 = no respondió, desde 2016) |
| T_VI | Total de ingresos no laborales de la persona (jubilación, subsidios, alquileres, etc.) |
| V*_M | Monto de cada fuente no laboral: jubilación, subsidio, beca, alquiler, etc. (ver grupo 9) |

## 5. Ingresos del hogar y posición en la distribución

| Código | Qué es |
|---|---|
| ITF | Ingreso total familiar: suma de ingresos de todos los integrantes |
| IPCF | Ingreso per cápita familiar: ITF dividido por la cantidad de integrantes |
| DECCFR | **(a confirmar)** Decil del hogar según su ingreso per cápita, en el total de aglomerados (1 = 10% más pobre, 10 = 10% más rico; 12 = no clasificable, desde 2016) -- no está en el codebook disponible, descripción inferida (ver `src/tipado_nucleo.py`) |
| DECIFR | **(a confirmar)** Igual que DECCFR, pero según el ingreso total del hogar -- mismo motivo |
| RDECCFR / ADECCFR | **(a confirmar)** Decil de ingreso per cápita calculado dentro de la región / dentro del aglomerado -- mismo motivo |

## 6. La vivienda: materiales y servicios

| Código | Pregunta |
|---|---|
| IV1 | Tipo de vivienda: casa, departamento, pieza de inquilinato, pieza de hotel o pensión, local no construido para habitación, otro |
| IV2 | ¿Cuántos ambientes o habitaciones tiene la vivienda? |
| IV3 | Material de los pisos interiores |
| IV4 | Material de la cubierta exterior del techo |
| IV5 | ¿El techo tiene cielorraso o revestimiento interior? |
| IV6 | ¿Tiene agua por cañería dentro de la vivienda, fuera de la vivienda pero dentro del terreno, o fuera del terreno? |
| IV7 | ¿El agua es de red pública, de perforación con bomba a motor, de perforación con bomba manual, o de otra fuente? |
| IV8 | ¿Tiene baño o letrina? |
| IV9 | ¿El baño está dentro de la vivienda, fuera pero en el terreno, o fuera del terreno? |
| IV10 | ¿El baño tiene inodoro con botón/mochila/cadena y arrastre de agua, inodoro sin botón/cadena pero con arrastre de agua (a balde), o es letrina (sin arrastre de agua)? |
| IV11 | ¿El desagüe va a la red pública (cloaca), a cámara séptica y pozo ciego, solo a pozo ciego, o a un hoyo? |

## 7. Entorno de la vivienda

| Código | Pregunta |
|---|---|
| IV12_1 | ¿La vivienda está a 3 cuadras o menos de un basural? |
| IV12_2 | ¿Está en una zona inundable (se inundó en los últimos 12 meses)? |
| IV12_3 | ¿Está en una villa de emergencia? |

## 8. El hogar: uso del espacio y tenencia

| Código | Pregunta |
|---|---|
| II1 | ¿Cuántos ambientes son de uso exclusivo del hogar? |
| II2 | De esos, ¿cuántos se usan habitualmente para dormir? |
| II3, II3_1 | ¿Usa alguno exclusivamente como lugar de trabajo? ¿Cuántos? |
| II4_1, II4_2, II4_3 | ¿Tiene cuarto de cocina? ¿Lavadero? ¿Garage? |
| II5, II5_1 | ¿Alguno de esos cuartos se usa también para dormir? ¿Cuántos? |
| II6, II6_1 | ¿Alguno se usa también para trabajar? ¿Cuántos? |
| II7 | Régimen de tenencia: propietario de la vivienda y el terreno, propietario solo de la vivienda, inquilino/arrendatario, ocupante por pago de impuestos/expensas, ocupante en relación de dependencia, ocupante gratuito (con permiso), ocupante de hecho (sin permiso), en sucesión, otra situación |
| II8 | Combustible para cocinar: gas de red, gas en garrafa, kerosene/leña/carbón, otro |
| II9 | ¿El baño es de uso exclusivo del hogar, compartido con otro/s hogar/es de la misma vivienda, compartido con otra/s vivienda/s, o no tiene baño? |

## 9. De qué vivió el hogar en los últimos tres meses

Preguntas sí/no al hogar. Los montos correspondientes están en la base de personas (V*_M).

| Código | En los últimos tres meses, ¿las personas del hogar vivieron de… | Tipo de estrategia |
|---|---|---|
| V1 | lo que ganan en el trabajo? | trabajo |
| V2 | alguna jubilación o pensión? | transferencia (previsional) |
| V2_01 / V2_02 / V2_03 | (desde 2023T4) jubilación por aportes / por moratoria (ama de casa) / otras pensiones | transferencia (previsional) |
| V21_0N / V22_0N | (desde 2023T4) aguinaldo / retroactivo de cada tipo de jubilación | transferencia (previsional) |
| V3 | indemnización por despido? | trabajo (pérdida) |
| V4 | seguro de desempleo? | transferencia |
| V5 | (confirmado contra los datos, 2003-2023T3) subsidio o ayuda social en dinero del gobierno, iglesias, etc.? -- campo único hasta 2023T3; reemplazado por los tres de abajo desde 2023T4 | transferencia (asistencial) |
| V5_01 / V5_02 / V5_03 | (desde 2023T4, confirmado contra los datos y el codebook p.11) …de la Asignación Universal por Hijo (AUH) y/o Asignación por Embarazo (incluye Tarjeta Alimentar)? / …de otro plan social o subsidio en dinero del gobierno? / …de ayuda en dinero de iglesias, parroquias u ONG? | transferencia (asistencial) |
| V6 | mercadería, ropa o alimentos del gobierno, iglesias, escuelas, etc.? | ayuda en especie |
| V7 | mercadería, ropa o alimentos de familiares, vecinos u otras personas? | ayuda en especie (redes) |
| V8 | algún alquiler de una propiedad? | renta |
| V9 | ganancias de algún negocio en el que no trabajan? | renta |
| V10 | intereses o rentas de plazos fijos o inversiones? | renta |
| V11 | (confirmado contra los datos, 2003-2023T3) una beca de estudio? -- campo único hasta 2023T3; reemplazado por los dos de abajo desde 2023T4 | transferencia |
| V11_01 / V11_02 | (desde 2023T4, confirmado contra los datos y el codebook p.11) …de una beca en dinero del gobierno para continuar o finalizar estudios (ej. beca Progresar)? / …de otra beca en dinero de instituciones no gubernamentales? | transferencia |
| V12 | cuota de alimentos o ayuda en dinero de personas que no viven en el hogar? | redes |
| V13 | gastar lo que tenían ahorrado? | desahorro |
| V14 | préstamos de familiares o amigos? | endeudamiento |
| V15 | préstamos de bancos o financieras? | endeudamiento |
| V16 | compras en cuotas o al fiado (tarjeta, libreta)? | endeudamiento |
| V17 | vender alguna de sus pertenencias? | desahorro |
| V18 | otros ingresos en efectivo (limosnas, juegos de azar, etc.)? | otros |
| V19_A / V19_B | ¿menores de 10 años ayudan con dinero trabajando? / ¿piden? | trabajo infantil |
| V21 / V22 | (hasta 2023T3) **(a confirmar)**: por el nombre podrían ser aguinaldo y retroactivo, pero D2 advierte que no hay continuidad confirmada | — |

Este bloque interesa especialmente por las **estrategias de subsistencia**. Desahorro, endeudamiento y venta de
pertenencias pueden anticipar un deterioro antes de que se vea en el ingreso.

## 10. Ponderadores

| Código | Qué es |
|---|---|
| PONDERA | Cuántas personas de la población representa cada registro |
| PONDIH | Ponderador del hogar corregido por no respuesta de ingresos (desde 2016; 0 = ingreso no declarado) |

---

## Códigos especiales (valen para casi todas las variables)

- **9, 99, 999…** = no sabe / no responde, salvo indicación contraria.
- **−9** = no respondió el monto (solo variables de ingreso, desde 2016).
- **0** = no corresponde (la pregunta no aplicaba a esa persona u hogar).
