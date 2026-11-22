# EXPERIMENTS.md — cuaderno de experimentos

Una entrada por cada cosa que lanzamos o decidimos, con fecha y con el JSON correspondiente
en `reports/`. El formato es el de `docs/EXPERIMENTS_plantilla.md`. Si algo no funcionó, lo
que vale es decir por qué creemos que no funcionó.

---

## Entradas

### 2026-11-19 · Fase P0–P2 · El parser de tool calls del servidor se come las llamadas

**Qué queríamos saber.** Si el backend `endpoint` (el servicio central Qwen3.8-27B-FP8 de la
DGX) servía para lo mismo que el `hf` local, es decir, si el bucle ReAct funcionaba igual con
los dos cerebros. Es la ablación C del plan y la que decide si podemos permitirnos un modelo
grande.

**Qué hicimos.** `ARCA_LLM_BACKEND=endpoint uv run python -m eval.smoke_agents`, con el bucle
ReAct tal cual: el protocolo era texto, el modelo escribe `<tool_call>{...}</tool_call>` y el
parser de `tool_use/parser.py` lo lee.

**Qué pasó.** 0 de 8 problemas y **cero tool calls**, con el campo `content` de la respuesta
vacío. El mismo código con el backend `hf` acertaba. La causa no era el modelo: la vLLM del
servicio lleva un *tool-call parser* activado, que detecta los bloques `<tool_call>` en la
generación, los saca de `content` y los pone en `tool_calls`. Como nosotros no mandábamos
`tools` en la petición, el servidor los descartaba: el modelo emitía la llamada y a nosotros
llegaba una respuesta vacía. Un fallo del servidor que se parece exactamente a "el modelo no
sabe usar herramientas", que es el fallo que veníamos a medir.

Al mirar las respuestas crudas apareció un segundo detalle: cuando `enable_thinking=1`, la
traza de razonamiento tampoco viene en `content`, sino en un campo aparte, y esta build la
llama **`reasoning`** y no `reasoning_content`, que es el nombre que usan otras. Si solo se
lee `reasoning_content`, el razonamiento se pierde sin avisar.

**Qué concluimos.** `generate(messages, max_new_tokens=None, tools=None)`: el `EndpointLLM`
manda las definiciones en `tools`, traduce el historial al formato nativo (la llamada del
asistente en `tool_calls` con su `id`, la observación como mensaje `role: "tool"` con el
`tool_call_id` que le corresponde) y **re-serializa la respuesta a `<tool_call>`**, de modo
que el bucle ReAct, el parser y los agentes no se enteran de qué protocolo se usó. El backend
`hf` acepta `tools` y los ignora: ya los lleva en el prompt de sistema. Se leen los dos
nombres del campo de razonamiento. Está fijado en `tests/agent_core/test_endpoint_native_tools.py`
con `requests` mockeado, porque es un fallo que solo se ve contra el servidor de verdad.

Lección para el informe: **medir una arquitectura contra un endpoint ajeno exige comprobar
primero que el transporte no te está tirando datos.** Habríamos escrito "el 27B no usa
herramientas" con toda la convicción.

---

### 2026-11-20 · Fase P2 · Smoke de los especialistas y del monolítico, 4 configuraciones

**Qué queríamos saber.** Antes de gastar GPU en el banco de los 90: ¿funciona la cadena
completa (modelo → tool call → argumentos → observación → respuesta final) con un modelo de
verdad, con los dos cerebros y con el *thinking* encendido y apagado? El routing está fijado
por el tipo del problema, así que un fallo aquí es del agente, nunca del router.

**Qué hicimos.** `eval/smoke_agents.py`: los 2 primeros problemas de cada uno de los 4 tipos
de `data/rlm_data_test.jsonl` (8 en total), cada uno por su especialista y por el monolítico
de ARCA (16 corridas por configuración), `max_steps=8`. Cuatro configuraciones:

    ARCA_LLM_BACKEND=hf|endpoint × ARCA_ENABLE_THINKING=0|1

JSON con la traza de cada paso en `reports/smoke_{hf,ep}_t{0,1}.json`.

**Qué pasó.**

| configuración | arquitectura | acierto | pasos | tokens | latencia (s) | no_tool |
|---|---|---|---|---|---|---|
| 1.7B think=0 | especialistas | 8/8 | 2.0 | 4126 | 22.3 | 0.00 |
| 1.7B think=0 | monolítico | 8/8 | 2.0 | 3784 | 57.1 | 0.00 |
| 1.7B think=1 | especialistas | **7/8** | 2.1 | 6161 | 81.3 | 0.00 |
| 1.7B think=1 | monolítico | 8/8 | 2.0 | 4587 | 40.8 | 0.00 |
| 27B think=0 | especialistas | 8/8 | 2.0 | 6532 | 2.0 | 0.00 |
| 27B think=0 | monolítico | 8/8 | 2.0 | 5350 | 2.6 | 0.00 |
| 27B think=1 | especialistas | 8/8 | 2.0 | 6815 | 4.3 | 0.00 |
| 27B think=1 | monolítico | 8/8 | 2.0 | 5628 | 4.9 | 0.00 |

Tres cosas que sí se pueden decir con 8 problemas:

1. **La cadena funciona con los dos cerebros.** `no_tool = 0.00` en todas las
   configuraciones: nunca pasó que el modelo contestara en prosa sin llamar a una tool. En la
   fase 2 de ARCA, 13 de 13 fallos fueron "no emite tool call". Ese fallo ha desaparecido, y
   es el resultado que de verdad importa de este smoke.
2. **Dos pasos por problema, siempre**: la tool y `final_answer`. Ninguna configuración gasta
   pasos de más.
3. **El *thinking* no sale gratis y no compra nada aquí.** En el 27B cuesta ~300 tokens y
   duplica la latencia (2.0 → 4.3 s) para el mismo 8/8. En el 1.7B, además, hacía fallar
   problemas que sin él salen bien (abajo). De ahí `ARCA_ENABLE_THINKING=0` por defecto.

**El fallo, que es lo interesante.** Con el 1.7B y `thinking=1`, la corrida del 19-11
(`reports/smoke_hf_thinking.json`) falló 3 de 16, y las tres por el mismo mecanismo. **La tool
devolvió el número correcto y el modelo contestó otro:**

| problema | la tool devolvió | el modelo contestó | qué contestó en realidad |
|---|---|---|---|
| `tae_00047` (especialista) | **7.48** | 6.80 | el `annual_rate_pct` que él mismo había pasado |
| `tae_00063` (especialista) | **8.18** | 6.68 | otra vez el `annual_rate_pct` |
| `cuota_00071` (monolítico) | **174.09** | 154.32 | inventado (ni es `principal/meses` = 154.51) |

No es un fallo de las cuentas (`credit_math` coincide al 100 % con los 600 problemas del
dataset) ni de los argumentos (en los tres casos los argumentos eran los correctos). Es que
`final_answer` **descarta la observación** y manda un número que el modelo se ha producido a
sí mismo. En los dos casos de TAE ese número es el tipo nominal, que es la confusión clásica
nominal/TAE; en el de cuota es una cifra directamente inventada. Con `thinking=1` la traza de
razonamiento es larga, la observación se queda lejos en el contexto y el modelo **re-deriva en
vez de copiar**. Las tres corridas con `thinking=0` no tienen ni un caso.

**Consecuencia para el crítico determinista del router (sección 1.3 del plan), que conviene
leer antes de implementarlo.** El crítico está especificado como "cada número de la respuesta
final tiene que aparecer en alguna observación". **Así especificado no caza `tae_00047` ni
`tae_00063`**, porque 6.80 y 6.68 *sí* aparecen en la observación: están dentro de `"inputs"`,
que es justamente el eco de lo que el modelo mandó. El crítico tiene que comparar el número
final contra el campo **`"result"`**, no contra el texto entero de la observación. Que las
tools devuelvan `{"result", "inputs"}` por separado es lo que lo hace posible.

**Qué concluimos.**

- `ARCA_ENABLE_THINKING=0` por defecto, en los dos backends.
- **Con 8 problemas no se concluye nada sobre especialistas frente a monolítico.** Las dos
  arquitecturas aciertan 8/8 en 3 de las 4 configuraciones: el banco es demasiado fácil y
  demasiado pequeño para distinguirlas. Lo único que mide este smoke es que la cadena no está
  rota. La comparación de verdad es el banco de los 90 (entrada siguiente), y la ablación A
  del plan necesita además las preguntas compuestas y las de routing, donde el monolítico no
  tiene con qué.
- Lo que el smoke sí deja medido y que vale para el informe: el coste. El monolítico gasta
  menos tokens por problema que el especialista (3784 vs 4126 en el 1.7B) porque su prompt de
  sistema lleva 2 tools en vez de 4. Si el multiagente tiene que justificarse, no va a ser por
  tokens en preguntas de un solo tipo.
- Queda propuesta, **sin aplicar**, una regla de `RULES` para el fallo de arriba, para poder
  medirla con y sin (está al final de este fichero).

**La repetición de 1.7B `thinking=1`, y una advertencia sobre fiarse de una sola corrida.**
Esa configuración se repitió el 20-11 con el comprobador arreglado
(`reports/smoke_hf_t1.json`). **El fallo principal se reprodujo exactamente**: `tae_00047`,
la tool devolvió 7.48 y el modelo volvió a contestar 6.80, el tipo nominal, con la misma
traza. Los otros dos (`tae_00063` y `cuota_00071`) salieron bien esta vez, así que la corrida
quedó en 7/8 y 8/8 en lugar de 6/8 y 7/8 del 19-11
(`reports/smoke_hf_thinking.json`, comprobador anterior).

Eso **no** es el comprobador: 6.80 cuando se espera 7.48 es un fallo con cualquier
comprobador. Es variación entre corridas, y conviene tenerla presente porque la generación es
*greedy* (`do_sample=False`) y uno esperaría que fuese determinista. No lo es bit a bit en una
GPU compartida: el orden de reducción de los kernels cambia, un logit se decide por un
margen mínimo y la traza se bifurca. Con `thinking=1` hay cientos de tokens de razonamiento
donde bifurcarse, y por eso la varianza se nota justo en esa configuración. **Conclusión
metodológica: 8 problemas y una corrida no bastan ni para ordenar dos configuraciones.** Las
cifras de thinking de este smoke valen para decidir el valor por defecto (el coste sí es
inequívoco: 6161 tokens frente a 4126, y 81.3 s frente a 22.3), no para afirmar que el
thinking empeora la exactitud. Para eso está el banco de los 90.

**Un bug del comprobador que sí era del comprobador.** La corrida del 19-11 con el 27B marcó
como fallo dos respuestas correctas: el modelo escribe "3.930,84 euros" (notación española) y
`numbers_in` lo leía como 3.930 y 84. Arreglado leyendo las dos notaciones, con la regla de
que un punto solo es separador de miles si todos los grupos que separa tienen tres cifras
(así "174.09" nunca se estropea). Fijado con 13 casos en
`tests/eval/test_smoke_agents.py::test_is_correct`. Es la razón de repetir las 4
configuraciones: **una tabla medida con dos comprobadores distintos no es una tabla.**

---

### 2026-11-20 · Fase P2 · Banco numérico de los 90, al lado de la fase 1 de ARCA

**Qué queríamos saber.** Lo que el smoke de 8 problemas no puede contestar: sobre los 90
problemas del split de test, ¿cuánto acierta cada arquitectura, con cada cerebro, y cuánto de
eso es mérito de las herramientas? La comparación que importa es contra la fase 1 de ARCA,
que son **estos mismos 90 problemas** resueltos por un 1.7B afinado que hacía las cuentas en
el prompt, sin herramientas.

**Qué hicimos.** `eval/numeric_bench.py`: los 90 de `data/rlm_data_test.jsonl`, cada uno por
su especialista y por el monolítico (180 corridas por configuración), `max_steps=8`, el
routing fijado por el tipo. Tres configuraciones, lanzadas con `nohup` en dos cadenas
paralelas (`eval/run_bench_local.sh` en la GPU local y `eval/run_bench_endpoint.sh` contra el
servicio central, que son recursos distintos; dentro de cada cadena, en serie). JSON en
`reports/numeric_{hf_t0,ep_t0,ep_t1}.json` y figura en `reports/numeric_bench.png`.

**Qué pasó.** Aciertos por tipo, con los denominadores a la vista:

| configuración | coste total | cuota | ratio | TAE | **TOTAL** |
|---|---|---|---|---|---|
| ARCA fase 1 · base (sin tools) | 4/19 | 0/23 | 0/27 | 0/21 | **4/90** |
| ARCA fase 1 · SFT (sin tools) | 13/19 | 0/23 | 0/27 | 0/21 | **13/90** |
| ARCA fase 1 · GRPO (sin tools) | 16/19 | 0/23 | 7/27 | 0/21 | **23/90** |
| 1.7B think=0 · especialistas | 19/19 | 23/23 | 27/27 | 21/21 | **90/90** |
| 1.7B think=0 · monolítico | 19/19 | **22/23** | 27/27 | 21/21 | **89/90** |
| 27B think=0 · especialistas | 19/19 | 23/23 | 27/27 | 21/21 | **90/90** |
| 27B think=0 · monolítico | 19/19 | 23/23 | 27/27 | 21/21 | **90/90** |
| 27B think=1 · especialistas | 19/19 | 23/23 | 27/27 | 21/21 | **90/90** |
| 27B think=1 · monolítico | 19/19 | 23/23 | 27/27 | 21/21 | **90/90** |

Coste y comportamiento (media por problema, 180 corridas por configuración):

| configuración | arquitectura | pasos | tokens | latencia (s) | no_tool |
|---|---|---|---|---|---|
| 1.7B think=0 | especialistas | 2.00 | 4081 | 6.4 | 0.000 |
| 1.7B think=0 | monolítico | 2.02 | 3826 | 11.7 | 0.000 |
| 27B think=0 | especialistas | 2.00 | 6519 | 2.0 | 0.000 |
| 27B think=0 | monolítico | 2.00 | 5323 | 2.5 | 0.000 |
| 27B think=1 | especialistas | 2.00 | 6800 | 4.3 | 0.000 |
| 27B think=1 | monolítico | 2.00 | 5658 | 5.1 | 0.000 |

**Qué concluimos.**

1. **El salto es de las herramientas, no del razonamiento.** De 23/90 (lo mejor de la fase 1,
   después de SFT y GRPO) a 90/90, y con el modelo *pequeño*: el 1.7B con tools acierta los 90.
   Los dos tipos que en la fase 1 daban **0/23 (cuota) y 0/21 (TAE) con los tres modelos** —
   los que llevan exponenciales y, en la TAE, una bisección — pasan a 23/23 y 21/21. No es
   que el modelo haya aprendido a calcular: es que ya no calcula. Elige la tool y pasa los
   argumentos, y la aritmética la hace una función de Python verificada contra los 600
   problemas del dataset. Ésta es la figura que justifica la fase 2 entera del proyecto.
2. **El fallo de "no emite tool call" ha desaparecido.** `no_tool = 0.000` en las 540
   corridas de las tres configuraciones. En la fase 2 de ARCA, 13 de 13 fallos eran ése. Lo
   que lo arregla es el diseño de tools (una por métrica, argumentos obligatorios) más el
   transporte nativo del endpoint; no hay ni una respuesta en prosa sin acción.
3. **Este banco no sirve para decidir la ablación A, y hay que decirlo.** Con el 27B las dos
   arquitecturas aciertan 90/90: hay techo. Con el 1.7B la diferencia es **90/90 frente a
   89/90**, o sea **un problema**, que con n=90 no es significativo de nada (test exacto de
   Fisher de 90/90 contra 89/90: p = 1.00). Lo honesto es decir que las dos
   arquitecturas empatan en preguntas de un solo tipo, que es justo lo que este banco mide.
   La ablación A se decidirá con las preguntas compuestas y las de routing (bancos 2 y 3 de la
   sección 2 del plan), donde el monolítico no tiene con qué: una sola tool no sabe de leyes,
   y no hay a quién descomponer la pregunta.
4. **Lo que sí separa a las dos arquitecturas es el coste, y va a favor del monolítico.**
   3826 tokens por problema frente a 4081 en el 1.7B (5323 vs 6519 en el 27B), porque su
   prompt de sistema lleva 2 tools en vez de 4. Si el multiagente se justifica, no será por
   tokens en preguntas simples. (La latencia del monolítico sale peor —11.7 s vs 6.4 s— pero
   eso es ruido de la GPU compartida, no una propiedad de la arquitectura: el 27B, que no
   comparte GPU con nosotros, da 2.5 vs 2.0 s.)
5. **El *thinking* sigue sin comprar nada.** En el 27B, mismo 90/90 por ~300 tokens más y el
   doble de latencia. Se queda apagado por defecto.

**El único fallo de las 540 corridas, con su traza.** `cuota_00106`, monolítico, 1.7B:

```
paso 1  loan_metrics(principal=17050.72, annual_rate_pct=9.26, months=36,
                     upfront_fee=254.37, monthly_income=1925.45, other_monthly_debts=351.43)
        -> {"monthly_payment": 544.27, "total_cost": 2797.37,
            "apr_pct": 10.79, "debt_ratio_pct": 46.52}
paso 2  final_answer(answer="124.32")          <- esperado 544.27
```

La tool acertó, los argumentos eran correctos y el modelo contestó **124.32**, que no es
ninguno de los cuatro campos de la observación ni ninguno de los argumentos: es un número
inventado. Es exactamente el mecanismo de los fallos del smoke del 19-11. **El mismo problema
por el especialista salió bien**, y la diferencia entre las dos observaciones se ve a simple
vista:

```
especialista:  {"result": 544.27, "inputs": {...}}          <- un número, y se llama "result"
monolítico:    {"monthly_payment": 544.27, "total_cost": ...,
                "apr_pct": ..., "debt_ratio_pct": ...}      <- cuatro, y hay que elegir campo
```

Con n=1 esto es una anécdota, no una medida, y como tal hay que contarlo. Pero es una
anécdota que apunta en la misma dirección que el diseño: **una observación con un solo número
no se puede leer mal**. Merece medirse a propósito (sería una variante de la ablación F:
`loan_metrics` con cuatro campos frente a una tool por métrica), no darlo por demostrado aquí.

Nota para el crítico determinista: este fallo **sí** lo cazaría la regla "el número final
tiene que aparecer en la observación", porque 124.32 no aparece en ningún sitio. Los dos
fallos de TAE del smoke **no**, porque el tipo nominal sí aparece, dentro de `"inputs"`. Las
dos cosas juntas dicen que la regla correcta es comparar contra `"result"`.

---

### 2026-11-20 · Decisión de diseño · Nos quedamos en 2 especialistas, y no hay ablación B

**Qué decidimos.** El sistema se queda con **dos especialistas**, `cuota_tae` y `coste_ratio`,
y no se parten en cuatro (ni en tres separando `cuota` de `tae`). Con ello **se cae la
ablación B del plan** ("¿cuántos agentes?", `docs/PLAN.md` §2) y **no se construye el banco
difícil**: no vamos a medir una variante que hemos decidido no mantener.

**Por qué.** Lo que de verdad tiene que acertar el sistema no es la aritmética —eso ya lo
hacen las tools, 600/600 contra el dataset— sino **el reparto**: el router tiene que elegir a
quién manda cada pregunta, y elegir entre 2 es más fácil que elegir entre 4. Cada especialista
que se añade mete dos clases más en el clasificador y duplica las fronteras donde se puede
equivocar, a cambio de nada en exactitud numérica: `cuota` y `tae` comparten los mismos datos
de entrada (principal, tipo, plazo, comisión) y las mismas tools, así que separarlos no
simplifica ni las reglas ni el prompt, solo reparte lo mismo en más cajas. El riesgo que
quitamos (un fallo de routing) es mayor que el que asumimos (un prompt de sistema con dos
tools en vez de una).

**Qué se pierde al decidirlo así, dicho claramente.** Es una decisión de ingeniería, no una
medición: no tenemos el número que diría cuánto se degrada el routing al pasar de 2 a 4
agentes, porque hemos decidido no gastar GPU en medir una arquitectura que no vamos a
entregar. Si en la defensa preguntan "¿y cómo sabéis que 4 es peor?", la respuesta honesta es
que no lo sabemos medido: sabemos que 4 añade clases al problema de routing sin añadir
capacidad, y que el presupuesto de experimentos se ha ido a las preguntas que sí deciden algo
(ablación A, el cerebro y el crítico).

---

### 2026-11-20 · Fase P2 · El banco entero: los 600 problemas con el 1.7B, test y train por separado

**Qué queríamos saber.** Si el 99-100 % de los 90 problemas de test era real o un espejismo de
tener solo 90 muestras. Con n=90 y un acierto del 99 %, el intervalo de confianza llega hasta
el 94 %: cabían perfectamente 5 puntos de caída sin que lo viéramos. La pregunta concreta era
si la exactitud se mantiene en los **600** (90 de test + 510 de train, que nadie había mirado)
y si en una muestra 6,7 veces mayor aparece alguna diferencia entre las dos arquitecturas.

**Qué hicimos.**

- Primero, comprobar el dataset: `data/rlm_data_test.jsonl` tiene 90 problemas y
  `data/rlm_data_train.jsonl` 510; **no se solapan** ni por `id`, ni por enunciado, ni por los
  parámetros del préstamo (`tipo` + `_debug_prestamo`), y no hay duplicados internos. Union =
  600 ids únicos, 150 de cada tipo (19+131 coste total, 23+127 cuota, 27+123 ratio, 21+129 TAE).
- `eval/numeric_bench.py` acepta ahora `--dataset` repetido y `--architectures`, y **cada fila
  guarda de qué fichero viene** (`dataset`, `split`), de modo que test y train se informan por
  separado sin correr dos veces. Si un `id` apareciera en los dos ficheros, el script se niega
  a arrancar: el sidecar se indexa por `id`, y un duplicado reanudaría mal.
- Dos procesos en paralelo en la GPU, uno por arquitectura (`eval/run_bench_600.sh`), con
  `nohup`, `PYTHONUNBUFFERED=1`, un JSON y un log cada uno y escritura incremental reanudable.
  Antes de lanzarlos medimos la memoria: la slice es una **MIG 1g.18gb de 16 GB** y una copia
  del 1.7B ocupa **3,45 GB**, así que caben dos de sobra (durante la corrida: 3984 + 3854 MiB).
- Los 600 desde cero, aunque ya teníamos los 90, para que todo salga de la misma corrida.
  1.7B local (`ARCA_LLM_BACKEND=hf`), `thinking=0`, `max_steps=8`. 1200 corridas en total.
  Ninguna de las dos falló: `exit=0` a la primera, sin usar el reintento (19:24 → 20:43 los
  especialistas, 19:24 → 21:14 el monolítico).

**Qué pasó.** Aciertos por tipo (`reports/numeric_hf_t0_600_stats.json`):

| arquitectura | split | coste total | cuota | ratio | TAE | **total** |
|---|---|---|---|---|---|---|
| especialistas | test (90) | 19/19 | 22/23 | 27/27 | 21/21 | **89/90 (98,9 %)** |
| monolítico | test (90) | 19/19 | 23/23 | 26/27 | 21/21 | **89/90 (98,9 %)** |
| especialistas | train (510) | 130/131 | 125/127 | 122/123 | 128/129 | **505/510 (99,0 %)** |
| monolítico | train (510) | 131/131 | 126/127 | 122/123 | 129/129 | **508/510 (99,6 %)** |
| especialistas | **600** | 149/150 | 147/150 | 149/150 | 149/150 | **594/600 (99,0 %)** |
| monolítico | **600** | 150/150 | 149/150 | 148/150 | 150/150 | **597/600 (99,5 %)** |

Intervalos de Wilson al 95 % del acierto total y McNemar exacto entre arquitecturas (mismos
problemas, pares discordantes):

| split | especialistas | monolítico | McNemar |
|---|---|---|---|
| test (90) | 98,89 % [93,97 ; 99,80] | 98,89 % [93,97 ; 99,80] | 1 vs 1 discordantes, p = 1,00 |
| train (510) | 99,02 % [97,73 ; 99,58] | 99,61 % [98,58 ; 99,89] | 2 vs 5, p = 0,45 |
| **600** | **99,00 % [97,84 ; 99,54]** | **99,50 % [98,54 ; 99,83]** | 3 vs 6, p = 0,51 |

Coste por problema (media de las 600 corridas de cada arquitectura):

| arquitectura | pasos | tokens | latencia (s) | no_tool | llamada repetida |
|---|---|---|---|---|---|
| especialistas | 2,02 | 4155 | 7,9 | 0,000 | 0,013 |
| monolítico | 2,00 | 3736 | 11,0 | 0,000 | 0,000 |

(Las latencias no son comparables entre sí: los dos procesos se repartían los SM de la misma
slice y no empezaron ni acabaron a la vez.)

**Los 9 fallos, por mecanismo** (`eval/stats_numeric.py` los clasifica; el detalle con traza
completa está en el JSON):

1. **Cifra inventada: 6 de 9** (5 de especialistas, 1 de monolítico). La tool devolvió el
   número correcto, el modelo contestó otro. No es un número de `inputs` ni otro campo de la
   observación: no aparece en ninguna parte de la traza.

   ```
   cuota_00043 · cuota_tae · esperado 321.35
   paso 1  calculate_monthly_payment(principal=10111.33, annual_rate_pct=8.96, months=36)
           -> {"result": 321.35, "inputs": {...}}
   paso 2  calculate_monthly_payment(...)   <- la repite, igual
           -> ERROR: duplicate call: ... use that result, or call final_answer
   paso 3  final_answer(answer="La cuota mensual es 306.44 euros.")   <- 306.44
   ```

   Los seis se parecen: 306,44 en vez de 321,35; 226,43 en vez de 256,29; 324,14 en vez de
   336,88; 1463,87 en vez de 1662,33; 13,44 en vez de 63,06; 424,54 en vez de 125,30. Son
   números **del orden de magnitud correcto**, lo que sugiere que el modelo rehace la cuenta
   por su cuenta y manda su resultado en vez de copiar el de la tool. Es exactamente el fallo
   que la propuesta de `RULES` pendiente ("el resultado es SIEMPRE el campo `result`") quiere
   atacar, y ahora tiene n=6 en vez de n=1.

2. **Omite un argumento opcional y la tool devuelve `null`: 2 de 9**, los dos del monolítico y
   los dos de ratio. `loan_metrics` acepta `monthly_income` como opcional, así que no puede
   rechazar la llamada: contesta `debt_ratio_pct: null` y el modelo se queda sin número.

   ```
   ratio_endeudamiento_00106 · monolithic · esperado 33.96
   paso 1  loan_metrics(principal=24847.9, annual_rate_pct=7.94, months=60,
                        other_monthly_debts=147.82)        <- falta monthly_income
           -> {"monthly_payment": 503.11, ..., "debt_ratio_pct": null}
   paso 2  final_answer(answer="Please provide your monthly income to calculate the debt ratio.")
   ```

   **Este es el fallo que el diseño de los especialistas elimina por construcción**: su
   `calculate_debt_ratio` tiene `monthly_income` obligatorio, así que una llamada sin ingresos
   no devuelve `null`, lanza `ValueError` y el modelo lo vuelve a intentar con el dato. 0 de
   600 en los especialistas, 2 de 600 en el monolítico. Es la ablación F del plan, medida de
   paso.

3. **Se agota sin contestar: 1 de 9** (`tae_00108`, especialistas). La tool dio 14,91 en el
   primer paso y el modelo repitió la misma llamada hasta quedarse sin pasos, sin llegar nunca
   a `final_answer`.

Nótese que **4 de los 6 "cifra inventada" y el único "se agota" vienen precedidos de una
llamada repetida** (el modelo reemite la misma tool call en vez de leer la observación). La
repetición aparece en el 1,3 % de las corridas de los especialistas y en ninguna del
monolítico; cuando aparece, suele acabar mal. No sabemos si es causa o síntoma del mismo
despiste.

**Qué concluimos.**

1. **La calidad se mantiene: no era un espejismo de n=90.** 99,0 % y 99,5 % sobre 600, con
   intervalos de Wilson que no bajan del 97,8 %. Los 510 de train, que nadie había visto, dan
   exactamente lo mismo que los 90 de test (99,0 % vs 98,9 % en especialistas; 99,6 % vs 98,9 %
   en monolítico). El salto frente a la fase 1 de ARCA (23/90 en el mejor caso) sigue en pie
   con 6,7 veces más muestras.
2. **Las dos arquitecturas siguen empatadas, y ahora con potencia para decirlo.** McNemar sobre
   los 600 da p = 0,51 (3 fallos exclusivos de los especialistas contra 6 del monolítico). Con
   n=600 ya no es "no sabemos porque son pocos": es que **no hay diferencia en exactitud
   numérica**, y la ablación A no se puede ganar aquí. Si acaso, la tendencia va **a favor del
   monolítico** (+0,5 puntos, no significativo), que además gasta menos tokens (3736 vs 4155).
   El multiagente tendrá que justificarse en los bancos de routing y de preguntas compuestas, o
   no justificarse.
3. **El modo de fallo dominante ha cambiado de sitio.** En ARCA era "no emite tool call" (13 de
   13); aquí es 0 de 1200. Ahora el fallo es **"llama bien, lee mal"**: 6 de 9 son el modelo
   pisando el resultado de la tool con un número suyo. Eso no lo arregla una tool mejor; lo
   arregla el **crítico determinista** comparando la respuesta final contra el campo `result`
   de la última observación. Los 6 casos lo cazaría: ninguno de los números inventados aparece
   en la traza. Es la ablación E, y este experimento le pone número al techo que puede subir:
   de 99,0 % a 99,8 % en los especialistas.
4. **Un argumento opcional sigue siendo un fallo esperando a pasar.** Los 2 fallos de `null` son
   el 100 % de la diferencia entre las dos arquitecturas en ratio, y son la consecuencia directa
   de que `loan_metrics` deje `monthly_income` por defecto. Confirma la regla del proyecto:
   argumento obligatorio siempre que un default silencioso dé una respuesta incorrecta.
5. **La corrida no es bit-reproducible, y hay que decirlo.** Con `do_sample=False` y la misma
   configuración, de las 180 corridas de los 90 de test que repiten las del 20-11, solo **146
   dan la respuesta literalmente idéntica** y **3 cambian de veredicto**: `cuota_00043`
   (especialistas) pasa de acertar a fallar, `ratio_endeudamiento_00106` (monolítico) de
   acertar a fallar, y `cuota_00106` (monolítico) — el famoso fallo del "124.32" de la entrada
   anterior — **ahora acierta**. Greedy no garantiza determinismo bit a bit en GPU (el orden de
   reducción de los kernels cambia con la ocupación, y aquí había dos procesos compartiendo la
   slice). Consecuencia práctica: **el 90/90 de la entrada del 20-11 y este 89/90 son la misma
   medida**, y ninguna anécdota construida sobre un solo caso (incluida la del 124.32) es
   reproducible. Las conclusiones hay que sacarlas de los 600, no de un problema.

**Dónde está todo.** `reports/numeric_hf_t0_600_specialists.json` y
`reports/numeric_hf_t0_600_monolithic.json` (una por proceso, con sus filas y su
`by_split_counts`), sus sidecars `.rows.jsonl`, el análisis en
`reports/numeric_hf_t0_600_stats.json` y la figura en `reports/numeric_bench_600.png` (panel de
arriba: los 90 de test con la fase 1 de ARCA en gris; panel de abajo: los 600). Los logs, en
`reports/logs/numeric_hf_t0_600_*.log`.

---

## Propuestas pendientes de medir

### Regla de `RULES` para "contesta con un número de `inputs` o inventado en vez de con `result`"

Sale del fallo del smoke del 20-11-2026 (3/16 con el 1.7B y `thinking=1`). **No está
aplicada**: la idea es medir con y sin ella, sobre el banco de los 90 y con el 1.7B, que es la
única configuración donde el fallo aparece. Si se aplica antes de medir, no sabremos si sirve.

Para `agents/cuota_tae.py`, añadir a `RULES`:

```
- The answer is ALWAYS the "result" field of the tool you just called. Copy that number
  digit by digit into final_answer. Never send a number taken from "inputs", never
  recompute it and never adjust it: "inputs" is only the echo of what you sent, so that
  the answer can be checked, and your own arithmetic is not more reliable than the tool's.
- Check one thing before calling final_answer: when upfront_fee is greater than 0, the APR
  is ALWAYS strictly greater than annual_rate_pct. If the number you are about to send is
  equal to annual_rate_pct, you are reporting the nominal rate instead of the APR: read
  the "result" of calculate_apr again and send that.
```

Para `agents/monolithic.py` (mismo fallo, un caso de 8), añadir a `RULES`:

```
- The answer is ALWAYS the field you read from loan_metrics' result, copied digit by digit.
  Never recompute it and never replace it with a number from your own reasoning: if your
  reasoning and the tool disagree, the tool is right.
- apr_pct is always strictly greater than annual_rate_pct when there is an opening fee.
```

Hipótesis a falsar: la regla no servirá de gran cosa, porque el fallo es de atención (la
observación queda lejos en un contexto largo de razonamiento) y no de desconocimiento de la
norma — `RULES` ya dice "Read \"result\"" y aun así falló. Si la hipótesis se cumple, el
arreglo bueno es el crítico determinista contra `"result"`, que no depende de que el modelo se
porte bien. Esto es exactamente la ablación E del plan, y merece ser una fila del informe.
