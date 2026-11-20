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
| 1.7B think=1 | especialistas | _(ver nota)_ | | | | |
| 1.7B think=1 | monolítico | _(ver nota)_ | | | | |
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

**Nota sobre la casilla que falta.** La configuración 1.7B `thinking=1` se relanzó el 20-11
con el comprobador arreglado, pero con la GPU compartida (64 GB ocupados por otro proceso)
tardaba 8.6 min por problema, 2.3 h para las 16 corridas. Se dejó para después del banco de
los 90, que es el que mide. Los números del 19-11 para esa configuración, con el comprobador
anterior, están en `reports/smoke_hf_thinking.json`: 6/8 especialistas y 7/8 monolítico, al
doble de tokens que sin thinking. Los 3 fallos analizados arriba son de esa corrida y no son
artefactos del comprobador (contestar 6.80 cuando se espera 7.48 es un fallo con cualquier
comprobador).

**Un bug del comprobador que sí era del comprobador.** La corrida del 19-11 con el 27B marcó
como fallo dos respuestas correctas: el modelo escribe "3.930,84 euros" (notación española) y
`numbers_in` lo leía como 3.930 y 84. Arreglado leyendo las dos notaciones, con la regla de
que un punto solo es separador de miles si todos los grupos que separa tienen tres cifras
(así "174.09" nunca se estropea). Fijado con 13 casos en
`tests/eval/test_smoke_agents.py::test_is_correct`. Es la razón de repetir las 4
configuraciones: **una tabla medida con dos comprobadores distintos no es una tabla.**

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
