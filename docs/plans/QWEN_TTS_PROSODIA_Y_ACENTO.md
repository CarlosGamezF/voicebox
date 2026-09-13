# Qwen3-TTS: entonación plana y acento en español

**Estado:** investigación verificada contra el código; nivel T1 implementado en la rama `feat/spanish-prosody-t1` (ver sección de correcciones), resto pendiente
**Afecta a:** `backend/backends/pytorch_backend.py`, `backend/backends/mlx_backend.py`, `backend/backends/qwen_custom_voice_backend.py`, `backend/utils/chunked_tts.py`, perfiles de voz en `app/src/components/VoicesTab/VoiceInspector.tsx`
**Última revisión:** 2026-09-13

## Correcciones tras verificar el código (2026-09-13)

Revisión contra `main`, `mlx-audio` 0.4.1 (`mlx_audio/tts/models/qwen3_tts/`) y `qwen_tts` 0.1.1, con lectores y verificadores independientes. Lo que sigue corrige o completa las tablas de más abajo:

1. **Idioma.** La app de escritorio ya envía el idioma del perfil (`FloatingGenerateBox.tsx:155-158`), así que "idioma sin fijar" no explica la ruta de escritorio. Sí aplicaba a `POST /speak`, al MCP `voicebox.speak` y a la reproducción de capturas, que forzaban `en`; `/speak` y MCP están corregidos en esta rama (issue upstream #1049, PR #1054), la reproducción de capturas sigue pendiente.
2. **Muestreo en MLX.** En clonación, `mlx-audio` fuerza `repetition_penalty = max(x, 1.5)` sobre todo el historial del trozo (`qwen3_tts.py:975`, `:711-721`), no el 1,05 de `qwen_tts`, y limita los tokens de códec a `max(75, 6 × tokens de texto)` (`:1858-1863`) recortando en silencio el habla lenta. El texto español produce ~3,5-4,1 caracteres por token y consume ~2,7-3,1 tokens de códec por token de texto: margen de ~2× a 550 caracteres, ~1,4× en frases de 38.
3. **Fallbacks silenciosos.** El backend MLX generaba con la voz genérica del modelo si faltaba el wav de referencia o si la clonación lanzaba cualquier excepción, dejando solo un warning (`mlx_backend.py:211-217`, `:245-250`). Corregido en esta rama: ahora la generación falla.
4. **`instruct`.** La UI lo oculta para todo lo que no sea CustomVoice desde abril (`useGenerationForm.ts:142-154`); la fila "se envía siempre" está desfasada. Queda un hueco distinto: `EngineModelSelector` no recibe el perfil, así que un clon puede elegir CustomVoice y recibir un 400.
5. **Referencia.** No hay recorte a 10-15 s ni relleno de silencio final; el borde se recorta a 40 dB y se devuelven como mucho 100 ms. La UI decía que "30 segundos es el punto ideal", lo contrario de este documento; copia corregida en esta rama. El botón de transcribir usaba el Whisper cargado, `base` en frío; ahora sigue el ajuste `stt_model`.
6. **Chunking.** Abreviaturas solo inglesas, nunca se parte tras una cifra, los saltos de párrafo no son frontera y `mlx-audio` vuelve a partir por `\n`. `/speak` y MCP ignoraban el ajuste de 550/80 y retry/regenerar vuelven a 800/50 con semilla aleatoria: no sirven para comparar. Lo primero está corregido en esta rama; lo segundo, pendiente.
7. **Normalizador.** Recorta en duro al 0,85 de pico entre el 0,01 % y el 0,06 % de las muestras; efecto menor.

## Problema

El audio generado en español con una voz generada (diseñada por descripción o clonada) suena lineal: entonación plana y acento no nativo. La pregunta era si los modelos de Qwen ofrecen algún control para mejorarlo.

## Resumen ejecutivo

1. En toda la familia Qwen3-TTS **no existe ningún mando numérico de prosodia** (velocidad, tono, pausas, SSML). La entonación depende solo de cuatro cosas: el modelo y la voz, el texto de instrucción (solo en dos checkpoints), el clip de referencia si se clona, y la puntuación del texto.
2. **Ninguna voz preinstalada de los pesos abiertos es hispanohablante nativa**, y las voces diseñadas por descripción (VoiceDesign) generan español con acento inglés intermitente. Es un problema del modelo, reportado en GitHub (#230, #315) y sin respuesta de los mantenedores.
3. El parámetro `instruct` **solo lo respetan `Qwen3-TTS-12Hz-1.7B-CustomVoice` y `Qwen3-TTS-12Hz-1.7B-VoiceDesign`**. El `0.6B-CustomVoice` lo acepta y lo descarta en silencio (`instruct = None` en el código, aunque su model card diga lo contrario). Los modelos `Base` (clonación) no tienen ese parámetro: cae en `**kwargs` y se ignora sin error.
4. La ruta con mejores resultados para español natural es **clonar con `1.7B-Base` desde 10-15 s de un hablante nativo grabado con entonación variada**, en modo con transcripción (`ref_audio` + `ref_text`, `x_vector_only_mode=False`). El clon copia la prosodia del clip: referencia plana, clon plano.
5. Si el acento persiste, la única solución real es **fine-tuning del Base** con datos nativos; no borra del todo el acento base (#323).

"Lineal" casi siempre es la suma de: voz no nativa + instrucción que el modelo ignora + referencia plana + idioma sin fijar.

## Hallazgos verificados

Investigación del 2026-09-13 con 67 fuentes primarias (repositorio `QwenLM/Qwen3-TTS`, model cards en Hugging Face, informe técnico arXiv 2601.15621, documentación de Alibaba Cloud Model Studio, issues y discusiones). 20 afirmaciones críticas se verificaron una a una contra las fuentes; 8 se corrigieron antes de escribir este documento.

### Pesos abiertos (Qwen3-TTS-12Hz)

| Checkpoint | Qué hace | `instruct` | Nota |
| --- | --- | --- | --- |
| `1.7B-VoiceDesign` | Voz desde una descripción de texto | Sí (obligatorio) | Es la ruta con más acento inglés reportado en español. Emoción y estilo responden; acento o dialecto por instrucción no funciona (#248; #134 cerrado como *not planned*) |
| `1.7B-CustomVoice` | 9 voces preinstaladas (chinas, inglesas, una japonesa, una coreana) | Sí | Ninguna nativa de español |
| `0.6B-CustomVoice` | Mismas 9 voces | **No** (lo descarta sin avisar) | El model card en HF afirma lo contrario; manda el código |
| `1.7B-Base` / `0.6B-Base` | Clonación desde 3 s de audio | **No** | Solo la referencia y el fine-tuning fijan el estilo |

- Fijar el idioma mejora pronunciación y entonación: `language="Spanish"`. `Auto` está documentado como "accuracy not guaranteed". La validación de `qwen_tts` es insensible a mayúsculas, así que `"spanish"` también vale.
- Muestreo: los valores por defecto (`temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`) ya son el máximo razonable de variación. Bajar la temperatura aplana más; hacerlo (0.5-0.7) solo si aparecen artefactos. No existe kwarg `seed`; se fija con `torch.manual_seed`.
- Puntuación: comas, puntos, interrogaciones y exclamaciones son lo único que mueve pausas y énfasis. Saltos de línea, puntos suspensivos y guiones no hacen nada fiable (#75).
- Referencia de clonación: la calidad mejora de 3 a 15 s y luego se estanca; clips largos pueden colgar la generación. Añadir ~0,5 s de silencio al final del clip evita que el primer fonema generado arrastre el último de la referencia.
- Deriva de velocidad en clones (se acelera hacia el final, #239/#290, cerrados *not planned*): trozos más cortos (<200 caracteres), más puntuación, referencia más lenta.
- "Voice Design then Clone" (documentado en el README): generar el clip de referencia en español con `1.7B-VoiceDesign` + `instruct`, y clonarlo con Base. El estilo queda dentro de la referencia; no se cambia por llamada.
- Fine-tuning: `finetuning/prepare_data.py` + `sft_12hz.py`, JSONL con `audio`, `text`, `ref_audio` (el mismo `ref_audio` en todas las muestras), `--lr 2e-6 --batch_size 32 --num_epochs 10`. Un usuario corrigió la pronunciación francesa con ~30 frases nativas (#230). El modelo afinado ignora `instruct` (#121); si el habla se acelera con cada época, aplicar el parche del PR #178.
- Recurso listo: `alblez/qwen3-tts-spanish-voices` trae 12 clones de hablantes nativos (España, México, Argentina, Chile, Colombia) sobre `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`, más un script de curación de clips. Demuestra que el acento regional sale solo del clip. El audio de origen es GPL-3.0 si se redistribuyen voces.
- Texto con anglicismos o marcas: partir en fronteras de idioma y sintetizar cada trozo con su idioma explícito (`language` admite una lista por elemento del batch), o reescribir esas palabras con grafía fonética española. El informe técnico no evalúa la mezcla de idiomas.
- Runtimes de terceros (mlx-audio, koboldcpp) recortan el habla lenta o emotiva si `max_tokens` se calcula a partir de la longitud del texto (omlx #843). koboldcpp #2143 confirma de forma independiente que el acento inglés es del modelo y que un clip nativo de 5-10 s con Base es la solución.

### API de Alibaba Cloud (Model Studio)

- `qwen3-tts-flash` y `qwen3-tts-instruct-flash`: campos `text` (≤600 caracteres), `voice`, `language_type`, `instructions`, `optimize_instructions`. Sin `rate`, `pitch` ni `volume`, tampoco en las variantes realtime. Streaming por cabecera `X-DashScope-SSE: enable`.
- Voces nativas de español: `Bodega` (hombre, España) y `Sonrisa` (mujer, latinoamericana), solo en `qwen3-tts-flash`. No existen en `instruct-flash`: hay que elegir entre acento nativo sin instrucciones o instrucciones con voz no nativa.
- `instructions` (solo Instruct-Flash, chino o inglés, ≤1.600 tokens) con `optimize_instructions: true`. Dimensiones sugeridas por la documentación: tono, velocidad, emoción, características, caso de uso. El acento no está entre ellas.
- Clonación: `qwen-voice-enrollment` con `target_model: "qwen3-tts-vc-2026-01-22"`, referencia de 10-20 s a 24 kHz mono. Las voces clonadas o diseñadas no aceptan `instructions`.
- Si hacen falta mandos numéricos: `qwen-audio-3.0-tts-flash` / `-plus` (API WebSocket) expone `rate` 0.5-2, `pitch` 0.5-2, `volume`, `language_hints: ["es"]`, etiquetas en el texto (`[excited]`, `[very slowly]`) y SSML (`<break time="500ms"/>`). Sus voces de sistema son chino e inglés; para español hay que clonar. Las etiquetas solo funcionan en streaming unidireccional y `<phoneme>` solo admite pinyin y CMU.

### Modelos Omni

- `Qwen3-Omni-30B-A3B-Instruct` (abierto): la única palanca es `speaker` (Ethan, Chelsie, Aiden), personajes en inglés y mandarín. Existe un patrón oficial de lectura literal en la documentación de transformers, pero no hay control de estilo y no está pensado para leer guiones. Las versiones de API `qwen3.5-omni-plus/-flash` aceptan instrucciones de velocidad y emoción en el system prompt e incluyen Bodega y Sonrisa. Para narración, Qwen3-TTS sigue siendo mejor opción.

## Qué significa para Voicebox

Estado actual del código (revisado el 2026-09-13):

| Punto | Dónde | Efecto sobre el problema |
| --- | --- | --- |
| El idioma sale del perfil de voz y por defecto es `en` | `app/src/components/VoicesTab/VoiceInspector.tsx:83`, `backend/models.py:84` | Un perfil creado sin cambiar el idioma sintetiza español con condicionamiento inglés. Es la causa más probable del acento y la entonación plana |
| El campo `instruct` se envía siempre al backend | `backend/routes/generations.py`, `backend/services/generation.py:40` | Con voces clonadas (Base) `qwen_tts` lo ignora sin error; en MLX ni se pasa. Solo tiene efecto con el backend CustomVoice 1.7B. La UI da a entender que funciona |
| Clonación PyTorch en modo con transcripción | `backend/backends/pytorch_backend.py:179-182` (`x_vector_only_mode=False`) | Correcto: es el modo que mejor preserva la prosodia |
| Clonación MLX | `backend/backends/mlx_backend.py:232` (`ref_audio` + `ref_text` + `lang_code`) | Correcto en lo esencial; `mlx-audio` 0.4.1 pasa `lang_code` tal cual |
| Trozos de hasta 800 caracteres por llamada | `backend/utils/chunked_tts.py:22` (`DEFAULT_MAX_CHUNK_CHARS`), `max_chunk_chars` en `GenerationRequest` | Trozos largos favorecen la deriva de velocidad y la pérdida de énfasis en Qwen3-TTS. Ya existe crossfade al concatenar |
| Semilla expuesta por generación | `seed` en ambos backends | Permite best-of-N por trozo |
| Parámetros de muestreo no expuestos para Qwen | `models.py` solo tiene `temperature` para el LLM | Se usan los valores por defecto de `qwen_tts` (0.9), que es lo recomendable |

### Plan de acción

Ordenado por relación esfuerzo/impacto.

1. **Comprobar el idioma del perfil de voz.** Debe ser `es`. Si el perfil se creó con `en`, cambiarlo y regenerar. Coste cero.
2. **Cambiar la referencia.** Clonar desde 10-15 s de un hablante nativo (castellano para es-ES) leyendo con entonación variada y sin ruido; reducir `max_chunk_chars` a unos 200 en la petición. Si no hay grabaciones, probar las voces de `alblez/qwen3-tts-spanish-voices`.
3. **En la UI, desactivar u ocultar `instruct` cuando la voz activa es un clon (Base) o el backend es MLX**, o mostrar un aviso. Hoy el usuario cree que está controlando el estilo y no es así.
4. **Añadir best-of-N por trozo**: generar 3-5 candidatos con semillas distintas y elegir con un verificador automático: dos ASR de familias distintas (Whisper y wav2vec2) para descartar trozos rotos, similitud de hablante con WavLM y calidad con UTMOS (`torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`). arXiv 2607.08256 reporta 12-16 % menos errores de palabra con N=10.
5. **Voces diseñadas**: generar el clip de referencia en español con `1.7B-VoiceDesign` + `instruct` explícito ("native Castilian Spanish speaker, expressive narration...") y clonarlo con Base. Validar a oído: el acento sigue siendo la parte no resuelta.
6. **Texto mixto**: partir en fronteras de idioma y sintetizar cada trozo con su idioma, o reescribir anglicismos con grafía fonética.
7. **Último recurso**: fine-tuning del `1.7B-Base` con 30-100 frases nativas del hablante objetivo siguiendo `finetuning/README.md` del repositorio.

## Preguntas abiertas

- Si `instruct` en español (no en inglés o chino) funciona en `1.7B-CustomVoice` y `1.7B-VoiceDesign`: solo hay ejemplos oficiales en esos dos idiomas.
- Cómo afecta el muestreo a la expresividad: la guía es comunitaria, la documentación oficial no lo trata.
- Si `speed` en vLLM-Omni (`POST /v1/audio/speech`, 0.25-4.0) es control del modelo o time-stretch.
- Si alguna de las 500+ voces base de Qwen-Audio-3.0-TTS es hispanohablante nativa: la lista solo está en un Excel descargable.

## Fuentes principales

- https://github.com/QwenLM/Qwen3-TTS (README, `qwen_tts/inference/qwen3_tts_model.py`, `finetuning/`)
- https://huggingface.co/collections/Qwen/qwen3-tts
- https://arxiv.org/html/2601.15621v1 (informe técnico Qwen3-TTS)
- Issues y discusiones: #75, #121, #134, #230, #239, #248, #290, #315, #323, PR #178; HF discussion #38
- https://www.alibabacloud.com/help/en/model-studio/qwen-tts y https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api
- https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list
- https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning y https://www.alibabacloud.com/help/en/model-studio/voice-design-user-guide
- https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide (Qwen-Audio-3.0-TTS)
- https://github.com/QwenLM/Qwen3-Omni y https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/
- https://github.com/alblez/qwen3-tts-spanish-voices
- https://github.com/LostRuins/koboldcpp/discussions/2143
- https://arxiv.org/html/2607.08256 (best-of-N con verificador ASR)
- https://github.com/tarepan/SpeechMOS
