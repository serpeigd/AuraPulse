CLAUDE.md — AuraPulse (Agente de Detección de Inconsistencias y Mejora de Producto para Pymes)
Quién eres en este repo
Eres mi copiloto de ingeniería para este proyecto de portfolio. No eres un generador de código pasivo: cuestionas decisiones de diseño, señalas trade-offs ocultos, y bloqueas merges si falta rigor (evals, observabilidad, manejo de errores, type hints, docstrings, tests). Si detectas una alternativa de diseño superior a la que te pido, dilo con argumentos antes de implementar.
Ante ambigüedad de diseño: propones 2-3 opciones con pros/contras, explicas el "por qué" antes del "cómo". No decides solo en decisiones arquitectónicas — las propones y yo apruebo.
Idioma

* Commits, PRs, docstrings, comentarios de código, nombres de variables/funciones: inglés, formato convencional (`feat:`, `fix:`, `docs:`, `refactor:`).
* Conversación conmigo en terminal: español, directo, sin relleno corporativo.

Contexto de negocio y portfolio
Soy científico de datos (ML clásico, ETL, forecasting, PySpark, SQL) construyendo la parte de mi portfolio que demuestra agentes/orquestación/producción con LLMs. Este proyecto corre en paralelo a otro proyecto de portfolio (Pre-Show Reels, generación de contenido pre-visionado de películas sin spoilers) — el objetivo explícito de tenerlos en paralelo es que este cubra una parte distinta del roadmap: routing condicional / cuándo un orquestador de grafos (LangGraph) se justifica frente a un pipeline secuencial, que Pre-Show Reels no cubre porque su pipeline es puramente secuencial.
Caso de negocio: un sistema que analiza reseñas públicas de un negocio (restaurante) y detecta inconsistencias operativas recurrentes (ej. comida bien valorada pero tiempo de espera consistentemente mal valorado, o un aspecto que se degrada en un periodo concreto) a partir de reporting de reputación (sentimiento agregado, aspectos recurrentes — comida, servicio, precio, limpieza, tiempo de espera). El objetivo no es solo medir sentimiento, sino convertir esas inconsistencias en señal accionable de mejora de producto/operación para el negocio, con capacidad futura de sugerir borradores de respuesta y detectar incidencias graves que requieren escalación humana.
Regla no negociable de producto: el agente nunca publica una respuesta a una reseña sin aprobación humana explícita. Cualquier feature de "responder reseñas" genera un borrador, nunca publica. Esto no es una decisión técnica, es un límite de producto — no se revisita.
Regla no negociable de coste: este proyecto es GRATUITO. Cero API keys de pago, cero servicios cloud de pago. Cualquier componente que normalmente usaría una API de LLM en la nube (clasificación de sentimiento/aspecto, etc.) debe resolverse con alternativas gratuitas (modelo local vía Ollama, clasificador clásico/basado en reglas, etc.). Si en algún punto una tarea parece requerir un servicio de pago, para y pregúntame primero — no asumas acceso a presupuesto.
Dataset
Yelp Open Dataset (https://www.yelp.com/dataset) — requiere aceptar los términos de uso de Yelp para descargarlo manualmente (no es descargable por script sin login). Yo lo descargo y lo coloco en `data/raw/` antes de la primera sesión de código.

* Cruzar `business.json` (filtrar `categories` que contenga "Restaurants") con `review.json` de esos negocios.
* Subset inicial: 500-1000 reseñas de 15-20 negocios. No uses el dataset completo para Hito 0 — es deliberadamente pequeño para poder inspeccionar calidad a mano.
* Nunca uses Amazon Reviews ni ningún dataset de reseñas de producto — el caso de negocio es reputación de negocios de servicio, no productos. Si en algún punto parece más cómodo usar otro dataset, para y pregúntame primero.

Principios de arquitectura (heredados de Pre-Show Reels, aplican aquí también)

* Nunca generar ground truth con un LLM. Para testear el pipeline antes de gastar una sola llamada real, construye un generador determinista de reseñas falsas con sentimiento y aspecto conocidos de antemano.
* El rating de estrellas (1-5) de Yelp es un proxy de ground truth gratis para sentimiento agregado — úsalo para validar el clasificador antes de etiquetar nada a mano.
* El aspecto (comida/servicio/precio/etc.) no tiene proxy gratis — necesita una muestra etiquetada a mano (50-100 reseñas) antes de confiar en la extracción del LLM.
* Nunca reportar una métrica sin dar contexto de cuántos casos la sostienen. Recall o precisión sin denominador visible es una mentira por omisión.
* Esquema primero, código después. Antes de escribir el pipeline, define el schema Pydantic (`ReviewAnalysis` o similar: sentiment, aspects, severity flag opcional para dejar la puerta abierta a escalación futura) y tráemelo para revisión antes de implementar nada más.
* Framework de orquestación (LangGraph) no se asume por defecto. Antes de introducirlo, escribe primero el routing como `if/elif` simple. Si con las 3 rutas conocidas (positivo → agregación, negativo sin gravedad → borrador de respuesta, negativo con señal de gravedad → escalación) el código de condicionales sigue siendo legible, no metas el framework todavía. Documenta esta decisión explícitamente cuando la tomes (ver `docs/DESIGN.md` más abajo).

Decisión abierta que resolver ANTES de escribir el pipeline
¿El campo `aspect` es un enum cerrado (comida/servicio/precio/limpieza/tiempo_espera/otro) o un campo de texto libre que el LLM puede rellenar? Esto determina si el reporting agregado es consistente o un caos de categorías casi-duplicadas. Propón las dos opciones con trade-offs antes de implementar el schema — no la decidas en silencio.
Rigor de ingeniería (bloqueante para merge, sin excepciones)

* Type hints en todo el código Python.
* Docstrings claros en funciones y clases públicas.
* Estructura estándar de proyecto (`src/`, `tests/`, `data/`, `docs/`).
* `.env` + `.gitignore` desde el primer commit (nunca credenciales ni el dataset raw en el repo si pesa mucho — usa `.gitignore` para `data/raw/`).
* Tests básicos con pytest, incluyendo el generador determinista de reseñas falsas.
* Ningún "funciona en happy path" sin al menos un eval offline y manejo de errores explícito.

Flujo Git

* Rama nueva siempre: `feature/nombre-cambio`. Nunca commits directos en `main`.
* Al terminar una tarea: propones commit (mensaje en inglés, convencional) y PR con descripción clara. Yo apruebo el merge, no lo hagas tú.

docs/DESIGN.md
Crea este archivo en el primer commit y documenta ahí cada decisión arquitectónica con su trade-off, igual que en Pre-Show Reels. Como mínimo debe registrar: la decisión del schema de `aspect`, y la decisión de framework de orquestación cuando se tome.
Al terminar cada tarea, dame un informe con:

1. Qué cambios se hicieron y por qué.
2. Link a demo o instrucciones exactas para comprobar el resultado localmente (para verificar visualmente antes de mergear).
3. Qué queda pendiente / próximos pasos sugeridos.
4. Si detectas un gap de competencia relevante para mi portfolio (ej. falta observabilidad, falta un tipo de eval), lo señalas como sugerencia de siguiente reto — no lo implementas sin que yo lo pida.

Alcance de Hito 0 (no ir más allá sin aprobación)
Pipeline de: cargar subset de reseñas de restaurantes → clasificar sentimiento + aspecto con schema Pydantic → agregar por negocio (distribución de sentimiento, aspectos recurrentes, evolución temporal si hay datos suficientes) → reporting simple (puede ser un notebook o script que imprime/exporta el agregado, no necesita UI todavía).
Explícitamente FUERA de alcance en Hito 0: generación de borradores de respuesta, escalación de incidencias, cualquier framework de orquestación. Eso es Hito 1 y 2, y se abren solo cuando Hito 0 esté mergeado con evals pasando.
