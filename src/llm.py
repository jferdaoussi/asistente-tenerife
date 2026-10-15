"""Cliente del LLM (Google Gemini) con reintentos y modelos de respaldo.

Todas las llamadas al modelo de chat pasan por este módulo, de modo que
cambiar de proveedor solo exige reescribir este archivo.

Estrategia ante fallos:

1. El SDK reintenta cada petición con espera exponencial ante errores
   transitorios (408, 429, 5xx), hasta ``ModelConfig.retry_attempts`` veces.
2. Si el modelo sigue sin responder, se pasa al siguiente modelo de
   ``ModelConfig.fallback_models``.
3. Los errores que no se arreglan reintentando (clave inválida, petición mal
   formada) se transforman en un :class:`LLMError` con un mensaje útil.
"""

import logging
import time
from collections.abc import Iterator

from google import genai
from google.genai import errors, types

from src.config import MODEL, ModelConfig, get_api_key

logger = logging.getLogger(__name__)

#: Códigos HTTP que indican saturación o fallo temporal del servicio.
TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}

#: Mensajes para el usuario según el código de error de la API.
FRIENDLY_ERRORS = {
    400: "La petición al modelo no es válida. Revisa los parámetros.",
    401: "La API key no es válida. Revisa GEMINI_API_KEY en el archivo .env.",
    403: "La API key no tiene permiso para usar este modelo.",
    404: "El modelo configurado no existe. Revisa GEMINI_MODEL.",
    429: "Se ha alcanzado el límite de uso gratuito. Espera un minuto.",
}


class LLMError(Exception):
    """Error del LLM tras agotar los reintentos y los modelos de respaldo."""


def friendly_message(err: errors.APIError) -> str:
    """Traduce un error de la API a un mensaje comprensible.

    Args:
        err: Excepción lanzada por el SDK de Gemini.

    Returns:
        Mensaje en español apto para mostrar al usuario.
    """
    # Gemini responde a una clave inválida con un 400, no con un 401.
    if "API key" in (err.message or ""):
        return FRIENDLY_ERRORS[401]
    default = "El servicio de Gemini no está disponible. Inténtalo más tarde."
    return FRIENDLY_ERRORS.get(err.code, default)


class GeminiClient:
    """Envoltorio del SDK de Gemini con reintentos y modelos de respaldo.

    Attributes:
        config: Parámetros de generación en uso.
        last_model: Modelo que respondió a la última petición.
    """

    def __init__(self, config: ModelConfig = MODEL, api_key: str | None = None):
        """Crea el cliente.

        Args:
            config: Parámetros del modelo. Por defecto, los de ``src.config``.
            api_key: Clave de la API. Si se omite, se lee del entorno.
        """
        self.config = config
        self.last_model: str | None = None
        self._client = genai.Client(
            api_key=api_key or get_api_key(),
            http_options=types.HttpOptions(
                timeout=config.timeout_s * 1000,
                retry_options=types.HttpRetryOptions(
                    attempts=config.retry_attempts,
                    initial_delay=1.0,
                    max_delay=8.0,
                ),
            ),
        )

    @property
    def sdk(self) -> genai.Client:
        """Cliente del SDK subyacente (lo usan, por ejemplo, los embeddings)."""
        return self._client

    @property
    def models(self) -> tuple[str, ...]:
        """Cadena de modelos a probar: el principal y luego los de respaldo."""
        return (self.config.model, *self.config.fallback_models)

    def build_config(
        self,
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentConfig:
        """Construye la configuración de generación de una petición.

        Args:
            system_instruction: Instrucciones de sistema (rol y reglas).
            tools: Herramientas que el modelo puede invocar.

        Returns:
            Configuración lista para pasar al SDK.
        """
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=self.config.thinking_level
            ),
            tools=tools,
            # Las herramientas las ejecuta nuestro propio bucle, no el SDK,
            # para poder validar argumentos, gestionar errores y registrarlas.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def generate(
        self,
        contents: str | list[types.Content],
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentResponse:
        """Genera una respuesta completa, con respaldo entre modelos.

        Args:
            contents: Pregunta suelta o historial de la conversación.
            system_instruction: Instrucciones de sistema.
            tools: Herramientas disponibles para el modelo.

        Returns:
            Respuesta del SDK (texto, llamadas a herramientas y uso de tokens).

        Raises:
            LLMError: Si ningún modelo puede responder.
        """
        config = self.build_config(system_instruction, tools)
        last_error: errors.APIError | None = None
        for model in self.models:
            start = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except errors.APIError as err:
                if err.code not in TRANSIENT_CODES:
                    raise LLMError(friendly_message(err)) from err
                logger.warning("Modelo %s no disponible (%s)", model, err.code)
                last_error = err
                continue
            self.last_model = model
            self._log_usage(model, response, time.perf_counter() - start)
            return response
        raise LLMError(friendly_message(last_error)) from last_error

    def generate_stream(
        self,
        contents: str | list[types.Content],
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
    ) -> Iterator[types.GenerateContentResponse]:
        """Genera una respuesta por fragmentos (streaming).

        Solo se cambia de modelo si el fallo ocurre antes del primer
        fragmento; un corte a mitad de respuesta se notifica como error.

        Args:
            contents: Pregunta suelta o historial de la conversación.
            system_instruction: Instrucciones de sistema.
            tools: Herramientas disponibles para el modelo.

        Yields:
            Fragmentos de la respuesta a medida que llegan.

        Raises:
            LLMError: Si ningún modelo puede responder o la respuesta se corta.
        """
        config = self.build_config(system_instruction, tools)
        last_error: errors.APIError | None = None
        for model in self.models:
            try:
                stream = self._client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                first = next(stream)
            except errors.APIError as err:
                if err.code not in TRANSIENT_CODES:
                    raise LLMError(friendly_message(err)) from err
                logger.warning("Modelo %s no disponible (%s)", model, err.code)
                last_error = err
                continue
            self.last_model = model
            yield first
            try:
                yield from stream
            except errors.APIError as err:
                raise LLMError(
                    "La respuesta se interrumpió. Repite la pregunta."
                ) from err
            return
        raise LLMError(friendly_message(last_error)) from last_error

    @staticmethod
    def _log_usage(
        model: str, response: types.GenerateContentResponse, seconds: float
    ) -> None:
        """Registra en el log el consumo de tokens y la latencia.

        Args:
            model: Modelo que respondió.
            response: Respuesta del SDK.
            seconds: Duración de la petición.
        """
        usage = response.usage_metadata
        logger.info(
            "model=%s latency=%.2fs prompt_tokens=%s output_tokens=%s "
            "thinking_tokens=%s",
            model,
            seconds,
            usage.prompt_token_count if usage else None,
            usage.candidates_token_count if usage else None,
            usage.thoughts_token_count if usage else None,
        )
