"""Tests del cliente del LLM ante fallos, sin llamar a la API."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from src.config import ModelConfig
from src.llm import GeminiClient, LLMError, friendly_message, is_transient


def api_error(code, message="fallo"):
    return errors.APIError(code, {"error": {"code": code, "message": message}})


@pytest.fixture()
def client():
    config = ModelConfig(model="principal", fallback_models=("respaldo",))
    return GeminiClient(config=config, api_key="clave-falsa")


def install(client, outcomes, method="generate_content"):
    """Sustituye la llamada del SDK por respuestas programadas por modelo."""
    calls = []

    def fake(model, contents, config):
        calls.append(model)
        outcome = outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._client = SimpleNamespace(models=SimpleNamespace(**{method: fake}))
    return calls


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (api_error(503), True),
        (api_error(429), True),
        (api_error(400), False),
        (httpx.ReadTimeout("lento"), True),
        (httpx.ConnectError("sin red"), True),
    ],
)
def test_transient_errors(err, expected):
    assert is_transient(err) is expected


def test_friendly_messages():
    assert "API key" in friendly_message(api_error(400, "API key not valid"))
    assert "a tiempo" in friendly_message(httpx.ReadTimeout("lento"))


def test_network_timeout_falls_back_to_next_model(client):
    ok = SimpleNamespace(usage_metadata=None, text="hola")
    calls = install(client, {"principal": httpx.ReadTimeout("lento"), "respaldo": ok})
    assert client.generate("hola") is ok
    assert calls == ["principal", "respaldo"]
    assert client.last_model == "respaldo"


def test_network_timeout_on_every_model_raises_llm_error(client):
    timeout = httpx.ReadTimeout("lento")
    install(client, {"principal": timeout, "respaldo": timeout})
    with pytest.raises(LLMError, match="a tiempo"):
        client.generate("hola")


def test_non_transient_error_does_not_try_other_models(client):
    calls = install(client, {"principal": api_error(400), "respaldo": None})
    with pytest.raises(LLMError):
        client.generate("hola")
    assert calls == ["principal"]


def test_stream_timeout_before_first_chunk_falls_back(client):
    def stream(*chunks):
        yield from chunks

    def broken():
        raise httpx.ReadTimeout("lento")
        yield  # pragma: no cover

    calls = install(
        client,
        {"principal": None, "respaldo": None},
        method="generate_content_stream",
    )
    outcomes = {"principal": broken(), "respaldo": stream("a", "b")}
    client._client.models.generate_content_stream = lambda model, contents, config: (
        calls.append(model) or outcomes[model]
    )
    assert list(client.generate_stream("hola")) == ["a", "b"]
    assert calls == ["principal", "respaldo"]


def test_stream_cut_midway_raises_llm_error(client):
    def cut():
        yield "a"
        raise httpx.ReadTimeout("lento")

    client._client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **kwargs: cut())
    )
    with pytest.raises(LLMError, match="interrumpió"):
        list(client.generate_stream("hola"))
