"""Unit tests for the pass-through thinking controls on the OpenAI
Compatible client: the `reasoning_effort` / `thinking_mode` request
parameters and the capture of the separate `reasoning_content` field
for the reasoning display.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import talemate.config.state as config_state
from talemate.client import openai_compat as openai_compat_module
from talemate.client.openai_compat import OpenAICompatibleClient


@pytest.fixture
def cfg_isolation():
    """Snapshot/restore the client config section these tests mutate."""
    saved_clients = dict(config_state.CONFIG.clients)
    yield
    config_state.CONFIG.clients.clear()
    config_state.CONFIG.clients.update(saved_clients)


def _register(name: str, **kwargs):
    cfg = openai_compat_module.ClientConfig(
        type="openai_compat",
        name=name,
        api_url="http://fake:5000",
        api_handles_prompt_template=True,
        **kwargs,
    )
    config_state.CONFIG.clients[name] = cfg
    return cfg


class _FakeCompletions:
    def __init__(self, message):
        self._message = message
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self._message)],
        )


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, base_url=None, api_key=None):
        self.chat = SimpleNamespace(completions=self._completions)
        _FakeAsyncOpenAI.instances.append(self)

    @classmethod
    def set_response(cls, message):
        cls._completions = _FakeCompletions(message)


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeAsyncOpenAI.instances = []
    monkeypatch.setattr(openai_compat_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    return _FakeAsyncOpenAI


class TestThinkingControls:
    async def test_reasoning_effort_sent_when_set(self, cfg_isolation, fake_openai):
        _register("oc_effort", reasoning_effort="low")
        fake_openai.set_response(SimpleNamespace(content="hi", reasoning_content=None))
        client = OpenAICompatibleClient(name="oc_effort")
        await client.generate("prompt", {"max_tokens": 64}, "conversation")
        kwargs = fake_openai.instances[0].chat.completions.calls[0]
        assert kwargs["reasoning_effort"] == "low"
        assert "thinking" not in kwargs

    async def test_thinking_mode_sent_as_type_object(
        self, cfg_isolation, fake_openai
    ):
        _register("oc_thinking", thinking_mode="disabled")
        fake_openai.set_response(SimpleNamespace(content="hi", reasoning_content=None))
        client = OpenAICompatibleClient(name="oc_thinking")
        await client.generate("prompt", {}, "conversation")
        kwargs = fake_openai.instances[0].chat.completions.calls[0]
        # provider-specific parameter travels through the SDK's extra_body
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in kwargs

    async def test_thinking_controls_omitted_when_unset(
        self, cfg_isolation, fake_openai
    ):
        _register("oc_plain")
        fake_openai.set_response(SimpleNamespace(content="hi", reasoning_content=None))
        client = OpenAICompatibleClient(name="oc_plain")
        await client.generate("prompt", {}, "conversation")
        kwargs = fake_openai.instances[0].chat.completions.calls[0]
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs


class TestReasoningContentCapture:
    async def test_reasoning_content_stored_for_display(
        self, cfg_isolation, fake_openai
    ):
        _register("oc_reasoned")
        fake_openai.set_response(
            SimpleNamespace(content="answer", reasoning_content="thinking it over")
        )
        client = OpenAICompatibleClient(name="oc_reasoned")
        response = await client.generate("prompt", {}, "conversation")
        assert response == "answer"
        assert client._reasoning_response == "thinking it over"

    async def test_no_reasoning_content_leaves_none(
        self, cfg_isolation, fake_openai
    ):
        _register("oc_plain2")
        # message objects from APIs without thinking carry no attribute at all
        fake_openai.set_response(SimpleNamespace(content="answer"))
        client = OpenAICompatibleClient(name="oc_plain2")
        response = await client.generate("prompt", {}, "conversation")
        assert response == "answer"
        assert client._reasoning_response is None


class TestDefaults:
    def test_reasoning_validation_pattern_defaults_to_think_tag(self):
        # With reasoning enabled, responses without inline <think> tags must
        # pass strip_reasoning cleanly (thinking arrived in the separate
        # reasoning_content field) instead of raising ReasoningResponseError.
        assert openai_compat_module.Defaults().reason_validation_pattern == "<think>"
