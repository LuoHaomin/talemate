"""Unit tests for the headless-disconnect rescue (game.general.headless_disconnect).

Covers the pure logic of `_finish_scene_after_disconnect`: waiting for
in-flight agent work, the force-stop timeout, save-once semantics and the
memory db close, plus the config flag default.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import talemate.server.api as api_module
from talemate.config.schema import Config


class FakeScene:
    def __init__(self, saved=False, filename="/tmp/scene.json", name=""):
        self.saved = saved
        self.filename = filename
        self.name = name
        self.loading = False
        self.active = True
        self.continue_scene = True
        self.save = AsyncMock()


@pytest.fixture
def agent_registry(monkeypatch):
    """Replace the global agent registry with a controllable fake."""
    agents = {}

    def _add(name, processing=0, processing_bg=0):
        agents[name] = SimpleNamespace(
            processing=processing, processing_bg=processing_bg
        )

    monkeypatch.setattr(api_module.instance, "AGENTS", agents)
    return _add


@pytest.fixture
def memory_agent(monkeypatch):
    agent = Mock()
    monkeypatch.setattr(api_module.instance, "get_agent", lambda _name: agent)
    return agent


class TestFinishSceneAfterDisconnect:
    async def test_saves_and_deactivates_when_idle(
        self, agent_registry, memory_agent
    ):
        agent_registry("narrator")  # exists but idle
        scene = FakeScene(saved=False)

        await api_module._finish_scene_after_disconnect(scene, timeout=5, poll=0.01)

        assert scene.active is False
        assert scene.continue_scene is False
        scene.save.assert_awaited_once_with(force=True)
        memory_agent.close_db.assert_called_once_with(scene)

    async def test_waits_for_busy_agents_then_saves(
        self, agent_registry, memory_agent
    ):
        agent_registry("conversation", processing=1)
        scene = FakeScene(saved=False)

        async def free_agents_later():
            await asyncio.sleep(0.1)
            agent_registry("conversation", processing=0)

        task = asyncio.create_task(free_agents_later())
        await api_module._finish_scene_after_disconnect(scene, timeout=5, poll=0.02)
        await task

        scene.save.assert_awaited_once()
        memory_agent.close_db.assert_called_once_with(scene)

    async def test_timeout_force_stops_and_still_saves(
        self, agent_registry, memory_agent
    ):
        # agent stays busy forever -> the rescue must time out, force-stop
        # the scene and still save
        agent_registry("conversation", processing=1)
        scene = FakeScene(saved=False)

        await api_module._finish_scene_after_disconnect(scene, timeout=0.2, poll=0.05)

        assert scene.active is False
        scene.save.assert_awaited_once_with(force=True)
        memory_agent.close_db.assert_called_once()

    async def test_no_save_when_already_saved(self, agent_registry, memory_agent):
        agent_registry("narrator")
        scene = FakeScene(saved=True)

        await api_module._finish_scene_after_disconnect(scene, timeout=5, poll=0.01)

        scene.save.assert_not_awaited()
        memory_agent.close_db.assert_called_once()

    async def test_save_failure_still_closes_memory(
        self, agent_registry, memory_agent
    ):
        agent_registry("narrator")
        scene = FakeScene(saved=False)
        scene.save.side_effect = RuntimeError("disk full")

        await api_module._finish_scene_after_disconnect(scene, timeout=5, poll=0.01)

        memory_agent.close_db.assert_called_once()


class TestShouldHeadlessRescue:
    def test_unnamed_placeholder_scene_does_not_qualify(self):
        # the handler's scene before anything is loaded: nothing to rescue,
        # and Scene.save would refuse it anyway
        assert api_module._should_headless_rescue(FakeScene()) is False
        assert api_module._should_headless_rescue(None) is False

    def test_named_loaded_scene_qualifies(self):
        scene = FakeScene()
        scene.name = "My Scene"
        assert api_module._should_headless_rescue(scene) is True

    def test_loading_scene_does_not_qualify(self):
        scene = FakeScene()
        scene.name = "My Scene"
        scene.loading = True
        assert api_module._should_headless_rescue(scene) is False


class TestConfigFlag:
    def test_default_off(self):
        config = Config.model_validate({})
        assert config.game.general.headless_disconnect is False
        assert config.game.general.auto_save is True

    def test_can_be_enabled(self):
        config = Config.model_validate(
            {"game": {"general": {"headless_disconnect": True}}}
        )
        assert config.game.general.headless_disconnect is True
