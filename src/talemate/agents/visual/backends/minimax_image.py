"""
MiniMax native image generation backend.

Talks directly to ``https://api.minimaxi.com/v1/image_generation``,
replacing the local OpenAI-images-to-MiniMax translation shim
(no OPENAI_BASE_URL hijack, no sidecar process).

The API key is read from the shared image API key slot
(``config.openai.api_key``) -- the same slot the OpenAI backend uses,
so a MiniMax key stored there powers this backend.

Limitations:
- text-to-image only (no editing, no analysis)
- prompts are clamped to 1400 characters at a sentence boundary,
  because the API rejects prompts longer than 1500 characters
"""

import httpx
import pydantic
import structlog

from talemate.agents.base import (
    AgentAction,
    AgentActionConfig,
    AgentActionConditional,
)
from talemate.agents.visual.schema import (
    FORMAT_TYPE,
    GEN_TYPE,
    GenerationRequest,
    GenerationResponse,
    PROMPT_TYPE,
    Resolution,
)
from talemate.config import get_config
from talemate.instance import get_agent
import talemate.agents.visual.backends as backends


log = structlog.get_logger("talemate.agents.visual.minimax")

BACKEND_NAME = "minimax"

API_URL = "https://api.minimaxi.com/v1/image_generation"

# The API hard limit is 1500 characters; clamp earlier so a sentence
# boundary can still be found.
MAX_PROMPT_CHARS = 1400

SENTENCE_SEPARATORS = ("。", "！", "？", ".", "!", "?", "\n", "，", ",")

ALLOWED_ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 1.5,
    "2:3": 2 / 3,
    "21:9": 21 / 9,
}


def clamp_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """
    Clamp a prompt to ``max_chars`` characters, cutting at the last
    sentence separator when one can be found reasonably late in the
    text (MiniMax rejects prompts over 1500 characters).
    """

    if len(prompt) <= max_chars:
        return prompt

    cut = prompt[:max_chars]
    for separator in SENTENCE_SEPARATORS:
        index = cut.rfind(separator)
        if index > max_chars * 0.6:
            return cut[: index + 1]
    return cut


def aspect_ratio_for(resolution: Resolution) -> str:
    """
    Snap a resolution to the closest aspect ratio MiniMax accepts.
    """

    if not resolution.width or not resolution.height:
        return "1:1"

    target = resolution.width / resolution.height
    return min(
        ALLOWED_ASPECT_RATIOS.items(), key=lambda item: abs(item[1] - target)
    )[0]


def resolution_for_format(fmt: FORMAT_TYPE) -> Resolution:
    if fmt == FORMAT_TYPE.LANDSCAPE:
        return Resolution(width=1600, height=900)
    if fmt == FORMAT_TYPE.PORTRAIT:
        return Resolution(width=900, height=1600)
    return Resolution(width=1024, height=1024)


def minimax_api_key() -> str:
    return get_config().openai.api_key or ""


@backends.register
class Backend(backends.Backend):
    name = BACKEND_NAME
    label = "MiniMax"
    image_create = True
    image_edit = False
    image_analyzation = False
    description = (
        "MiniMax image generation (image-01) via the native API. "
        "Text to image only. Prompts over 1400 characters are clamped "
        "at a sentence boundary."
    )

    model: str = "image-01"
    prompt_type: PROMPT_TYPE = PROMPT_TYPE.DESCRIPTIVE

    @property
    def instance_label(self) -> str:
        return self.model

    @property
    def generator_label(self) -> str | None:
        return self.model

    @property
    def generate_timeout(self) -> int:
        return get_agent("visual").generate_timeout

    async def ready(self) -> backends.BackendStatus:
        try:
            await self.ensure_test_connection_task()
        except Exception as e:
            self.status = backends.BackendStatus(
                type=backends.BackendStatusType.ERROR, message=str(e)
            )
        return self.status

    async def test_connection(self, timeout: int = 5) -> backends.BackendStatus:
        api_key = minimax_api_key()
        if not api_key:
            return backends.BackendStatus(
                type=backends.BackendStatusType.ERROR,
                message="MiniMax API Key not set (stored in the shared image API key)",
            )
        try:
            async with httpx.AsyncClient() as http:
                # The endpoint only accepts POST; any structured 4xx
                # response still proves reachability and auth handling.
                response = await http.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": self.model, "prompt": ""},
                    timeout=timeout,
                )
            if response.status_code in (401, 403):
                return backends.BackendStatus(
                    type=backends.BackendStatusType.ERROR,
                    message=f"MiniMax rejected the API key ({response.status_code})",
                )
            return backends.BackendStatus(type=backends.BackendStatusType.OK)
        except Exception as e:
            log.error("minimax.test_connection.error", error=str(e))
            return backends.BackendStatus(
                type=backends.BackendStatusType.ERROR, message=str(e)
            )

    async def generate(
        self, request: GenerationRequest, response: GenerationResponse
    ) -> GenerationResponse:
        if request.gen_type != GEN_TYPE.TEXT_TO_IMAGE:
            raise ValueError("MiniMax backend only supports text to image")

        api_key = minimax_api_key()
        if not api_key:
            raise ValueError("MiniMax API Key not set")

        prompt = clamp_prompt(request.prompt)
        aspect_ratio = aspect_ratio_for(
            request.resolution or Resolution(width=1024, height=1024)
        )

        log.debug(
            "minimax.generate",
            model=self.model,
            aspect_ratio=aspect_ratio,
            prompt_chars=len(request.prompt),
            clamped=len(request.prompt) != len(prompt),
        )

        async with httpx.AsyncClient() as http:
            result = await http.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "response_format": "url",
                },
                timeout=self.generate_timeout,
            )
            result.raise_for_status()
            payload = result.json()

        urls = (payload.get("data") or {}).get("image_urls") or []
        if not urls:
            raise ValueError(
                f"MiniMax image response missing data.image_urls: {payload}"
            )

        async with httpx.AsyncClient() as http:
            download = await http.get(urls[0], timeout=self.generate_timeout)
            download.raise_for_status()
            response.generated = download.content
        return response


class MinimaxHandler(pydantic.BaseModel):
    backend: Backend
    action: AgentAction


class MinimaxMixin:
    @classmethod
    def minimax_shared_config(cls) -> dict[str, AgentActionConfig]:
        return {
            "api_key": AgentActionConfig(
                type="unified_api_key",
                value="openai.api_key",
                label="MiniMax API Key",
                scene_overridable=False,
            ),
            "model": AgentActionConfig(
                type="text",
                value="image-01",
                choices=[
                    {"label": "image-01", "value": "image-01"},
                ],
                label="Model",
                description="MiniMax image model",
                save_on_change=True,
                scene_overridable=False,
            ),
        }

    @classmethod
    def add_actions(cls, actions: dict[str, AgentAction]):
        actions["minimax_image_create"] = AgentAction(
            enabled=True,
            container=True,
            icon="mdi-image",
            label="MiniMax",
            subtitle="Text to image",
            description="Native MiniMax text to image generation.",
            condition=AgentActionConditional(
                attribute="_config.config.backend", value=BACKEND_NAME
            ),
            config=cls.minimax_shared_config(),
        )
        return actions

    async def minimax_backend(
        self, old_config: dict | None = None, force: bool = False
    ) -> Backend:
        model = self.actions["minimax_image_create"].config["model"].value
        backend_exists = isinstance(self.backend, Backend)

        try:
            model_changed = old_config["minimax_image_create"].config["model"].value != model
        except Exception:
            model_changed = False

        if force or not backend_exists:
            log.debug("reinitializing minimax backend")
            return Backend(model=model, gen_type=GEN_TYPE.TEXT_TO_IMAGE)

        self.backend.model = model
        return self.backend

    async def minimax_prepare_generation(self, request: GenerationRequest) -> Backend:
        request.resolution = resolution_for_format(request.format)
        return self.backend
