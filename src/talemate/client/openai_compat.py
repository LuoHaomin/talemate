import random

from openai import AsyncOpenAI

from talemate.client.api_handles import (
    ApiHandlesPromptTemplateConfig,
    ApiHandlesPromptTemplateMixin,
    api_handles_prompt_template_extra_fields,
)
from talemate.client.base import ClientBase, ExtraField, FieldGroup
from talemate.client.registry import register
from talemate.client.toggleable_parameters import (
    ToggleableParametersMixin,
    toggleable_parameters_config,
    toggleable_parameters_extra_fields,
)
from talemate.config.schema import Client as BaseClientConfig

EXPERIMENTAL_DESCRIPTION = """Use this client if you want to connect to a service implementing an OpenAI-compatible API. Success is going to depend on the level of compatibility. Use the actual OpenAI client if you want to connect to OpenAI's API."""

# Sampler parameters whose inclusion in the request payload is user-toggleable.
# Some OpenAI-compatible APIs hard-error when these are sent for certain models,
# so they need to be omitted entirely (not just zeroed out).
TOGGLEABLE_PARAMETERS = ("temperature", "top_p", "presence_penalty")

ToggleableParametersConfig = toggleable_parameters_config(TOGGLEABLE_PARAMETERS)

REASONING_FIELD_GROUP = FieldGroup(
    name="reasoning",
    label="Reasoning",
    description="Thinking controls are only sent when explicitly set here.",
    icon="mdi-brain",
)


class Defaults(ToggleableParametersConfig, ApiHandlesPromptTemplateConfig):
    api_url: str = "http://localhost:5000"
    api_key: str = ""
    max_token_length: int = 8192
    model: str = ""
    double_coercion: str = None
    rate_limit: int | None = None
    reasoning_effort: str = ""
    thinking_mode: str = ""
    # When reasoning is enabled, only treat the response as inline reasoning
    # if it actually carries <think> tags; APIs that return thinking in a
    # separate reasoning_content field then pass through cleanly instead of
    # tripping the missing-reasoning retry loop.
    reason_validation_pattern: str | None = "<think>"


class ClientConfig(
    ToggleableParametersConfig, ApiHandlesPromptTemplateConfig, BaseClientConfig
):
    reasoning_effort: str = ""
    thinking_mode: str = ""
    # See Defaults.reasoning_validation_pattern — placed here as well so
    # existing configs get the safe default on validation, not just new
    # clients created through the UI.
    reason_validation_pattern: str | None = "<think>"


@register()
class OpenAICompatibleClient(
    ApiHandlesPromptTemplateMixin, ToggleableParametersMixin, ClientBase
):
    client_type = "openai_compat"
    conversation_retries = 0
    config_cls = ClientConfig
    toggleable_parameters = TOGGLEABLE_PARAMETERS

    class Meta(ClientBase.Meta):
        title: str = "OpenAI Compatible API"
        name_prefix: str = "OpenAI Compatible API"
        experimental: str = EXPERIMENTAL_DESCRIPTION
        enable_api_auth: bool = True
        manual_model: bool = True
        defaults: Defaults = Defaults()
        self_hosted: bool | None = None
        extra_fields: dict[str, ExtraField] = {
            **api_handles_prompt_template_extra_fields(
                description="The API handles the prompt template, meaning your choice in the UI for the prompt template below will be ignored. This is not recommended and should only be used if the API does not support the `completions` andpoint or you don't know which prompt template to use.",
            ),
            **toggleable_parameters_extra_fields(TOGGLEABLE_PARAMETERS),
            "reasoning_effort": ExtraField(
                name="reasoning_effort",
                type="text",
                label="Reasoning Effort",
                choices=["", "low", "medium", "high"],
                description="Sent as the reasoning_effort parameter for APIs that support it (e.g. MiniMax M3.1, OpenAI o-series). Controls thinking depth; lower is faster and cheaper. Leave empty to not send it.",
                group=REASONING_FIELD_GROUP,
                required=False,
            ),
            "thinking_mode": ExtraField(
                name="thinking_mode",
                type="text",
                label="Thinking Mode",
                choices=["", "enabled", "adaptive", "disabled"],
                description="Sent as the thinking object (\"type\") for APIs that support it (e.g. MiniMax M3). 'disabled' turns thinking off where the model allows it; M3.1-Flash forces thinking on and M2.x cannot disable it. Leave empty to not send it.",
                group=REASONING_FIELD_GROUP,
                required=False,
            ),
        }

    @property
    def experimental(self):
        return EXPERIMENTAL_DESCRIPTION

    @property
    def can_be_coerced(self):
        """
        Determines whether or not his client can pass LLM coercion. (e.g., is able
        to predefine partial LLM output in the prompt)
        """
        return not self.reason_enabled

    @property
    def reasoning_effort(self) -> str:
        return self.client_config.reasoning_effort or ""

    @property
    def thinking_mode(self) -> str:
        return self.client_config.thinking_mode or ""

    async def get_model_name(self):
        return self.model

    async def generate(self, prompt: str, parameters: dict, kind: str):
        """
        Generates text from the given prompt and parameters.
        """

        client = AsyncOpenAI(base_url=self.api_url, api_key=self.api_key)

        if self.api_handles_prompt_template:
            # OpenAI API handles prompt template
            # Use the chat completions endpoint
            self.log.debug(
                "generate (chat/completions)",
                prompt=prompt[:128] + " ...",
                parameters=parameters,
            )

            messages, coercion_prompt = self.chat_messages_for_coercion(prompt, kind)

            if coercion_prompt:
                # continue the pre-fill via the (non-standard) prefix flag
                messages[-1]["prefix"] = True

            request_kwargs = dict(parameters)

            # Pass-through thinking controls. Only sent when explicitly
            # configured — arbitrary OpenAI-compatible backends may reject
            # unknown parameters. reasoning_effort is a standard SDK
            # parameter; provider-specific ones like MiniMax's `thinking`
            # object must travel through extra_body.
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
            extra_body = {}
            if self.thinking_mode:
                extra_body["thinking"] = {"type": self.thinking_mode}
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                **request_kwargs,
            )
            message = response.choices[0].message

            # Some APIs (MiniMax M3/M3.1, DeepSeek) return the thinking in a
            # separate reasoning_content field instead of inline <think> tags;
            # surface it for the reasoning display in the prompt log.
            reasoning = getattr(message, "reasoning_content", None)
            self._reasoning_response = reasoning if reasoning else None

            return message.content
        else:
            # Talemate handles prompt template
            # Use the completions endpoint
            self.log.debug(
                "generate (completions)",
                prompt=prompt[:128] + " ...",
                parameters=parameters,
            )
            parameters["prompt"] = prompt
            response = await client.completions.create(
                model=self.model_name, stream=False, **parameters
            )
            return response.choices[0].text

    def jiggle_randomness(self, prompt_config: dict, offset: float = 0.3) -> dict:
        """
        adjusts temperature and presence penalty
        by random values using the base value as a center
        """

        temp = prompt_config["temperature"]

        min_offset = offset * 0.3

        prompt_config["temperature"] = random.uniform(temp + min_offset, temp + offset)

        try:
            presence_penalty = prompt_config["presence_penalty"]
            prompt_config["presence_penalty"] = round(
                random.uniform(presence_penalty + 0.1, presence_penalty + offset), 1
            )
        except KeyError:
            pass
