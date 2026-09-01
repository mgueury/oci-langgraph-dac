import json
import os
import sys
import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"

# langchain_oci warning
warnings.filterwarnings(
    "ignore",
    message=r"GenericProvider could not extract text and returned an empty string.*",
    category=UserWarning,
    module=r"langchain_oci\.chat_models\.oci_generative_ai",
)

# langgraph warning in the Cloud Shell
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change.*",
    module=r"langgraph\.cache\.base",
)


# -- format_trace_message ----------------------------------------------------
def supports_color() -> bool:
    """Return whether the current terminal can display ANSI colors."""
    return not os.getenv("NO_COLOR") and sys.stdout.isatty()


# -- colorize ----------------------------------------------------------------
def colorize(text: str, color: str, enabled: bool | None = None) -> str:
    """Apply an ANSI color only when terminal output supports it."""
    use_color = supports_color() if enabled is None else enabled
    if use_color:
        return f"{color}{text}{RESET}"
    return text


# -- is_error_result ---------------------------------------------------------
def is_error_result(content: str) -> bool:
    """Identify the human-readable failure messages returned by example tools."""
    return any(phrase in content.lower() for phrase in (
        "could not", "failed", "unavailable", "not configured", "not found",
    ))


# -- remove_none_values ------------------------------------------------------
def remove_none_values(value: Any) -> Any:
    """Recursively remove ``None`` values from usage metadata dictionaries."""
    if isinstance(value, dict):
        return {key: remove_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [remove_none_values(item) for item in value if item is not None]
    return value


# -- sanitize_langfuse_response ----------------------------------------------
def sanitize_langfuse_response(response: Any) -> Any:
    """Copy an LLM result and remove OCI's null token-detail values for Langfuse."""
    sanitized = deepcopy(response)
    if isinstance(getattr(sanitized, "llm_output", None), dict):
        for key in ("token_usage", "usage"):
            if isinstance(sanitized.llm_output.get(key), dict):
                sanitized.llm_output[key] = remove_none_values(sanitized.llm_output[key])
    for generation_group in getattr(sanitized, "generations", []):
        for generation in generation_group:
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                message.usage_metadata = remove_none_values(usage_metadata)
            generation_info = getattr(generation, "generation_info", None)
            if isinstance(generation_info, dict) and isinstance(generation_info.get("usage_metadata"), dict):
                generation_info["usage_metadata"] = remove_none_values(generation_info["usage_metadata"])
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                for key in ("usage", "amazon-bedrock-invocationMetrics"):
                    if isinstance(response_metadata.get(key), dict):
                        response_metadata[key] = remove_none_values(response_metadata[key])
    return sanitized


# -- get_langfuse_callback ---------------------------------------------------
def get_langfuse_callback() -> Any | None:
    """Create a Langfuse callback when complete Langfuse settings are configured."""
    settings = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    if not all(os.getenv(setting) for setting in settings):
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as error:
        raise RuntimeError("Langfuse is configured but not installed. Run: pip install -r requirements.txt") from error
    class OCICompatibleCallbackHandler(CallbackHandler):
        """Avoid Langfuse usage parsing errors from nullable OCI token details."""

        def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
            return super().on_llm_end(sanitize_langfuse_response(response), **kwargs)

    return OCICompatibleCallbackHandler()


# -- format_trace_message ----------------------------------------------------
def format_trace_message(
    message: Any,
    indent: int = 0,
    color: bool | None = None,
) -> list[str]:
    """Return readable trace lines for a model tool call or a tool response."""
    prefix = " " * indent
    if isinstance(message, AIMessage) and message.tool_calls:
        return [colorize(
            f"{prefix}\N{RIGHTWARDS ARROW} {call['name']}({json.dumps(call['args'], default=str, ensure_ascii=False)})",
            YELLOW,
            color,
        )
            for call in message.tool_calls
        ]

    if isinstance(message, ToolMessage):
        name = message.name or "tool"
        content = str(message.content)
        lines = content.splitlines() or [""]
        response_color = RED if is_error_result(content) else GREEN
        response_lines = [
            f"{prefix}\N{LEFTWARDS ARROW} {name}: {lines[0]}",
            *[f"{prefix}  {line}" for line in lines[1:]],
        ]
        return [colorize(line, response_color, color) for line in response_lines]

    return []


# -- trace_agent -------------------------------------------------------------
def trace_agent(
    agent: Any,
    payload: dict[str, Any],
    *,
    label: str | None = None,
    indent: int = 0,
) -> dict[str, Any]:
    """Stream an agent, print tool activity, and return its accumulated result state."""
    prefix = " " * indent
    if label:
        print(colorize(f"{prefix}[{label}]", CYAN))

    result = dict(payload)
    result["messages"] = list(payload.get("messages", []))
    config: dict[str, Any] = {"run_name": label or "agent"}
    if callback := get_langfuse_callback():
        config["callbacks"] = [callback]
    for update in agent.stream(payload, config=config, stream_mode="updates"):
        for state_update in update.values():
            if not isinstance(state_update, dict):
                continue
            for key, value in state_update.items():
                if key == "messages":
                    messages: Iterable[Any] = value if isinstance(value, list) else [value]
                    for message in messages:
                        for line in format_trace_message(message, indent):
                            print(line)
                        result["messages"].append(message)
                else:
                    result[key] = value

    return result
