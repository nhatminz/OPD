from __future__ import annotations

from typing import Any


MATH_USER_INSTRUCTION = (
    r"Let's think step by step and output the final answer within \boxed{}."
)


def _problem_text(value: Any) -> str:
    """Extract the raw math problem from a string or a chat-message field."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    if isinstance(value, (list, tuple)):
        for message in reversed(value):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return message["content"]
        if len(value) == 1 and isinstance(value[0], str):
            return value[0]
    raise TypeError(
        "Math prompt must be a string or contain a user message with string content; "
        f"got {type(value).__name__}"
    )


def build_math_user_prompt(problem: Any) -> str:
    """Return the one idempotent verl/EOPD-style math user prompt."""
    text = _problem_text(problem).strip()
    if not text:
        raise ValueError("Math problem text must not be empty")
    # Collapse any repeated canonical suffix before appending exactly one. This
    # makes the builder safe for both raw and already-preformatted datasets.
    while text.endswith(MATH_USER_INSTRUCTION):
        text = text[: -len(MATH_USER_INSTRUCTION)].rstrip()
    if not text:
        raise ValueError("Math problem contains only the output instruction")
    return f"{text} {MATH_USER_INSTRUCTION}"


def math_messages(problem: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_math_user_prompt(problem)}]


def render_math_prompt(tokenizer, problem: Any, data_config: dict[str, Any]) -> str:
    """Apply the configured chat template to the shared canonical prompt."""
    return tokenizer.apply_chat_template(
        math_messages(problem),
        tokenize=False,
        add_generation_prompt=True,
        **dict(data_config.get("chat_template_kwargs", {})),
    )
