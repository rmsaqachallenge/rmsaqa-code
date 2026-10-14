"""Shared two-stage conversation construction for MiDashengLM."""
from __future__ import annotations

from typing import Any, Mapping

from .layered_mcq import build_mcq_prompt_body


def normalize_history_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def build_followup_user_prompt(
    question: str, options: Mapping[str, str]
) -> str:
    return (
        "Based on the same audio clip and the previous answer, answer the "
        "follow-up question.\n"
        + build_mcq_prompt_body(question=question, options=dict(options))
    )


def build_layer2_messages(
    backbone: Any,
    *,
    layer1_question: str,
    layer1_options: Mapping[str, str],
    layer1_answer: object,
    layer2_question: str,
    layer2_options: Mapping[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": backbone.build_user_prompt(
                layer1_question, dict(layer1_options)
            ),
        },
        {"role": "assistant", "content": normalize_history_text(layer1_answer)},
        {
            "role": "user",
            "content": build_followup_user_prompt(
                layer2_question, layer2_options
            ),
        },
    ]
