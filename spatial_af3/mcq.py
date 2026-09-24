from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence, Tuple


def _option_keys(options: Dict[str, str] | Iterable[str]) -> list[str]:
    keys = options.keys() if isinstance(options, dict) else options
    normalized = [str(key).strip().upper() for key in keys if str(key).strip()]
    return sorted(dict.fromkeys(normalized))


def format_mcq_options(options: Dict[str, str]) -> str:
    """Format options in the parenthesized style used by Audio Flamingo MCQ prompts."""
    if not options:
        raise ValueError("MCQ options are required.")
    lines = []
    for key in _option_keys(options):
        if key not in options:
            raise ValueError(f"Missing option text for {key}.")
        lines.append(f"({key}) {str(options[key]).strip()}")
    return "\n".join(lines)


def format_mcq_target(correct_option: str, options: Dict[str, str], answer_text: str = "") -> str:
    key = str(correct_option or "").strip().upper()
    if not key:
        raise ValueError("MCQ correct_option is required.")
    if key not in options:
        raise ValueError(f"Correct option {key!r} is not present in options: {sorted(options)}")
    return f"<Final answer>: ({key}) {str(options[key]).strip()}"


def build_mcq_prompt_body(question: str, options: Dict[str, str]) -> str:
    question = str(question or "").strip()
    return (
        f"Question: {question}\n"
        "Choose the only correct option from the following options:\n"
        f"{format_mcq_options(options)}"
    )


def _valid_option_set(valid_options: Dict[str, str] | Sequence[str] | Iterable[str]) -> set[str]:
    valid = set(_option_keys(valid_options))
    if not valid:
        raise ValueError("At least one valid MCQ option is required for extraction.")
    return valid


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _valid_matches(matches: Iterable[str], valid: set[str]) -> list[str]:
    return [match.upper() for match in matches if match and match.upper() in valid]


def _last_unambiguous(candidates: list[str]) -> Optional[str]:
    if not candidates:
        return None
    if len(set(candidates)) == 1:
        return candidates[-1]
    return None


def extract_mcq_option(
    generated_text: str,
    valid_options: Dict[str, str] | Sequence[str] | Iterable[str],
) -> Tuple[Optional[str], str]:
    """Extract the selected MCQ option from AF3-style generated text."""
    valid = _valid_option_set(valid_options)
    text = _clean_text(generated_text)
    if not text:
        return None, "empty_generation"

    tag_matches = re.findall(r"<answer>\s*\(?([A-Za-z])\)?(?:\s|[).:,;]|</answer>)", text, flags=re.IGNORECASE)
    tag_candidates = _valid_matches(tag_matches, valid)
    if tag_candidates:
        return tag_candidates[-1], "answer_tag"

    answer_patterns = [
        r"(?:<\s*final\s+answer\s*>|final\s+answer|the\s+answer|answer|selected\s+option|option)\s*>?\s*(?:is|:|-)?\s*\(?([A-Za-z])\)?\b",
        r"therefore[^.?!]*?\banswer\s*>?\s*(?:is|:|-)?\s*\(?([A-Za-z])\)?\b",
    ]
    answer_candidates: list[str] = []
    for pattern in answer_patterns:
        answer_candidates.extend(_valid_matches(re.findall(pattern, text, flags=re.IGNORECASE), valid))
    if answer_candidates:
        return answer_candidates[-1], "regex_match"

    line_candidates = _valid_matches(
        re.findall(r"(?:^|[\n\r])\s*\(?([A-Za-z])\)?\s*(?:[).:]|$)", generated_text, flags=re.IGNORECASE),
        valid,
    )
    if line_candidates:
        return line_candidates[-1], "regex_match"

    generic_candidates = []
    generic_candidates.extend(_valid_matches(re.findall(r"\(([A-Za-z])\)", text, flags=re.IGNORECASE), valid))
    generic_candidates.extend(_valid_matches(re.findall(r"\b([A-Za-z])\s*[).]", text, flags=re.IGNORECASE), valid))
    generic_candidates.extend(_valid_matches(re.findall(r"\b([A-Za-z])\b", text, flags=re.IGNORECASE), valid))
    generic_choice = _last_unambiguous(generic_candidates)
    if generic_choice is not None:
        return generic_choice, "regex_match"

    return None, "no_option_found"
