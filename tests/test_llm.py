"""Tests for LLM module (prompts and summarizer structure)."""

from src.llm.prompts import (
    CATEGORY_TRENDS_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    PRODUCT_COMPARISON_PROMPT,
)


def test_executive_prompt_has_data_placeholder():
    assert "{data}" in EXECUTIVE_SUMMARY_PROMPT


def test_category_prompt_has_data_placeholder():
    assert "{data}" in CATEGORY_TRENDS_PROMPT


def test_chain_of_thought_prompt_has_steps():
    """Chain of thought prompt should include step-by-step instructions."""
    assert "step by step" in CHAIN_OF_THOUGHT_PROMPT.lower()
    assert "{data}" in CHAIN_OF_THOUGHT_PROMPT


def test_product_comparison_prompt():
    assert "{data}" in PRODUCT_COMPARISON_PROMPT


def test_prompt_formatting():
    """Prompts should be formattable with a data string."""
    import json

    sample = json.dumps({"test": "data"})
    result = EXECUTIVE_SUMMARY_PROMPT.format(data=sample)
    assert "test" in result
    assert "{data}" not in result


def test_chain_of_thought_formatting():
    import json

    sample = json.dumps({"categories": ["A", "B"]})
    result = CHAIN_OF_THOUGHT_PROMPT.format(data=sample)
    assert "categories" in result
