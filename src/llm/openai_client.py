"""
OpenAI-compatible LLM adapter.

Provides the same interface as the Gemini summarizer but targets
any OpenAI-compatible endpoint (OpenAI, Azure, local vLLM, etc.).
Dossier: OpenAI API as alternative LLM backend.

Usage:
    Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in .env.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


def generate_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "(OpenAI adapter: no OPENAI_API_KEY set.)"

    try:
        from openai import OpenAI

        base_url = os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        return "(openai package not installed. pip install openai)"
    except Exception as exc:
        return f"(OpenAI error: {exc})"


if __name__ == "__main__":
    from src.llm.prompts import EXECUTIVE_SUMMARY_PROMPT

    sample = {"top_categories": ["merchandise", "combos"], "best_shop": "Dan-O's"}
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(data=json.dumps(sample, indent=2))
    print(generate_openai(prompt))
