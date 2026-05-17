#!/usr/bin/env python3
"""
LangChain summary demo.

Demonstrates using LangChain to orchestrate an LLM call for summarising
analytics outputs, as suggested in the dossier (Étape 5).

Requires: pip install langchain langchain-google-genai  (or langchain-openai)
          and GEMINI_API_KEY (or OPENAI_API_KEY) in .env.

Usage:
    python scripts/langchain_summary_demo.py
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print(
        "LangChain not installed.\n"
        "  pip install langchain langchain-core langchain-google-genai\n"
        "Main pipeline uses a lean, custom summarizer instead."
    )
    sys.exit(0)

TEMPLATE = """You are an eCommerce analyst. Based on the following structured analytics summary,
write a short executive summary (3–5 sentences) for a decision-maker.
Be precise and only use the facts provided. Do not invent numbers or categories.

Data:
{data}

Summary:"""


def get_llm():
    """Try Gemini first, then OpenAI."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key,
                temperature=0.3,
            )
        except ImportError:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_key,
                temperature=0.3,
            )
        except ImportError:
            pass

    return None


def main():
    llm = get_llm()
    if llm is None:
        print(
            "No LLM key found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env.\n"
            "Main pipeline uses src/llm/summarizer.py with google-genai."
        )
        return

    sample_data = {
        "top_categories": ["merchandise", "cheesonings", "combos"],
        "best_shop": "Dan-O's Seasoning",
        "best_shop_avg_score": 0.42,
        "n_top_products": 50,
        "cluster_summary": {"0": 92, "1": 45, "2": 50, "3": 33},
    }

    prompt = PromptTemplate(input_variables=["data"], template=TEMPLATE)
    # Modern LangChain API (RunnableSequence instead of deprecated LLMChain)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"data": json.dumps(sample_data, indent=2)})
    print("LangChain summary result:\n")
    print(result)


if __name__ == "__main__":
    main()
