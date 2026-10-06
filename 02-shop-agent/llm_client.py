import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

# Every provider speaks the OpenAI API; only these three things differ
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}


def get_client() -> tuple[OpenAI, str]:
    """Return (client, model) for the provider set in LLM_PROVIDER."""
    name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {list(PROVIDERS)}")

    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise ValueError(f"{cfg['api_key_env']} is not set in .env")

    # Allow overriding the model without touching code
    model = os.getenv("LLM_MODEL", cfg["model"])
    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), model


def get_langchain_model(temperature: float = 0.3):
    """Same provider config, wrapped as a LangChain chat model."""
    from langchain_openai import ChatOpenAI

    name = os.getenv("LLM_PROVIDER", "gemini").lower()
    cfg = PROVIDERS[name]
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", cfg["model"]),
        api_key=os.getenv(cfg["api_key_env"]),
        base_url=cfg["base_url"],
        temperature=temperature,
        max_retries=5,
        timeout=60,
    )