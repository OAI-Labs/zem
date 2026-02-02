from xfmr_zem.server import ZemServer
from typing import Any, List, Optional
import os
import json
import requests

mcp = ZemServer("LLM-Curation")

def _call_llm(prompt: str, provider: str = "ollama", model: Optional[str] = None) -> str:
    """Helper to route LLM calls."""
    if provider == "ollama":
        model = model or "llama3"
        try:
            url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
            response = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
            return response.json().get("response", "")
        except Exception:
            return f"[Ollama Error] Could not connect to {model}. Fallback: processed {prompt[:20]}..."
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: return "[OpenAI Error] Missing API Key."
        # Placeholder for real openai client call
        return f"[OpenAI Mock] Classification/Response for: {prompt[:20]}..."
    return f"[Mock] {prompt}"

@mcp.tool()
def mask_pii(data: Any, provider: str = "ollama") -> List[Any]:
    """Smart PII masking using LLM."""
    dataset = mcp.get_data(data)
    for item in dataset:
        if "text" in item:
            prompt = f"Remove all PII from this text and return ONLY the cleaned text: {item['text']}"
            item["text"] = _call_llm(prompt, provider=provider)
    return dataset

@mcp.tool()
def classify_domain(data: Any, categories: List[str] = ["Tech", "Finance", "Legal"], provider: str = "ollama") -> List[Any]:
    """Classify data domain using LLM."""
    dataset = mcp.get_data(data)
    for item in dataset:
        if "text" in item:
            prompt = f"Classify this text into one of {categories}. Return ONLY the category name: {item['text']}"
            item["domain"] = _call_llm(prompt, provider=provider).strip()
    return dataset

if __name__ == "__main__":
    mcp.run()
