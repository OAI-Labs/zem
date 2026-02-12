import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OpikCloudModel:
    """Simple provider for Opik model evaluation with auto API key loading"""
    
    # API key environment variables
    API_KEYS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }
    
    def __init__(self):
        self._cache = {}
    
    def get_model(
        self, 
        provider: str, 
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get model for Opik evaluation
        
        Args:
            provider: Provider name (openai, anthropic, google, etc.)
            model_id: Model ID (gpt-4o, claude-3-5-sonnet-latest, etc.)
            api_key: Optional API key (loads from env if not provided)
            
        Returns:
            Model string for Opik metrics
        """
        provider = provider.lower()
        
        # Load API key from environment if not provided
        if not api_key and provider in self.API_KEYS:
            api_key = os.getenv(self.API_KEYS[provider])
            
        # Validate API key for providers that need it
        if provider in ["openai", "anthropic", "google", "azure"] and not api_key:
            raise ValueError(f"API key required for {provider}. Set {self.API_KEYS[provider]} environment variable.")
        
        # Cache the configuration
        cache_key = f"{provider}_{model_id}"
        if cache_key not in self._cache:
            self._cache[cache_key] = {
                "provider": provider,
                "model_id": model_id,
                "api_key": api_key,
                **kwargs
            }
            logger.info(f"Configured model: {provider}/{model_id}")
        
        # Return model ID for Opik (Opik auto-detects provider)
        return model_id
    
    def get_config(self, provider: str, model_id: str) -> Dict[str, Any]:
        """Get full model configuration"""
        cache_key = f"{provider.lower()}_{model_id}"
        return self._cache.get(cache_key, {})
    
    def list_providers(self) -> list[str]:
        """List supported providers"""
        return list(self.API_KEYS.keys())

class EvaluateModelFactory:
    @staticmethod
    def get_model(engine: str, provider: str, model_id: str) -> str:
        engine = engine.lower()
        
        # 1. Cloud Engine
        if engine == "cloud":  
            if provider in OpikCloudModel().list_providers():
                provider = cloud_aliases.get(engine, engine)
                try:
                    return OpikCloudModel().get_model(provider, model_id)
                except Exception as e:
                    raise ErrorValue(f"Cannot load {model_id} from provider {provider}")
            else:
                raise ErrorValue(f"Unsupported provider {provider}")
        # 3. Future Engines
        raise ValueError(f"Unsupported evaluation engine: {engine}")

# Usage
if __name__ == "__main__":
    provider = OpikCloudModel()
    
    # Simple usage
    model = provider.get_model("gemini", "gemini-2.5-flash")
    print(f"Model: {model}")
    
    # Use with Opik metrics
    from opik.evaluation.metrics import Hallucination
    metric = Hallucination(model=model)
