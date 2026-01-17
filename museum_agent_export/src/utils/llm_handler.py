"""
LLM Handler for Museum Dialogue Generation

Supports multiple LLM providers:
- Groq (default, free tier available)
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)

Configuration via environment variables:
- LLM_PROVIDER: "groq", "openai", or "anthropic" (default: "groq")
- LLM_MODEL: Model name (provider-specific)
- GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY: API keys
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMCriticalError(Exception):
    """Raised when LLM encounters a critical, unrecoverable error."""
    pass


class BaseLLMHandler(ABC):
    """Base class for LLM handlers."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 250,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from prompt."""
        pass


class GroqLLMHandler(BaseLLMHandler):
    """Handler for Groq API (Llama models)."""
    
    def __init__(self, model_name: str = "llama-3.1-8b", **kwargs):
        super().__init__(model_name, **kwargs)
        self._initialize()
    
    def _initialize(self):
        """Initialize Groq client."""
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            raise LLMCriticalError(
                "GROQ_API_KEY environment variable not set.\n"
                "Get a free API key at: https://console.groq.com/\n"
                "Then set it: export GROQ_API_KEY=your_key_here"
            )
        
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            print(f"[OK] Groq API initialized with model: {self.model_name}")
        except ImportError:
            raise LLMCriticalError(
                "groq package not installed. Run: pip install groq"
            )
        except Exception as e:
            raise LLMCriticalError(f"Error initializing Groq: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Groq API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Map model names to Groq model IDs
        model_map = {
            "llama-3.3": "llama-3.3-70b-versatile",
            "llama-3.1": "llama-3.1-70b-versatile",
            "llama-3.1-8b": "llama-3.1-8b-instant",
        }
        groq_model = model_map.get(self.model_name.lower(), self.model_name)
        
        try:
            response = self.client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            critical_keywords = [
                'spend_limit_reached', 'insufficient_quota',
                'authentication_error', 'invalid_api_key',
            ]
            if any(kw in error_str for kw in critical_keywords):
                raise LLMCriticalError(f"Groq API error: {e}")
            print(f"[WARN] Groq API error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."


class OpenAILLMHandler(BaseLLMHandler):
    """Handler for OpenAI API (GPT models)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise LLMCriticalError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Get an API key at: https://platform.openai.com/api-keys\n"
                "Then set it: export OPENAI_API_KEY=your_key_here"
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"[OK] OpenAI API initialized with model: {self.model_name}")
        except ImportError:
            raise LLMCriticalError(
                "openai package not installed. Run: pip install openai"
            )
        except Exception as e:
            raise LLMCriticalError(f"Error initializing OpenAI: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            critical_keywords = [
                'insufficient_quota', 'authentication_error',
                'invalid_api_key', 'rate_limit',
            ]
            if any(kw in error_str for kw in critical_keywords):
                raise LLMCriticalError(f"OpenAI API error: {e}")
            print(f"[WARN] OpenAI API error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."


class AnthropicLLMHandler(BaseLLMHandler):
    """Handler for Anthropic API (Claude models)."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307", **kwargs):
        super().__init__(model_name, **kwargs)
        self._initialize()
    
    def _initialize(self):
        """Initialize Anthropic client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise LLMCriticalError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Get an API key at: https://console.anthropic.com/\n"
                "Then set it: export ANTHROPIC_API_KEY=your_key_here"
            )
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            print(f"[OK] Anthropic API initialized with model: {self.model_name}")
        except ImportError:
            raise LLMCriticalError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        except Exception as e:
            raise LLMCriticalError(f"Error initializing Anthropic: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic API."""
        # Anthropic uses different message format
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt if system_prompt else "",
                messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            critical_keywords = [
                'authentication_error', 'invalid_api_key',
                'rate_limit', 'insufficient_quota',
            ]
            if any(kw in error_str for kw in critical_keywords):
                raise LLMCriticalError(f"Anthropic API error: {e}")
            print(f"[WARN] Anthropic API error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."


# Backward compatibility alias
FreeLLMHandler = GroqLLMHandler


# Global LLM handler instance
_llm_handler: Optional[BaseLLMHandler] = None


def get_llm_handler(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLMHandler:
    """
    Get global LLM handler instance (singleton).
    
    Args:
        provider: LLM provider ("groq", "openai", "anthropic")
                 If None, reads from LLM_PROVIDER env var (default: "groq")
        model_name: Model name (provider-specific)
                   If None, reads from LLM_MODEL env var
        **kwargs: Additional arguments for handler (temperature, max_tokens, etc.)
        
    Returns:
        BaseLLMHandler instance
        
    Examples:
        # Use Groq (default)
        handler = get_llm_handler()
        
        # Use OpenAI
        handler = get_llm_handler(provider="openai", model_name="gpt-4")
        
        # Use Anthropic
        handler = get_llm_handler(provider="anthropic", model_name="claude-3-sonnet-20240229")
    """
    global _llm_handler
    
    # Determine provider
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    
    # Determine model name
    if model_name is None:
        model_name = os.environ.get("LLM_MODEL")
        if model_name is None:
            # Default models per provider
            defaults = {
                "groq": "llama-3.1-8b",
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku-20240307",
            }
            model_name = defaults.get(provider, "llama-3.1-8b")
    
    # Create handler if needed
    if _llm_handler is None or _llm_handler.model_name != model_name:
        if provider == "groq":
            _llm_handler = GroqLLMHandler(model_name=model_name, **kwargs)
        elif provider == "openai":
            _llm_handler = OpenAILLMHandler(model_name=model_name, **kwargs)
        elif provider == "anthropic":
            _llm_handler = AnthropicLLMHandler(model_name=model_name, **kwargs)
        else:
            raise LLMCriticalError(
                f"Unknown LLM provider: {provider}. "
                f"Supported: groq, openai, anthropic"
            )
    
    return _llm_handler


def reset_llm_handler():
    """Reset global LLM handler instance."""
    global _llm_handler
    _llm_handler = None
