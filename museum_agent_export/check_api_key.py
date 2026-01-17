#!/usr/bin/env python3
"""
Quick script to check if your API key is configured correctly.
"""

import os
import sys
from pathlib import Path

# Load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"[INFO] Found .env file: {env_file}")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
                    print(f"[LOADED] {key} from .env file")
else:
    print(f"[INFO] No .env file found at {env_file}")

print("\n" + "="*60)
print("API Key Configuration Check")
print("="*60)

# Check each provider
providers = {
    "Groq": ("GROQ_API_KEY", None),
    "OpenAI": ("OPENAI_API_KEY", "LLM_PROVIDER"),
    "Anthropic": ("ANTHROPIC_API_KEY", "LLM_PROVIDER"),
}

configured = False

for provider_name, (api_key_var, provider_var) in providers.items():
    api_key = os.environ.get(api_key_var)
    provider_setting = os.environ.get(provider_var) if provider_var else None
    
    if api_key:
        if provider_var and provider_setting:
            if provider_setting.lower() == provider_name.lower():
                print(f"\n[OK] {provider_name} is configured!")
                print(f"  {api_key_var}: {api_key[:20]}...")
                print(f"  {provider_var}: {provider_setting}")
                configured = True
                break
        elif not provider_var:  # Groq doesn't need LLM_PROVIDER
            print(f"\n[OK] {provider_name} is configured!")
            print(f"  {api_key_var}: {api_key[:20]}...")
            configured = True
            break
    else:
        print(f"\n[FAIL] {provider_name} not configured")
        print(f"  Missing: {api_key_var}")

if not configured:
    print("\n" + "="*60)
    print("[ERROR] NO API KEY FOUND!")
    print("="*60)
    print("\nTo fix this:")
    print("\n1. EASIEST: Create a .env file")
    print("   - Copy env.example to .env")
    print("   - Edit .env and add: GROQ_API_KEY=your-key-here")
    print("\n2. OR set environment variable:")
    print("   PowerShell: $env:GROQ_API_KEY=\"your-key-here\"")
    print("   CMD:        set GROQ_API_KEY=your-key-here")
    print("   Linux/Mac:  export GROQ_API_KEY=\"your-key-here\"")
    sys.exit(1)
else:
    print("\n" + "="*60)
    print("[SUCCESS] API key is configured correctly!")
    print("="*60)
    sys.exit(0)

