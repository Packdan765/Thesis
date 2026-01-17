#!/usr/bin/env python3
"""
Museum Dialogue Agent - Interactive CLI

Chat with the trained museum guide agent.

Usage:
    python run_agent.py                    # Use default model
    python run_agent.py --model <path>     # Use specific model
    python run_agent.py --list-models      # List available models

Interactive Commands:
    <message>        - Chat with the agent
    exhibits         - List all exhibits and fact coverage
    exhibit N        - Switch to exhibit number N (1-5)
    exhibit <name>   - Switch to exhibit by name
    reset            - Reset conversation
    quit / exit      - Exit program

LLM Provider Configuration:
    Set environment variables for your chosen LLM provider:
    
    Groq (default):
        export GROQ_API_KEY="your-key"
    
    OpenAI:
        export LLM_PROVIDER="openai"
        export OPENAI_API_KEY="your-key"
    
    Anthropic:
        export LLM_PROVIDER="anthropic"
        export ANTHROPIC_API_KEY="your-key"
    
    See README.md for details.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if it exists
def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
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

_load_env_file()

from agent_api import MuseumAgent


def list_models():
    """List available model files."""
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        print("No models/ directory found")
        return
    
    models = sorted(models_dir.glob("*.pt"))
    if models:
        print("Available models:")
        for m in models:
            print(f"  - {m.name}")
    else:
        print("No .pt files found in models/")


def main():
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="Museum Dialogue Agent CLI")
    parser.add_argument('--model', type=str, default="models/H2_MDP_Augmented.pt",
                       help='Path to model checkpoint (relative to script directory)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    # Resolve model path relative to script directory
    if not Path(args.model).is_absolute():
        model_path = script_dir / args.model
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Model not found: {args.model}")
        print(f"Resolved to: {model_path}")
        list_models()
        return
    
    # Load agent
    print("\n" + "="*50)
    print("MUSEUM DIALOGUE AGENT")
    print("="*50)
    print(f"Loading model: {args.model}")
    
    # Pass relative path to MuseumAgent (both scripts are in same directory, so relative paths work)
    # If absolute path was provided, pass it as-is
    if Path(args.model).is_absolute():
        agent = MuseumAgent(model_path=str(model_path))
    else:
        agent = MuseumAgent(model_path=args.model)
    
    print(f"Model type: {agent.model_type.upper()}")
    print(f"Exhibits: {agent.get_exhibits()}")
    print("\nCommands: exhibits, exhibit N, reset, quit")
    print("="*50)
    
    # Select starting exhibit
    print("\nAvailable exhibits:")
    for i, ex in enumerate(agent.exhibits, 1):
        facts = len(agent.kg.get_exhibit_facts(ex))
        print(f"  {i}. {ex} ({facts} facts)")
    
    choice = input("\nSelect exhibit (1-5) or Enter for default: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(agent.exhibits):
        agent.current_exhibit = agent.exhibits[int(choice) - 1]
    
    print(f"\nStarting at: {agent.current_exhibit}")
    print("Type your message to chat!\n")
    
    # Chat loop
    while True:
        try:
            user_input = input(f"[{agent.current_exhibit}] You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset()
                print("Conversation reset.\n")
                continue
            
            if user_input.lower() == 'exhibits':
                print("\nExhibits:")
                for i, ex in enumerate(agent.exhibits, 1):
                    total = len(agent.kg.get_exhibit_facts(ex))
                    mentioned = len(agent.facts_mentioned[ex])
                    current = " <--" if ex == agent.current_exhibit else ""
                    print(f"  {i}. {ex}: {mentioned}/{total} facts{current}")
                print()
                continue
            
            if user_input.lower().startswith('exhibit '):
                ref = user_input[8:].strip()
                try:
                    idx = int(ref) - 1
                    if 0 <= idx < len(agent.exhibits):
                        agent.current_exhibit = agent.exhibits[idx]
                        print(f"Switched to: {agent.current_exhibit}\n")
                    else:
                        print(f"Invalid exhibit number. Use 1-{len(agent.exhibits)}\n")
                except ValueError:
                    # Try name match
                    for ex in agent.exhibits:
                        if ref.lower() in ex.lower():
                            agent.current_exhibit = ex
                            print(f"Switched to: {agent.current_exhibit}\n")
                            break
                    else:
                        print(f"Unknown exhibit: {ref}\n")
                continue
            
            # Get response
            result = agent.respond(user_input)
            
            print(f"\n[{result['action']}]")
            print(f"Agent: {result['response']}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
