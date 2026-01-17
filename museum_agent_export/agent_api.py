"""
Museum Dialogue Agent - Python API

Simple interface for using the trained dialogue agent.

Example:
    from agent_api import MuseumAgent
    
    agent = MuseumAgent()
    result = agent.respond("What is this painting?", exhibit="King_Caspar")
    print(result["response"])

LLM Provider Configuration:
    The agent supports multiple LLM providers. Set environment variables:
    
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
import re
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import FlatActorCriticAgent, ActorCriticAgent
from src.utils import SimpleKnowledgeGraph, get_dialoguebert_recognizer, build_prompt, get_llm_handler


def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and value and key not in os.environ:
                        os.environ[key] = value


# Load .env file on import
_load_env_file()


def _detect_model_type(checkpoint: dict) -> str:
    """Detect if model is flat (MDP) or hierarchical (SMDP)."""
    if 'network' in checkpoint:
        network_keys = checkpoint['network'].keys()
    elif 'agent_state_dict' in checkpoint:
        network_keys = checkpoint['agent_state_dict'].keys()
    else:
        return 'flat'
    
    for key in network_keys:
        if 'intra_option_policies' in key or 'termination_functions' in key:
            return 'hrl'
    return 'flat'


# Default configuration
OPTIONS = ["Explain", "AskQuestion", "OfferTransition", "Conclude"]
SUBACTIONS = {
    "Explain": ["ExplainNewFact", "RepeatFact", "ClarifyFact"],
    "AskQuestion": ["AskOpinion", "AskMemory", "AskClarification"],
    "OfferTransition": ["SummarizeAndSuggest"],
    "Conclude": ["WrapUp"]
}


class MuseumAgent:
    """
    Simple API for the trained museum dialogue agent.
    
    The agent selects dialogue actions (explain, ask questions, etc.) based on
    its learned policy, then generates natural language using an LLM.
    
    Inputs needed:
        - user_message: What the visitor said (string)
        - exhibit: Which painting they're looking at (string, e.g. "King_Caspar")
    
    Available exhibits:
        - King_Caspar
        - Turban
        - Dom_Miguel
        - Pedro_Sunda
        - Diego_Bemba
    """
    
    def __init__(self, model_path: str = "models/H2_MDP_Augmented.pt"):
        """
        Load the trained agent.
        
        Args:
            model_path: Path to model checkpoint (.pt file), relative to script directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.script_dir = Path(__file__).parent
        
        # Load knowledge graph (exhibit facts)
        kg_path = self.script_dir / "data" / "museum_knowledge_graph.json"
        self.kg = SimpleKnowledgeGraph(str(kg_path))
        self.exhibits = self.kg.get_exhibit_names()
        
        # Resolve model path relative to script directory
        if not Path(model_path).is_absolute():
            model_path = self.script_dir / model_path
        else:
            model_path = Path(model_path)
        
        # Load model
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model_type = _detect_model_type(checkpoint)
        
        state_dim = checkpoint.get('state_dim', 143)
        options = checkpoint.get('options', OPTIONS)
        subactions = checkpoint.get('subactions', SUBACTIONS)
        self.options = options
        self.subactions = subactions
        
        # Create agent (flat or hierarchical based on model)
        if self.model_type == 'hrl':
            self.agent = ActorCriticAgent(
                state_dim=state_dim, options=options, subactions=subactions,
                hidden_dim=256, lstm_hidden_dim=128, use_lstm=True, device=self.device
            )
        else:
            self.agent = FlatActorCriticAgent(
                state_dim=state_dim, options=options, subactions=subactions,
                hidden_dim=256, lstm_hidden_dim=128, use_lstm=True, device=self.device
            )
        
        # Load weights
        if 'agent_state_dict' in checkpoint:
            self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        else:
            self.agent.network.load_state_dict(checkpoint['network'])
        self.agent.network.eval()
        
        # Load DialogueBERT and LLM
        self.bert = get_dialoguebert_recognizer()
        self.llm = get_llm_handler()
        
        # Projection matrix for DialogueBERT (768d -> 64d)
        np.random.seed(42)
        self._proj = np.random.randn(64, 768).astype(np.float32) / np.sqrt(768)
        
        # Initialize conversation state
        self.reset()
    
    def reset(self):
        """Reset conversation to start fresh."""
        self.current_exhibit = self.exhibits[0]
        self.dialogue_history = []
        self.facts_mentioned = defaultdict(set)
        self.option_counts = defaultdict(int)
        self.turn = 0
        self.agent.reset()
    
    def respond(self, user_message: str, exhibit: Optional[str] = None) -> Dict[str, Any]:
        """
        Get agent response to user message.
        
        Args:
            user_message: What the visitor said
            exhibit: Which exhibit they're at (optional, keeps current if not provided)
        
        Returns:
            Dictionary with:
                - response: Agent's natural language response
                - action: What action was selected (e.g. "Explain/ExplainNewFact")
                - exhibit: Current exhibit name
                - facts_remaining: How many facts left to share
        
        Example:
            result = agent.respond("Tell me about this", exhibit="Turban")
            print(result["response"])
        """
        # Update exhibit if provided
        if exhibit and exhibit in self.exhibits:
            self.current_exhibit = exhibit
        
        exhibit_idx = self.exhibits.index(self.current_exhibit)
        n = len(self.exhibits)
        
        # === BUILD STATE VECTOR ===
        # 1. Focus: one-hot of current exhibit
        focus = np.zeros(n + 1, dtype=np.float32)
        focus[exhibit_idx] = 1.0
        
        # 2. History: fact coverage + action usage
        history = np.zeros(n + len(self.options), dtype=np.float32)
        for i, ex in enumerate(self.exhibits):
            total = len(self.kg.get_exhibit_facts(ex))
            mentioned = len(self.facts_mentioned[ex])
            history[i] = mentioned / total if total > 0 else 0.0
        total_actions = sum(self.option_counts.values()) or 1
        for i, opt in enumerate(self.options):
            history[n + i] = self.option_counts[opt] / total_actions
        
        # 3. Intent: encode user message with DialogueBERT
        intent_768 = self.bert.get_intent_embedding(user_message, role="user", turn_number=self.turn)
        intent_64 = (self._proj @ intent_768).astype(np.float32)
        
        # 4. Context: encode recent dialogue
        context_768 = self.bert.get_dialogue_context(self.dialogue_history, max_turns=3)
        context_64 = (self._proj @ context_768).astype(np.float32)
        
        # Combine into state
        state = np.concatenate([focus, history, intent_64, context_64])
        
        # === GET AVAILABLE ACTIONS ===
        available_options = list(self.options)
        available_subs = {opt: list(subs) for opt, subs in self.subactions.items()}
        
        # Mask ExplainNewFact if no facts left
        all_facts = self.kg.get_exhibit_facts(self.current_exhibit)
        mentioned_ids = self.facts_mentioned[self.current_exhibit]
        has_new = any(self.kg.extract_fact_id(f) not in mentioned_ids for f in all_facts)
        if not has_new and "ExplainNewFact" in available_subs.get("Explain", []):
            available_subs["Explain"].remove("ExplainNewFact")
        if not available_subs.get("Explain"):
            available_options.remove("Explain") if "Explain" in available_options else None
        
        # === SELECT ACTION ===
        action = self.agent.select_action(
            state=state,
            available_options=available_options,
            available_subactions_dict=available_subs,
            deterministic=True
        )
        option, subaction = action['option_name'], action['subaction_name']
        self.option_counts[option] += 1
        
        # === GENERATE RESPONSE ===
        used_facts = [f for f in all_facts if self.kg.extract_fact_id(f) in mentioned_ids]
        
        target_exhibit = None
        if option == "OfferTransition":
            # Pick least-discussed exhibit
            best, best_remaining = None, -1
            for ex in self.exhibits:
                if ex == self.current_exhibit:
                    continue
                remaining = len(self.kg.get_exhibit_facts(ex)) - len(self.facts_mentioned[ex])
                if remaining > best_remaining:
                    best, best_remaining = ex, remaining
            target_exhibit = best
        
        coverage = {ex: {"total": len(self.kg.get_exhibit_facts(ex)),
                        "mentioned": len(self.facts_mentioned[ex]),
                        "coverage": len(self.facts_mentioned[ex]) / len(self.kg.get_exhibit_facts(ex)) 
                                   if self.kg.get_exhibit_facts(ex) else 0}
                   for ex in self.exhibits}
        
        prompt = build_prompt(
            option=option, subaction=subaction, ex_id=self.current_exhibit,
            last_utt=user_message, facts_all=all_facts, facts_used=used_facts,
            selected_fact=None, dialogue_history=self.dialogue_history,
            exhibit_names=self.exhibits, knowledge_graph=self.kg,
            target_exhibit=target_exhibit, coverage_dict=coverage
        )
        
        system_prompt = f"""You are a natural, conversational museum guide. Your role is {option}/{subaction}.

IMPORTANT GUIDELINES:
- Respond naturally and conversationally - be concise and engaging
- DO NOT quote or repeat what the visitor said verbatim (avoid phrases like "You said...", "I see you...")
- Use conversation history to maintain natural flow and continuity
- Reference past topics naturally when relevant, but don't quote them
- Be warm, informative, and engaging"""
        
        response = self.llm.generate(prompt, system_prompt=system_prompt)
        
        # Track mentioned facts
        for fid in re.findall(r'\[([A-Z]{2}_\d{3})\]', response):
            for ex in self.exhibits:
                prefix = ''.join(c for c in ex if c.isupper())[:2] or ex[:2].upper()
                if fid.startswith(prefix):
                    self.facts_mentioned[ex].add(fid)
                    break
        
        # Update dialogue history
        self.dialogue_history.append(("user", user_message, self.turn))
        self.turn += 1
        self.dialogue_history.append(("agent", response, self.turn))
        self.turn += 1
        
        # Handle transition
        if option == "OfferTransition" and target_exhibit:
            self.current_exhibit = target_exhibit
        
        return {
            "response": response,
            "action": f"{option}/{subaction}",
            "exhibit": self.current_exhibit,
            "facts_remaining": len(all_facts) - len(mentioned_ids)
        }
    
    def get_exhibits(self) -> List[str]:
        """Get list of available exhibit names."""
        return self.exhibits.copy()


# Simple test when run directly
if __name__ == "__main__":
    print("Testing MuseumAgent...")
    agent = MuseumAgent()
    
    print(f"\nExhibits: {agent.get_exhibits()}")
    
    result = agent.respond("Hello! What can you tell me about this painting?", exhibit="King_Caspar")
    print(f"\nAction: {result['action']}")
    print(f"Response: {result['response']}")
