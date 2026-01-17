"""
DialogueBERT Intent Recognition Module

Provides DialogueBERT-based intent recognition for the museum dialogue agent.
Uses contextual encoders that interpret utterances with speaker role and dialogue history.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple
import logging
import os

from .dialoguebert_model import DialogueBERTModel

logger = logging.getLogger(__name__)


class DialogueBERTIntentRecognizer:
    """
    DialogueBERT based intent recognition for museum dialogue agent.
    
    Implements DialogueBERT architecture with:
    - Turn embeddings: Track turn position in dialogue (0-indexed)
    - Role embeddings: Distinguish user (0) vs system/agent (1)
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 device: Optional[str] = None,
                 mode: str = "dialoguebert"):
        """
        Initialize DialogueBERT intent recognizer.
        
        Args:
            model_name: HuggingFace model name for base BERT model
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            mode: "dialoguebert" or "standard_bert"
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode.lower()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.mode == "standard_bert":
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Standard BERT loaded on {self.device}")
            else:
                self.model = DialogueBERTModel(
                    base_model_name=model_name,
                    max_turns=50,
                    embedding_dim=768
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"DialogueBERT loaded on {self.device}")
                
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using fallback embeddings.")
            self.model = None
            self.tokenizer = None
    
    def get_intent_embedding(self, utterance: str, role: str = "user", 
                            turn_number: Optional[int] = None) -> np.ndarray:
        """
        Generate intent embedding for an utterance.
        
        Args:
            utterance: Utterance text
            role: Speaker role ("user" or "system")
            turn_number: Turn number in dialogue (0-indexed)
            
        Returns:
            Intent embedding vector of shape (768,)
        """
        if not utterance or not utterance.strip():
            return np.zeros(768, dtype=np.float32)
        
        if self.model is None:
            return self._fallback_intent_embedding(utterance)
        
        try:
            inputs = self.tokenizer(
                utterance,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            if self.mode == "standard_bert":
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    intent_embedding = outputs.last_hidden_state[:, 0, :]
                return intent_embedding.cpu().numpy().flatten()
            else:
                batch_size, seq_len = input_ids.shape
                
                if turn_number is not None:
                    turn_ids = torch.full((batch_size, seq_len), turn_number, dtype=torch.long, device=self.device)
                else:
                    turn_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
                
                role_id = 0 if role.lower() == "user" else 1
                role_ids = torch.full((batch_size, seq_len), role_id, dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    intent_embedding = self.model.get_pooled_output(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        turn_ids=turn_ids,
                        role_ids=role_ids
                    )
                
                return intent_embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Error generating intent embedding: {e}")
            return self._fallback_intent_embedding(utterance)
    
    def get_dialogue_context(self, recent_utterances: List[Tuple[str, str, int]], 
                           max_turns: int = 3) -> np.ndarray:
        """
        Generate dialogue context embedding.
        
        Args:
            recent_utterances: List of (role, text, turn_number) tuples
            max_turns: Maximum number of turns to include
            
        Returns:
            Dialogue context embedding vector of shape (768,)
        """
        if not recent_utterances:
            return np.zeros(768, dtype=np.float32)
        
        if self.model is None:
            texts = [u[1] if len(u) >= 2 else u[0] for u in recent_utterances]
            return self._fallback_context_embedding(texts)
        
        try:
            context_utterances = recent_utterances[-max_turns:]
            
            dialogue_parts = []
            turn_numbers = []
            role_ids_list = []
            
            for utterance_tuple in context_utterances:
                if len(utterance_tuple) == 3:
                    role, text, turn_num = utterance_tuple
                elif len(utterance_tuple) == 2:
                    role, text = utterance_tuple
                    turn_num = 0
                else:
                    continue
                
                dialogue_parts.append(text)
                turn_numbers.append(turn_num)
                role_id = 0 if role.lower() == "user" else 1
                role_ids_list.append(role_id)
            
            context_text = " [SEP] ".join(dialogue_parts)
            
            inputs = self.tokenizer(
                context_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            if self.mode == "standard_bert":
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    context_embedding = outputs.last_hidden_state[:, 0, :]
                return context_embedding.cpu().numpy().flatten()
            else:
                batch_size, seq_len = input_ids.shape
                avg_turn = int(sum(turn_numbers) / len(turn_numbers)) if turn_numbers else 0
                turn_ids = torch.full((batch_size, seq_len), avg_turn, dtype=torch.long, device=self.device)
                last_role_id = role_ids_list[-1] if role_ids_list else 0
                role_ids = torch.full((batch_size, seq_len), last_role_id, dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    context_embedding = self.model.get_pooled_output(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        turn_ids=turn_ids,
                        role_ids=role_ids
                    )
                
                return context_embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Error generating context embedding: {e}")
            texts = [u[1] if len(u) >= 2 else u[0] for u in recent_utterances]
            return self._fallback_context_embedding(texts)
    
    def classify_intent_category(self, utterance: str) -> str:
        """Classify utterance into intent categories"""
        if not utterance or not utterance.strip():
            return "silence"
        
        utterance_lower = utterance.lower()
        
        if any(word in utterance_lower for word in ["?", "what", "how", "why", "when", "where", "who", "which"]):
            return "question"
        if any(word in utterance_lower for word in ["don't understand", "confused", "unclear"]):
            return "confusion"
        if any(word in utterance_lower for word in ["interesting", "fascinating", "amazing", "wow", "beautiful"]):
            return "interest"
        if any(word in utterance_lower for word in ["tell me", "explain", "show me"]):
            return "request"
        
        return "statement"
    
    def _fallback_intent_embedding(self, utterance: str) -> np.ndarray:
        """Fallback intent embedding when model is unavailable"""
        intent_category = self.classify_intent_category(utterance)
        embedding = np.zeros(768, dtype=np.float32)
        
        if intent_category == "question":
            embedding[:100] = 0.1
        elif intent_category == "confusion":
            embedding[100:200] = 0.1
        elif intent_category == "interest":
            embedding[200:300] = 0.1
        elif intent_category == "request":
            embedding[300:400] = 0.1
        elif intent_category == "statement":
            embedding[400:500] = 0.1
        
        return embedding
    
    def _fallback_context_embedding(self, utterances: List[str]) -> np.ndarray:
        """Fallback context embedding when model is unavailable"""
        embedding = np.zeros(768, dtype=np.float32)
        
        if not utterances:
            return embedding
        
        num_utterances = len(utterances)
        embedding[500:600] = num_utterances / 10.0
        
        recent_intents = [self.classify_intent_category(u) for u in utterances[-3:]]
        
        if "question" in recent_intents:
            embedding[600:650] = 0.1
        if "confusion" in recent_intents:
            embedding[650:700] = 0.1
        if "interest" in recent_intents:
            embedding[700:750] = 0.1
        
        return embedding


# Global DialogueBERT recognizer instance
_dialoguebert_recognizer = None


def get_dialoguebert_recognizer() -> DialogueBERTIntentRecognizer:
    """Get global DialogueBERT recognizer instance (singleton)"""
    global _dialoguebert_recognizer
    if _dialoguebert_recognizer is None:
        bert_mode = os.environ.get('BERT_MODE', 'dialoguebert').lower()
        mode = 'standard_bert' if bert_mode == 'standard' else 'dialoguebert'
        _dialoguebert_recognizer = DialogueBERTIntentRecognizer(mode=mode)
    return _dialoguebert_recognizer


def reset_dialoguebert_recognizer():
    """Reset global DialogueBERT recognizer instance"""
    global _dialoguebert_recognizer
    _dialoguebert_recognizer = None

