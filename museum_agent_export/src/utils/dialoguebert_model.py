"""
DialogueBERT Model Implementation

DialogueBERT extends BERT with turn and role embeddings for dialogue understanding.
Based on: "DialogueBERT: A Self-Supervised Learning based Dialogue Pre-training Encoder"
by Zhang et al., 2021.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class DialogueBERTModel(nn.Module):
    """
    DialogueBERT model with turn and role embeddings.
    
    Extends BERT with:
    - Turn embeddings: Track position in dialogue
    - Role embeddings: Distinguish user vs system
    """
    
    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        max_turns: int = 50,
        embedding_dim: int = 768
    ):
        super().__init__()
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(base_model_name)
        
        # Turn embedding (0 to max_turns-1)
        self.turn_embedding = nn.Embedding(max_turns, embedding_dim)
        
        # Role embedding (0 = user, 1 = system/agent)
        self.role_embedding = nn.Embedding(2, embedding_dim)
        
        # Layer norm for combined embeddings
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        
        # Pooler for [CLS] token
        self.pooler = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
        
        self._init_additional_embeddings()
    
    def _init_additional_embeddings(self):
        """Initialize turn and role embeddings"""
        nn.init.normal_(self.turn_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        turn_ids: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass with turn and role embeddings.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            turn_ids: Turn position IDs (batch, seq_len)
            role_ids: Role IDs - 0=user, 1=agent (batch, seq_len)
            return_dict: Return as dict
        """
        # Get BERT embeddings
        bert_embeddings = self.bert.embeddings(input_ids)
        
        # Add turn and role embeddings if provided
        if turn_ids is not None:
            bert_embeddings = bert_embeddings + self.turn_embedding(turn_ids)
        
        if role_ids is not None:
            bert_embeddings = bert_embeddings + self.role_embedding(role_ids)
        
        # Apply layer norm
        bert_embeddings = self.embedding_layer_norm(bert_embeddings)
        
        # Pass through BERT encoder
        encoder_outputs = self.bert.encoder(
            bert_embeddings,
            attention_mask=self.bert.get_extended_attention_mask(
                attention_mask, input_ids.shape, input_ids.device
            ) if attention_mask is not None else None
        )
        
        sequence_output = encoder_outputs.last_hidden_state
        
        if return_dict:
            return {"last_hidden_state": sequence_output}
        return sequence_output
    
    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        turn_ids: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get pooled [CLS] token output.
        
        Returns:
            Pooled output (batch, embedding_dim)
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            turn_ids=turn_ids,
            role_ids=role_ids,
            return_dict=True
        )
        
        # Get [CLS] token (first token)
        cls_output = outputs["last_hidden_state"][:, 0, :]
        
        # Apply pooler
        pooled_output = self.pooler(cls_output)
        
        return pooled_output

