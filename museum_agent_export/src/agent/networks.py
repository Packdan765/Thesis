"""
Neural Networks for Museum Dialogue Agent

Includes:
- FlatPolicyNetwork: Simple flat action space (MDP models)
- ActorCriticNetwork: Option-Critic architecture (SMDP/HRL models)
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class FlatPolicyNetwork(nn.Module):
    """
    Simple Actor-Critic network producing a single joint policy over the flat
    action space and a state value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm

        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=state_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            encoder_dim = lstm_hidden_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            encoder_dim = hidden_dim

        # Shared torso for policy/value heads
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.hidden_state = None

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: shape (batch, state_dim) or (batch, seq_len, state_dim)
        """
        if self.use_lstm:
            if state.dim() == 2:
                state = state.unsqueeze(1)

            if self.hidden_state is not None:
                encoded, self.hidden_state = self.encoder(state, self.hidden_state)
            else:
                encoded, self.hidden_state = self.encoder(state)

            encoded = encoded[:, -1, :]
        else:
            encoded = self.encoder(state)

        logits = self.policy_head(encoded)
        values = self.value_head(encoded).squeeze(-1)

        return {
            "action_logits": logits,
            "state_value": values,
        }

    def reset_hidden_state(self):
        self.hidden_state = None


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for Option-Critic Architecture (SMDP/HRL models).
    
    Actor learns:
    - π_Ω(ω|s): Option policy
    - π_o(a|s): Intra-option policies  
    - β_o(s): Termination functions
    
    Critic learns:
    - Q_Ω(s,ω): Option-value function
    - Q_U(s,ω,a): Action-value function
    - V(s): State value function
    """
    
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        num_subactions: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_options = num_options
        self.num_subactions = num_subactions
        self.use_lstm = use_lstm
        
        # Shared encoder
        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=state_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True
            )
            encoder_dim = lstm_hidden_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            encoder_dim = hidden_dim
        
        # Option policy π_Ω(ω|s)
        self.option_policy = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Intra-option policies π_o(a|s) - one per option
        self.intra_option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subactions)
            )
            for _ in range(num_options)
        ])
        
        # Termination functions β_o(s) - one per option
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_options)
        ])
        
        # Option-value Q_Ω(s,ω)
        self.option_value = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Action-value Q_U(s,ω,a) - one per option
        self.action_value = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subactions)
            )
            for _ in range(num_options)
        ])
        
        # State value V(s)
        self.state_value = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_state = None
        self._initialize_value_functions()
    
    def _initialize_value_functions(self):
        """Initialize value function heads to predict small values near zero."""
        for module in self.state_value.modules():
            if isinstance(module, nn.Linear) and module.out_features == 1:
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
        
        for module in self.option_value.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == self.num_options:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
        
        for action_value_net in self.action_value:
            for module in action_value_net.modules():
                if isinstance(module, nn.Linear):
                    if module.out_features == self.num_subactions:
                        nn.init.orthogonal_(module.weight, gain=0.01)
                        nn.init.constant_(module.bias, 0.0)
                    else:
                        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                        nn.init.constant_(module.bias, 0.0)
        
    def forward(self, state: torch.Tensor, reset_hidden: bool = False):
        """Forward pass through Actor-Critic network."""
        if reset_hidden or self.hidden_state is None:
            self.hidden_state = None
        
        if self.use_lstm:
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            
            if self.hidden_state is not None:
                encoded, self.hidden_state = self.encoder(state, self.hidden_state)
            else:
                encoded, self.hidden_state = self.encoder(state)
            
            encoded = encoded[:, -1, :]
        else:
            encoded = self.encoder(state)
        
        # Actor outputs
        option_logits = self.option_policy(encoded)
        intra_option_logits = [policy(encoded) for policy in self.intra_option_policies]
        termination_probs = torch.cat([term(encoded) for term in self.termination_functions], dim=-1)
        
        # Critic outputs
        option_values = self.option_value(encoded)
        action_values = [value(encoded) for value in self.action_value]
        state_value = self.state_value(encoded).squeeze(-1)
        
        return {
            'encoded': encoded,
            'option_logits': option_logits,
            'intra_option_logits': intra_option_logits,
            'termination_probs': termination_probs,
            'option_values': option_values,
            'action_values': action_values,
            'state_value': state_value
        }
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state for new episode."""
        self.hidden_state = None


__all__ = ["FlatPolicyNetwork", "ActorCriticNetwork"]

