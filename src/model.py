"""PyTorch neural network architectures and quantitative loss functions.

Includes:
- LSTMForecaster: Multi-layer recurrent model accepting (B, L, F) sequential tensors
- DifferentialSharpeLoss: Custom differentiable objective maximizing portfolio Sharpe Ratio
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


# ==============================================================================
# 1. Custom Quantitative Loss Functions
# ==============================================================================

class DifferentialSharpeLoss(nn.Module):
    """Differentiable Sharpe Ratio loss module for direct portfolio optimization.

    Given raw model output logits z_t, positions are mapped to continuous exposure:
        s_t = tanh(z_t) in [-1, 1]

    Strategy returns across batch size B with forward asset returns r_{t->t+T}:
        R^{strat}_t = s_t * r_{t->t+T}

    Sample statistics:
        mu = (1 / B) * sum(R^{strat}_t)
        sigma = sqrt((1 / (B - 1)) * sum((R^{strat}_t - mu)^2) + eps)

    Objective to minimize:
        Loss = - mu / sigma
    """

    def __init__(self, eps: float = 1e-6, turnover_penalty: float = 0.0):
        """
        Args:
            eps: Numerical stability constant to prevent division by zero variance.
            turnover_penalty: Optional regularizer penalizing large position swings.
        """
        super().__init__()
        self.eps = eps
        self.turnover_penalty = turnover_penalty

    def forward(
        self,
        logits: torch.Tensor,
        forward_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the negative batch Sharpe Ratio.

        Args:
            logits: Unconstrained model output tensor of shape (batch_size, 1).
            forward_returns: Forward realized return tensor of shape (batch_size, 1).

        Returns:
            Scalar loss tensor.
        """
        # Map logits to continuous position allocation [-1, 1]
        positions = torch.tanh(logits).view(-1)
        rets = forward_returns.view(-1)

        # Batch strategy returns
        strat_returns = positions * rets

        batch_size = strat_returns.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Sample mean and unbiased sample standard deviation
        mean_return = torch.mean(strat_returns)
        var_return = torch.var(strat_returns, unbiased=True)
        std_return = torch.sqrt(var_return + self.eps)

        sharpe = mean_return / std_return
        loss = -sharpe

        # Optional penalty on quadratic position changes across the batch
        if self.turnover_penalty > 0.0 and batch_size > 1:
            turnover = torch.mean((positions[1:] - positions[:-1]) ** 2)
            loss = loss + self.turnover_penalty * turnover

        return loss


# ==============================================================================
# 2. LSTM Forecaster Architecture
# ==============================================================================

class LSTMForecaster(nn.Module):
    """Long Short-Term Memory (LSTM) network for sequential financial forecasting.

    Accepts 3D tensors: (batch_size, sequence_length, num_features)
    Outputs:
    - Raw logits for binary cross-entropy (BCEWithLogitsLoss) or Sharpe loss
    - Probability estimates via sigmoid
    - Continuous position exposure via tanh
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        fc_dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Args:
            num_features: Number of input features F per time step.
            hidden_size: Number of hidden units in each LSTM layer.
            num_layers: Depth of stacked LSTM layers.
            lstm_dropout: Inter-layer dropout probability for LSTM (if num_layers > 1).
            fc_dropout: Dropout probability applied to the dense head.
            bidirectional: Whether to use bidirectional LSTM (default False to preserve causality).
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Dense Head
        lstm_out_dim = hidden_size * num_directions
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(p=fc_dropout),
            nn.Linear(lstm_out_dim, 32),
            nn.GELU(),
            nn.Dropout(p=fc_dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through the LSTM and projection head.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
            state: Optional initial (h_0, c_0) hidden states.

        Returns:
            Raw logit tensor of shape (batch_size, 1).
        """
        # lstm_out: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x, state)

        # Extract representation from the final sequence step
        final_step_repr = lstm_out[:, -1, :]

        # Project to scalar logit
        logits = self.head(final_step_repr)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict positive return probabilities via Sigmoid activation: P(y=1 | x).

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Probability tensor of shape (batch_size, 1) in range (0, 1).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_position(self, x: torch.Tensor) -> torch.Tensor:
        """Predict continuous portfolio exposure via Hyperbolic Tangent: s in [-1, 1].

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Position allocation tensor of shape (batch_size, 1) in range [-1, 1].
        """
        logits = self.forward(x)
        return torch.tanh(logits)
