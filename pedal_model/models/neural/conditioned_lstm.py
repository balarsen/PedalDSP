"""Conditioned LSTM: knob-position embedding injected into the LSTM input."""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel


class ConditionedLSTMModel(nn.Module, PedalModel):
    """LSTM with continuous knob-position conditioning.

    Knob vector (e.g. [drive, tone, level]) is embedded and concatenated with
    each audio sample before entering the LSTM. One model covers the full
    parameter space — requires multi-setting captures.

    Capture protocol: record the pedal on a grid of knob positions (e.g. 5×5
    drive/tone grid = 25 captures) and pass the knob vector alongside the audio
    during training.
    """

    def __init__(
        self,
        n_knobs: int = 3,
        knob_embedding_dim: int = 8,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        """
        Args:
            n_knobs: Number of continuous knob parameters (e.g. drive, tone, level).
            knob_embedding_dim: Output dimension of the knob embedding MLP.
            hidden_size: LSTM hidden size.
            num_layers: Number of stacked LSTM layers.
        """
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.knob_embed = nn.Sequential(
            nn.Linear(n_knobs, knob_embedding_dim),
            nn.Tanh(),
        )
        lstm_input_dim = 1 + knob_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    @property
    def name(self) -> str:
        return f"ConditionedLSTM-{self.hidden_size}"

    @property
    def receptive_field(self) -> int:
        return 4096

    def forward(
        self,
        x: torch.Tensor,
        knobs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a chunk with knob conditioning.

        Args:
            x: Audio chunk, shape (batch, time, 1).
            knobs: Knob positions, shape (batch, n_knobs). Constant per chunk.
            hidden: Previous (h, c), or None.

        Returns:
            (output, (h, c)) — output shape (batch, time, 1).
        """
        embed = self.knob_embed(knobs)                           # (batch, embed_dim)
        embed = embed.unsqueeze(1).expand(-1, x.shape[1], -1)   # (batch, time, embed_dim)
        inp = torch.cat([x, embed], dim=-1)                     # (batch, time, 1 + embed_dim)
        out, hidden = self.lstm(inp, hidden)
        return self.fc(out), hidden

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(
        self,
        x: np.ndarray,
        knobs: np.ndarray,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Inference with a fixed knob position.

        Args:
            x: Dry input, shape (N,), float32.
            knobs: Knob values, shape (n_knobs,), float32, range [0, 1].
            chunk_size: Samples per chunk.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        out = np.zeros(len(x), dtype=np.float32)
        k_tensor = torch.tensor(knobs, dtype=torch.float32).unsqueeze(0)
        hidden = None
        with torch.no_grad():
            for start in range(0, len(x), chunk_size):
                chunk = x[start : start + chunk_size]
                t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                y, hidden = self(t, k_tensor, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                out[start : start + len(chunk)] = y.squeeze().cpu().numpy()
        return out
