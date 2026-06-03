"""GRU model: fewer parameters than LSTM, often comparable accuracy."""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel


class GRUModel(nn.Module, PedalModel):
    """GRU variant of the recurrent pedal model.

    GRU has no separate cell state (fewer parameters than LSTM).
    Often trains faster; compare directly against LSTMModel on your captures.
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 1) -> None:
        """
        Args:
            hidden_size: GRU hidden dimension.
            num_layers: Number of stacked GRU layers.
        """
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    @property
    def name(self) -> str:
        return f"GRU-{self.hidden_size}x{self.num_layers}"

    @property
    def receptive_field(self) -> int:
        return 4096

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a chunk of samples.

        Args:
            x: Input, shape (batch, time, 1).
            hidden: h from previous chunk, shape (num_layers, batch, hidden_size).

        Returns:
            (output, h) — output shape (batch, time, 1).
        """
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Chunked inference with carried hidden state.

        Args:
            x: Dry input, shape (N,), float32.
            chunk_size: Samples per chunk.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        out = np.zeros(len(x), dtype=np.float32)
        hidden = None
        with torch.no_grad():
            for start in range(0, len(x), chunk_size):
                chunk = x[start : start + chunk_size]
                t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                y, hidden = self(t, hidden)
                hidden = hidden.detach()
                out[start : start + len(chunk)] = y.squeeze().cpu().numpy()
        return out
