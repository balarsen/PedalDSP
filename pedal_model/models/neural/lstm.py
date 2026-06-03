"""LSTM model: processes one sample at a time, hidden state carries circuit memory."""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel


class LSTMModel(nn.Module, PedalModel):
    """Single or stacked LSTM for sample-by-sample pedal modelling.

    Hidden state h[n] propagates circuit memory (e.g. capacitor charge)
    forward in time. Train with TBPTT (truncated backprop, chunk_size=2048).
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 1) -> None:
        """
        Args:
            hidden_size: LSTM hidden dimension. Sweet spot: 32. Max meaningful: 64.
            num_layers: Number of stacked LSTM layers.
        """
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    @property
    def name(self) -> str:
        return f"LSTM-{self.hidden_size}x{self.num_layers}"

    @property
    def receptive_field(self) -> int:
        # LSTM has theoretically infinite receptive field; return a practical estimate
        return 4096

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a chunk of samples.

        Args:
            x: Input chunk, shape (batch, time, 1).
            hidden: (h, c) from previous chunk, or None to start fresh.

        Returns:
            (output, (h, c)) — output shape (batch, time, 1).
        """
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

    # ------------------------------------------------------------------ #
    # PedalModel interface                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Run inference in chunks, carrying hidden state between chunks.

        Args:
            x: Dry input, shape (N,), float32.
            chunk_size: Samples per forward-pass chunk.

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
                # Detach hidden state to prevent graph accumulation
                hidden = (hidden[0].detach(), hidden[1].detach())
                out[start : start + len(chunk)] = y.squeeze().cpu().numpy()
        return out
