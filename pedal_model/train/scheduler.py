"""Learning rate scheduling utilities."""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def build_scheduler(
    optimizer: Optimizer,
    n_epochs: int,
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup → cosine annealing schedule.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        n_epochs: Total number of training epochs.
        warmup_epochs: Number of linear warmup epochs before cosine decay starts.

    Returns:
        A scheduler compatible with scheduler.step() called once per epoch.
    """
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
