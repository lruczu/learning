import torch


def compute_loss(positions: torch.tensor, probs: torch.tensor) -> torch.tensor:
    """
    Args:
         positions: of shape (batch_size, 1)
         probs: of shape (batch_size, T), where T is number of tokens
    """
    pass


def qa_loss(
    start_positions: torch.tensor,
    start_probs: torch.tensor,
    end_positions: torch.tensor,
    end_probs: torch.tensor,
) -> torch.tensor:
    start_loss = compute_loss(start_positions, start_probs)
    end_loss = compute_loss(end_positions, end_probs)

    return (start_loss + end_loss) / 2.0
