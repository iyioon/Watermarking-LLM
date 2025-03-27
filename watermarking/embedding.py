# watermarking/embedding.py
import torch
import hashlib


def simple_watermark_bias(logits, secret_key, step, bias_val=2.0):
    """
    Applies a simple bias to the logits based on a secret key and the current step.

    Args:
        logits (torch.Tensor): The raw logits for the next token (1D tensor of shape [vocab_size]).
        secret_key (str): The secret watermark key.
        step (int): Current generation step.
        bias_val (float): Amount to boost biased tokens.

    Returns:
        torch.Tensor: The modified logits.
    """
    # Compute a deterministic hash from the key and step.
    combined = f"{secret_key}_{step}"
    h = hashlib.sha256(combined.encode()).hexdigest()

    # Use part of the hash to determine a threshold index.
    vocab_size = logits.size(0)
    threshold = int(h[:4], 16) % vocab_size

    # Bias tokens with index below threshold.
    logits[:threshold] += bias_val
    return logits
