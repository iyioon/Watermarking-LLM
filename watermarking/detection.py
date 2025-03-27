# watermarking/detection.py
def simple_watermark_detection(token_ids, watermark_key):
    """
    A dummy detection function that computes an average 'bias score'
    over token IDs as a proxy for the watermark signal.

    Args:
        token_ids (list or iterable): List of generated token indices.
        watermark_key (str): The secret watermark key.

    Returns:
        float: A score representing the watermark signal.
    """
    score = 0.0
    for i, token in enumerate(token_ids):
        # Example: use hash to compute an expected threshold per token.
        expected_threshold = int(hash(watermark_key + "_" + str(i)) % 100)
        if token < expected_threshold:
            score += 1
    return score / len(token_ids)
