import torch
from watermarking.mersenne import mersenne_rng
try:
    from tqdm import tqdm
except ImportError:
    # Define a simple fallback if tqdm isn't installed
    def tqdm(iterable, **kwargs):
        print(f"Starting {kwargs.get('desc', 'process')}...")
        result = iterable
        print(f"Completed {kwargs.get('desc', 'process')}.")
        return result


def permutation_test(tokens, vocab_size, n, k, seed, test_stat, n_runs=100, max_seed=100000):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    # First convert tokens to long type for consistent indexing
    tokens = tokens.long()

    print("Computing test statistic for original tokens...")
    test_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size,
                            verbose=False)  # Add verbose parameter

    p_val = 0
    print(f"Running {n_runs} permutation tests...")

    # Add progress bar for the permutation test
    for run in tqdm(range(n_runs), desc="Permutation tests", ncols=100):
        pi = torch.randperm(vocab_size)
        shuffled_tokens = torch.argsort(pi)[tokens]

        seed = torch.randint(high=max_seed, size=(1,)).item()
        generator.manual_seed(int(seed))
        null_result = test_stat(tokens=shuffled_tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size,
                                null=True,
                                verbose=False)  # Add verbose parameter
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float() / n_runs

    return p_val


def adjacency(tokens, xi, dist, k, vocab_size, verbose=True):
    m = len(tokens)
    n = len(xi)

    A = torch.empty(size=(m-(k-1), n))

    # Use smaller batches for efficiency
    batch_size = 10
    total_batches = (m-(k-1) + batch_size - 1) // batch_size

    # Only show progress bar if verbose is True
    if verbose:
        pbar_context = tqdm(total=total_batches,
                            desc="Adjacency computation", ncols=100)
    else:
        # Use a dummy context manager when not verbose
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def update(self, *args): pass
        pbar_context = DummyContext()

    with pbar_context as pbar:
        for batch_start in range(0, m-(k-1), batch_size):
            batch_end = min(batch_start + batch_size, m-(k-1))

            for i in range(batch_start, batch_end):
                for j in range(n):
                    # Convert tokens to one-hot vectors to match xi dimensions
                    token_vectors = []
                    for t in range(i, i+k):
                        # Create a one-hot vector for each token
                        one_hot = torch.zeros(vocab_size)

                        # Convert token to proper integer index
                        token_idx = int(tokens[t].item())

                        if token_idx < vocab_size:  # Safety check
                            one_hot[token_idx] = 1.0
                        token_vectors.append(one_hot)

                    # Stack the one-hot vectors
                    token_tensor = torch.stack(token_vectors)

                    try:
                        # Handle different dimensions more robustly
                        xi_pattern = xi[(j+torch.arange(k)) % n]
                        if xi_pattern.ndim == 2 and token_tensor.size(1) != xi_pattern.size(1):
                            # Adjust dimensions for comparison
                            min_size = min(token_tensor.size(1),
                                           xi_pattern.size(1))
                            A[i][j] = dist(
                                token_tensor[:, :min_size], xi_pattern[:, :min_size])
                        else:
                            A[i][j] = dist(token_tensor, xi_pattern)
                    except Exception as e:
                        # Fallback in case of dimension mismatch
                        A[i][j] = float('inf')  # Use worst possible score

            pbar.update(1)

    return A


def phi(tokens, n, k, generator, key_func, vocab_size, dist, null=False, normalize=False, verbose=True):
    # Ensure tokens are in long format for indexing
    tokens = tokens.long()

    if null:
        tokens = torch.unique(tokens, return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]

    if normalize:
        tokens = tokens.float() / vocab_size

    # Pass verbose parameter to adjacency
    A = adjacency(tokens, xi, dist, k, vocab_size, verbose=verbose)
    closest = torch.min(A, axis=1)[0]

    return torch.min(closest)


def detect_watermark(text, tokenizer, n=256, k=4, key=42):
    """
    Run watermark detection on the given text.

    Args:
        text (str): Text to check for watermark
        tokenizer: The tokenizer to use
        n (int): Length of watermark sequence
        k (int): Sequence comparison length
        key (int): Key for watermark generation

    Returns:
        float: p-value indicating how likely the text is watermarked
    """
    print(f"\nStarting watermark detection with parameters:")
    print(f"- Text length: {len(text)} characters")
    print(f"- Key: {key}")
    print(f"- n: {n}, k: {k}")

    # L1 distance function that handles different tensor shapes
    def l1_dist(x, y):
        try:
            # Try to compute L1 distance directly
            return torch.sum(torch.abs(x - y))
        except:
            # If shapes don't match, use minimum common dimensions
            min_dim0 = min(x.size(0), y.size(0))
            min_dim1 = min(x.size(1) if x.ndim > 1 else 1,
                           y.size(1) if y.ndim > 1 else 1)

            x_resized = x[:min_dim0,
                          :min_dim1] if x.ndim > 1 else x[:min_dim0].unsqueeze(1)
            y_resized = y[:min_dim0,
                          :min_dim1] if y.ndim > 1 else y[:min_dim0].unsqueeze(1)

            return torch.sum(torch.abs(x_resized - y_resized))

    # Function for generating watermark patterns based on the key
    def key_func(generator, n, vocab_size, eff_vocab_size):
        # IMPORTANT: Match generation.py's random pattern exactly
        rng = mersenne_rng(int(generator.initial_seed()))
        xi = torch.tensor([rng.rand()
                          for _ in range(n*vocab_size)]).view(n, vocab_size)
        pi = torch.randperm(vocab_size, generator=generator)
        return xi, pi

    # Test statistic function using phi
    def phi_test(tokens, n, k, generator, vocab_size, null=False, verbose=True):
        return phi(tokens, n, k, generator, key_func, vocab_size, l1_dist, null, normalize=True, verbose=verbose)

    # Tokenize the text
    print("Tokenizing text...")
    tokens = torch.tensor(tokenizer.encode(text))
    print(f"Text tokenized to {len(tokens)} tokens")

    # If text is too short for detection
    if len(tokens) < k:
        print("Text too short for detection.")
        return 1.0  # Not watermarked

    # Run permutation test
    print("Running permutation test...")
    p_value = permutation_test(
        tokens=tokens,
        vocab_size=len(tokenizer),
        n=n,
        k=k,
        seed=key,
        test_stat=phi_test,
        n_runs=20
    )

    print(f"Detection complete. P-value: {p_value.item()}")
    return p_value.item()
