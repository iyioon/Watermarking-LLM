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


def fast_permutation_test(tokens, vocab_size, n, k, seed, test_stat, null_results):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    print("Computing test statistic...")
    test_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size,
                            verbose=False)

    # Find where test_result falls in the null distribution
    p_val = torch.searchsorted(
        null_results, test_result, right=True) / len(null_results)
    return p_val


def generate_null_distribution(vocab_size, n, k, n_samples=1000, max_seed=100000):
    """
    Generate a null distribution for fast permutation testing.

    Args:
        vocab_size (int): Size of the vocabulary
        n (int): Length of watermark sequence
        k (int): Sequence comparison length
        n_samples (int): Number of samples for null distribution
        max_seed (int): Maximum seed value

    Returns:
        torch.Tensor: Sorted null distribution values
    """
    print(f"Generating null distribution with {n_samples} samples...")
    null_results = []

    # Define L1 distance function
    def l1_dist(x, y):
        try:
            return torch.sum(torch.abs(x - y))
        except:
            min_dim0 = min(x.size(0), y.size(0))
            min_dim1 = min(x.size(1) if x.ndim > 1 else 1,
                           y.size(1) if y.ndim > 1 else 1)

            x_resized = x[:min_dim0,
                          :min_dim1] if x.ndim > 1 else x[:min_dim0].unsqueeze(1)
            y_resized = y[:min_dim0,
                          :min_dim1] if y.ndim > 1 else y[:min_dim0].unsqueeze(1)

            return torch.sum(torch.abs(x_resized - y_resized))

    # Key function
    def key_func(generator, n, vocab_size, eff_vocab_size):
        rng = mersenne_rng(int(generator.initial_seed()))
        xi = torch.tensor([rng.rand()
                          for _ in range(n*vocab_size)]).view(n, vocab_size)
        pi = torch.randperm(vocab_size, generator=generator)
        return xi, pi

    # Test statistic function
    def phi_test(tokens, n, k, generator, vocab_size, null=False, verbose=False):
        return phi(tokens, n, k, generator, key_func, vocab_size, l1_dist, null, normalize=True, verbose=verbose)

    # Generate random samples and compute test statistics
    generator = torch.Generator()

    for run in tqdm(range(n_samples), desc="Generating null distribution"):
        # Generate random tokens
        random_length = k + 20  # Add some buffer
        tokens = torch.randint(0, vocab_size, (random_length,))

        # Random seed for each sample
        seed = torch.randint(high=max_seed, size=(1,)).item()
        generator.manual_seed(int(seed))

        # Compute test statistic
        null_result = phi_test(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            vocab_size=vocab_size,
            null=True,
            verbose=False
        )

        null_results.append(null_result)

    # Sort the null results for binary search in fast_permutation_test
    null_results = torch.sort(torch.tensor(null_results)).values
    print(f"Null distribution generated with {len(null_results)} values")
    return null_results


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


def generate_null_distribution(vocab_size, n, k, n_samples=300, max_seed=100000):
    """
    Generate a null distribution for fast permutation testing.

    Args:
        vocab_size (int): Size of the vocabulary
        n (int): Length of watermark sequence
        k (int): Sequence comparison length
        n_samples (int): Number of samples for null distribution
        max_seed (int): Maximum seed value

    Returns:
        torch.Tensor: Sorted null distribution values
    """
    print(f"Generating null distribution with {n_samples} samples...")
    null_results = []

    # Define L1 distance function
    def l1_dist(x, y):
        try:
            return torch.sum(torch.abs(x - y))
        except:
            min_dim0 = min(x.size(0), y.size(0))
            min_dim1 = min(x.size(1) if x.ndim > 1 else 1,
                           y.size(1) if y.ndim > 1 else 1)

            x_resized = x[:min_dim0,
                          :min_dim1] if x.ndim > 1 else x[:min_dim0].unsqueeze(1)
            y_resized = y[:min_dim0,
                          :min_dim1] if y.ndim > 1 else y[:min_dim0].unsqueeze(1)

            return torch.sum(torch.abs(x_resized - y_resized))

    # Key function
    def key_func(generator, n, vocab_size, eff_vocab_size):
        rng = mersenne_rng(int(generator.initial_seed()))
        xi = torch.tensor([rng.rand()
                          for _ in range(n*vocab_size)]).view(n, vocab_size)
        pi = torch.randperm(vocab_size, generator=generator)
        return xi, pi

    # Test statistic function
    def phi_test(tokens, n, k, generator, vocab_size, null=False, verbose=False):
        return phi(tokens, n, k, generator, key_func, vocab_size, l1_dist, null, normalize=True, verbose=verbose)

    # Generate random samples and compute test statistics
    generator = torch.Generator()

    for run in tqdm(range(n_samples), desc="Generating null distribution"):
        # Generate random tokens
        random_length = k + 20  # Add some buffer
        tokens = torch.randint(0, vocab_size, (random_length,))

        # Random seed for each sample
        seed = torch.randint(high=max_seed, size=(1,)).item()
        generator.manual_seed(int(seed))

        # Compute test statistic
        null_result = phi_test(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            vocab_size=vocab_size,
            null=True,
            verbose=False
        )

        null_results.append(null_result)

    # Sort the null results for binary search in fast_permutation_test
    null_results = torch.sort(torch.tensor(null_results)).values
    print(f"Null distribution generated with {len(null_results)} values")
    return null_results


def detect_watermark(text, tokenizer, n=128, k=4, key=42, use_fast=False, null_distribution=None, verbose=False):
    """
    Run watermark detection on the given text.

    Args:
        text (str): Text to check for watermark
        tokenizer: The tokenizer to use
        n (int): Length of watermark sequence
        k (int): Sequence comparison length
        key (int): Key for watermark generation
        use_fast (bool): Whether to use fast permutation test
        null_distribution (torch.Tensor): Pre-computed null distribution
        verbose (bool): Whether to print progress information

    Returns:
        float: p-value indicating how likely the text is watermarked
    """
    if verbose:
        print(f"\nStarting watermark detection with parameters:")
        print(f"- Text length: {len(text)} characters")
        print(f"- Key: {key}")
        print(f"- n: {n}, k: {k}")
        print(
            f"- Method: {'Fast' if use_fast else 'Standard'} permutation test")

    # L1 distance function
    def l1_dist(x, y):
        try:
            return torch.sum(torch.abs(x - y))
        except:
            min_dim0 = min(x.size(0), y.size(0))
            min_dim1 = min(x.size(1) if x.ndim > 1 else 1,
                           y.size(1) if y.ndim > 1 else 1)

            x_resized = x[:min_dim0,
                          :min_dim1] if x.ndim > 1 else x[:min_dim0].unsqueeze(1)
            y_resized = y[:min_dim0,
                          :min_dim1] if y.ndim > 1 else y[:min_dim0].unsqueeze(1)

            return torch.sum(torch.abs(x_resized - y_resized))

    # Key function
    def key_func(generator, n, vocab_size, eff_vocab_size):
        rng = mersenne_rng(int(generator.initial_seed()))
        xi = torch.tensor([rng.rand()
                          for _ in range(n*vocab_size)]).view(n, vocab_size)
        pi = torch.randperm(vocab_size, generator=generator)
        return xi, pi

    # Test statistic function
    def phi_test(tokens, n, k, generator, vocab_size, null=False, verbose=verbose):
        return phi(tokens, n, k, generator, key_func, vocab_size, l1_dist, null, normalize=True, verbose=verbose)

    # Tokenize the text
    if verbose:
        print("Tokenizing text...")
    tokens = torch.tensor(tokenizer.encode(text))
    if verbose:
        print(f"Text tokenized to {len(tokens)} tokens")

    # Check if text is too short
    if len(tokens) < k:
        if verbose:
            print("Text too short for detection.")
        return 1.0  # Not watermarked

    # Choose detection method
    if use_fast and null_distribution is not None:
        if verbose:
            print("Running fast permutation test...")
        p_value = fast_permutation_test(
            tokens=tokens,
            vocab_size=len(tokenizer),
            n=n,
            k=k,
            seed=key,
            test_stat=phi_test,
            null_results=null_distribution
        )
    else:
        # If fast requested but no null distribution, or if standard requested
        if use_fast and verbose:
            print(
                "No null distribution provided. Falling back to standard permutation test.")

        if verbose:
            print("Running standard permutation test...")
        p_value = permutation_test(
            tokens=tokens,
            vocab_size=len(tokenizer),
            n=n,
            k=k,
            seed=key,
            test_stat=phi_test,
            n_runs=100
        )

    if verbose:
        print(
            f"Detection complete. P-value: {p_value.item() if hasattr(p_value, 'item') else p_value}")

    return p_value.item() if hasattr(p_value, 'item') else p_value
