import argparse
import os
import json
import random
import numpy as np
import torch
import time

from paraphraser.paraphrase import paraphrase_text, paraphrase_without_watermark


def generate_multiple_paraphrases(
    text,
    num_versions=5,
    model_name="facebook/opt-iml-1.3b",
    n=256,
    keys=None,
    include_unwatermarked=True,
    device=None,
    seed=None,
    verbose=False
):
    """
    Generate multiple paraphrases of a text, each with a different watermark key.

    Args:
        text (str): Text to paraphrase
        num_versions (int): Number of paraphrased versions to generate
        model_name (str): HuggingFace model ID
        n (int): Length of watermark sequence
        keys (list): Optional specific keys to use. If None, random keys are generated.
        include_unwatermarked (bool): Whether to include an unwatermarked version
        device (torch.device): Computing device
        seed (int): Random seed
        verbose (bool): Print generation details

    Returns:
        list: List of dicts with paraphrased text and key information
    """
    results = []

    # Set the seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate random keys if not provided
    if keys is None:
        keys = random.sample(range(10000, 1000000), num_versions)
    else:
        # Use provided keys but ensure we have enough
        if len(keys) < num_versions:
            additional_keys = random.sample(
                range(10000, 1000000), num_versions - len(keys))
            keys = keys + additional_keys

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Generating {num_versions} paraphrased versions...")

    # Generate each paraphrase with a different key
    for i, key in enumerate(keys[:num_versions]):
        print(f"\nGenerating paraphrase {i+1}/{num_versions} with key={key}")
        start_time = time.time()

        # Generate paraphrased text with watermarking
        paraphrased_text = paraphrase_text(
            text=text,
            model_name=model_name,
            n=n,
            key=key,
            device=device,
            seed=seed + i if seed is not None else None,
            verbose=verbose
        )

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f} seconds")

        # Store result
        results.append({
            "version": i+1,
            "key": key,
            "watermarked": True,
            "text": paraphrased_text
        })

    # Add non-watermarked version if requested
    if include_unwatermarked:
        print("\nGenerating non-watermarked paraphrase...")
        start_time = time.time()

        unwatermarked_text = paraphrase_without_watermark(
            text=text,
            model_name=model_name,
            device=device,
            seed=seed + num_versions if seed is not None else None,
            verbose=verbose
        )

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f} seconds")

        results.append({
            "version": len(results) + 1,
            "key": None,
            "watermarked": False,
            "text": unwatermarked_text
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple paraphrased versions with different watermark keys"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to text file to paraphrase"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of paraphrased versions to generate"
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-iml-1.3b",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=256,
        help="Length of watermark sequence"
    )
    parser.add_argument(
        "--keys",
        type=int,
        nargs="+",
        help="Specific keys to use (optional)"
    )
    parser.add_argument(
        "--no-unwatermarked",
        action="store_false",
        dest="include_unwatermarked",
        help="Don't include an unwatermarked version"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print token-by-token generation details"
    )
    parser.add_argument(
        "--output",
        help="Output file path (JSON format)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for individual text files"
    )

    args = parser.parse_args()

    # Read input text
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"Input text ({len(text)} chars):")
        print(f"{text[:100]}..." if len(text) > 100 else text)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Generate paraphrases
    results = generate_multiple_paraphrases(
        text=text,
        num_versions=args.num,
        model_name=args.model,
        n=args.n,
        keys=args.keys,
        include_unwatermarked=args.include_unwatermarked,
        seed=args.seed,
        verbose=args.verbose
    )

    # Print results summary
    print("\n--- Results Summary ---")
    for result in results:
        version = result["version"]
        key = result["key"] if result["key"] is not None else "None"
        watermarked = "Yes" if result["watermarked"] else "No"
        text_sample = result["text"][:50] + \
            "..." if len(result["text"]) > 50 else result["text"]
        print(f"Version {version}: Key={key}, Watermarked={watermarked}")
        print(f"  {text_sample}")
        print()

    # Save to JSON file if output specified
    if args.output:
        output_data = {
            "original_text": text,
            "model": args.model,
            "paraphrases": results
        }

        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")

    # Save individual files if output directory specified
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)

            # Save original text
            with open(os.path.join(args.output_dir, "original.txt"), "w", encoding="utf-8") as f:
                f.write(text)

            # Save each paraphrase
            for result in results:
                version = result["version"]
                key = result["key"] if result["key"] is not None else "none"
                filename = f"paraphrase_{version}_key_{key}.txt"

                with open(os.path.join(args.output_dir, filename), "w", encoding="utf-8") as f:
                    f.write(result["text"])

            # Save keys file
            with open(os.path.join(args.output_dir, "keys.txt"), "w", encoding="utf-8") as f:
                for result in results:
                    version = result["version"]
                    key = result["key"] if result["key"] is not None else "none"
                    watermarked = "Yes" if result["watermarked"] else "No"
                    f.write(
                        f"Version {version}: Key={key}, Watermarked={watermarked}\n")

            print(f"Individual files saved to {args.output_dir}")
        except Exception as e:
            print(f"Error saving individual files: {e}")


if __name__ == "__main__":
    main()

# python -m experiments.paraphrase_with_keys --file data/email.txt --num 3 --output out/email_paraphrases.json
