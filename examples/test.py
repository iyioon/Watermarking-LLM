import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarking.detection import detect_watermark, generate_null_distribution
from paraphraser.paraphrase import paraphrase_text, paraphrase_without_watermark

# Explicitly disable tokenizer parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(
        description="Compare watermarked vs non-watermarked paraphrasing")
    parser.add_argument("--file", required=True, type=str,
                        help="File with text to paraphrase")
    parser.add_argument("--model", default="facebook/opt-iml-1.3b",
                        type=str, help="Model name")
    parser.add_argument("--key", default=42, type=int, help="Watermarking key")
    parser.add_argument("--n", default=128, type=int,
                        help="Watermark sequence length")
    parser.add_argument("--k", default=4, type=int,
                        help="Sequence comparison length")
    parser.add_argument("--seed", default=12345, type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Show generation process")
    parser.add_argument("--output", type=str, help="Output file for results")

    parser.add_argument("--null-file", type=str,
                        help="Pre-computed null distribution file")
    parser.add_argument("--null-samples", type=int, default=300,
                        help="Number of samples for null distribution")
    parser.add_argument("--save-null", type=str,
                        help="Save generated null distribution to file")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Read input text
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Reading input from: {args.file}")
    print(f"Text length: {len(text)} characters")
    print(f"Tokenized length: {len(text.split())} tokens")

    # Set up device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    print(f"Using model: {args.model}")

    print(f"\n\n=== Original text ===\n {text}")

    # Initialize tokenizer (needed for both paraphrasing and detection)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generate watermarked paraphrase
    print("\n\n=== Generating watermarked paraphrase ===")
    watermarked_text = paraphrase_text(
        text=text,
        model_name=args.model,
        n=args.n,
        key=args.key,
        device=device,
        seed=args.seed,
        verbose=args.verbose
    )

    # Generate non-watermarked paraphrase
    print("\n\n=== Generating non-watermarked paraphrase ===")
    normal_text = paraphrase_without_watermark(
        text=text,
        model_name=args.model,
        device=device,
        seed=args.seed,
        verbose=args.verbose
    )

    # Get or generate null distribution
    null_distribution = None
    if args.null_file and os.path.exists(args.null_file):
        print(f"\n\n=== Loading null distribution from {args.null_file}...===")
        null_distribution = torch.load(args.null_file)
        print(
            f"Loaded null distribution with {len(null_distribution)} samples")
    else:
        print(
            f"\nGenerating null distribution with {args.null_samples} samples...")
        null_distribution = generate_null_distribution(
            vocab_size=len(tokenizer),
            n=args.n,
            k=args.k,
            n_samples=args.null_samples
        )

        # Save null distribution if requested
        if args.save_null:
            print(f"Saving null distribution to {args.save_null}...")
            torch.save(null_distribution, args.save_null)

    # Detect watermarks in both texts
    print("\n\n=== Running watermark detection ===")
    print("This may take some time...")
    print("Detecting watermarks in watermarked text...")
    watermarked_p_value = detect_watermark(
        watermarked_text,
        tokenizer,
        n=args.n,
        k=args.k,
        key=args.key,
        use_fast=True,
        null_distribution=null_distribution,
        verbose=True
    )

    print("\nDetecting watermarks in non-watermarked text...")
    normal_p_value = detect_watermark(
        normal_text,
        tokenizer,
        n=args.n,
        k=args.k,
        key=args.key,
        use_fast=True,
        null_distribution=null_distribution,
        verbose=True
    )
    # Display results
    print("\n\n=== RESULTS ===")
    print(f"\nWatermarked paraphrase (p-value: {watermarked_p_value:.6f}):")
    print("-" * 50)
    print(watermarked_text[:200] +
          "..." if len(watermarked_text) > 200 else watermarked_text)

    print(f"\nNon-watermarked paraphrase (p-value: {normal_p_value:.6f}):")
    print("-" * 50)
    print(normal_text[:200] + "..." if len(normal_text) > 200 else normal_text)

    print("\n=== INTERPRETATION ===")
    print("Lower p-values indicate stronger evidence of watermarking.")
    print(
        f"Watermarked text p-value: {watermarked_p_value:.6f} - {'WATERMARKED' if watermarked_p_value < 0.05 else 'NOT DETECTED'}")
    print(
        f"Normal text p-value: {normal_p_value:.6f} - {'WATERMARKED' if normal_p_value < 0.05 else 'NOT DETECTED'}")

    # Calculate ratio of detection strength
    detection_ratio = "N/A"
    if normal_p_value > 0:
        detection_ratio = f"{normal_p_value/watermarked_p_value:.2f}x"

    print(
        f"\nThe watermarked text shows {detection_ratio} stronger watermark signal than the normal text.")

    # Save results to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("=== WATERMARKING TEST RESULTS ===\n\n")
            f.write(f"Original text length: {len(text)} characters\n")
            f.write(
                f"Watermark parameters: n={args.n}, k={args.k}, key={args.key}\n\n")

            f.write("=== WATERMARKED TEXT ===\n")
            f.write(f"p-value: {watermarked_p_value:.6f}\n")
            f.write(watermarked_text + "\n\n")

            f.write("=== NON-WATERMARKED TEXT ===\n")
            f.write(f"p-value: {normal_p_value:.6f}\n")
            f.write(normal_text + "\n\n")

            f.write("=== INTERPRETATION ===\n")
            f.write(
                f"Watermarked detection: {'DETECTED' if watermarked_p_value < 0.05 else 'NOT DETECTED'}\n")
            f.write(
                f"Normal text detection: {'DETECTED' if normal_p_value < 0.05 else 'NOT DETECTED'}\n")
            f.write(f"Detection strength ratio: {detection_ratio}\n")

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

# python -m examples.test --file data/email.txt --verbose
