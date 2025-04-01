from watermarking.detection import detect_watermark
import argparse
import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def detect_watermark_keys(text, json_file_path, model_name="facebook/opt-iml-1.3b",
                          n=256, k=4, significance_threshold=0.05,
                          visualize=False, output_file=None):
    """
    Test if text contains a watermark from any of the keys in the JSON file.

    Args:
        text (str): Text to check for watermarks
        json_file_path (str): Path to JSON file with paraphrase information
        model_name (str): Model name for tokenizer
        n (int): Watermark sequence length
        k (int): Sequence comparison length
        significance_threshold (float): P-value threshold for detecting watermarks
        visualize (bool): Whether to create a visualization of results
        output_file (str): File to save visualization to (if visualize=True)

    Returns:
        tuple: (matched_keys, all_results)
    """
    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return [], {}

    # Extract the keys from the JSON
    keys = []
    for paraphrase in data.get('paraphrases', []):
        if paraphrase.get('watermarked', False) and paraphrase.get('key') is not None:
            keys.append({
                'version': paraphrase.get('version'),
                'key': paraphrase.get('key'),
                'text': paraphrase.get('text', '')[:50] + '...'
            })

    if not keys:
        print("No watermark keys found in the JSON file")
        return [], {}

    print(f"Found {len(keys)} watermark keys to test")

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return [], {}

    # Test each key against the text
    results = {}
    matched_keys = []

    print("\nTesting each key against the text:")
    print("-" * 60)
    print(f"{'Version':<10} {'Key':<10} {'P-value':<10} {'Result'}")
    print("-" * 60)

    for key_info in keys:
        version = key_info['version']
        key = key_info['key']

        p_value = detect_watermark(
            text=text,
            tokenizer=tokenizer,
            n=n,
            k=k,
            key=key
        )

        is_watermarked = p_value < significance_threshold
        status = "WATERMARKED" if is_watermarked else "Not detected"

        print(f"{version:<10} {key:<10} {p_value:<10.6f} {status}")

        results[key] = {
            'version': version,
            'p_value': p_value,
            'watermarked': is_watermarked
        }

        if is_watermarked:
            matched_keys.append({
                'version': version,
                'key': key,
                'p_value': p_value
            })

    # Summary of results
    print("\nSummary:")
    if matched_keys:
        print(
            f"Text appears to be watermarked with {len(matched_keys)} key(s):")
        for match in matched_keys:
            print(
                f"  - Version {match['version']}, Key: {match['key']}, P-value: {match['p_value']:.6f}")
    else:
        print("No watermarks detected in the text.")

    # Create visualization if requested
    if visualize:
        create_visualization(results, significance_threshold, output_file)

    return matched_keys, results


def create_visualization(results, significance_threshold, output_file=None):
    """Create a visualization of the p-values for each tested key."""
    keys = list(results.keys())
    p_values = [results[key]['p_value'] for key in keys]
    versions = [results[key]['version'] for key in keys]

    plt.figure(figsize=(10, 6))

    # Bar chart of p-values
    bars = plt.bar(range(len(keys)), p_values, color=[
                   'green' if p < significance_threshold else 'gray' for p in p_values])

    # Add a horizontal line at the significance threshold
    plt.axhline(y=significance_threshold, color='red', linestyle='--',
                label=f'Threshold ({significance_threshold})')

    # Add labels
    plt.xlabel('Key Version')
    plt.ylabel('P-value')
    plt.title('Watermark Detection Results')
    plt.xticks(range(len(keys)), [f"V{v}" for v in versions], rotation=45)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{p_values[i]:.4f}',
                 ha='center', va='bottom', rotation=0)

    # Add legend
    plt.legend(['Significance Threshold',
               'P-value (lower = stronger watermark)'])

    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Detect which watermark key was used in a text."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to check for watermarks"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file containing text to check for watermarks"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to JSON file with paraphrase information"
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-iml-1.3b",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=256,
        help="Watermark sequence length"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Sequence comparison length"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="P-value threshold for detecting watermarks"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create a visualization of results"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="File to save visualization to"
    )
    parser.add_argument(
        "--test-json-text",
        action="store_true",
        help="Test using text from the JSON file itself"
    )

    args = parser.parse_args()

    # Get the text to check
    if args.test_json_text:
        print("Using text from the JSON file for testing...")
        try:
            with open(args.json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # For testing, use the first watermarked text from the JSON
            watermarked_items = [p for p in data.get(
                'paraphrases', []) if p.get('watermarked', False)]
            if watermarked_items:
                test_item = watermarked_items[0]
                text = test_item['text']
                expected_key = test_item['key']
                print(
                    f"Using watermarked text from version {test_item['version']} (key={expected_key}):")
                print(f"{text[:100]}..." if len(text) > 100 else text)
                print()
            else:
                print("No watermarked text found in the JSON file")
                return

        except Exception as e:
            print(f"Error loading JSON for testing: {e}")
            return
    elif args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Error: Either --text, --file, or --test-json-text must be provided")
        return

    # Run detection
    detect_watermark_keys(
        text=text,
        json_file_path=args.json,
        model_name=args.model,
        n=args.n,
        k=args.k,
        significance_threshold=args.threshold,
        visualize=args.visualize,
        output_file=args.output
    )


if __name__ == "__main__":
    main()


# python -m experiments.detect_key_if_watermarked --file data/email_to_test.txt --json out/email_paraphrases.json --visualize --output out/detection_results.png
