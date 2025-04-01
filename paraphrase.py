import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarking.generation import generate_shift


def main(args):
    # Set reproducibility and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # Read the input text from the provided file
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Construct a paraphrasing prompt.
    prompt = (
        "Rewrite the following text in a different style:\n\n"
        f"{text}\n\nRewritten version:"
    )

    # Tokenize the prompt. Adjust max_length as needed.
    tokens = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )

    # Generate watermarked text using the provided key.
    watermarked_tokens = generate_shift(
        model, tokens, len(tokenizer), args.n, args.m, args.key
    )[0]
    watermarked_text = tokenizer.decode(
        watermarked_tokens, skip_special_tokens=True
    )

    print(watermarked_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paraphrase text from a file using watermarked generation."
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-1.3b",
        type=str,
        help="HuggingFace model ID."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Path to the text file to paraphrase."
    )
    parser.add_argument(
        "--m",
        default=80,
        type=int,
        help="Number of tokens to generate (default: 80)."
    )
    parser.add_argument(
        "--n",
        default=256,
        type=int,
        help="Length of the watermark sequence (default: 256)."
    )
    parser.add_argument(
        "--key",
        default=42,
        type=int,
        help="Key for watermark generation (default: 42)."
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for reproducibility (default: 0)."
    )
    args = parser.parse_args()
    main(args)
