import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarking.generation import generate_shift
from watermarking.generation import generate_rnd

EMAIL_PARAPHRASE_PROMPT_TEMPLATE = """
TASK: Reword the following email using different vocabulary but EXACTLY preserve its complete meaning.
IMPORTANT: You must maintain the EXACT same:
- Greeting (Hi Team)
- Structure (same paragraphs)
- All information and details
- Professional tone
- Signature format

DO NOT respond to the email or write a new email. ONLY rewrite it with different words.

EXAMPLE INPUT:
Hi Team,
I need your reports by Friday.
Thanks,
John

EXAMPLE OUTPUT:
Hi Team,
Please submit your documents before the weekend.
Appreciated,
John

Now rewrite this email:

{text}

Rewritten version:
"""


def create_paraphrase_prompt(text):
    """Create a paraphrasing prompt using the global template."""
    return EMAIL_PARAPHRASE_PROMPT_TEMPLATE.format(text=text)


def paraphrase_text(
    text,
    model_name="facebook/opt-iml-1.3b",
    n=256,
    key=42,
    device=None,
    seed=None,
    verbose=False
):
    """
    Paraphrase text using a language model with watermarking.

    Args:
        text (str): The text to paraphrase
        model_name (str): HuggingFace model ID
        n (int): Length of the watermark sequence
        key (int): Key for watermark generation
        device (torch.device): Device to run the model on (default: auto-detect)
        seed (int): Random seed for reproducibility
        verbose (bool): Print tokens as they're generated

    Returns:
        str: The paraphrased text with watermarking
    """
    # Set reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Device selection
    if device is None:
        # Auto-detect the best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Construct a paraphrasing prompt
    prompt = create_paraphrase_prompt(text)

    # Tokenize the prompt
    tokens = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )

    # Generate watermarked text
    watermarked_tokens = generate_shift(
        model, tokens, len(tokenizer), n, key, tokenizer=tokenizer, verbose=verbose
    )[0]
    watermarked_text = tokenizer.decode(
        watermarked_tokens, skip_special_tokens=True
    )

    # Extract only the paraphrased part
    result = watermarked_text.replace(prompt, "").strip()

    return result


def main(args):
    # Read the input text from the provided file
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Call the paraphrase function
    result = paraphrase_text(
        text=text,
        model_name=args.model,
        n=args.n,
        key=args.key,
        device=device,
        seed=args.seed,
        verbose=args.verbose
    )

    # Print the result (or save to file if output is specified)
    if hasattr(args, "output") and args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paraphrase text from a file using watermarked generation."
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-iml-1.3b",
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
        default=np.random.randint(0, 10000),
        type=int,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tokens as they're generated"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save paraphrased text (optional)."
    )
    args = parser.parse_args()
    main(args)


def paraphrase_without_watermark(
    text,
    model_name="facebook/opt-iml-1.3b",
    device=None,
    seed=None,
    verbose=False
):
    """
    Paraphrase text using a language model WITHOUT watermarking.

    Args:
        text (str): The text to paraphrase
        model_name (str): HuggingFace model ID
        device (torch.device): Device to run the model on
        seed (int): Random seed for reproducibility
        verbose (bool): Print tokens as they're generated

    Returns:
        str: The paraphrased text without watermarking
    """
    # Set reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Device selection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Construct a paraphrasing prompt - same as the watermarked version
    prompt = create_paraphrase_prompt(text)

    # Tokenize the prompt
    tokens = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )

    # Generate non-watermarked text
    normal_tokens = generate_rnd(
        model, tokens, tokenizer=tokenizer, max_tokens=100, verbose=verbose
    )[0]
    normal_text = tokenizer.decode(
        normal_tokens, skip_special_tokens=True
    )

    # Extract only the paraphrased part
    result = normal_text.replace(prompt, "").strip()

    return result

# python -m paraphraser.paraphrase --file data/email.txt
