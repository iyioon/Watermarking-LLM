# examples/compare_demo.py
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your watermark modules
from watermarking.embedding import simple_watermark_bias
from watermarking.detection import simple_watermark_detection

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def generate_normal_text(prompt, model_name="gpt2", max_new_tokens=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
        logits = outputs.logits[:, -1, :].squeeze(0)

        # Normal sampling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

    return tokenizer.decode(generated[0])


def generate_watermarked_text(prompt, secret_key="demo_key", model_name="gpt2", max_new_tokens=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
        logits = outputs.logits[:, -1, :].squeeze(0)

        # Apply watermark bias
        biased_logits = simple_watermark_bias(logits, secret_key, step)
        probs = torch.softmax(biased_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

    return tokenizer.decode(generated[0])


def highlight_watermark_tokens(text, watermark_key, model_name="gpt2"):
    """
    Tokenize the text, run detection on each token, and highlight
    tokens that meet the watermark condition (token < threshold).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_ids = tokenizer.encode(text)

    highlighted_tokens = []
    for i, token_id in enumerate(token_ids):
        expected_threshold = int(hash(watermark_key + "_" + str(i)) % 100)
        token_str = tokenizer.decode([token_id])

        # If the token is 'under the threshold', color it red
        if token_id < expected_threshold:
            highlighted_tokens.append(f"{RED}{token_str}{RESET}")
        else:
            highlighted_tokens.append(token_str)

    # Join with spaces (or you could just join with '' if you want a continuous text)
    return " ".join(highlighted_tokens)


if __name__ == "__main__":
    prompt_text = "Once upon a time, I wanted"

    # 1) Generate normal vs. watermarked
    normal_text = generate_normal_text(prompt_text)
    watermarked_text = generate_watermarked_text(prompt_text)

    # 2) Detect watermark presence (naive detection score)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    normal_ids = tokenizer.encode(normal_text)
    watermarked_ids = tokenizer.encode(watermarked_text)

    normal_score = simple_watermark_detection(normal_ids, "demo_key")
    watermarked_score = simple_watermark_detection(watermarked_ids, "demo_key")

    print("\n--- Normal Text ---")
    print(normal_text)
    print(f"Detection Score (Normal): {normal_score:.3f}")

    print("\n--- Watermarked Text ---")
    print(watermarked_text)
    print(f"Detection Score (Watermarked): {watermarked_score:.3f}")

    # 3) Highlight the watermarked text tokens that meet the watermark condition
    print("\n--- Highlighted Watermark Tokens ---")
    colored_wm = highlight_watermark_tokens(watermarked_text, "demo_key")
    print(colored_wm)
