# Source https://github.com/jthickstun/watermark/blob/main/demo/generate.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermarking.mersenne import mersenne_rng


def generate_shift(model, prompt, vocab_size, n, key, tokenizer=None, verbose=False):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand()
                      for _ in range(n*vocab_size)]).view(n, vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    eos_token_id = model.config.eos_token_id

    # Last decoded text position
    last_pos = len(prompt[0])

    # Safety counter to prevent infinite loops
    i = 0
    max_tokens = prompt.shape[1] + 100
    while i < max_tokens:
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(
            output.logits[:, -1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs, xi[(shift+i) % n, :]).to(model.device)

        # Print token as it's generated
        if verbose and tokenizer:
            # Add the token to decode
            temp_input = inputs.clone()
            temp_input = torch.cat([temp_input, token], dim=-1)

            # Decode only the new part (from last_pos to end)
            new_text = tokenizer.decode(
                temp_input[0][last_pos:], skip_special_tokens=True)
            print(new_text, end="", flush=True)
            last_pos = len(temp_input[0])

        # Check if the generated token is an EOS token
        if token.item() == eos_token_id:
            inputs = torch.cat([inputs, token], dim=-1)
            break

        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        i += 1

    return inputs.detach().cpu()


def exp_sampling(probs, u):
    return torch.argmax(u ** (1/probs), axis=1).unsqueeze(-1)

# generate unwatermarked completions of token length m given list of prompts


def generate_rnd(model, prompt, tokenizer=None, max_tokens=100, verbose=False):
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    # Get EOS token ID from model config
    eos_token_id = model.config.eos_token_id

    # Track last position for verbose printing
    last_pos = len(prompt[0])

    # Safety counter to prevent infinite loops
    i = 0
    while i < max_tokens:
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1], dim=-1)
        token = torch.multinomial(probs, 1)

        # Print token as it's generated
        if verbose and tokenizer:
            # Add the token to decode
            temp_input = inputs.clone()
            temp_input = torch.cat([temp_input, token], dim=-1)

            # Decode only the new part
            new_text = tokenizer.decode(
                temp_input[0][last_pos:], skip_special_tokens=True)
            print(new_text, end="", flush=True)
            last_pos = len(temp_input[0])

        # Check if the generated token is an EOS token
        if token.item() == eos_token_id:
            inputs = torch.cat([inputs, token], dim=1)
            break

        inputs = torch.cat([inputs, token], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        i += 1

    return inputs.detach().cpu()
