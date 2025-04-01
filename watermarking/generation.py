# Source https://github.com/jthickstun/watermark/blob/main/demo/generate.py

import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_shift(model, prompt, vocab_size, n, m, key):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand()
                      for _ in range(n*vocab_size)]).view(n, vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(
            output.logits[:, -1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs, xi[(shift+i) % n, :]).to(model.device)
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()


def exp_sampling(probs, u):
    return torch.argmax(u ** (1/probs), axis=1).unsqueeze(-1)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    tokens = tokenizer.encode(
        args.prompt, return_tensors='pt', truncation=True, max_length=2048)

    watermarked_tokens = generate_shift(
        model, tokens, len(tokenizer), args.n, args.m, args.key)[0]
    watermarked_text = tokenizer.decode(
        watermarked_tokens, skip_special_tokens=True)

    print(watermarked_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate text watermarked with a key')
    parser.add_argument('--model', default='gpt2', type=str,
                        help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--prompt', default='', type=str,
                        help='an optional prompt for generation')
    parser.add_argument('--m', default=80, type=int,
                        help='the requested length of the generated text')
    parser.add_argument('--n', default=256, type=int,
                        help='the length of the watermark sequence')
    parser.add_argument('--key', default=42, type=int,
                        help='a key for generating the random watermark sequence')
    parser.add_argument('--seed', default=0, type=int,
                        help='a seed for reproducibile randomness')

    main(parser.parse_args())


class mersenne_rng(object):
    def __init__(self, seed=5489):
        self.state = [0]*624
        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1 << 31)-1
        self.upper_mask = 1 << 31

        # update state
        self.state[0] = seed
        for i in range(1, 624):
            self.state[i] = self.int_32(
                self.f*(self.state[i-1] ^ (self.state[i-1] >> 30)) + i)

    def twist(self):
        for i in range(624):
            temp = self.int_32(
                (self.state[i] & self.upper_mask)+(self.state[(i+1) % 624] & self.lower_mask))
            temp_shift = temp >> 1
            if temp % 2 != 0:
                temp_shift = temp_shift ^ 0x9908b0df
            self.state[i] = self.state[(i+self.m) % 624] ^ temp_shift
        self.index = 0

    def int_32(self, number):
        return int(0xFFFFFFFF & number)

    def randint(self):
        if self.index >= 624:
            self.twist()
        y = self.state[self.index]
        y = y ^ (y >> self.u)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        self.index += 1
        return self.int_32(y)

    def rand(self):
        return self.randint()*(1.0/4294967296.0)

    def randperm(self, n):
        # Fisher-Yates shuffle
        p = list(range(n))
        for i in range(n-1, 0, -1):
            j = self.randint() % i
            p[i], p[j] = p[j], p[i]

        return p


if __name__ == "__main__":
    rng = mersenne_rng(10)
    for i in range(1000000):
        rng.rand()

    for i in range(10):
        print(rng.rand())
