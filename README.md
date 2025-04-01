# Multi-Key Paraphrasing Watermarking for Document Distribution

This repository provides an implementation of a distortion‑free watermarking scheme for paraphrasing a confidential text into multiple watermarked versions. Each version is generated using a different secret key so that if one version is leaked, the corresponding key can be used to trace the source of the leak. A typical use case is detecting the leakage point of a confidential email distributed within a company.

## Overview

The watermarking method is inspired by the paper:

> **Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang.**  
> _Robust Distortion‑free Watermarks for Language Models._  
> arXiv:2307.15593v3, July 2023.  
> [https://arxiv.org/abs/2307.15593](https://arxiv.org/abs/2307.15593)

In this approach, a secret key is used to generate a deterministic sequence of uniform random numbers and a permutation over the vocabulary. During text generation, the model’s token probabilities are reordered according to this permutation and inverse‑transform sampling is applied using the precomputed random numbers. Although the token‐by‐token selections are steered by the key, the overall output distribution remains unchanged (i.e. distortion‑free). Later, the same key is used in detection by aligning the observed tokens with the watermark sequence, computing an alignment cost, and deriving a p‑value that indicates whether the watermark is present.

## Features

- Added EOS TOken Support
- Use opt-iml-1.3b model (instruction-tuned version of OPT)

## Environment Setup

```
pip install -r requirements.txt
```

## Deletion

Delete the cache in

```
~/.cache/huggingface/hub
```
