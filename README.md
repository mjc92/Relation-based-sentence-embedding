# Attentive Relations Network

## Description

A joint effort between Kenta Iwasaki and Minje Choi on creating self-attentive sentence embeddings using relation networks with multi-head attention.

The model works by computing a relations map between permutations of words in a sentence, and applying inter-attention on every single feature vector within this relations matrix.

Cosine similarity is also provided as a feature within the relations map.

All implementations are done within PyTorch.

## Datasets

- bAbI
- Normal <-> Simple Wikipedia

## Citations

Vaswani, Ashish, et al. "Attention Is All You Need." arXiv preprint arXiv:1706.03762 (2017).

Santoro, Adam, et al. "A simple neural network module for relational reasoning." arXiv preprint arXiv:1706.01427 (2017).