# Mathematics of the Updraft-LM Architecture

Updraft-LM version 2.0.0 utilizes advanced architectural components introduced by LLaMA and other modern Transformer variants. This document provides formal derivations for these components.

## 1. RMSNorm (Root Mean Square Normalization)

Unlike standard LayerNorm which computes both mean and variance, RMSNorm normalizes activations strictly by their root mean square, which reduces computational overhead without sacrificing performance.

Given an input vector $x \in \mathbb{R}^d$, the output $y$ is computed as:

$$y = \frac{x}{\text{RMS}(x)} \odot \gamma$$

Where the Root Mean Square is defined as:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

And $\gamma \in \mathbb{R}^d$ is a learnable scaling parameter. There is no bias term, nor is there a mean-subtraction step, differentiating it from traditional Layer Normalization.

## 2. SwiGLU Activation

The traditional Feed-Forward Network (FFN) uses a sequence of two linear layers separated by a ReLU or GELU activation. Updraft-LM employs the SwiGLU formulation, an improvement introduced by the GLU Variants paper, which increases the expressive capacity of the model.

For an input $x$, SwiGLU uses three learnable weight matrices $W_1, W_2, W_3$:

$$\text{FFN}_{SwiGLU}(x) = (\text{Swish}_{1}(x W_1) \odot (x W_3)) W_2$$

The Swish function (specifically with parameter $\beta=1$, making it equivalent to SiLU) is defined as:

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

The intermediate hidden dimension is typically adjusted (e.g., to $\frac{8}{3}d$) to preserve parameter count equality with standard FFNs.

## 3. Rotary Positional Embeddings (RoPE)

Instead of supplementing token embeddings with absolute positional vectors, RoPE encodes absolute positions with a rotation matrix and directly embeds relative positional information into the self-attention process.

Let $x_q$ and $x_k$ be the query and key vectors for an attention head, each of dimension $d$. We consider them as a set of $d/2$ pairs of complex numbers. The rotation for position $m$ is applied as follows:

$$f_q(x_m, m) = (x_{m,1} e^{im\theta_1}, x_{m,2} e^{im\theta_2}, ..., x_{m,d/2} e^{im\theta_{d/2}})$$
$$f_k(x_n, n) = (x_{n,1} e^{in\theta_1}, x_{n,2} e^{in\theta_2}, ..., x_{n,d/2} e^{in\theta_{d/2}})$$

Where $\theta_i = \theta_{base}^{-2i/d}$, and typically $\theta_{base} = 10000$.

When the dot product corresponding to the attention score between query at position $m$ and key at position $n$ is computed, the relative distance $(m - n)$ naturally factors in:

$$\langle f_q(x_m, m), f_k(x_n, n) \rangle = \dots = \text{Re}\left( \sum_{j=1}^{d/2} (x_{m,j} e^{im\theta_j}) (x_{n,j} e^{-in\theta_j}) \right)$$
$$= \text{Re}\left( \sum x_{m,j} x_{n,j} e^{i(m-n)\theta_j} \right)$$

This demonstrates that the attention score strictly depends on the relative positional offset.

## 4. Grouped-Query Attention (GQA)

Updraft-LM uses Grouped-Query Attention to optimize the memory bandwidth bottleneck during autoregressive decoding.

Instead of a single Key-Value (KV) head (Multi-Query Attention) or $N$ parallel KV heads (Multi-Head Attention), GQA allocates $G$ KV heads, where $1 < G < N$.

Queries are divided into $G$ groups, where each group of $\frac{N}{G}$ query heads shares a single KV pair. This dramatically decreases the dimension of the Key-Value (KV) cache, yielding a generation speedup comparable to MQA, while maintaining linguistic capability comparable to full MHA.
