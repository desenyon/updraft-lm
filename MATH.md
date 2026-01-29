# Mathematical Foundations of Updraft-LM

## 1. Transformer Architecture

### 1.1 Multi-Head Self-Attention

The self-attention mechanism computes attention scores between all pairs of positions in a sequence.

**Scaled Dot-Product Attention:**

Given input sequence $X \in \mathbb{R}^{n \times d_{model}}$, we compute:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}$ are learnable weight matrices.

The attention output is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing too large.

**Multi-Head Attention:**

Instead of computing a single attention function, multi-head attention computes $h$ attention functions in parallel:

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d_{model} \times d_k}$ and $W^O \in \mathbb{R}^{hd_k \times d_{model}}$.

In our implementation:
- $d_{model} = 768$
- $h = 12$ (number of heads)
- $d_k = d_{model} / h = 64$

### 1.2 Position-wise Feed-Forward Networks

Each transformer block contains a fully connected feed-forward network applied independently to each position:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

where:
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- $d_{ff} = 3072$ (typically $4 \times d_{model}$)

**GELU Activation:**

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

### 1.3 Layer Normalization

Layer normalization normalizes across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{d_{model}}\sum_{i=1}^{d_{model}} x_i$ (mean)
- $\sigma^2 = \frac{1}{d_{model}}\sum_{i=1}^{d_{model}} (x_i - \mu)^2$ (variance)
- $\gamma, \beta$ are learnable parameters
- $\epsilon = 10^{-5}$ for numerical stability

### 1.4 Residual Connections

Each sub-layer uses a residual connection followed by layer normalization:

$$\text{Output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

This formulation (pre-norm) has been shown to be more stable than the original post-norm variant.

## 2. Positional Encoding

Since self-attention has no notion of position, we add positional encodings to the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

where:
- $pos$ is the position in the sequence
- $i$ is the dimension index
- $d_{model} = 768$

## 3. Full Transformer Block

A complete transformer block combines all components:

$$\begin{align}
z^{(\ell)} &= \text{LayerNorm}(x^{(\ell-1)}) \\
x^{(\ell)} &= x^{(\ell-1)} + \text{MultiHead}(z^{(\ell)}) \\
y^{(\ell)} &= \text{LayerNorm}(x^{(\ell)}) \\
h^{(\ell)} &= x^{(\ell)} + \text{FFN}(y^{(\ell)})
\end{align}$$

## 4. GPT-1 Model Architecture

### 4.1 Input Embedding

The input tokens are embedded into continuous vectors:

$$E = \text{Embedding}(X) + \text{PositionalEncoding}$$

where $E \in \mathbb{R}^{n \times d_{model}}$ and $X \in \mathbb{Z}^n$ are the input token IDs.

### 4.2 Causal Masking

For autoregressive generation, we apply a causal mask to prevent attention to future positions:

$$M_{ij} = \begin{cases} 
0 & \text{if } i < j \\
1 & \text{if } i \geq j
\end{cases}$$

The masked attention becomes:

$$\text{Attention}_{\text{masked}} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where positions with $M_{ij} = 0$ are set to $-\infty$ before softmax.

### 4.3 Output Layer

After $L = 12$ transformer blocks, we apply final layer normalization and project to vocabulary:

$$\text{logits} = \text{LayerNorm}(h^{(L)})W_{vocab}$$

where $W_{vocab} \in \mathbb{R}^{d_{model} \times V}$ and $V = 50257$ is the vocabulary size.

### 4.4 Loss Function

The model is trained with cross-entropy loss:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log P(x_i | x_{<i})$$

where $P(x_i | x_{<i})$ is computed by applying softmax to the logits:

$$P(x_i | x_{<i}) = \frac{\exp(\text{logit}_i)}{\sum_{j=1}^{V}\exp(\text{logit}_j)}$$

## 5. Training Procedure

### 5.1 Optimizer: AdamW

We use AdamW optimizer with the following update rules:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_{t-1}\right)$$

where:
- $\beta_1 = 0.9, \beta_2 = 0.999$
- $\alpha$ is the learning rate (with warmup and cosine decay)
- $\lambda = 0.01$ is the weight decay coefficient
- $\epsilon = 10^{-8}$

### 5.2 Learning Rate Schedule

We use a warmup followed by cosine annealing:

$$\eta_t = \begin{cases}
\frac{t}{T_{warmup}} \cdot \eta_{max} & \text{if } t < T_{warmup} \\
0.1 \cdot \eta_{max} + 0.45 \cdot \eta_{max} \left(1 + \cos\left(\pi \frac{t - T_{warmup}}{T_{max} - T_{warmup}}\right)\right) & \text{otherwise}
\end{cases}$$

where:
- $T_{warmup} = 2000$ steps
- $\eta_{max} = 2.5 \times 10^{-4}$
- Minimum learning rate is $0.1 \times \eta_{max}$

### 5.3 Gradient Clipping

To prevent gradient explosion, we clip gradients by global norm:

$$\tilde{g} = \begin{cases}
g & \text{if } \|g\| \leq \theta \\
\theta \frac{g}{\|g\|} & \text{otherwise}
\end{cases}$$

where $\theta = 1.0$ is the clipping threshold.

## 6. Sampling Strategies

### 6.1 Temperature Sampling

Temperature controls the randomness of predictions:

$$P(x_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

- $T < 1$: More deterministic (sharper distribution)
- $T = 1$: Standard softmax
- $T > 1$: More random (flatter distribution)

### 6.2 Top-k Sampling

Only sample from the top $k$ most likely tokens:

$$P(x_i) = \begin{cases}
\frac{\exp(\text{logit}_i)}{\sum_{j \in V_k} \exp(\text{logit}_j)} & \text{if } i \in V_k \\
0 & \text{otherwise}
\end{cases}$$

where $V_k$ is the set of $k$ tokens with highest logits.

### 6.3 Nucleus (Top-p) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

$$V_p = \min\{V' : \sum_{i \in V'} P(x_i) \geq p\}$$

where tokens are ordered by probability.

## 7. Model Statistics

**Total Parameters:** ~117M (comparable to GPT-1)

**Architecture:**
- Layers: 12
- Attention heads: 12
- Embedding dimension: 768
- FFN dimension: 3072
- Max sequence length: 512
- Vocabulary size: 50,257

**Memory Requirements:**
- Model parameters: ~468 MB (FP32)
- Training batch (64 × 512): ~96 MB
- Optimizer states: ~936 MB
- Total training memory: ~1.5 GB (excluding activations)

## 8. Computational Complexity

### Time Complexity per Layer:

- **Self-Attention:** $O(n^2 \cdot d_{model})$
- **Feed-Forward:** $O(n \cdot d_{model} \cdot d_{ff})$
- **Total per layer:** $O(n^2 \cdot d_{model} + n \cdot d_{model} \cdot d_{ff})$

For the full model with $L = 12$ layers:
$$O(L \cdot (n^2 \cdot d_{model} + n \cdot d_{model} \cdot d_{ff}))$$

With $n = 512, d_{model} = 768, d_{ff} = 3072, L = 12$:
- Self-attention: ~2.4B operations
- Feed-forward: ~9.5B operations
- Total: ~143B FLOPs per forward pass
