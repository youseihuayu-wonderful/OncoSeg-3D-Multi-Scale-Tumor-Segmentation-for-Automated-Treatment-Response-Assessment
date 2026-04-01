# OncoSeg — AI & Technical Knowledge Fundamentals

> A comprehensive reference of every AI, mathematical, and domain-specific concept used in the OncoSeg project.

---

## Table of Contents

1. [Mathematics Foundations](#1-mathematics-foundations)
   - 1.1 Linear Algebra
   - 1.2 Calculus
   - 1.3 Probability & Statistics
   - 1.4 Information Theory
2. [Deep Learning Fundamentals](#2-deep-learning-fundamentals)
   - 2.1 Neural Network Basics
   - 2.2 Activation Functions
   - 2.3 Weight Initialization
   - 2.4 Regularization Techniques
   - 2.5 Normalization Techniques
3. [Optimization](#3-optimization)
   - 3.1 Gradient Descent Variants
   - 3.2 Adam & AdamW Optimizer
   - 3.3 Learning Rate Scheduling
   - 3.4 Gradient Clipping
4. [Convolutional Neural Networks (CNN)](#4-convolutional-neural-networks-cnn)
   - 4.1 Convolution Operation (3D)
   - 4.2 Transposed Convolution
   - 4.3 Receptive Field
   - 4.4 Feature Maps & Channels
   - 4.5 Downsampling & Upsampling
5. [U-Net Architecture](#5-u-net-architecture)
   - 5.1 Encoder-Decoder Design
   - 5.2 Skip Connections
   - 5.3 3D U-Net Extension
6. [Transformer & Attention Mechanism](#6-transformer--attention-mechanism)
   - 6.1 Self-Attention
   - 6.2 Scaled Dot-Product Attention
   - 6.3 Multi-Head Attention
   - 6.4 Cross-Attention
   - 6.5 Positional Encoding
   - 6.6 Swin Transformer
7. [Loss Functions](#7-loss-functions)
   - 7.1 Cross-Entropy Loss
   - 7.2 Dice Loss
   - 7.3 Combined DiceCE Loss
   - 7.4 Deep Supervision Loss
8. [Evaluation Metrics](#8-evaluation-metrics)
   - 8.1 Dice Score (DSC)
   - 8.2 Hausdorff Distance 95%
   - 8.3 Average Surface Distance
   - 8.4 Sensitivity & Specificity
9. [Uncertainty Quantification](#9-uncertainty-quantification)
   - 9.1 Monte Carlo Dropout
   - 9.2 Predictive Entropy
   - 9.3 Bayesian Deep Learning
10. [Medical Image Processing](#10-medical-image-processing)
    - 10.1 NIfTI Format
    - 10.2 Voxel Spacing & Affine Matrix
    - 10.3 Orientation Standardization
    - 10.4 Intensity Normalization
    - 10.5 Sliding Window Inference
11. [Data Augmentation](#11-data-augmentation)
    - 11.1 Spatial Augmentations
    - 11.2 Intensity Augmentations
    - 11.3 Why Augmentation Matters
12. [Clinical Knowledge (RECIST 1.1)](#12-clinical-knowledge-recist-11)
    - 12.1 Tumor Measurement Standards
    - 12.2 Response Categories
    - 12.3 Connected Component Analysis
13. [Software Engineering & Experiment Management](#13-software-engineering--experiment-management)
14. [Knowledge Coverage Map](#14-knowledge-coverage-map)

---

## 1. Mathematics Foundations

### 1.1 Linear Algebra

**Tensors**

The fundamental data structure in deep learning. In OncoSeg, data flows as 5D tensors:

```
[Batch, Channels, Height, Width, Depth]

Example: [2, 4, 128, 128, 128]
         2 samples, 4 MRI modalities, 128³ voxel volume
```

A scalar is a 0D tensor, a vector is 1D, a matrix is 2D, a 3D volume is 3D, a multi-channel volume is 4D, and a batch of multi-channel volumes is 5D — this is what we work with.

**Matrix Multiplication**

Every neural network layer performs matrix multiplication at its core:

```
y = W · x + b

Where:
  x = input features    (shape: [in_features])
  W = weight matrix      (shape: [out_features, in_features])
  b = bias vector        (shape: [out_features])
  y = output features    (shape: [out_features])
```

In our Transformer's attention mechanism:
```
Q = X · W_Q    (project input to queries)
K = X · W_K    (project input to keys)
V = X · W_V    (project input to values)
```

Each is a matrix multiplication that transforms the input into different representational spaces.

**Dot Product**

The attention score between two positions is computed via dot product:

```
score(q, k) = q · k = Σ(q_i × k_i)

A large dot product means the query and key are similar (aligned),
so that position receives high attention weight.
```

**Transpose**

Used in attention: `Q · K^T` computes pairwise similarity between all query-key pairs. The transpose flips rows and columns so the matrix dimensions align for multiplication:

```
Q: [seq_len, d_k]
K^T: [d_k, seq_len]
Q · K^T: [seq_len, seq_len]  ← attention score matrix
```

**Eigenvalues (indirect use)**

Weight initialization methods (Kaiming, Xavier) are designed based on the spectral properties of random matrices to ensure that signal variance is preserved through layers — preventing vanishing or exploding activations.

---

### 1.2 Calculus

**Partial Derivatives**

Neural network training requires computing the gradient of the loss with respect to every parameter:

```
∂L/∂w_ij = how much does the loss change if we slightly change weight w_ij?
```

For a network with millions of parameters, we need millions of partial derivatives computed efficiently.

**Chain Rule**

The backbone of backpropagation. For composed functions:

```
If L = f(g(h(x))), then:

∂L/∂x = (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

In a deep network, the loss is a composition of dozens of layers. The chain rule lets us decompose the gradient computation layer by layer, propagating from the output back to the input.

Example in OncoSeg:
```
Loss ← Softmax ← Conv3D ← LeakyReLU ← ConvTranspose3D ← CrossAttention ← SwinTransformer ← PatchEmbed ← Input

Backprop applies the chain rule through each of these operations in reverse.
```

**Gradient Descent**

The fundamental optimization algorithm:

```
w_new = w_old - learning_rate × ∂L/∂w

Move each weight in the direction that reduces the loss,
scaled by the learning rate.
```

**Gradient Clipping**

In OncoSeg, we clip gradients to prevent training instability:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

If ||∇L|| > 1.0:
    ∇L = ∇L × (1.0 / ||∇L||)    # Scale down to unit norm
```

This prevents a single bad batch from causing an extremely large parameter update that destabilizes training.

---

### 1.3 Probability & Statistics

**Softmax Function**

Converts raw network outputs (logits) into a valid probability distribution:

```
P(class_i) = exp(z_i) / Σ_j exp(z_j)

Properties:
  - All outputs are in (0, 1)
  - Outputs sum to 1
  - Preserves ordering (largest logit → highest probability)

In OncoSeg: applied to each voxel independently
  Input:  logits [B, 4, H, W, D]  (4 classes: bg, NCR, edema, ET)
  Output: probs  [B, 4, H, W, D]  (probabilities per class)
```

**Cross-Entropy**

Measures the "distance" between two probability distributions:

```
H(p, q) = -Σ p(x) × log(q(x))

Where:
  p = true distribution (one-hot label)
  q = predicted distribution (softmax output)

For a single voxel with true class k:
  CE = -log(q_k)

If the model predicts class k with probability 0.95:
  CE = -log(0.95) = 0.05  (low loss — good prediction)

If the model predicts class k with probability 0.01:
  CE = -log(0.01) = 4.6   (high loss — bad prediction)
```

**Bayesian Inference (MC Dropout)**

Classical (frequentist) neural networks give a single point prediction. Bayesian deep learning estimates a distribution over predictions:

```
p(y|x, D) = ∫ p(y|x, w) × p(w|D) dw

This integral is intractable for neural networks.

MC Dropout approximation:
  1. Keep dropout active at test time
  2. Run N forward passes → N different predictions
  3. Mean of predictions ≈ expected output
  4. Variance of predictions ≈ model uncertainty
```

In OncoSeg, we apply dropout to the bottleneck features N times and compute the predictive entropy of the averaged softmax outputs.

**Statistical Testing**

When comparing models (OncoSeg vs baselines), we need to know if differences are statistically significant:

```
Wilcoxon Signed-Rank Test:
  - Non-parametric paired test
  - H₀: OncoSeg and baseline have the same median Dice score
  - If p < 0.05: the difference is statistically significant
  - Applied per-subject across the test set
```

Multi-seed evaluation: train with seeds {42, 123, 456}, report mean ± std to show robustness.

---

### 1.4 Information Theory

**Entropy**

Measures uncertainty in a probability distribution:

```
H(p) = -Σ p_i × log(p_i)

Maximum entropy: uniform distribution (most uncertain)
  H([0.25, 0.25, 0.25, 0.25]) = log(4) = 1.39

Minimum entropy: deterministic (no uncertainty)
  H([1.0, 0.0, 0.0, 0.0]) = 0

In OncoSeg: we compute the predictive entropy at each voxel to create
an uncertainty map. High-entropy voxels are where the model is unsure
— typically at tumor boundaries.
```

**Cross-Entropy Loss (Information-Theoretic View)**

```
CE(p, q) = H(p) + D_KL(p || q)

Where:
  H(p) = entropy of true distribution (constant for one-hot labels = 0)
  D_KL = Kullback-Leibler divergence (how different q is from p)

Minimizing CE ≡ minimizing KL divergence ≡ making predictions match truth
```

---

## 2. Deep Learning Fundamentals

### 2.1 Neural Network Basics

A neural network is a composition of parameterized functions (layers):

```
f(x) = f_L(f_{L-1}(...f_2(f_1(x))))

Each layer: y = activation(W · x + b)
  - W, b = learnable parameters
  - activation = non-linear function

Training: adjust W, b to minimize loss on training data
  1. Forward pass: compute predictions
  2. Loss computation: compare predictions to ground truth
  3. Backward pass: compute gradients via backpropagation
  4. Parameter update: w = w - lr × gradient
```

OncoSeg has approximately 25 million trainable parameters spread across the Swin Transformer encoder, cross-attention skip connections, and CNN decoder.

---

### 2.2 Activation Functions

Non-linear functions applied after linear transformations — without these, the entire network would collapse to a single linear function.

**LeakyReLU (used in CNN decoder)**
```
LeakyReLU(x) = x        if x > 0
             = 0.01x    if x ≤ 0

- Allows small gradient for negative inputs (prevents "dead neurons")
- Computationally efficient
- Used in: CNNDecoder3D
```

**GELU (used in Swin Transformer)**
```
GELU(x) = x × Φ(x)    where Φ is the standard normal CDF

- Smoother than ReLU
- Probabilistic interpretation: scale x by the probability that x is positive
- Standard activation in Transformers
- Used in: Swin Transformer's MLP blocks
```

**Softmax (output layer)**
```
Softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

- Converts logits to probability distribution
- Used as the final activation for classification
- Applied independently per voxel
```

---

### 2.3 Weight Initialization

How parameters are initialized significantly affects training convergence:

**Kaiming Initialization (for ReLU/LeakyReLU layers)**
```
W ~ Normal(0, √(2 / fan_in))

- fan_in = number of input connections
- Designed to preserve variance through ReLU layers
- Used for: Conv3d, ConvTranspose3d layers
```

**Truncated Normal (for Transformer layers)**
```
W ~ TruncatedNormal(0, 0.02)

- Values beyond 2σ are resampled
- Prevents extreme initial weights
- Used for: attention projections, embeddings
```

---

### 2.4 Regularization Techniques

Prevent overfitting — when the model memorizes training data instead of learning general patterns.

**Dropout**
```python
Dropout3d(p=0.1)

During training: randomly set 10% of feature channels to zero
During inference: use all channels (or multiple stochastic passes for MC Dropout)

Why it works:
  - Forces redundancy — no single channel can be relied upon
  - Approximate ensemble of 2^N sub-networks
  - In OncoSeg: applied to bottleneck features for uncertainty estimation
```

**Weight Decay (L2 Regularization)**
```
L_total = L_task + λ × Σ w²

- Penalizes large weights, encouraging simpler models
- AdamW implements "decoupled" weight decay:
  w = w × (1 - lr × λ) - lr × gradient
- λ = 1e-5 in OncoSeg
```

**Data Augmentation**

The most effective regularizer for medical imaging (see Section 11). Artificially increases training set diversity.

---

### 2.5 Normalization Techniques

Stabilize training by normalizing intermediate activations.

**Instance Normalization (used in CNN decoder)**
```
For each sample and each channel independently:

InstanceNorm(x) = (x - μ) / √(σ² + ε) × γ + β

Where:
  μ, σ² = mean, variance computed over spatial dims (H, W, D) only
  γ, β = learnable affine parameters
  ε = small constant for numerical stability

Why Instance Norm (not Batch Norm)?
  - Medical imaging: batch size is small (1-2) due to large 3D volumes
  - Batch Norm statistics are unreliable with small batches
  - Instance Norm is independent of batch size
```

**Layer Normalization (used in Swin Transformer)**
```
LayerNorm(x) = (x - μ) / √(σ² + ε) × γ + β

Where:
  μ, σ² = computed over the feature/channel dimension
  
- Standard in Transformers
- Normalizes across features, not spatial dimensions
- Stable regardless of batch size
```

---

## 3. Optimization

### 3.1 Gradient Descent Variants

**Stochastic Gradient Descent (SGD)**
```
w = w - lr × ∇L(w; x_batch)

- Compute gradient on a mini-batch, not entire dataset
- Noisy gradients help escape local minima
- Foundation — all modern optimizers build on this
```

**Momentum**
```
v = β × v + ∇L(w)
w = w - lr × v

- Accumulates past gradients to smooth updates
- Accelerates convergence in consistent gradient directions
- β = 0.9 is standard (90% of previous velocity retained)
```

---

### 3.2 Adam & AdamW Optimizer

**Adam (Adaptive Moment Estimation)**
```
Maintains two moving averages per parameter:

m = β₁ × m + (1 - β₁) × g          (1st moment: mean of gradients)
v = β₂ × v + (1 - β₂) × g²         (2nd moment: mean of squared gradients)

m̂ = m / (1 - β₁^t)                   (bias correction)
v̂ = v / (1 - β₂^t)                   (bias correction)

w = w - lr × m̂ / (√v̂ + ε)           (update)

- Adaptive learning rate per parameter
- Parameters with large gradients get smaller updates (and vice versa)
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (standard)
```

**AdamW (used in OncoSeg)**
```
Adam + Decoupled Weight Decay:

w = w × (1 - lr × λ) - lr × m̂ / (√v̂ + ε)

- Standard Adam applies weight decay inside the adaptive step (incorrect)
- AdamW applies it separately (mathematically correct L2 regularization)
- λ = 1e-5 in OncoSeg
```

---

### 3.3 Learning Rate Scheduling

**Cosine Annealing (used in OncoSeg)**
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))

Where:
  lr_max = 1e-4  (initial learning rate)
  lr_min = 1e-6  (minimum learning rate)
  T = 300        (total epochs)
  t = current epoch

Epoch   1: lr = 1.0e-4  (start high — explore aggressively)
Epoch  75: lr = 8.5e-5  (gentle decay)
Epoch 150: lr = 5.1e-5  (midpoint)
Epoch 225: lr = 1.6e-5  (fine-tuning)
Epoch 300: lr = 1.0e-6  (converge precisely)

Why cosine?
  - Smooth, gradual decay (no sudden drops)
  - Spends more time at low learning rates for fine-grained convergence
  - Empirically outperforms step decay for Transformer training
```

---

### 3.4 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

If the total gradient norm exceeds 1.0:
  all_gradients *= (1.0 / total_norm)

Why?
  - Transformer attention can produce extreme gradients
  - 3D medical volumes have high variance across subjects
  - One bad sample shouldn't crash training
  - Especially important with deep supervision (multiple loss sources)
```

---

## 4. Convolutional Neural Networks (CNN)

### 4.1 Convolution Operation (3D)

```
Standard 2D convolution slides a kernel over a 2D image.
3D convolution slides a 3D kernel over a 3D volume.

Conv3d(in_channels=96, out_channels=192, kernel_size=3, padding=1)

For each output position (x, y, z):
  output[x,y,z] = Σ over kernel Σ over in_channels (
      input[x+i, y+j, z+k, c] × kernel[i, j, k, c]
  ) + bias

Properties:
  - Parameter sharing: same kernel applied everywhere (translation equivariance)
  - Local connectivity: each output depends only on a local neighborhood
  - 3D captures spatial relationships in all three dimensions

In OncoSeg: used in the CNN decoder for upsampling and feature refinement
```

---

### 4.2 Transposed Convolution

```
Also called "deconvolution" (technically incorrect) — learns to upsample.

ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=2, stride=2)

Regular convolution:  input [96, 32, 32, 32] → output [192, 16, 16, 16]  (downsample)
Transposed convolution: input [192, 16, 16, 16] → output [96, 32, 32, 32]  (upsample)

How it works:
  - Insert zeros between input elements (stride > 1 creates gaps)
  - Apply convolution on the expanded input
  - Result: output is larger than input

In OncoSeg: used in the CNN decoder to progressively upsample from 8³ back to 128³
```

---

### 4.3 Receptive Field

```
The region of the input that influences a given output position.

Layer 1 (3×3×3 conv): receptive field = 3×3×3
Layer 2 (3×3×3 conv): receptive field = 5×5×5
Layer 3 (3×3×3 conv): receptive field = 7×7×7
...with pooling/stride, receptive field grows even faster

Why this matters:
  - Small receptive field → captures local texture (tumor edges)
  - Large receptive field → captures global context (is this region brain or tumor?)
  - OncoSeg uses Swin Transformer for global context + CNN for local precision
```

---

### 4.4 Feature Maps & Channels

```
Each convolutional layer produces multiple feature maps (channels).

In OncoSeg encoder:
  Stage 1: 48 channels   (low-level: edges, textures)
  Stage 2: 96 channels   (mid-level: tumor boundaries)
  Stage 3: 192 channels  (high-level: tumor regions)
  Stage 4: 384 channels  (abstract: tumor type classification)

More channels = richer representation but more computation
Channel doubling at each stage is a standard design pattern
```

---

### 4.5 Downsampling & Upsampling

```
Encoder (downsampling):
  128³ → 64³ → 32³ → 16³ → 8³
  Each step: spatial resolution halved, channels doubled
  Purpose: compress spatial info into semantic features

Decoder (upsampling):
  8³ → 16³ → 32³ → 64³ → 128³
  Each step: spatial resolution doubled, channels halved
  Purpose: restore spatial resolution for pixel-level prediction

The encoder answers "what is this?" (tumor vs normal)
The decoder answers "where exactly?" (precise boundaries)
```

---

## 5. U-Net Architecture

### 5.1 Encoder-Decoder Design

```
The U-Net is the most influential architecture in medical image segmentation (Ronneberger et al., 2015).

Shape of the data flow forms a "U":

    128³ ──[Enc1]──→ 64³ ──[Enc2]──→ 32³ ──[Enc3]──→ 16³ ──[Enc4]──→ 8³
      │                │                │                │          (Bottleneck)
      │                │                │                │              │
    128³ ←──[Dec1]── 64³ ←──[Dec2]── 32³ ←──[Dec3]── 16³ ←──[Dec4]──┘
      │
    Output: segmentation map
```

---

### 5.2 Skip Connections

```
The critical innovation of U-Net: connect encoder to decoder at matching resolutions.

Without skip connections:
  - Decoder must reconstruct spatial detail from abstract bottleneck features
  - Fine boundaries are lost through downsampling

With skip connections:
  - Encoder passes high-resolution spatial features directly to decoder
  - Decoder combines: abstract semantics (from bottleneck) + spatial detail (from encoder)

Standard U-Net: concatenation skip
  decoder_input = concat(encoder_features, upsampled_decoder_features)

OncoSeg innovation: cross-attention skip
  decoder_input = CrossAttention(Q=decoder_features, K=encoder_features, V=encoder_features)
  
  The decoder SELECTIVELY attends to relevant encoder information,
  rather than receiving everything through concatenation.
```

---

### 5.3 3D U-Net Extension

```
Original U-Net: 2D (single image slices)
3D U-Net: processes entire volumes — captures inter-slice relationships

Why 3D matters for brain tumors:
  - Tumors are 3D structures, not flat
  - Slice-by-slice (2D) segmentation has discontinuities between slices
  - 3D convolutions model spatial continuity in all directions
  
Tradeoff: 3D is ~10x more memory-intensive than 2D
  → Requires smaller input sizes (128³ instead of 512²)
  → Requires sliding window inference for full-resolution volumes
```

---

## 6. Transformer & Attention Mechanism

### 6.1 Self-Attention

```
Self-attention allows every position to "look at" every other position
in the input sequence:

Input: X = [x₁, x₂, ..., x_n]  (n positions, each a feature vector)

For each position i:
  "How much should I pay attention to position j?"
  attention_weight(i,j) = softmax(q_i · k_j / √d_k)
  
  "What information should I gather?"
  output_i = Σ_j attention_weight(i,j) × v_j

This captures long-range dependencies:
  - CNNs: each position sees only a local neighborhood
  - Self-attention: each position sees the entire input
  - A voxel on one side of a tumor can directly attend to the other side
```

---

### 6.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Step by step:
  1. Q · K^T        → raw similarity scores [n × n matrix]
  2. / √d_k         → scale to prevent softmax saturation
  3. softmax(...)    → normalize to probability distribution (rows sum to 1)
  4. ... · V         → weighted combination of value vectors

Why scale by √d_k?
  Without scaling, dot products grow with dimension d_k.
  Large values → softmax outputs approach one-hot → gradients vanish.
  Scaling keeps softmax in a well-behaved range.
  
  d_k = 16 (for 48-dim embeddings with 3 heads): scale = √16 = 4
```

---

### 6.3 Multi-Head Attention

```
Instead of one attention function, use H parallel attention heads:

MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_H) · W_O

Where each head:
  head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)

In OncoSeg (stage-dependent):
  Stage 1: 3 heads  (48-dim / 3 = 16-dim per head)
  Stage 2: 6 heads  (96-dim / 6 = 16-dim per head)
  Stage 3: 12 heads (192-dim / 12 = 16-dim per head)
  Stage 4: 24 heads (384-dim / 24 = 16-dim per head)

Why multiple heads?
  - Each head can learn a different attention pattern
  - Head 1: attend to similar intensity regions
  - Head 2: attend to nearby spatial positions
  - Head 3: attend to tumor boundary features
  - The combination captures richer relationships than a single attention
```

---

### 6.4 Cross-Attention

```
Self-attention: Q, K, V all come from the SAME input
Cross-attention: Q comes from one source, K and V from another

In OncoSeg skip connections:
  Q = decoder features  ("I'm looking for information to help me upsample")
  K = encoder features   ("Here's what I contain")
  V = encoder features   ("Here's the actual information")

CrossAttention(Q_decoder, K_encoder, V_encoder)
  = "Decoder selectively retrieves relevant encoder features"

Why this is better than concatenation:
  Concatenation: decoder receives ALL encoder info, must learn to filter
  Cross-attention: decoder queries for SPECIFIC information it needs
  
  Result: more efficient feature fusion, better segmentation boundaries
```

---

### 6.5 Positional Encoding

```
Attention is permutation-invariant — it doesn't know spatial position.
Positional encoding injects spatial information.

Standard Transformer: absolute sinusoidal or learned position embeddings
Swin Transformer: relative position bias

  Attention(Q, K) = softmax((Q · K^T + B) / √d_k) · V
  
  B = learnable relative position bias table
  B[i,j] depends on the relative spatial offset between positions i and j

Why relative > absolute for 3D medical imaging?
  - Translational generalization: tumor at position (10,20,30)
    should be processed the same as at (50,60,70)
  - Relative positions capture "how far apart" without "where exactly"
```

---

### 6.6 Swin Transformer

```
Full name: Shifted Window Transformer (Liu et al., 2021)

Problem: Standard self-attention has O(n²) complexity.
  For a 128³ volume: n = 128³ = 2,097,152 positions
  n² = 4.4 trillion operations — completely infeasible!

Solution: Window-based local attention

Step 1: Partition volume into non-overlapping 7×7×7 windows
  128³ / 7³ ≈ 6,100 windows, each with 343 positions
  Attention within each window: 343² = 117,649 operations (feasible!)

Step 2: Shifted windows
  Problem: windows can't communicate with each other
  Solution: shift the window partition by (3,3,3) in alternate layers
  
  Layer 1: [Regular windows]  → attention within windows
  Layer 2: [Shifted windows]  → attention crosses original boundaries
  Layer 3: [Regular windows]  → and so on...

Step 3: Hierarchical stages (patch merging)
  Stage 1: 128/4 = 32³ tokens, 48-dim     → local features
  Stage 2: 32/2 = 16³ tokens, 96-dim      → regional features  
  Stage 3: 16/2 = 8³ tokens, 192-dim      → global features
  Stage 4: 8/2 = 4³ tokens, 384-dim       → abstract features

Patch merging: concatenate 2×2×2 neighboring tokens, project to halve channels
  Analogous to pooling in CNNs, but learned
```

---

## 7. Loss Functions

### 7.1 Cross-Entropy Loss

```
CE = -Σ_c y_c × log(p_c)

For a single voxel with true class k (one-hot encoded y):
  CE = -log(p_k)

Measures: "how surprised is the model by the true label?"
  p_k = 0.99 → CE = 0.01  (confident and correct → low loss)
  p_k = 0.50 → CE = 0.69  (uncertain → moderate loss)
  p_k = 0.01 → CE = 4.60  (confident and wrong → high loss)

Properties:
  + Smooth, well-defined gradients everywhere
  + Strong gradient signal for incorrect predictions
  - Treats all voxels equally (problematic when 98% are background)
```

---

### 7.2 Dice Loss

```
Dice Score = 2 × |Pred ∩ Truth| / (|Pred| + |Truth|)

Dice Loss = 1 - Dice Score

Differentiable (soft) version using probabilities:
  Dice = 2 × Σ(p_i × y_i) / (Σp_i² + Σy_i²)

Where:
  p_i = predicted probability at voxel i
  y_i = ground truth label at voxel i

Why Dice is essential for medical imaging:
  In a brain MRI with 128³ = 2M voxels:
    Background: ~1,950,000 voxels (97.5%)
    Tumor:      ~50,000 voxels    (2.5%)
  
  Cross-entropy: a model predicting "all background" achieves 97.5% accuracy!
  Dice loss: same model gets Dice = 0 (no overlap with tumor)
  
  Dice directly measures overlap, so it forces the model to find the tumor
  regardless of class imbalance.
```

---

### 7.3 Combined DiceCE Loss

```python
DiceCELoss = 0.5 × DiceLoss + 0.5 × CrossEntropyLoss

Why combine?
  Dice Loss alone:
    + Handles class imbalance perfectly
    - Noisy gradients when predictions are near-zero (early training)
    
  CE Loss alone:
    + Smooth, stable gradients
    - Dominated by the majority class (background)
    
  Together:
    + Dice handles imbalance
    + CE provides stable training signal
    = Better convergence than either alone (empirically proven in nnU-Net, BraTS winners)
```

---

### 7.4 Deep Supervision Loss

```
Standard training: loss computed only at the final output

Deep supervision: additional losses at intermediate decoder stages

L_total = L_main + w₁ × L_aux1 + w₂ × L_aux2

In OncoSeg:
  L_main: segmentation loss at full resolution (128³)
  L_aux1: segmentation loss at 64³ (upsampled for comparison)
  L_aux2: segmentation loss at 32³ (upsampled for comparison)
  
  Weights: w_i = 0.5^i (geometrically decaying)
    w₁ = 0.5, w₂ = 0.25

Why this helps:
  - Provides gradient signal directly to intermediate layers
  - Without it: gradients must flow through many layers → vanishing gradients
  - Forces intermediate features to be semantically meaningful
  - Proven to improve convergence in deep segmentation networks
```

---

## 8. Evaluation Metrics

### 8.1 Dice Score (DSC)

```
Dice(P, G) = 2 × |P ∩ G| / (|P| + |G|)

Range: 0 (no overlap) to 1 (perfect overlap)
Equivalent to F1-score for segmentation

In OncoSeg, computed for three BraTS evaluation regions:
  Enhancing Tumor (ET): label 3 only
  Tumor Core (TC):      labels 1 + 3
  Whole Tumor (WT):     labels 1 + 2 + 3

Example:
  Prediction: 5000 voxels predicted as ET
  Truth:      4500 voxels are actually ET
  Overlap:    4000 voxels correct
  
  Dice = 2 × 4000 / (5000 + 4500) = 8000 / 9500 = 0.842
```

---

### 8.2 Hausdorff Distance 95% (HD95)

```
Measures the worst-case boundary error (in mm).

HD(P, G) = max(max_p∈∂P min_g∈∂G d(p,g),  max_g∈∂G min_p∈∂P d(g,p))

HD95 uses the 95th percentile instead of maximum:
  - Robust to outlier predictions
  - Clinical relevance: "the boundary is within X mm for 95% of points"

Lower is better:
  HD95 = 1.0mm → boundary accurate to ~1 voxel
  HD95 = 5.0mm → some boundary regions off by 5mm (clinically concerning)
  HD95 = 20mm  → major segmentation failures
```

---

### 8.3 Average Surface Distance (ASD)

```
ASD = (1/|∂P| × Σ_p min_g d(p,g) + 1/|∂G| × Σ_g min_p d(g,p)) / 2

- Mean distance between predicted and true surfaces
- Less sensitive to outliers than Hausdorff
- Gives the "typical" boundary error in mm

Lower is better:
  ASD = 0.5mm → excellent boundary accuracy
  ASD = 2.0mm → acceptable for clinical use
  ASD = 5.0mm → needs improvement
```

---

### 8.4 Sensitivity & Specificity

```
Sensitivity (Recall / True Positive Rate):
  TP / (TP + FN) = "Of all actual tumor voxels, what fraction did we detect?"
  
  Sensitivity = 0.95 → we find 95% of the tumor (miss 5%)
  Critical for: not missing tumor → under-segmentation is dangerous

Specificity (True Negative Rate):
  TN / (TN + FP) = "Of all non-tumor voxels, what fraction did we correctly exclude?"
  
  Specificity = 0.99 → only 1% of healthy tissue wrongly marked as tumor
  Critical for: not over-treating → false positives waste clinical resources
```

---

## 9. Uncertainty Quantification

### 9.1 Monte Carlo Dropout

```
Standard inference:
  - Dropout disabled (deterministic)
  - Single forward pass → single prediction
  - No confidence information

MC Dropout inference:
  - Dropout ENABLED at test time
  - N forward passes → N different predictions (stochastic)
  - Each pass drops different features → different outputs
  
In OncoSeg (N = 10):
  Pass 1: drop features {A, C, F} → prediction_1
  Pass 2: drop features {B, D, G} → prediction_2
  ...
  Pass 10: drop features {A, E, H} → prediction_10

Final prediction = mean(prediction_1, ..., prediction_10)
Uncertainty = entropy of the mean prediction
```

---

### 9.2 Predictive Entropy

```
Given N MC Dropout samples, compute mean prediction:

  p̄(class_c | x) = (1/N) × Σ_n softmax(f_θ_n(x))_c

Predictive entropy:

  H[p̄] = -Σ_c p̄_c × log(p̄_c)

Interpretation:
  H ≈ 0:    model consistently predicts the same class → high confidence
  H ≈ log(C): model predictions disagree across samples → high uncertainty

Clinical value:
  - Tumor boundaries: high uncertainty (expected — boundaries are ambiguous)
  - Clear tumor core: low uncertainty (model is confident)
  - Artifact regions: high uncertainty (model recognizes it can't tell)
  
Uncertainty maps help radiologists focus review on unreliable regions.
```

---

### 9.3 Bayesian Deep Learning

```
Standard neural network: learns point estimate of weights W*
Bayesian neural network: learns a distribution p(W|D) over weights

Full Bayesian inference:
  p(y|x) = ∫ p(y|x,W) × p(W|D) dW    (intractable for deep networks)

MC Dropout approximation (Gal & Ghahramani, 2016):
  - Dropout at test time ≈ sampling from approximate posterior
  - Each dropout mask defines a sub-network ≈ a sample from p(W|D)
  - Average over samples ≈ Bayesian model averaging

This gives us uncertainty "for free" — no architecture changes,
just keep dropout active and run multiple forward passes.
```

---

## 10. Medical Image Processing

### 10.1 NIfTI Format

```
NIfTI (Neuroimaging Informatics Technology Initiative) — .nii / .nii.gz

Structure:
  - Header: metadata (dimensions, voxel spacing, orientation, data type)
  - Data: 3D or 4D array of voxel intensities

In OncoSeg:
  Image files: [H, W, D] or [H, W, D, 4] (4 MRI modalities)
  Label files: [H, W, D] with integer labels (0, 1, 2, 3)

Typical brain MRI: ~240 × 240 × 155 voxels ≈ 8.9M voxels per modality
```

---

### 10.2 Voxel Spacing & Affine Matrix

```
Voxels are NOT necessarily cubic. Physical size varies:

  pixdim = (1.0mm, 1.0mm, 1.0mm)  → isotropic (ideal)
  pixdim = (0.5mm, 0.5mm, 2.0mm)  → anisotropic (common in clinical CT)

Affine matrix (4×4): maps voxel indices to physical coordinates (mm)

  [x_mm]     [a₁₁ a₁₂ a₁₃ t₁] [i]
  [y_mm]  =  [a₂₁ a₂₂ a₂₃ t₂] [j]
  [z_mm]     [a₃₁ a₃₂ a₃₃ t₃] [k]
  [ 1  ]     [ 0   0   0   1 ] [1]

Includes rotation, scaling, and translation.

In OncoSeg: we resample all inputs to 1.0mm isotropic spacing
  → Consistent scale across all subjects and scanners
  → RECIST measurements in mm are physically meaningful
```

---

### 10.3 Orientation Standardization

```
MRI scanners can acquire data in different orientations.
We standardize all volumes to RAS orientation:

  R = Right    (x-axis: left → right)
  A = Anterior (y-axis: posterior → anterior)
  S = Superior (z-axis: inferior → superior)

This ensures:
  - Consistent spatial layout across all subjects
  - Augmentations (flip, rotate) have consistent meaning
  - Model doesn't need to learn orientation invariance — it's preprocessed away
```

---

### 10.4 Intensity Normalization

```
MRI intensities are arbitrary — they vary between:
  - Different scanners (GE vs Siemens vs Philips)
  - Different acquisition protocols
  - Different patients

Z-score normalization (per channel, non-zero voxels only):

  x_normalized = (x - μ_nonzero) / σ_nonzero

  Why non-zero only?
    Background (air) has intensity 0 in brain MRI.
    Including zeros would skew the mean and std.
    We only normalize the brain tissue intensities.

After normalization: mean ≈ 0, std ≈ 1 for each modality.
This allows the model to learn from intensity patterns
rather than absolute scanner-dependent values.
```

---

### 10.5 Sliding Window Inference

```
Problem: training uses 128³ crops, but test volumes are ~240×240×155

Solution: process the full volume in overlapping 128³ patches

  1. Place a 128³ window at the top-left corner
  2. Run model → get prediction for this patch
  3. Slide window by stride (overlap=0.5 → stride=64)
  4. Repeat until entire volume is covered
  5. Average predictions in overlapping regions

          ┌──────────┐
          │ Patch 1  │
          │    ┌─────┼────┐
          │    │OVER │    │
          └────┼─LAP─┘    │
               │ Patch 2  │
               └──────────┘

  Overlapping regions: use Gaussian-weighted averaging
    → Center of each patch weighted more (most reliable)
    → Edges weighted less (boundary artifacts)

This allows:
  - Training on small crops (fits in GPU memory)
  - Inference on arbitrarily large volumes
  - No resolution loss at test time
```

---

## 11. Data Augmentation

### 11.1 Spatial Augmentations

```
Random Flip (probability=0.5 per axis):
  Flip along H, W, or D axis independently
  → Model learns that tumors can appear on either hemisphere
  
  Original:    [Brain with tumor on left]
  Flipped:     [Brain with tumor on right]

Random Rotate 90° (probability=0.5):
  Rotate volume by 0°, 90°, 180°, or 270° around a random axis
  → Orientation invariance

Random Spatial Crop:
  Extract a random 128³ sub-volume from the full volume
  → Model sees different spatial contexts
  → Acts as a form of spatial dropout
```

---

### 11.2 Intensity Augmentations

```
Random Scale Intensity (±10%):
  x_aug = x × (1 + scale),  scale ~ Uniform(-0.1, 0.1)
  → Robustness to scanner intensity calibration differences

Random Shift Intensity (±10%):
  x_aug = x + offset,  offset ~ Uniform(-0.1, 0.1)
  → Robustness to brightness variations
```

---

### 11.3 Why Augmentation Matters

```
Medical imaging datasets are SMALL:
  - MSD Brain Tumor: ~484 subjects
  - Compare to ImageNet: 1.2 million images

Augmentation artificially multiplies the effective dataset size:
  Each epoch, every subject appears with different:
    - Crop position
    - Flip/rotation
    - Intensity variation
  
  484 subjects × ~100 augmentation variants = ~48,400 unique training examples

Without augmentation:
  - Model memorizes the 484 training cases
  - Fails on new patients (overfitting)

With augmentation:
  - Model learns general tumor patterns
  - Generalizes to unseen patients
```

---

## 12. Clinical Knowledge (RECIST 1.1)

### 12.1 Tumor Measurement Standards

```
RECIST 1.1 (Response Evaluation Criteria In Solid Tumors)
  — Standard clinical protocol for measuring tumor response to treatment

Measurement process:
  1. Identify all measurable lesions
  2. Select up to 5 target lesions (max 2 per organ)
  3. Measure longest diameter on axial (cross-sectional) slice
  4. Sum all target lesion diameters = "Sum of Longest Diameters" (SLD)
  5. Compare baseline SLD to follow-up SLD

In OncoSeg automation:
  1. Connected component labeling → detect individual lesions
  2. For each lesion: find axial slice with maximum tumor area
  3. On that slice: compute Feret diameter (maximum distance between boundary points)
  4. Convert from voxels to mm using voxel spacing
  5. Sum across lesions → automated SLD
```

---

### 12.2 Response Categories

```
Compare baseline SLD to follow-up SLD:

Complete Response (CR):
  All target lesions have disappeared
  SLD_followup = 0

Partial Response (PR):
  ≥30% decrease in SLD from baseline
  (SLD_followup - SLD_baseline) / SLD_baseline ≤ -0.30

Progressive Disease (PD):
  ≥20% increase in SLD from baseline, OR
  Appearance of new lesions
  (SLD_followup - SLD_baseline) / SLD_baseline ≥ +0.20

Stable Disease (SD):
  Neither PR nor PD criteria met

Clinical significance:
  CR/PR → treatment is working → continue current therapy
  SD    → treatment may be working → monitor closely
  PD    → treatment is failing → change therapy
```

---

### 12.3 Connected Component Analysis

```
Given a binary segmentation mask, identify individual lesions:

scipy.ndimage.label(mask > 0):
  - Assigns unique integer to each connected component
  - Uses 26-connectivity in 3D (face, edge, and corner neighbors)

Example:
  Input mask: two separate tumor regions
  
  [0 0 0 0 0 0 0]
  [0 1 1 0 0 0 0]   → Component 1 (lesion A)
  [0 1 1 0 0 0 0]
  [0 0 0 0 0 0 0]
  [0 0 0 0 1 1 0]   → Component 2 (lesion B)
  [0 0 0 0 1 1 0]
  [0 0 0 0 0 0 0]

Each component is measured independently:
  Lesion A: longest_diameter = 12.5mm, volume = 450mm³
  Lesion B: longest_diameter = 8.3mm,  volume = 280mm³
  SLD = 12.5 + 8.3 = 20.8mm
```

---

## 13. Software Engineering & Experiment Management

```
Hydra (Configuration Management):
  - YAML-based config files with hierarchical composition
  - Override any parameter from command line
  - Automatic output directory management
  - Enables reproducible experiments

  oncoseg-train model=oncoseg data=brats2023 training.lr=1e-4

Weights & Biases (Experiment Tracking):
  - Log training loss, validation metrics every epoch
  - Compare runs across hyperparameters
  - Visualize training curves, attention maps
  - Reproducibility: every run's config is recorded

PyTorch:
  - Dynamic computational graphs (define-by-run)
  - Automatic differentiation (autograd)
  - GPU acceleration (CUDA)
  - Rich ecosystem (MONAI, torchvision)

MONAI (Medical Open Network for AI):
  - Medical imaging transforms (LoadNIfTI, Spacing, CropForeground)
  - Pre-built architectures (SwinUNETR, UNet)
  - Medical metrics (Dice, Hausdorff, Surface Distance)
  - Sliding window inference utility

pytest (Testing):
  - Unit tests verify model output shapes
  - Integration tests verify RECIST measurements
  - Reproducibility: fixed seeds, deterministic operations
```

---

## 14. Knowledge Coverage Map

```
OncoSeg AI Knowledge Taxonomy
│
├── Mathematics
│   ├── Linear Algebra
│   │   ├── Tensor operations (5D)
│   │   ├── Matrix multiplication
│   │   ├── Dot product
│   │   └── Transpose
│   ├── Calculus
│   │   ├── Partial derivatives
│   │   ├── Chain rule (backpropagation)
│   │   └── Gradient descent
│   ├── Probability & Statistics
│   │   ├── Softmax / probability distributions
│   │   ├── Bayesian inference (MC Dropout)
│   │   ├── Statistical significance testing
│   │   └── Mean / Standard deviation
│   └── Information Theory
│       ├── Cross-entropy
│       ├── KL Divergence
│       └── Predictive entropy
│
├── Deep Learning
│   ├── Neural Network Basics
│   │   ├── Forward pass / Backward pass
│   │   ├── Activation functions (LeakyReLU, GELU, Softmax)
│   │   ├── Weight initialization (Kaiming, Truncated Normal)
│   │   └── Normalization (Instance Norm, Layer Norm)
│   ├── Regularization
│   │   ├── Dropout / Dropout3D
│   │   ├── Weight decay (L2)
│   │   └── Data augmentation
│   └── Optimization
│       ├── SGD / Momentum
│       ├── Adam / AdamW
│       ├── Cosine annealing LR schedule
│       └── Gradient clipping
│
├── CNN (Convolutional Neural Networks)
│   ├── 3D Convolution
│   ├── Transposed Convolution (upsampling)
│   ├── Receptive field
│   ├── Feature maps / Channels
│   └── Encoder-decoder architecture
│
├── Transformer / Attention
│   ├── Self-attention (Query / Key / Value)
│   ├── Scaled dot-product attention
│   ├── Multi-head attention
│   ├── Cross-attention (our innovation)
│   ├── Positional encoding (relative position bias)
│   └── Swin Transformer
│       ├── Window-based local attention
│       ├── Shifted window mechanism
│       ├── Hierarchical feature maps
│       └── Patch merging
│
├── U-Net Architecture
│   ├── Encoder-decoder with skip connections
│   ├── Standard skip (concatenation)
│   ├── Cross-attention skip (our method)
│   └── 3D extension for volumetric data
│
├── Loss Functions
│   ├── Cross-entropy loss
│   ├── Dice loss (handles class imbalance)
│   ├── Combined DiceCE loss
│   └── Deep supervision loss
│
├── Evaluation Metrics
│   ├── Dice Score (volume overlap)
│   ├── Hausdorff Distance 95% (boundary error)
│   ├── Average Surface Distance (mean boundary error)
│   ├── Sensitivity / Recall (detection rate)
│   └── Specificity (false positive control)
│
├── Uncertainty Quantification
│   ├── Monte Carlo Dropout
│   ├── Predictive entropy
│   └── Bayesian deep learning
│
├── Medical Image Processing
│   ├── NIfTI format
│   ├── Voxel spacing / Affine matrix
│   ├── Orientation standardization (RAS)
│   ├── Intensity normalization (Z-score)
│   ├── Foreground cropping
│   └── Sliding window inference
│
├── Data Augmentation
│   ├── Random flip (3 axes)
│   ├── Random rotation (90°)
│   ├── Random intensity scale
│   ├── Random intensity shift
│   └── Random spatial cropping
│
├── Clinical Knowledge (RECIST 1.1)
│   ├── Longest axial diameter measurement
│   ├── Sum of longest diameters
│   ├── Response categories (CR / PR / SD / PD)
│   └── Connected component analysis
│
└── Software Engineering
    ├── Hydra configuration management
    ├── Weights & Biases experiment tracking
    ├── PyTorch / MONAI frameworks
    └── pytest unit testing
```

---

## 15. Purpose of Each Section in OncoSeg — Where & Why Each Is Used

### Section 1: Mathematics Foundations — The Engine Under Everything

Every computation in this project reduces to math operations.

| Math Concept | Specific Use in OncoSeg | File |
|---|---|---|
| Matrix multiplication | Every `nn.Linear`, every `Conv3d`, every attention `Q·K^T` | All model files |
| Tensor operations | Data flows as 5D tensors `[B,C,H,W,D]` through the entire pipeline | Everywhere |
| Dot product | Attention scores between decoder queries and encoder keys | `cross_attention_skip.py` |
| Partial derivatives | PyTorch autograd computes gradients of DiceCE loss w.r.t. ~25M parameters | `trainer.py` → `loss.backward()` |
| Chain rule | Gradients propagate backwards: Loss → Softmax → Decoder → Cross-Attention → Swin Encoder | `trainer.py` → `loss.backward()` |
| Softmax | Convert logits to probabilities at every voxel: `torch.softmax(logits, dim=1)` | `oncoseg.py`, `trainer.py`, `inference.py` |
| Entropy | Uncertainty map: `H = -Σ p̄ × log(p̄)` from MC Dropout samples | `oncoseg.py:_mc_uncertainty()` |
| Cross-entropy | Half of our loss function: `-Σ y × log(p)` per voxel | `losses.py:DiceCELoss` |
| Bayesian inference | MC Dropout ≈ sampling from weight posterior → uncertainty estimation | `oncoseg.py:_mc_uncertainty()` |
| Wilcoxon test | Compare OncoSeg vs baselines: "is the Dice difference statistically significant?" | Notebook cell 40 |

**Without this:** Nothing works. Math is the language the model thinks in.

---

### Section 2: Deep Learning Fundamentals — How the Model Learns

These are the building blocks that make a neural network trainable.

| Concept | Specific Use | File |
|---|---|---|
| Forward pass | Input MRI → Swin Encoder → Cross-Attention → CNN Decoder → Segmentation | `oncoseg.py:forward()` |
| Backpropagation | `loss.backward()` computes all gradients automatically | `trainer.py:train_epoch()` |
| LeakyReLU | Activation in every CNN decoder block | `cnn_decoder.py:DecoderBlock` |
| GELU | Activation inside Swin Transformer MLP blocks | `swin_encoder.py` (via MONAI) |
| Instance Normalization | Normalize features in decoder (stable with batch_size=2) | `cnn_decoder.py:DecoderBlock` |
| Layer Normalization | Normalize features in Transformer (standard for attention) | `swin_encoder.py` (via MONAI) |
| Dropout3d | MC Dropout on bottleneck features for uncertainty | `oncoseg.py:self.mc_dropout` |
| Weight decay | L2 regularization `λ=1e-5` prevents overfitting on 484 subjects | `trainer.py:AdamW(weight_decay=1e-5)` |
| Kaiming init | Initialize Conv3d weights properly for LeakyReLU | MONAI handles automatically |

**Without this:** Model either can't learn (vanishing gradients), overfits (memorizes 484 cases), or produces garbage (bad initialization).

---

### Section 3: Optimization — How Parameters Get Updated

Controls how the model improves from each training batch.

| Concept | Specific Use | File |
|---|---|---|
| AdamW | Our optimizer: adaptive learning rates + correct weight decay | `trainer.py:AdamW(lr=1e-4, weight_decay=1e-5)` |
| Cosine annealing | LR schedule: `1e-4 → 1e-6` over 300 epochs, smooth decay | `trainer.py:CosineAnnealingLR(T_max=300)` |
| Gradient clipping | Cap gradient norm at 1.0, prevents training explosion | `trainer.py:clip_grad_norm_(max_norm=1.0)` |
| Momentum (inside Adam) | `β₁=0.9` — smooth gradient estimates across batches | `trainer.py` (PyTorch default) |

**Without this:** Training either diverges (too fast), stalls (too slow), or oscillates wildly (no clipping).

---

### Section 4: CNN — The Decoder's Core Technology

The CNN decoder restores spatial resolution and produces the final segmentation.

| Concept | Specific Use | File |
|---|---|---|
| 3D Convolution | Feature refinement in decoder: `Conv3d(in_ch, out_ch, kernel_size=3)` | `cnn_decoder.py:DecoderBlock` |
| Transposed Convolution | Upsample: 8³→16³→32³→64³→128³ via `ConvTranspose3d(stride=2)` | `cnn_decoder.py:DecoderBlock` |
| Feature maps | Decoder channels: 384→192→96→48→4 (4 output classes) | `cnn_decoder.py:CNNDecoder3D` |
| 1×1×1 Convolution | Final segmentation head: `Conv3d(48, 4, kernel_size=1)` — classify each voxel | `cnn_decoder.py:CNNDecoder3D` |
| Receptive field | Decoder combines local CNN features with global Transformer context via skip connections | Architecture design |

**Without this:** The Transformer encoder captures what the tumor IS, but without the CNN decoder we can't recover WHERE the exact boundary is at full resolution.

---

### Section 5: U-Net Architecture — The Overall Blueprint

U-Net is the structural foundation of the entire model.

| Concept | Specific Use | File |
|---|---|---|
| Encoder path | Swin Transformer: 128³→64³→32³→16³→8³ (downsample, extract features) | `swin_encoder.py` |
| Decoder path | CNN: 8³→16³→32³→64³→128³ (upsample, restore resolution) | `cnn_decoder.py` |
| Skip connections | Cross-attention fusion at 3 scales (64³, 32³, 16³) | `cross_attention_skip.py` |
| Bottleneck | 8³ × 384 channels — most compressed representation of the input | `oncoseg.py:encoder_features[-1]` |
| 3D extension | Process entire volumetric MRI, not slice-by-slice | All model files use 3D ops |

**Without this:** No architecture to put the pieces together. U-Net is the skeleton that Swin Transformer encoder, cross-attention skips, and CNN decoder all hang on.

---

### Section 6: Transformer & Attention — The Encoder's Core Technology

Captures global context — understands spatial relationships across the entire brain volume.

| Concept | Specific Use | File |
|---|---|---|
| Self-attention | Every voxel in a 7³ window attends to all 343 neighbors | `swin_encoder.py` (MONAI SwinTransformer) |
| Multi-head attention | 3→6→12→24 heads across 4 stages, each learns different patterns | `swin_encoder.py`, `cross_attention_skip.py` |
| Cross-attention | **Our innovation:** decoder queries encoder for relevant features at skip connections | `cross_attention_skip.py:CrossAttentionSkip` |
| Q/K/V projections | `W_Q`, `W_K`, `W_V` linear layers in both self-attention and cross-attention | `cross_attention_skip.py` |
| Window attention | Local 7×7×7 windows keep computation feasible (343² vs 2M²) | `swin_encoder.py` (MONAI) |
| Shifted windows | Alternate window positions so information flows across boundaries | `swin_encoder.py` (MONAI) |
| Patch embedding | Input: `[B,4,128,128,128]` → patches: `[B,32³,48]` via 4×4×4 projection | `swin_encoder.py` (MONAI) |
| Patch merging | Downsample: concatenate 2×2×2 neighbors, project — like learned pooling | `swin_encoder.py` (MONAI) |
| Relative position bias | Learned bias `B[Δx,Δy,Δz]` added to attention scores — encodes spatial structure | `swin_encoder.py` (MONAI) |

**Without this:** Pure CNN (UNet3D) only sees local neighborhoods. Transformer attention lets a voxel at one end of a tumor understand its relationship to the other end — critical for large, irregular brain tumors.

---

### Section 7: Loss Functions — What the Model Optimizes

Tells the model what "correct" means — the mathematical definition of success.

| Loss | Specific Use | File |
|---|---|---|
| Cross-entropy | Per-voxel classification loss: penalizes wrong predictions | `losses.py:DiceCELoss` |
| Dice loss | Overlap-based loss: directly optimizes the Dice metric, handles class imbalance | `losses.py:DiceCELoss` |
| Combined DiceCE | `0.5 × Dice + 0.5 × CE` — stable gradients + imbalance handling | `losses.py:DiceCELoss` |
| Deep supervision | Auxiliary losses at 64³ and 32³ scales with weights [0.5, 0.25] | `losses.py:DeepSupervisionLoss` |

**Why this specific combination matters for brain tumors:**
```
Brain MRI volume: 128³ = 2,097,152 voxels
Typical tumor:    ~50,000 voxels (2.4%)
Background:       ~2,047,152 voxels (97.6%)

CE alone: model learns "predict background everywhere" → 97.6% accuracy but useless
Dice alone: forces finding the tumor, but unstable gradients early in training
DiceCE: finds the tumor AND trains stably
Deep supervision: prevents vanishing gradients through 20+ layer network
```

---

### Section 8: Evaluation Metrics — How We Measure Success

Quantify model performance from multiple clinical perspectives.

| Metric | What It Answers | Applied To | File |
|---|---|---|---|
| Dice Score | "How much overlap between our prediction and truth?" | ET, TC, WT regions separately | `metrics.py` |
| HD95 | "What's the worst boundary error (excluding 5% outliers)?" | ET, TC, WT in mm | `metrics.py` |
| ASD | "What's the average boundary error?" | ET, TC, WT in mm | `metrics.py` |
| Sensitivity | "Did we find all the tumor?" (miss rate) | ET, TC, WT | `metrics.py` |
| Specificity | "Did we avoid marking healthy tissue as tumor?" (false alarm rate) | ET, TC, WT | `metrics.py` |

**Why 5 metrics, not just Dice?**
```
Scenario: Model A and Model B both have Dice = 0.85

Model A: smooth boundaries, consistent performance
  → HD95 = 3mm, ASD = 1mm

Model B: mostly correct but has one region where boundary is 15mm off
  → HD95 = 15mm, ASD = 3mm

Clinically, Model A is far safer. Dice alone can't distinguish them.
HD95 catches dangerous boundary failures.
```

---

### Section 9: Uncertainty Quantification — "How Confident Is the Model?"

Tells doctors which parts of the segmentation they should trust vs. review manually.

| Concept | Specific Use | File |
|---|---|---|
| MC Dropout | Keep `Dropout3d(0.1)` active, run 10 forward passes | `oncoseg.py:_mc_uncertainty()` |
| Predictive entropy | `H = -Σ p̄ log(p̄)` from averaged softmax outputs | `oncoseg.py:_mc_uncertainty()` |
| Uncertainty map | Save as NIfTI alongside segmentation for radiologist review | `inference.py:predict_and_save()` |

**Clinical value:**
```
Tumor core:       low uncertainty  → model is confident → trust the segmentation
Tumor boundary:   high uncertainty → model is unsure    → radiologist should review
Imaging artifact: high uncertainty → model flags it     → don't measure this region

This is the difference between "here's a segmentation" and
"here's a segmentation, and HERE are the parts I'm not sure about"
```

---

### Section 10: Medical Image Processing — Preparing Raw MRI for the Model

Raw MRI scans are messy — different scanners, orientations, resolutions. Preprocessing standardizes everything.

| Step | What It Does | Why Needed | File |
|---|---|---|---|
| NIfTI loading | Read `.nii.gz` 3D volumes into numpy arrays | Standard medical image format | `transforms.py:LoadImaged` |
| Orientation (RAS) | Rotate volume to standard Right-Anterior-Superior | Different scanners use different orientations | `transforms.py:Orientationd` |
| Resampling (1mm³) | Resample to isotropic 1.0mm voxels | Scanners have different resolutions (0.5mm–3mm) | `transforms.py:Spacingd` |
| Z-score normalization | `(x-μ)/σ` per modality, non-zero voxels only | MRI intensities are arbitrary, scanner-dependent | `transforms.py:NormalizeIntensityd` |
| Foreground cropping | Remove empty air around the brain | Reduces volume size → faster training, less wasted computation | `transforms.py:CropForegroundd` |
| Sliding window | Process full volume in overlapping 128³ patches | Full volume doesn't fit in GPU memory | `trainer.py`, `evaluator.py`, `inference.py` |

**Without this:** Feeding raw MRI directly to the model would be like feeding a mix of English, Chinese, and French text to a language model with no translation — garbage in, garbage out.

---

### Section 11: Data Augmentation — Making 484 Subjects Feel Like 48,400

Prevent overfitting on our small dataset by showing the model artificial variations of each subject.

| Augmentation | What It Simulates | File |
|---|---|---|
| Random flip (3 axes) | Tumor can appear on left or right hemisphere | `transforms.py:RandFlipd` |
| Random rotate 90° | Different patient head orientations in scanner | `transforms.py:RandRotate90d` |
| Random intensity scale ±10% | Scanner calibration differences between hospitals | `transforms.py:RandScaleIntensityd` |
| Random intensity shift ±10% | Brightness variations across acquisitions | `transforms.py:RandShiftIntensityd` |
| Random spatial crop | Different spatial contexts around the tumor | `transforms.py:RandSpatialCropd` |

**Impact:**
```
Without augmentation: model sees the same 484 images every epoch → overfits
With augmentation: each epoch generates unique combinations:
  Subject 1: flipped_x=yes, flipped_y=no, rotate=90°, scale=1.05, shift=-0.03
  Subject 1: flipped_x=no,  flipped_y=yes, rotate=0°,  scale=0.97, shift=+0.08
  ... effectively infinite variations

This is the #1 most important technique for small medical datasets.
```

---

### Section 12: Clinical Knowledge (RECIST 1.1) — Bridging AI to Medicine

Transform raw segmentation masks into measurements that oncologists actually use in clinical practice.

| Concept | Specific Use | File |
|---|---|---|
| Connected component labeling | Detect individual lesions in binary mask | `recist.py:measure_lesions()` |
| Longest axial diameter | Find max-area slice → compute max Feret distance in mm | `recist.py:longest_axial_diameter()` |
| Volume measurement | Count tumor voxels × voxel spacing = volume in mm³ | `recist.py:volume_mm3()` |
| Sum of longest diameters | Sum across all lesions — the standard RECIST measurement | `classifier.py:classify()` |
| Response classification | Compare baseline vs follow-up: CR/PR/SD/PD | `classifier.py:classify()` |

**Why this matters:**
```
Without RECIST:
  Doctor gets a segmentation mask → "okay, the tumor is... somewhere... some size"
  Still needs to manually measure diameter on screen

With RECIST:
  Doctor gets: "Enhancing tumor: 3 lesions, SLD = 42.3mm, Volume = 15,230mm³"
  Follow-up:  "SLD = 28.1mm, -33.6% change → Partial Response (PR)"
  
  This is the actual clinical workflow — automated end-to-end.
  Saves hours of manual measurement per patient.
```

---

### Section 13: Software Engineering — Making It Reproducible and Maintainable

Without proper engineering, experiments are unreproducible, code is unmaintainable, and results are unreliable.

| Tool | Specific Use | Why Needed |
|---|---|---|
| Hydra configs | `configs/*.yaml` — all hyperparameters in structured YAML files | Change experiments without editing code. `oncoseg-train model=unet3d` switches models instantly |
| W&B tracking | `wandb.log({"train/loss": loss, "val/dice_et": dice})` | Compare 10+ training runs visually. See which hyperparameters work best |
| PyTorch | Entire model and training loop | Industry-standard deep learning framework, GPU acceleration |
| MONAI | Transforms, metrics, SwinTransformer, sliding window | Medical imaging library built on PyTorch — don't reinvent medical preprocessing |
| pytest | `tests/test_models.py` — verify output shapes, RECIST math | Catch bugs before training (a 300-epoch run takes hours — don't waste it on a shape bug) |
| Modular `src/` package | Separate model, data, training, evaluation, response modules | Each component is independent, testable, and reusable |

---

### Section 14: Knowledge Coverage Map — The Big Picture

Shows how all 13 sections connect together in the end-to-end pipeline:

```
Patient gets brain MRI scan
        │
        ▼
[10. Medical Image Processing]     Load NIfTI, standardize orientation/spacing/intensity
        │
        ▼
[11. Data Augmentation]            Flip, rotate, scale for training robustness
        │
        ▼
[1. Mathematics]                   Tensor operations, matrix multiplications
        │
        ▼
[4. CNN] + [6. Transformer]        OncoSeg: Swin encoder + CNN decoder
connected by [5. U-Net]            with cross-attention skip connections
        │
        ▼
[2. Deep Learning Fundamentals]    Forward pass, activation functions, normalization
        │
        ▼
[7. Loss Functions]                DiceCE + Deep Supervision → compute error
        │
        ▼
[3. Optimization]                  AdamW + cosine LR + gradient clipping → update weights
        │
        ▼
[8. Evaluation Metrics]            Dice, HD95, ASD, Sensitivity, Specificity
        │
        ▼
[9. Uncertainty]                   MC Dropout → "how confident is the model?"
        │
        ▼
[12. Clinical Knowledge]           RECIST measurement → "tumor is 42mm, response = PR"
        │
        ▼
[13. Software Engineering]         Hydra configs, W&B tracking, reproducible results
        │
        ▼
[14. Knowledge Map]                This overview — ties everything together
```

---

*This document is a living reference for the OncoSeg project. Every concept listed here is directly used in the codebase and can be traced to specific source files.*
