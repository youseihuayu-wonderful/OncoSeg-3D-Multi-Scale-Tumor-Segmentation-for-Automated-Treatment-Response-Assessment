# OncoSeg — Interview Questions: AI & Technical Knowledge Fundamentals

Questions a hiring manager would ask to test your understanding of the AI/ML concepts behind this project. Each answer ties the theory back to OncoSeg.

---

## 1. Mathematics & Linear Algebra

**Q1: What is a tensor and how are tensors used in your project?**
> A tensor is a multi-dimensional array — the fundamental data structure in deep learning. In OncoSeg, data flows as 5D tensors: [Batch, Channels, Height, Width, Depth]. For example, an input MRI is [1, 4, 128, 128, 128] — 1 sample, 4 modalities, 128³ voxels. The output is [1, 3, 128, 128, 128] — 3 tumor region predictions.

**Q2: What is a matrix multiplication and where does it appear in your model?**
> Matrix multiplication is the core operation in both linear layers and attention. In the cross-attention skip connections, Q·K^T computes attention scores between decoder queries and encoder keys. It's a [N_dec × d] × [d × N_enc] multiplication that produces [N_dec × N_enc] attention weights.

**Q3: What is an eigenvalue and why does it matter in deep learning?**
> Eigenvalues describe how a matrix scales along its principal directions. In deep learning, the eigenvalues of the Hessian (second-derivative matrix of the loss) determine the loss landscape's curvature. Large eigenvalue spread → ill-conditioned optimization → Adam/AdamW helps by adapting per-parameter learning rates.

**Q4: What is the dot product and how is it used in attention?**
> The dot product measures similarity between two vectors. In scaled dot-product attention: score = Q·K^T / √d_k. Higher dot product = more similar query and key = higher attention weight. The √d_k scaling prevents the softmax from becoming too peaked when dimensionality is large.

---

## 2. Calculus & Optimization

**Q5: What is backpropagation and how does it train your model?**
> Backpropagation computes gradients of the loss with respect to every parameter by applying the chain rule backwards through the computation graph. In OncoSeg, the DiceCE loss is computed on the output, then gradients flow backwards through the CNN decoder, cross-attention skips, and Swin Transformer encoder. PyTorch's autograd handles this automatically.

**Q6: What is the chain rule and why is it important?**
> The chain rule says ∂L/∂w = ∂L/∂y · ∂y/∂x · ∂x/∂w for a chain of operations. It's how gradients propagate through deep networks — each layer multiplies its local gradient. Without it, we couldn't train networks with hundreds of layers like our Swin Transformer encoder.

**Q7: What is the vanishing gradient problem and how does your architecture handle it?**
> When gradients are multiplied through many layers, they can shrink to near-zero, stopping learning in early layers. OncoSeg handles this three ways: (1) skip connections provide gradient shortcuts from loss to encoder, (2) deep supervision injects loss at intermediate layers, (3) LayerNorm in Transformer blocks stabilizes gradient magnitudes.

**Q8: What is the exploding gradient problem and what do you do about it?**
> The opposite — gradients grow exponentially. Transformers are prone to this, especially early in training. We use gradient clipping at max_norm=1.0: if the total gradient norm exceeds 1.0, all gradients are scaled down proportionally. This caps the step size without changing gradient direction.

**Q9: Explain gradient descent. What variant do you use?**
> Gradient descent updates parameters in the direction that reduces the loss: w = w - lr · ∂L/∂w. We use AdamW, which maintains per-parameter running averages of the gradient (first moment) and squared gradient (second moment). This adapts the learning rate for each parameter — parameters with noisy gradients get smaller steps.

**Q10: What is the difference between Adam and AdamW?**
> Adam applies weight decay to the gradient before the adaptive step, which couples regularization with the learning rate. AdamW decouples them — it applies weight decay directly to the weights after the adaptive step. This is mathematically correct L2 regularization and gives better generalization, especially for Transformers.

**Q11: Explain your learning rate schedule. Why cosine annealing?**
> CosineAnnealingLR decreases the learning rate following a cosine curve from 1e-4 to 1e-6. It starts with large steps for fast initial learning, then gradually fine-tunes with smaller steps. Unlike step decay, there are no sudden drops that can destabilize training. It's simple and works well with AdamW.

**Q12: What is a loss landscape and why does it matter?**
> The loss landscape is the surface of loss values over all possible parameter settings. A smooth landscape with a wide minimum generalizes better than a sharp one. AdamW's weight decay biases toward smaller weights (wider minima). Cosine annealing helps explore the landscape before settling.

---

## 3. Probability & Statistics

**Q13: What is Bayes' theorem and how does it relate to MC Dropout?**
> Bayes' theorem: P(model|data) ∝ P(data|model) · P(model). In Bayesian deep learning, we want the posterior distribution over model weights. MC Dropout approximates this — each dropout mask samples a different sub-network, approximating sampling from the posterior. The mean of predictions approximates the Bayesian model average.

**Q14: What is the difference between epistemic and aleatoric uncertainty?**
> Epistemic uncertainty is model uncertainty — "I don't have enough data to be sure." It decreases with more training data. MC Dropout captures this. Aleatoric uncertainty is data noise — "this image is inherently ambiguous." It doesn't decrease with more data. OncoSeg currently only estimates epistemic uncertainty.

**Q15: How do you use statistical tests in your evaluation?**
> Paired Wilcoxon signed-rank tests compare per-subject Dice scores between models. It's non-parametric — doesn't assume normal distribution, which is important because Dice scores are bounded [0,1] and often skewed. Alpha=0.05 with correction for multiple comparisons.

**Q16: What is a p-value and when is a result statistically significant?**
> The p-value is the probability of observing results as extreme as ours if there were no real difference between models. If p < 0.05, we reject the null hypothesis (no difference) and say the improvement is statistically significant. But significance doesn't mean clinically meaningful — a 0.01 Dice improvement could be significant but irrelevant in practice.

---

## 4. Convolutional Neural Networks

**Q17: How does a 3D convolution work?**
> A 3D kernel (e.g., 3×3×3) slides through a volume, computing the dot product at each position. It captures local spatial patterns in all three dimensions. In OncoSeg's decoder, 3D convolutions refine features at each resolution level. With in_channels input channels and out_channels filters, one layer has kernel_size³ × in_channels × out_channels parameters.

**Q18: What is a transposed convolution and why do you use it?**
> Transposed convolution (sometimes called deconvolution) upsamples feature maps. It's the mathematical transpose of a strided convolution — instead of reducing spatial dimensions, it increases them. In our decoder, stride-2 transposed convolutions double the resolution at each level. Unlike bilinear upsampling, the upsampling is learned.

**Q19: What is a receptive field and why does it matter for tumor segmentation?**
> The receptive field is the region of the input that influences a particular output voxel. For tumor segmentation, we need a large receptive field because tumors can be large and context matters (surrounding tissue helps identify tumor boundaries). CNNs have limited receptive fields; the Swin Transformer encoder provides a global receptive field through attention.

**Q20: What is the difference between valid, same, and full padding?**
> Valid padding = no padding, output shrinks. Same padding = pad so output matches input size. In OncoSeg, we use padding=1 with kernel_size=3 (same padding) in decoder convolutions to maintain spatial dimensions. The Swin Transformer uses its own padding for window partitioning.

**Q21: What are feature maps and what do they represent?**
> Feature maps are the output channels of a convolution layer. Early layers detect edges and textures; deeper layers detect complex patterns like tumor boundaries. In OncoSeg, the encoder produces feature maps at 4 resolutions with increasing channels: 48, 96, 192, 384 (with embed_dim=48). Each channel captures a different learned pattern.

---

## 5. U-Net Architecture

**Q22: Explain the U-Net architecture and why it's used for segmentation.**
> U-Net has an encoder (downsampling path) that captures context and a decoder (upsampling path) that enables precise localization. Skip connections bridge the two, preserving fine-grained spatial details that are lost during downsampling. The "U" shape comes from the symmetric encoder-decoder structure. It's the foundation of medical image segmentation since 2015.

**Q23: Why are skip connections important for segmentation?**
> During encoding, spatial resolution decreases — fine details like tumor boundaries are lost. Skip connections pass high-resolution features from the encoder directly to the decoder, which can combine them with the semantic (contextual) features from the deeper layers. Without skips, the decoder would have to reconstruct spatial details from scratch.

**Q24: How does your cross-attention skip differ from the original U-Net skip?**
> Original U-Net concatenates encoder and decoder features. This is unselective — all encoder information is passed through, including noise. Our cross-attention skip lets the decoder query the encoder: "which encoder features are relevant to what I'm trying to decode here?" This selective fusion filters noise and focuses on useful information.

**Q25: How does 3D U-Net extend 2D U-Net?**
> Replace 2D convolutions with 3D convolutions, 2D max-pooling with 3D, and 2D upsampling with 3D. The architecture is the same; operations work in three dimensions. The computational cost increases dramatically — a 3×3 kernel becomes 3×3×3 (27 values vs 9), and feature maps have an extra spatial dimension.

---

## 6. Transformer & Attention

**Q26: What is self-attention and how does it work?**
> Self-attention computes relationships between all positions in a sequence. Each position generates a Query (what am I looking for?), Key (what do I contain?), and Value (what information do I share). Attention weight = softmax(Q·K^T/√d). Output = weighted sum of Values. Complexity: O(n²) for sequence length n.

**Q27: What is scaled dot-product attention? Why the scaling?**
> Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V. The scaling by √d_k prevents the dot products from growing too large with high dimensionality. Large dot products push softmax into regions with tiny gradients (saturation), making learning very slow. The scaling keeps values in a well-behaved range.

**Q28: What is multi-head attention and why use multiple heads?**
> Instead of one attention with d dimensions, use h heads each with d/h dimensions. Each head can learn different attention patterns — one might focus on texture, another on shape. Results are concatenated and projected. In OncoSeg's cross-attention, we use max(dim//48, 1) heads per level.

**Q29: What is cross-attention and how is it different from self-attention?**
> In self-attention, Q, K, V all come from the same input. In cross-attention, Q comes from one source (decoder features) and K, V from another (encoder features). This lets the decoder selectively attend to encoder information. OncoSeg uses cross-attention in skip connections — decoder queries encoder for relevant features.

**Q30: Explain the Swin Transformer. Why "Swin"?**
> "Swin" = Shifted Windows. It computes attention within local windows (e.g., 7×7×7) instead of globally — O(n) vs O(n²). Shifted windows: alternate between regular and shifted window positions so that tokens in different windows can communicate. This gives a global receptive field with linear complexity. Hierarchical structure produces multi-scale feature maps like a CNN.

**Q31: What are positional embeddings and why do Transformers need them?**
> Attention is permutation-invariant — it doesn't know the order or position of tokens. Positional embeddings inject spatial information. Swin Transformer uses relative position bias: a learned bias added to attention scores based on the relative position between two tokens. This is more flexible than absolute position embeddings and works for varying input sizes.

**Q32: What is the complexity difference between ViT and Swin Transformer?**
> ViT: global self-attention is O(n²) where n = number of tokens. For a 128³ volume with patch_size=16, n = 8³ = 512 tokens — manageable. But for finer patches or larger volumes, it's prohibitive. Swin: windowed attention is O(n·w²) where w = window size (fixed, e.g., 7). Linear in n. This is why we chose Swin over ViT for the encoder.

---

## 7. Normalization Techniques

**Q33: What is BatchNorm and why don't you use it?**
> BatchNorm normalizes across the batch dimension — computes mean/variance over all samples in a mini-batch for each channel. With batch_size=1 (common in 3D medical imaging — volumes are too large for bigger batches), the statistics are just the single sample. This is noisy and unreliable. That's why we use InstanceNorm instead.

**Q34: What is InstanceNorm and why is it right for your model?**
> InstanceNorm normalizes per-sample, per-channel — independent of batch size. For each channel in each sample, it computes mean and variance over the spatial dimensions (H, W, D) and normalizes. Works perfectly with batch_size=1. Standard in medical image segmentation.

**Q35: What is LayerNorm and where is it used?**
> LayerNorm normalizes across the feature dimension for each token in a sequence. It's the standard normalization in Transformers because attention operates on sequences, not spatial grids. In OncoSeg, the Swin Transformer encoder uses LayerNorm throughout. The cross-attention skip connections also use LayerNorm on the flattened feature sequences.

**Q36: When would you use GroupNorm?**
> GroupNorm splits channels into groups and normalizes within each group. It's a middle ground between InstanceNorm (1 channel per group) and LayerNorm (all channels in one group). Useful when batch_size is small but not 1. We don't use it in OncoSeg — InstanceNorm is simpler and sufficient for our case.

---

## 8. Activation Functions

**Q37: Why LeakyReLU instead of ReLU in the decoder?**
> ReLU(x) = max(0, x) — kills negative activations completely. Neurons can "die" if they always produce negatives (zero gradient, never recover). LeakyReLU(x) = x if x > 0, else 0.01x — preserves a small gradient for negatives. More robust in deeper networks with many layers.

**Q38: What is GELU and where is it used?**
> GELU (Gaussian Error Linear Unit) = x · Φ(x) where Φ is the normal CDF. It smoothly interpolates between passing and blocking values. Used in the FFN inside our cross-attention skip connections and in the Swin Transformer. It's the standard activation for Transformers — smoother than ReLU, works better in practice.

**Q39: Why does the model output raw logits instead of probabilities?**
> The model outputs logits (raw scores before activation). BCEWithLogitsLoss applies sigmoid internally, which is numerically more stable than separate sigmoid + BCE. For inference, we apply torch.sigmoid() explicitly. DiceLoss with sigmoid=True also handles this internally.

---

## 9. Loss Functions

**Q40: Explain Dice Loss mathematically.**
> Dice = 2|A∩B| / (|A| + |B|). Dice Loss = 1 - Dice. For soft predictions: numerator = 2 · Σ(p·g) + smooth, denominator = Σp + Σg + smooth, where p = predicted probabilities, g = ground truth. It directly optimizes the overlap metric. Handles class imbalance naturally because it normalizes by region size.

**Q41: Explain Binary Cross-Entropy Loss.**
> BCE = -[y·log(p) + (1-y)·log(1-p)] averaged over all voxels. For each voxel, it penalizes the model for incorrect confidence. With logits: BCEWithLogitsLoss applies sigmoid internally for numerical stability (log-sum-exp trick avoids log(0)). It treats each voxel independently.

**Q42: Why combine Dice and BCE instead of using one?**
> Dice Loss: optimizes region overlap but gradients are noisy, especially when predictions are very wrong (early training). BCE: stable per-voxel gradients but dominated by the majority class (background). Combined: Dice handles class imbalance while BCE provides stable gradients. This is the standard combination used in nnU-Net and most medical segmentation.

**Q43: What is label smoothing and do you use it?**
> Label smoothing replaces hard labels (0/1) with soft labels (0.05/0.95). It prevents the model from becoming overconfident. We don't use it in OncoSeg — the smooth_nr/smooth_dr in Dice Loss serves a similar regularization purpose, and BCE already provides well-calibrated probabilities with sigmoid.

---

## 10. Regularization

**Q44: What is dropout and how does it prevent overfitting?**
> Dropout randomly zeros out neurons during training with probability p. This prevents co-adaptation — neurons can't rely on specific other neurons and must learn robust features independently. In OncoSeg, we use Dropout3d(p=0.1) at the bottleneck. At inference, dropout is normally disabled, but we keep it on for uncertainty estimation.

**Q45: What is weight decay and how does it regularize?**
> Weight decay adds λ·||w||² to the loss, penalizing large weights. This biases the model toward simpler functions. In AdamW, weight decay is applied directly to weights: w = w - lr·λ·w. We use λ=1e-5 — light regularization that prevents weights from growing unbounded without over-constraining the model.

**Q46: What is early stopping and do you use it?**
> Early stopping monitors validation performance and stops training when it stops improving. We don't use it explicitly — we save the best checkpoint by validation Dice and train for all 50 epochs. The best checkpoint serves the same purpose: we use the weights from the epoch with best generalization.

**Q47: What other regularization does your model use?**
> Data augmentation (implicit regularization — presents different views of the same data), weight decay in AdamW, dropout at the bottleneck, and InstanceNorm (normalizes activations, reducing internal covariate shift). These are all complementary.

---

## 11. Evaluation Metrics

**Q48: What is the Dice Score and why is it the primary metric for segmentation?**
> Dice = 2|A∩B| / (|A| + |B|), ranges from 0 (no overlap) to 1 (perfect). It's the standard in medical image segmentation because it directly measures what matters: how well the predicted region matches the ground truth. It handles class imbalance better than accuracy (a model predicting "no tumor everywhere" would get 99% accuracy but 0 Dice).

**Q49: What is Hausdorff Distance and why use the 95th percentile?**
> Hausdorff Distance = max distance from any point on one surface to the nearest point on the other surface. It measures worst-case boundary error. The full Hausdorff is sensitive to single outlier voxels. HD95 uses the 95th percentile — ignores the worst 5%, giving a robust measure of boundary quality. Clinically meaningful: tells you the worst boundary error a radiologist would see.

**Q50: What is the difference between sensitivity and specificity?**
> Sensitivity (recall) = TP / (TP + FN) — what fraction of actual tumor was detected. Specificity = TN / (TN + FP) — what fraction of healthy tissue was correctly excluded. In tumor segmentation, high sensitivity means you don't miss tumor; high specificity means you don't over-segment. There's a trade-off controlled by the prediction threshold (we use 0.5).

**Q51: What is IoU (Jaccard) and how does it relate to Dice?**
> IoU = |A∩B| / |A∪B|. Dice = 2·IoU / (1+IoU). They're monotonically related — if one goes up, so does the other. Dice is always ≥ IoU for the same prediction. We use Dice because it's the BraTS convention. Some object detection challenges prefer IoU. Functionally equivalent for ranking models.

---

## 12. Uncertainty Quantification

**Q52: How does Monte Carlo Dropout estimate uncertainty?**
> Run N forward passes (e.g., 10) with dropout enabled at inference. Each pass uses a different random dropout mask → slightly different predictions. Stack predictions → compute mean (final prediction) and variance (uncertainty). High variance = the model gives different answers depending on which neurons are dropped = it's uncertain.

**Q53: What are the limitations of MC Dropout for uncertainty?**
> It only captures epistemic (model) uncertainty, not aleatoric (data) uncertainty. The quality depends on N (number of passes) and dropout rate. It's N times slower than standard inference. The dropout must be placed strategically — we put it at the bottleneck because that's the information bottleneck. Dropout at every layer would be too aggressive.

**Q54: How would you add aleatoric uncertainty estimation?**
> Have the model predict two outputs per voxel: mean and variance. The variance head learns per-voxel aleatoric uncertainty. Train with a heteroscedastic loss that weights the data term by the predicted variance. This captures uncertainty from ambiguous boundaries, imaging noise, etc.

**Q55: Why is uncertainty important in clinical applications?**
> A confident wrong prediction is dangerous — a radiologist might trust it. An uncertain prediction with a flag saying "please review" is safe. Uncertainty maps can highlight: ambiguous tumor boundaries, regions where the model lacks training data, and potential artifacts. This builds trust and enables human-AI collaboration.

---

## 13. Medical Imaging Concepts

**Q56: What is a NIfTI file and what metadata does it contain?**
> NIfTI (Neuroimaging Informatics Technology Initiative) is the standard format for brain imaging data. It stores: the 3D/4D voxel array, an affine matrix (maps voxel coordinates to real-world mm), voxel spacing (e.g., 1mm × 1mm × 1mm), data type, and orientation information. MONAI's LoadImage reads NIfTI natively.

**Q57: What is voxel spacing and why does it matter?**
> Voxel spacing is the physical size of each voxel in mm (e.g., 1.0 × 1.0 × 1.0). Different scanners produce different spacings. A "3×3×3 kernel" covers 3mm³ with 1mm spacing but 9mm³ with 3mm spacing. Resampling to isotropic spacing ensures the model processes consistent physical scales. We resample to 1mm isotropic.

**Q58: What is an affine matrix in medical imaging?**
> A 4×4 matrix that maps voxel indices [i, j, k] to real-world coordinates [x, y, z] in mm. It encodes: voxel spacing, rotation/orientation, and translation (origin position). Essential for correct spatial alignment between images from different scanners or time points. MONAI preserves this as MetaTensor metadata.

**Q59: What are the 4 MRI modalities and what information does each provide?**
> - **FLAIR**: suppresses cerebrospinal fluid, highlights edema (peritumoral edema appears bright)
> - **T1**: structural anatomy, good gray/white matter contrast
> - **T1ce**: T1 with gadolinium contrast agent — enhancing tumor lights up (blood-brain barrier breakdown)
> - **T2**: fluid-sensitive, shows edema and tumor — complementary to FLAIR
> Each modality reveals different aspects of the tumor; the model learns to fuse all four.

**Q60: What is sliding window inference and why is it necessary?**
> A 128³ volume with 4 channels is 128⁴ × 4 = 33M voxels × 4 bytes = ~134MB per sample. With model activations, GPU memory needed is many GB. Sliding window: process small overlapping patches (roi_size), then stitch results. Overlapping regions are averaged for smooth boundaries. MONAI implements this efficiently with configurable overlap and batch size.

---

## 14. Data Augmentation Theory

**Q61: Why is data augmentation important for medical imaging?**
> Medical datasets are small (484 subjects in MSD vs millions in ImageNet). Augmentation creates synthetic variations of existing data, effectively increasing dataset size. It also teaches the model invariances — "a tumor flipped horizontally is still a tumor." This improves generalization to unseen data.

**Q62: Why spatial augmentation but not elastic deformation?**
> Random flips and rotations are fast and capture natural anatomical variation (brains aren't perfectly symmetric). Elastic deformation simulates tissue deformation but is very expensive in 3D — computing the deformation field for a 128³ volume is memory-intensive. The marginal benefit doesn't justify the cost for brain tumors, which have relatively rigid anatomy.

**Q63: What is the risk of too much augmentation?**
> Over-aggressive augmentation creates unrealistic training samples that confuse the model. For example, extreme intensity scaling could make healthy tissue look like tumor. The model wastes capacity learning unrealistic variations instead of real patterns. We keep augmentation moderate: 10% intensity variation, 50% probability flips.

---

## 15. Softmax vs Sigmoid

**Q64: What is softmax and when should you use it?**
> Softmax converts logits to probabilities that sum to 1: softmax(z_i) = exp(z_i) / Σexp(z_j). Use it for mutually exclusive classes — e.g., "this voxel is exactly one of: cat, dog, bird." Each voxel gets assigned to the highest-probability class.

**Q65: What is sigmoid and when should you use it?**
> Sigmoid converts each logit independently to a probability in [0,1]: σ(z) = 1 / (1 + exp(-z)). Use it for multi-label classification where classes can overlap — e.g., "this voxel is both TC and WT." Each channel is a separate binary decision.

**Q66: Why is sigmoid correct for BraTS tumor regions?**
> BraTS tumor regions are nested: WT ⊃ TC ⊃ ET. A voxel in the enhancing tumor is simultaneously ET, TC, and WT. Softmax would force a single label per voxel. Sigmoid allows all three to be 1 simultaneously. This matches the biological reality that these regions overlap.

---

## 16. Training Dynamics

**Q67: What is overfitting and how do you detect it?**
> Overfitting = model memorizes training data but fails on new data. Detection: training loss decreases but validation Dice stops improving or decreases. Our trainer tracks both. Mitigation: data augmentation, weight decay, dropout, and saving the best validation checkpoint.

**Q68: What is underfitting and what causes it?**
> Underfitting = model can't learn the training data well. Causes: model too small, learning rate too high, insufficient training. Detection: both training loss and validation Dice are poor. If embed_dim=24 (M1 constraint) limits capacity, the model might underfit compared to embed_dim=48.

**Q69: What is the bias-variance tradeoff?**
> High bias = model too simple, underfits (e.g., linear model for segmentation). High variance = model too complex, overfits to training noise. The sweet spot minimizes total error. OncoSeg with 2.9M parameters is relatively lightweight — we rely on augmentation and regularization rather than model size to manage this tradeoff.

**Q70: What is a learning rate and what happens if it's too high or too low?**
> The learning rate controls step size in parameter space. Too high: training diverges, loss explodes or oscillates. Too low: training converges very slowly, may get stuck in bad minima. We start at 1e-4 (standard for AdamW) and decay to 1e-6. The cosine schedule transitions smoothly from exploration to refinement.

---

## 17. Model Architecture Theory

**Q71: What is the encoder-decoder paradigm?**
> Encoder compresses input into a compact representation (captures "what" is in the image). Decoder expands it back to full resolution (recovers "where" things are). The bottleneck forces the model to learn a compressed, semantic representation. For segmentation, we need both: understanding what a tumor looks like (encoder) and precisely delineating its boundary (decoder).

**Q72: What are residual connections and why do they help?**
> Residual connections add the input to the output: y = F(x) + x. The network only needs to learn the residual F(x) = y - x, which is easier than learning y directly. They solve the degradation problem (deeper networks performing worse than shallow ones) and improve gradient flow. Used in our cross-attention skip (residual after attention) and implicitly in skip connections.

**Q73: What is the information bottleneck in your model?**
> The deepest encoder layer has the lowest spatial resolution but highest channels (384 with embed_dim=48). This is the bottleneck — all information about the input must pass through this compressed representation. We apply MC Dropout here because: (1) uncertainty at the bottleneck propagates to all output voxels, (2) it's the most information-dense point in the network.

**Q74: What is the difference between a parameter and a hyperparameter?**
> Parameters are learned during training (weights, biases) — OncoSeg has 2.9M of them. Hyperparameters are set before training: learning rate (1e-4), embed_dim (24), dropout rate (0.1), roi_size (96), batch_size (1). Hyperparameters determine the search space; parameters are found by optimization within that space.

---

## 18. Practical Deep Learning

**Q75: Why do you need a GPU for training?**
> 3D convolutions and attention involve massive matrix operations. A single forward pass on a 128³ volume involves billions of floating point operations. GPUs have thousands of cores optimized for parallel matrix math — 10-100x faster than CPUs. An M1 MPS GPU handles it in ~1.5 hours per epoch vs potentially 10+ hours on CPU.

**Q76: What is mixed precision training and would it help?**
> Use FP16 for most operations, FP32 for critical ones (loss, gradients). Halves memory usage, speeds up computation on modern GPUs with tensor cores. Would definitely help — we could double roi_size or embed_dim within the same memory. MPS support for mixed precision is limited, but CUDA benefits greatly.

**Q77: What is gradient accumulation?**
> Simulate a larger batch by accumulating gradients over N mini-batches before updating weights. With batch_size=1 and accumulation=4, effective batch_size=4 without 4x memory. We don't currently use it, but it would help on memory-constrained devices.

**Q78: What is transfer learning and could you use it?**
> Use a model pretrained on a large dataset, then fine-tune on your task. MONAI provides Swin Transformer weights pretrained on large medical imaging datasets. We could initialize our encoder with these weights for faster convergence and potentially better results, especially with limited data.

---

## 19. Clinical & Domain Knowledge

**Q79: What is RECIST 1.1 and why is it the standard?**
> Response Evaluation Criteria in Solid Tumors, version 1.1 (2009). Standardizes how tumor response is measured in clinical trials. Defines target lesions, measurement methods (longest axial diameter), and response categories (CR/PR/SD/PD). Used by FDA for drug approval decisions. Without RECIST, comparing treatment effects across trials would be impossible.

**Q80: What are the RECIST response categories?**
> - **CR (Complete Response)**: all target lesions disappeared
> - **PR (Partial Response)**: ≥30% decrease in sum of diameters
> - **SD (Stable Disease)**: neither PR nor PD criteria met
> - **PD (Progressive Disease)**: ≥20% increase in sum of diameters, or new lesions
> These thresholds are clinically validated and universally used.

**Q81: What is the blood-brain barrier and why does it matter for T1ce?**
> The blood-brain barrier normally prevents contrast agents from entering the brain. When a tumor breaks down this barrier (angiogenesis in aggressive tumors), gadolinium contrast agent leaks in and lights up on T1ce MRI. This is the enhancing tumor (ET) — the most clinically aggressive component. Detecting ET accurately is critical for treatment decisions.

**Q82: What is peritumoral edema?**
> Swelling around the tumor caused by fluid leaking from abnormal blood vessels. It appears bright on FLAIR and T2 images. In BraTS, it's label 1 (part of whole tumor but not tumor core). It's clinically relevant — extensive edema may indicate more aggressive disease, but it's not tumor tissue itself.

---

## 20. Software Engineering & MLOps

**Q83: How do you ensure reproducibility in ML experiments?**
> Fixed random seeds (42) for PyTorch, NumPy. Hydra logs the complete config for every experiment. W&B tracks metrics, hyperparameters, and code state. Git for code versioning. Requirements pinned in pyproject.toml. Docker could further lock the environment. Anyone can reproduce our results by running the same config.

**Q84: What is the difference between a unit test and an integration test?**
> Unit test: tests one component in isolation (e.g., CrossAttentionSkip produces correct output shape). Integration test: tests components working together (e.g., full forward pass through OncoSeg with real-shaped input). We have both — module tests are unit tests, model tests are integration tests. An ideal addition: a mini training loop test on 2 samples.

**Q85: What is CI/CD and why does it matter for ML projects?**
> Continuous Integration: automatically run tests, linting, formatting on every push. Continuous Deployment: automatically deploy new versions. Our GitHub Actions CI catches regressions before they reach main. For ML specifically, CI prevents: broken model code from being committed, style inconsistencies, and import errors.

---

## 21. Conceptual / Big Picture

**Q86: What is the difference between classification, detection, and segmentation?**
> Classification: "is there a tumor?" (one label per image). Detection: "where is the tumor?" (bounding box). Segmentation: "which exact voxels are tumor?" (per-voxel labels). Segmentation is the most precise and what OncoSeg does. For RECIST, you need segmentation-level precision to measure diameters accurately.

**Q87: What is semantic segmentation vs instance segmentation?**
> Semantic: label every voxel with a class (all tumor voxels get "tumor"). Instance: distinguish individual objects (tumor 1 vs tumor 2). OncoSeg does semantic segmentation — every voxel gets TC/WT/ET labels. For RECIST measurement, we then apply connected component analysis to separate individual lesions (instance-level).

**Q88: What is multi-task learning and does your model do it?**
> Multi-task learning trains one model on multiple objectives simultaneously. OncoSeg has elements of this: the main segmentation task plus deep supervision auxiliary tasks at different resolutions. The temporal attention module (when used) adds treatment response comparison as another task. Shared representations across tasks improve feature learning.

**Q89: If you could only use one metric to evaluate your model, which would you choose and why?**
> Mean Dice across all three regions (TC, WT, ET). It directly measures overlap quality, handles class imbalance, is the BraTS standard, and a single number makes comparison easy. But in practice, you always want multiple metrics — HD95 catches boundary errors that Dice misses, and per-region breakdown reveals which tumor component is hardest.

**Q90: What would happen if you trained on MSD but tested on BraTS?**
> Domain shift — different scanners, protocols, patient populations. Performance would likely drop. Both are brain tumors with similar labels, so the drop shouldn't be catastrophic, but it would be measurable. This is why multi-dataset evaluation (KiTS23, LiTS, BTCV) is planned — to test generalization across different tumor types and anatomies.
