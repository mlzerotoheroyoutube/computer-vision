# Convolutional Neural Networks

**Level:** Beginner  ·  **Topic:** Computer Vision

▶️ **Watch the video:** https://youtu.be/2gSITaqftKY

> A CNN stacks convolution, non-linearity and downsampling layers so that early layers detect edges, middle layers detect parts and late layers detect whole objects, ending in a classifier or a dense-prediction head. Weight sharing and locality make CNNs data-efficient; they dominated vision from 2012 and remain the backbone of efficient and dense-prediction models in 2026.

## The intuition
Think of how you recognise a friend in a crowd. You don't compare every pixel of their face to a memory. First your eyes notice simple things: lines, edges, colours. Then your brain groups those into shapes: an eye, a nose, a curl of hair. Then it groups the shapes into "that's Priya".

A CNN works the same way, in layers. Layer 1 is a set of small stencils (filters) that slide over the picture looking for tiny patterns, like a vertical edge or a patch of green. Layer 2 slides new stencils over the *results* of layer 1, so it finds combinations: a corner, a circle, a stripe. After every few layers the network **shrinks** the picture, for example keeping only the strongest value in each 2×2 block, so later stencils cover more of the original image. By the last layer, one number might light up for "cat face" and another for "wheel".

Concrete example: a 28×28 photo of a handwritten digit (784 pixels) goes in. After two rounds of "stencil + shrink" it might become 16 small 4×4 maps (256 numbers) that describe strokes and loops. A final simple layer turns those 256 numbers into 10 scores, one per digit, and the highest score is the guess. The network learns all its stencils by being shown thousands of labelled digits and nudged every time it is wrong.

## Core notes
- Canonical block: **Conv → Norm → ReLU/GELU**; stages separated by stride-2 downsampling that halve $H,W$ and typically double $C$.
- Typical shape trajectory (ResNet-50, 224 input): 3×224² → 64×112² → 256×56² → 512×28² → 1024×14² → 2048×7² → GAP → 2048 → 1000 logits.
- Total downsampling factor ("output stride") is usually 32 for classification, 8–16 for segmentation.
- **Receptive field** grows linearly with depth and multiplicatively with stride; global context needs depth, dilation or attention.
- **Global average pooling** replaced huge FC heads after Network-in-Network / GoogLeNet: fewer params, any input size.
- Training staples: Batch Normalization, He init (Weight Initialization), Data Augmentation, SGD+momentum or AdamW, cosine Learning Rate Schedules, label smoothing.
- CNNs are **texture-biased** and not robust to small shifts or adversarial perturbations by default.
- Pretrained CNNs are strong feature extractors for Transfer Learning.
- Still the default for real-time detection (YOLO), segmentation decoders (U-Net) and on-device models (Edge and On-Device ML).

## The math
**Layer.** For feature map $h^{(l-1)} \in \mathbb{R}^{C_{l-1}\times H_{l-1}\times W_{l-1}}$:
$$
h^{(l)} = \sigma\!\left(\operatorname{Norm}\!\left(W^{(l)} \star h^{(l-1)} + b^{(l)}\right)\right)
$$
where $\star$ is multi-channel cross-correlation, $W^{(l)}\in\mathbb{R}^{C_l\times C_{l-1}\times K\times K}$, and $\sigma$ is ReLU or GELU.

**Classifier head.** Global average pooling $z_c = \frac{1}{HW}\sum_{i,j} h^{(L)}_{c,i,j}$, logits $s = Az + a$, and loss
$$
\mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N} \log \frac{e^{s_{y_n}}}{\sum_{k} e^{s_k}}.
$$

**Receptive field** of the network after $L$ layers with kernel $K_l$ and stride $S_l$:
$$
r_L = 1 + \sum_{l=1}^{L} (K_l - 1)\prod_{m=1}^{l-1} S_m.
$$

**Parameter efficiency.** A conv layer has $C_l C_{l-1} K^2$ weights; a dense layer between the same feature maps has $C_l H_l W_l\, C_{l-1} H_{l-1} W_{l-1}$. For $64\times56\times56$ to $64\times56\times56$: about $3.7\times10^4$ vs $4.0\times10^{10}$.

**Compute per stage.** Halving $H,W$ and doubling $C$ keeps $H W C^2 K^2$ constant, which is why stages are balanced in FLOPs.

## Run the code
```bash
pip install torch torchvision
python convolutional_neural_networks.py
```

## Check yourself
<details><summary>Why do CNN stages halve spatial resolution while doubling channels?</summary>

Conv FLOPs scale as $HWC^2K^2$. Halving $H$ and $W$ divides cost by 4 while doubling $C$ multiplies it by 4, so each stage costs about the same. Downsampling enlarges the receptive field and builds abstraction; more channels give capacity to represent the larger variety of higher-level features. The pattern also yields a natural multi-scale feature pyramid for detection and segmentation.
</details>

<details><summary>What replaced the large fully connected layers at the end of AlexNet and VGG, and why?</summary>

Global average pooling followed by a single linear layer. VGG-16's FC layers hold roughly 120M of its ~138M parameters and overfit easily; GAP has no parameters, enforces a correspondence between channels and class evidence (enabling class activation maps), and allows arbitrary input resolution.
</details>

<details><summary>When would you choose a CNN over a Vision Transformer in 2026?</summary>

For limited labelled data without strong pretrained ViTs, for latency- or memory-constrained deployment (mobile, embedded, real-time detection), for high-resolution dense prediction where attention's quadratic cost hurts, and where robust off-the-shelf tooling matters. ViTs win for large-scale pretraining, multimodal alignment and when you can reuse a strong foundation model like DINOv2 or SigLIP. Modern CNNs such as ConvNeXt are competitive at matched recipes, so the training recipe often matters more than the family.
</details>

<details><summary>Compute the receptive field of three 3×3 stride-1 convs followed by a 2×2 stride-2 max-pool and another 3×3 conv.</summary>

Use $r_l = r_{l-1} + (K_l-1)\,j_{l-1}$, $j_l = j_{l-1}S_l$. Starting at $r=1,j=1$: three 3×3 convs give $r = 3, 5, 7$ with $j=1$. The pool gives $r = 7 + 1 = 8$, $j = 2$. The final 3×3 conv gives $r = 8 + 2\cdot2 = 12$. So each output unit sees a 12×12 input window.
</details>

<details><summary>Your CNN gets 99% train accuracy but 70% validation. What do you try?</summary>

Treat it as overfitting: stronger Data Augmentation (RandAugment, Mixup/CutMix), weight decay, dropout or stochastic depth, label smoothing, and early stopping. Switch to a pretrained backbone and fine-tune with a lower learning rate. Check for train/val distribution shift or leakage and do Error Analysis on misclassified examples. If data is tiny, freeze most layers and train only the head.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.