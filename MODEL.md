# Thesis Context: Detection of Digital Content Manipulation Using Deep Hashing

## Project Summary

This thesis builds a deep hashing model for image forensics. The model takes a single image as input and outputs a compact binary hash code. That hash code encodes two things at once: whether the image has been tampered with, and if so, what type of manipulation was used. The model is trained on the CASIA v2 dataset.

---

## The Problem

Digital images are routinely manipulated before being shared online. Two manipulation types dominate real-world forgery:

- **Copy-move**: A region is copied from one part of an image and pasted elsewhere within the same image, usually to hide or duplicate something.
- **Splicing**: A region from one image is inserted into a different image entirely.

Existing detection methods fall into two categories:

1. **Binary classifiers** (authentic vs. tampered): These work reasonably well but throw away forensic information. They cannot tell you what kind of manipulation occurred.
2. **Pixel-level localization models** (e.g. segmentation networks): These can identify exactly which pixels were tampered, but they are computationally heavy, produce no compact representation, and cannot be efficiently queried against a large database.

Neither category produces a **compact, queryable forensic signature** that encodes manipulation type.

---

## What Deep Hashing Is

A deep hashing model maps an image through a neural network to produce a short binary string, called a hash code, for example 64 bits. Hash codes can be compared using Hamming distance (counting differing bits), which is extremely fast even at database scale.

In standard image retrieval, hashing is used to find similar images quickly. This thesis repurposes the hashing framework for forensics: instead of encoding visual similarity, the hash encodes forensic class membership.

The key property being exploited: **similar hash codes mean same forensic class, distant hash codes mean different forensic class**.

---

## The Specific Gap Being Filled

No published paper has trained a deep hashing model where the learning objective explicitly structures the hash space by manipulation type (authentic, copy-move, splicing) using the CASIA v2 dataset. Papers that use deep hashing for tampering detection treat it as a binary problem. Papers that distinguish manipulation types do not produce a compact binary hash. This thesis closes that gap.

The closest related work:

- **SmartHash (CIKM 2024)**: DCT-based perceptual hashing on CASIA v2, binary (authentic vs. tampered), not learned end-to-end.
- **SFTA-Net (PeerJ 2025)**: Uses triplet loss for copy-move and splicing detection, but is not a hashing model and produces no binary hash code.
- **MADPHash (ACM MM 2025)**: Manipulation-aware deep perceptual hashing, but binary only.
- **Dual-branch spatial+frequency on CASIA 2.0 (arXiv 2509.05281, 2025)**: Binary classification, 77.9% accuracy, no manipulation-type encoding.

---

## The Model: Triplet Deep Hashing with Manipulation-Aware Metric Space (TDHMAM)

### Core Idea

Train a CNN to produce binary hash codes such that the Hamming distance between two hashes reflects their forensic relationship:

- Two authentic images: **small** Hamming distance
- Two copy-move images: **small** Hamming distance
- Two splicing images: **small** Hamming distance
- An authentic vs. a copy-move image: **large** Hamming distance
- An authentic vs. a splicing image: **large** Hamming distance
- A copy-move vs. a splicing image: **large** Hamming distance

This is learned through triplet supervision. Each training triplet consists of:

- **Anchor**: any image
- **Positive**: an image from the same forensic class as the anchor
- **Negative**: an image from a different forensic class

The loss pushes the anchor closer to the positive than to the negative in Hamming space.

### Three Classes

The CASIA v2 dataset provides three natural classes:

| Class | Count |
|---|---|
| Authentic | 7,491 images |
| Copy-move tampered | 3,295 images |
| Splicing tampered | 1,828 images |

### Architecture

```
Input Image (256x256 RGB)
        |
   [Preprocessing]
   - Resize to 256x256
   - Normalize (ImageNet mean/std)
        |
   [Backbone: ResNet-50]
   - Pretrained on ImageNet
   - Remove final classification layer
   - Output: 2048-dimensional feature vector
        |
   [Hash Layer]
   - Fully connected: 2048 -> N bits (e.g. 64)
   - Tanh activation (continuous relaxation during training)
   - Sign function at inference (binarization)
        |
   Output: N-bit binary hash code
```

The backbone is frozen for the first few epochs, then unfrozen for end-to-end fine-tuning. This is standard practice when fine-tuning pretrained CNNs on small forensic datasets.

### Loss Function

The training loss has two components:

**1. Triplet loss (metric learning)**

For each triplet (anchor a, positive p, negative n):

```
L_triplet = max(0, d(a, p) - d(a, n) + margin)
```

Where d() is Hamming distance (approximated by L2 distance on continuous hash values during training). The margin is a hyperparameter, typically 0.5.

**2. Classification loss (auxiliary)**

A small classification head is attached to the hash layer during training. It receives the hash code and outputs a 3-class softmax prediction. Cross-entropy loss is computed against the ground truth forensic class.

```
L_total = L_triplet + lambda * L_classification
```

Lambda controls the balance between metric learning and classification. Typical value: 0.1 to 0.5.

The classification head is removed at inference. Only the hash code is used.

**Why both losses?** Triplet loss alone can produce degenerate solutions where the network learns one large cluster separation but does not form compact intra-class clusters. The classification loss forces the hash code to be directly predictive of the forensic class, which prevents this.

### Inference

At inference time:

1. Feed the query image through the backbone and hash layer.
2. Apply sign() to get the binary hash code.
3. Compute Hamming distance to cluster centroids (precomputed from training set).
4. Assign to the nearest centroid class: authentic, copy-move, or splicing.

This is O(N) in hash length and O(K) in number of classes. For the 3-class case it is essentially instantaneous.

Alternatively: compute Hamming distance to all database hashes and retrieve the K nearest neighbors. This enables use as a forensic retrieval system.

---

## Why Copy-Move and Splicing Should Be Separable

Copy-move and splicing leave statistically different traces:

- **Copy-move**: Creates self-correlation within the image. The source and destination regions share pixel statistics, texture, noise, and DCT quantization patterns. Block-matching features and self-similarity maps respond strongly to this.
- **Splicing**: Creates cross-image inconsistency. The inserted region has different camera noise profiles, lighting direction, color temperature, and JPEG compression history than the host image. These inconsistencies show up in frequency domain statistics and noise analysis.

A CNN trained with class-discriminative loss should learn to pick up on these different artifact signatures without explicitly being told what to look for. This is the working hypothesis of the model and it is supported by the fact that classification-based models (ResNet, VGG) have achieved above-chance three-class performance on CASIA v2 even without hashing.

---

## Inputs and Outputs

| Property | Value |
|---|---|
| Input | Single RGB image, any resolution (resized to 256x256) |
| Output at training | Continuous hash vector (tanh-activated) |
| Output at inference | Binary hash code (N bits, e.g. 64) |
| Classification output | One of: authentic, copy-move, splicing |
| Hash comparison metric | Hamming distance |

---

## Dataset: CASIA v2

- Source: Chinese Academy of Sciences Institute of Automation
- 7,491 authentic images (JPEG and TIFF, various content)
- 3,295 copy-move tampered images
- 1,828 splicing tampered images
- Total: 12,614 images
- Ground truth masks are available (but not used by this model, which operates at image level)
- Known issues: some label noise, inconsistent forgery quality, some duplicate images

The dataset class imbalance (authentic >> splicing) must be handled during training, either by weighted sampling or by weighted loss.

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| 3-class accuracy | Overall correct classification |
| Per-class precision and recall | Performance on each forensic class separately |
| Hamming distance distributions | Whether the three classes actually separate in hash space |
| Mean Average Precision (mAP) | Quality of hash-based retrieval |
| Hash bit variance | Whether bits are informative (degenerate bits are always 0 or 1) |

The Hamming distance distribution plots are especially important for this thesis because they directly show whether the core claim, that the hash space is structured by manipulation type, is true.

---

## What Success Looks Like

A successful model shows:

1. Above 80% three-class classification accuracy on the test set.
2. Clearly separated Hamming distance distributions between classes (intra-class distances smaller than inter-class distances on average).
3. Hash bit variance close to 0.5 per bit (meaning bits are informative, not degenerate).
4. The copy-move vs. splicing discrimination is above chance (above 50% for that binary sub-problem).

A model that achieves high authentic-vs-tampered accuracy but cannot distinguish copy-move from splicing is a partial success, not a full success. This distinction should be reported honestly.

---

## What This Model Is Not

- It does not localize the tampered region at pixel level.
- It does not produce a confidence score (binary hash only).
- It does not detect AI-generated (deepfake) content, only traditional copy-move and splicing.
- It is not a watermarking scheme. It requires no modification of the original image.
- It will not match the localization accuracy of dedicated segmentation models like MantraNet or MVSS-Net. That is not its purpose.

---

## Key References

- Du et al. (2018). Image Hashing for Tamper Detection with Multiview Embedding and Perceptual Saliency. Advances in Multimedia.
- Hussain et al. (2022). An Efficient Supervised Deep Hashing Method for Image Retrieval. Entropy.
- Samanta and Jain (2024). SmartHash: Perceptual Hashing for Image Tampering Detection and Authentication. CIKM.
- Alabrah (2025). SFTA-Net: Triplet Loss and Spatial Attention for Copy-Move and Splicing Detection. PeerJ CS.
- Dong et al. (2013). CASIA Image Tampering Detection Evaluation Database.