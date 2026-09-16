# BSc Thesis - Companion Code

**Vedad Gaštan · ETF Sarajevo · 2026**

Companion code for the bachelor thesis
*"Detection of Digital Content Tampering Using Deep Hashing"*.

This repository contains the Jupyter notebooks that train, evaluate, and ablate
the `ForensicHashNet` model - an EfficientNet-B0 backbone with a hash head and a
classification head, trained with a Central Similarity Quantization (CSQ) loss
adapted to two antipodal class centers plus an intra-class diversity term.

---

## 1. Repository layout

Every file is a self-contained Jupyter notebook. There is no shared package:
each notebook sets its own hyperparameters, downloads the dataset, defines the
model and loss, trains, and evaluates, so that an ablation differs from the
baseline in **exactly one** design choice.

| Notebook | Role | What changes vs. baseline |
|---|---|---|
| `test_v11_128_bits.ipynb` | **Baseline** (K=128) | - |
| `test_v11_no_csq.ipynb` | loss ablation | `LAMBDA_CSQ = 0` (CSQ term removed) |
| `test_v11_no_IntraDiversity.ipynb` | loss ablation | `LAMBDA_DIV = 0` (diversity term removed) |
| `test_v11_2x_IntraDiversity.ipynb` | loss ablation | `LAMBDA_DIV = 0.6` (doubled from 0.3) |
| `test_v11_no_cls_head.ipynb` | architecture ablation | classification head removed; classes assigned by **nearest CSQ center** (`classify_from_hash`) |
| `test_v11_no_weighted_sampler.ipynb` | training ablation | `WeightedRandomSampler` → plain `shuffle=True` |
| `test_v11_new_backbone.ipynb` | architecture ablation | EfficientNet-B0 → **ResNet-50** (`timm.create_model("resnet50")`, feat_dim 2048) |
| `test_v11_32_bits.ipynb` | code-length sweep | `HASH_BITS = 32` |
| `test_v11_64_bits.ipynb` | code-length sweep | `HASH_BITS = 64` |
| `test_v11_TARGET_INTRA_D=16.ipynb` | diversity-margin sweep | `TARGET_INTRA_D = HASH_BITS / 8` |
| `test_v11_TARGET_INTRA_D=64.ipynb` | diversity-margin sweep | `TARGET_INTRA_D = HASH_BITS / 2` |
| `test_v11_CSQ_SCALE=1_0.ipynb` | CSQ-scale sweep | `CSQ_SCALE = 1.0` |
| `test_v11_CSQ_SCALE=4_0.ipynb` | CSQ-scale sweep | `CSQ_SCALE = 4.0` |
| `test_v11_LAMBDA_CSQ=1_0.ipynb` | CSQ-weight sweep | `LAMBDA_CSQ = 1.0` |
| `test_v11_LAMBDA_CSQ=3_0.ipynb` | CSQ-weight sweep | `LAMBDA_CSQ = 3.0` |
| `test_v11_larger_LR_BACKBONE.ipynb` | training ablation | `LR_BACKBONE = 5e-5` (from 1e-5) |
| `test_v11_DHC_loss.ipynb` | loss-family comparison | CSQ replaced by **Deep Cauchy Hashing** (pairwise Cauchy + Cauchy quantization, `GAMMA_CAUCHY = HASH_BITS/8`) |
| `test_v11_diff_seed.ipynb` | split-robustness check | `SPLIT_SEED = 123` for `train_test_split` (training seed stays 42) |
| `test_v11_post_hoc.ipynb` | post-hoc analysis | baseline model + threshold sweep, Precision@k/mAP vs k, bit-truncation retrieval (top-K PCA-ranked bits) |

> The 19 notebooks map one-to-one onto the rows of the ablation table in
> Chapter 6 (Results) of the thesis.

---

## 2. Environment

- **Python** 3.10+
- **PyTorch** 2.x with CUDA (GPU required for training)
- Key packages:

```
pip install torch torchvision timm scikit-learn kagglehub scipy matplotlib pillow numpy tqdm pandas
```

- **Hardware used:** single NVIDIA Tesla T4, 15.6 GB VRAM.
- Training is **not** supported on CPU-only machines (a full run is ~20–40 min
  on the T4; CPU would be impractically slow).
- `torch.backends.cudnn.deterministic = True` is set; with `SEED = 42` fixed
  throughout, runs are reproducible on the same hardware. Minor numerical
  differences may appear across GPU architectures.

---

## 3. Dataset

CASIA v2 is downloaded at runtime via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset")
DATA_ROOT = Path(path) / "CASIA2"      # contains Au/ and Tp/
```

- **Kaggle auth:** set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables,
  or place a `kaggle.json` in `~/.kaggle/`.
- **Contents:** 12,614 images - 7,491 authentic (`Au/`) + 5,123 tampered (`Tp/`).
- **Split:** 70 / 15 / 15, stratified by the authentic/tampered label only,
  `random_state = SEED` (= 42; = `SPLIT_SEED` only in `test_v11_diff_seed`).
  Resulting counts: Train 8,829 (Au 5,243 / Tp 3,586) · Val 1,892 · Test 1,893
  (Au 1,124 / Tp 769).

>
> **Known caveats (also documented in the thesis):**
> 1. **Format/compression confound.** CASIA v2 authentic and tampered images
>    differ systematically in file format (tampered are largely TIFF, authentic
>    are JPEG/BMP). A metadata-only classifier can separate the classes, so the
>    raw split does **not** give a clean estimate of forensic capability. See
>    the "Revisiting the CASIA v2 Caveat" section of the thesis.
> 2. **Source-image leakage.** The split is stratified by label only; tampered
>    filenames encode their authentic source image, so a tampered test image
>    frequently has its source in the training set - which is also the retrieval
>    gallery. This inflates retrieval metrics. The audit notebook
>    `test_v11_source_leakage_count.ipynb` quantifies the overlap.
>

---

## 4. Reproducing a result

1. Set up Kaggle credentials (see §3).
2. Install dependencies (see §2).
3. Open the notebook for the configuration you want (see §1 table), and
   **Run All** top to bottom. `kagglehub` downloads the dataset on first run
   and caches it; subsequent runs reuse the cache.
4. Each notebook ends by printing the same metric block used to fill the thesis
   results table:
   - **Classification:** Accuracy, Balanced Accuracy, AUC-ROC, F1, Sensitivity,
     Specificity, Confusion Matrix.
   - **Hash quality:** bit saturation, intra/inter-class Hamming distance,
     separability ratio, zero-distance collapse rate.
   - **Retrieval:** Precision@k (k ∈ {1,5,10,20,50,100,200,500}), mAP, PR curve.
   - **Figures:** training curves, confusion matrix, ROC, hash-quality
     histograms, PCA/t-SNE of codes, retrieval examples.

`torch.save(model.state_dict(), "best_model.pt")` writes the best checkpoint
into the notebook's working directory.

---

## 5. Baseline hyperparameters (`test_v11_128_bits.ipynb`)

```python
SEED            = 42
HASH_BITS       = 128
BATCH_SIZE      = 32
IMG_SIZE        = 224

EPOCHS_FROZEN   = 5        # frozen-backbone phase
EPOCHS_FINETUNE = 15       # discriminative-LR fine-tuning
LR_HEAD         = 3e-4
LR_BACKBONE     = 1e-5     # raised to 5e-5 in test_v11_larger_LR_BACKBONE
WEIGHT_DECAY    = 1e-4
PATIENCE        = 6        # early stopping on val loss
LOSS_SLACK      = 0.02     # slack in early-stopping criterion

LAMBDA_CLS      = 1.0      # classification cross-entropy
LAMBDA_CSQ      = 1.5      # central-similarity quantization
LAMBDA_DIV      = 0.3      # intra-class diversity hinge
LAMBDA_QUANT    = 0.5      # tanh-quantization regularizer
LAMBDA_BALAN    = 0.5      # bit-balance regularizer
CSQ_SCALE       = 2.0      # CSQ logit scale
TARGET_INTRA_D  = HASH_BITS / 4.0   # diversity hinge margin (32 bits at K=128)
```

---

## 6. Model and loss (summary)

**`ForensicHashNet`** - EfficientNet-B0 (`timm`, pretrained, `num_classes=0`,
`global_pool="avg"`) → fusion MLP (`feat_dim → 1024`, BN+GELU+Dropout) →
two heads:
- `hash_head`: `1024 → 512 → HASH_BITS` (with final BatchNorm).
- `cls_head`: `1024 → 2` (branches from fused features, **not** from the hash,
  so classification gradients do not reshape the code).

In `test_v11_new_backbone`, the backbone is `resnet50` (`feat_dim = 2048`).
In `test_v11_no_cls_head`, `cls_head` is removed and labels are assigned by
`classify_from_hash(h, centers)` = `argmax(h @ centersᵀ / K)`, which for the
antipodal ±1 centers is monotonic in Hamming distance.

**`ForensicHashLoss`** has five parts:
- **CSQ** - pull each hash toward its class center and away from the other,
  using fixed **antipodal** binary centers at maximum Hamming distance
  (`centers[0] = [+1…+1, −1…−1]`, `centers[1] = −centers[0]`; verified at
  runtime as `128/128 bits` apart).
- **IntraDiversity** - hinge on the *soft* Hamming distance of same-class
  pairs, active only when a pair is closer than `TARGET_INTRA_D`
  (`relu(target_d − d_ij)²`). Prevents collapse to a single code per class.
- **Quantization** - `tanh`-based regularizer pushing activations to ±1.
- **Balance** - penalizes per-bit mean drift from 0 (keeps bits balanced).
- **Classification** - standard cross-entropy on `cls_head` logits.

In `test_v11_DHC_loss`, the CSQ center term is replaced by the **Deep Cauchy
Hashing** pairwise loss (Cao et al., CVPR 2018): Cauchy cross-entropy on
pairwise same/different labels + Cauchy quantization, with
`GAMMA_CAUCHY = HASH_BITS / 8`.

---

## 7. Evaluation protocol

- **Classification** - single forward pass on the 1,893-image test set with
  the best checkpoint (chosen by validation loss with early stopping).
- **Hash quality** - intra/inter-class Hamming distances and the separability
  ratio are estimated from a random sample of pairs on the **binarized**
  (`sign()`) test codes; `n_pairs = min(3000, len(au), len(tp)) = 769` for
  the inter-class estimate at K=128.
- **Retrieval** - the **training set is the gallery** (8,829 images) and the
  **test set is the query set** (1,893 images), ranked by Hamming distance.
  Relevance is **shared class label** (authentic/tampered), **not**
  instance-level near-duplicate. mAP and Precision@k are computed in-code via
  cumulative-relevance / rank. The chance baseline for this class-label
  relevance is `E[mAP|random] = 0.594² + 0.406² ≈ 0.518`.

---

## 8. Confidence intervals and the DeLong test

Single-run results carry sampling uncertainty. The thesis reports binomial
95% Wilson confidence intervals for test accuracy (≈ ±2.0 p.p. at n = 1,893,
p = 0.72) and, for the baseline-vs-`larger_LR_BACKBONE` AUC comparison, a
**paired DeLong test** on saved per-image predicted probabilities.

To reproduce the DeLong test:

1. In `test_v11_128_bits.ipynb` and `test_v11_larger_LR_BACKBONE.ipynb`, after
   the test-evaluation cell, save the predicted probabilities and labels:
   ```python
   import numpy as np
   np.save("preds_baseline_probs.npy", np.asarray(all_probs))
   np.save("preds_baseline_labels.npy", np.asarray(all_labels))
   ```
   (and `preds_lr5e5_*.npy` for the larger-LR notebook).
2. Run `delong_ci.py` (provided alongside this README), which loads both
   prediction sets and prints the paired DeLong `z`, two-sided `p`, and the
   per-model AUC confidence intervals. The two notebooks must share the
   **same test set** - guaranteed because both use `SEED = 42` for the split;
   the script asserts this.

---

## 9. Source-image leakage audit

`test_v11_source_leakage_count.ipynb` reconstructs the exact `SEED = 42` split,
parses the source token from each tampered filename, and counts how many of the
769 tampered test queries have at least one authentic source image in the
8,829-image training gallery. The number it prints is reported in the
"Source-Image Leakage" section of the thesis.

---

## 10. License and attribution

- The dataset is CASIA v2 by Dong, Wang & Tan (2013), redistributed on Kaggle
  by user `divg07`; use of the dataset is governed by its original license.
- The model uses `timm` (Ross Wightman) and PyTorch.
- The CSQ loss follows Yuan et al. (2020); the DCH loss follows Cao et al.
  (2018). Full references are in the thesis bibliography.
- This companion code is released for academic reproducibility of the thesis.
