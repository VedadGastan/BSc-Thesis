# Thesis Structure Guide
## "Detection of Digital Content Tampering Using Deep Hashing"

This document is a working blueprint, not thesis prose. Its job is to tell you — and any LLM you paste it into later — exactly what goes in each chapter, which numbers from `test_v11_128_bits.ipynb` back each claim, and which of your 19 `prompts.txt` runs feed which section. Use it as a checklist. When you sit down to write Chapter 6, you paste the relevant subsection of this guide plus your actual run logs into the LLM, not the whole notebook.

**Ground truth constraint**: only ONE run currently has real logged numbers — the baseline (`test_v11_128_bits.ipynb`, HASH_BITS=128, SEED=42, EPOCHS_FINETUNE=15). Every ablation number referenced below as "TBD" must come from you actually running that config and pasting the printed metrics back — do not let an LLM invent plausible-looking numbers for unrun experiments. That is fabrication of results, not a writing problem.

---

## Chapter 1 — Introduction

**Length**: ~4-6 pages. **Purpose**: motivate the problem, state what you built, state what you're claiming.

### 1.1 Problem description
- Digital image manipulation (splicing, copy-move, retouching) is now trivial with consumer tools; forensic detection has to keep pace.
- Two related but distinct problems exist in image forensics: (a) **binary tamper detection** — is this image authentic or manipulated, and (b) **content-based retrieval / near-duplicate detection** — given a suspect image, find visually/semantically similar images in a large database quickly. Your thesis sits at the intersection: you use **deep hashing** to produce compact binary codes that simultaneously support classification (a) and fast Hamming-space retrieval (b).
- State plainly why hashing specifically: real forensic pipelines (reverse image search, media provenance databases) need sub-linear search over millions of images — a plain CNN classifier gives you (a) but not (b). Learning-to-hash gives both from one forward pass.

### 1.2 Motivation
- Cite the scale problem: exhaustive nearest-neighbor search in float feature space is O(N·d); Hamming-space search over binary codes is O(N) with cheap XOR+popcount, and is further accelerated by hash-table lookups. This is the standard motivation in the learning-to-hash literature (Wang et al., CSQ paper, etc.) — you should cite the actual CSQ paper here (Yuan et al., "Central Similarity Quantization for Efficient Image and Video Retrieval," CVPR 2020) since your loss function is a direct, disclosed adaptation of it.
- State the gap you're addressing: most deep hashing work targets category-level retrieval (CIFAR, NUS-WIDE, ImageNet) with many classes; you're applying the same machinery to a **2-class forensic problem** (authentic vs. tampered) on CASIA v2, where class semantics are about manipulation status, not object category. This reframing is your novelty claim — be honest that it's an application/adaptation, not a new hashing algorithm.

### 1.3 Objectives
State as a numbered list, matched to what you actually did:
1. Design and train a CNN-based hashing network (EfficientNet-B0 backbone) that jointly outputs a binary hash code and a classification decision for tamper detection on CASIA v2.
2. Adapt Central Similarity Quantization (CSQ) to the 2-class forensic setting, adding an explicit intra-class diversity term to prevent hash collapse (all same-class images mapping to one code).
3. Evaluate detection performance (accuracy, AUC, F1) and retrieval performance (Precision@k, mAP) jointly, since a hash used only for classification does not need to be evaluated as a hash.
4. Run a systematic ablation study over loss components, architecture choices, and hyperparameters to identify which design choices actually drive performance versus which are cosmetic.
5. Characterize the failure modes and limitations of the approach, particularly around CASIA v2's known dataset artifacts (see 1.5).

### 1.4 Scope and non-goals
Be explicit about what this thesis does NOT claim, to preempt examiner pushback:
- Not a general-purpose tamper localization method (no pixel-level mask output — this is image-level binary classification only).
- Not evaluated cross-dataset (no NIST16, Columbia, COVERAGE, or IMD2020 generalization test) — flag this as a limitation, not hide it.
- Not compared against a modern forensic-specific baseline (e.g., a noiseprint/SRM-stream detector) in the current codebase — your model explicitly dropped an SRM residual stream ("Ablation: SRM forensic stream removed entirely" — this is in your Cell 4 docstring verbatim). This is worth a full paragraph: you started from a design that included a forensic noise-residual branch and removed it. State why (simplicity, RGB-only baseline, isolate what a plain fine-tuned classifier can do) and flag it as future work to reintroduce.

### 1.5 A note on CASIA v2 that you must address somewhere in the thesis
This is not optional — any examiner who knows the forensics literature will ask about it. CASIA v2 has a well-documented artifact: authentic and tampered images differ systematically in JPEG compression history / format metadata in a way that lets classifiers achieve deceptively high accuracy from low-level compression artifacts rather than genuine tamper cues (this is documented in multiple forensics papers examining CASIA v2 bias). You should:
- State this explicitly, ideally in the Introduction or in Chapter 6's discussion.
- Note that your own preprocessing (`transforms.Resize`, JPEG re-decode via PIL, no format-preserving pipeline) likely destroys some but not all of this signal.
- Frame your ~72% test accuracy / 0.79 AUC honestly against this backdrop — it is a modest, believable number for RGB-only detection on this dataset, not a state-of-the-art forensic result, and that's fine to say.

### 1.6 Thesis structure
One short paragraph, one sentence per chapter, listing Ch2 through Ch8 (or however many you land on — see the chapter list below).

---

## Chapter 2 — Related Work

**Length**: ~5-8 pages. Two literature threads to cover, kept clearly separated so the reader sees the intersection is deliberate.

### 2.1 Image tampering / forgery detection
- Classical approaches: noise-inconsistency, JPEG double-compression / block-artifact analysis, copy-move detection (block matching, SIFT/ORB-based).
- Deep learning approaches: CNN classifiers on RGB, dual-stream RGB+noise-residual networks (this is the direct precedent for the SRM branch your model removed — cite works like ManTra-Net, RGB-N / dual-stream forgery networks, and note your architecture's docstring explicitly references this lineage even though the branch was dropped).
- Position your work: image-level binary classification, not pixel-level localization.

### 2.2 Learning to hash / deep hashing for image retrieval
- Pairwise/triplet-based hashing (DPSH, DHN) — briefly.
- Central Similarity Quantization (CSQ) — this needs real depth since your loss function IS an adaptation of it. Explain: fixed class-specific hash centers at maximum Hamming distance, pull-toward-own-center / push-from-other-center loss using a sigmoid-based similarity term. Your `ForensicHashLoss` centers construction (`centers[0, :K//2]=1, centers[0, K//2:]=-1, centers[1]=-centers[0]`) is exactly the 2-class special case of CSQ's Hadamard/Bernoulli center generation — say so explicitly, this is good scholarship, not a weakness.
- Quantization loss (push continuous activations toward ±1) and bit-balance loss (push mean activation per bit toward 0) — standard components in DPSH/HashNet-family losses; cite accordingly.
- Note what's *not* standard CSQ: your `IntraDiversity` loss term is not part of the original CSQ formulation. Say clearly this is your own addition, explain the motivation (prevent within-class code collapse to a near-single code, which CSQ's original formulation doesn't explicitly guard against when a class has high intra-class visual diversity — exactly the situation with "tampered" images, which can look almost like anything). This is your actual novel contribution — make sure it's stated as such, not buried.

### 2.3 Positioning statement (closing paragraph)
One paragraph: "This thesis combines X and Y in application Z, with novel contribution W" — write this last, after the rest of the thesis is drafted, so it accurately reflects what you ended up doing including whichever ablations turned out to matter.

---

## Chapter 3 — Deep Hashing Fundamentals (Theoretical Background I)

**Length**: ~4-6 pages. This is the "teach the reader the math" chapter, written at the same level of rigor as the Bayesian-optimization reference thesis you supplied (equations numbered, one concept per subsection, a figure per concept where useful).

### 3.1 The retrieval problem and Hamming space
- Define Hamming distance, explain why binary codes + Hamming distance give O(1) per-pair comparison (XOR + popcount) versus O(d) for float cosine/L2 distance.
- Explain the core tension: hashing is a *discretization* problem — you need codes to be discriminative (semantically similar → close in Hamming space) AND binary (±1 or {0,1}), and the sign function is non-differentiable, which is the central technical difficulty the whole field works around.

### 3.2 Continuous relaxation and quantization loss
- Explain the standard trick: output continuous activations `h = tanh(z)` in [-1,1], train with a continuous surrogate loss, binarize at inference with `sign(h)`.
- Derive/explain the quantization loss term used in your Cell 5: `(|h| - 1)^2` — penalizes activations that sit away from ±1, i.e., penalizes information lost when you eventually threshold. Show this is exactly what's in `ForensicHashLoss.forward`: `quant_loss = (h.abs() - 1).pow(2).mean()`.
- Explain bit-balance loss: `h.mean(dim=0).pow(2).mean()` — pushes each bit's mean activation across the batch toward 0, which maximizes the entropy of that bit (each bit should be "used," roughly half +1 / half -1 across the dataset) — cite the standard argument that unbalanced bits carry less than 1 bit of information each.

### 3.3 Central Similarity Quantization (CSQ)
- Full derivation of the center-based loss you use, matched line-by-line to Cell 5/11 code:
  - Center construction: for 2 classes, centers at maximum Hamming distance (Hamming distance = HASH_BITS, i.e., exact complements). Show the `centers[0,:K//2]=1; centers[0,K//2:]=-1; centers[1]=-centers[0]` snippet and verify (as your own Cell 11 output does) that Hamming distance between centers = 128/128.
  - Similarity term: `sim = h @ centers.T / K`, an average per-bit cosine-like similarity in [-1,1].
  - Loss: `-logsigmoid(scale·own_sim) - logsigmoid(-scale·other_sim)` — explain this as a pairwise logistic loss pulling toward the own center and away from the other, with `CSQ_SCALE` controlling loss sharpness/temperature. Show the effect of scale as a hyperparameter — this sets up your D4 ablation (CSQ_SCALE=1.0 vs 2.0 vs 4.0) later.

### 3.4 Your intra-class diversity term (this is the thesis's own contribution — write this carefully)
- Motivate: CSQ alone optimizes only *inter-class* separation via fixed centers; nothing stops all same-class samples from collapsing onto (near-)identical codes, since the center-pull loss is satisfied equally whether within-class codes are diverse or identical.
- Derive the loss from Cell 5/11: pairwise Hamming distance `d = (K - h·hᵀ)/2` for all same-class pairs in a batch, penalize `relu(target_d - d)^2` — i.e., only penalize pairs that are *closer* than a target distance `TARGET_INTRA_D` (default HASH_BITS/4 = 32 bits). Explain why this is a hinge-style loss (no penalty once the target margin is met) rather than a naive "maximize all pairwise distances" loss (which would actively fight the CSQ pull-to-center term).
- This section should end with the empirical justification straight from your own notebook: cite the collapse-gradient sanity check from Cell 11's own output — `Div gradient at collapse: 8.9280 (should be well above 0)` — as a validation that the term is functioning as intended before any training happens.

### 3.5 Evaluation metrics for hashing
Define precisely, since you use all of these in Chapter 6:
- Precision@k, mAP (mean average precision) for retrieval — give the formula and explain your own implementation from Cell 35 (rank gallery by Hamming distance, compute cumulative relevance / rank).
- Bit balance, bit saturation, degenerate-bit count, intra/inter-class Hamming distance, separability ratio — these are the exact diagnostic metrics printed by your Cell 11 hash-quality analysis; define each one formally here so Chapter 6 can just report numbers without re-explaining.

---

## Chapter 4 — Neural Network Fundamentals (Theoretical Background II)

**Length**: ~4-5 pages. This mirrors Chapter 5 of your reference thesis (activation functions, loss functions, optimizers, CNN basics) — don't over-invest here, it's supporting material, not your contribution. Keep it tighter than the reference thesis's Chapter 5 since your reader already got the hashing math in Ch3.

### 4.1 CNNs and transfer learning (brief)
- Convolution/pooling recap (1 paragraph, cite standard references, don't reinvent LeNet diagrams — your reference thesis already has a good one, you can produce an analogous one if required by your program, but don't over-invest).
- Transfer learning: pretrained ImageNet backbone (EfficientNet-B0), rationale for freeze-then-finetune. Explain compound scaling briefly (EfficientNet's actual contribution: jointly scaling depth/width/resolution) since you should be able to justify *why* EfficientNet-B0 specifically and not a plain ResNet — note that you also ran a ResNet-50 comparison (ablation C1) precisely to test this choice; forward-reference it here.

### 4.2 Loss functions used
- Cross-entropy with label smoothing (0.05) for the classification head — explain label smoothing's regularization effect briefly.
- Tie back to Ch3 for the hash losses; this section should mainly cover the *combination*: your total loss is a weighted sum `LAMBDA_CLS·L_cls + LAMBDA_CSQ·L_csq + LAMBDA_DIV·L_div + LAMBDA_QUANT·L_q + LAMBDA_BALAN·L_b`, with baseline weights (1.0, 1.5, 0.3, 0.5, 0.5). State plainly that this is a 5-term weighted sum with no automatic weighting (no GradNorm, no uncertainty weighting) — the weights were chosen manually/by search, which is itself a limitation worth naming and which motivates your D2/D4/D6 hyperparameter sweeps.

### 4.3 Optimization
- AdamW, weight decay, gradient clipping (`clip_grad_norm_`, max_norm=1.0) — explain why clipping matters for a multi-term loss with potentially different gradient scales.
- Two-phase training schedule: explain explicitly, matched to Cell 7/Cell 15 code — Phase 1 (backbone frozen, LR_HEAD=3e-4, 5 epochs) warms up new heads without corrupting pretrained features; Phase 2 (full fine-tune, discriminative LRs: backbone at LR_BACKBONE=1e-5, heads at LR_HEAD=3e-4, `CosineAnnealingWarmRestarts` schedule, `EPOCHS_FINETUNE`=15) lets the backbone adapt. This discriminative-LR design is itself a hyperparameter choice you ablate (D5: LR_BACKBONE=5e-5).
- Early stopping logic: your Cell 15 code saves the checkpoint only when val AUC improves **and** val loss is within `LOSS_SLACK` (2%) of its best-ever value — explain this guards against the failure mode where AUC improves while the model is actually becoming miscalibrated/overfit on loss. State the patience value (6) and that training can stop before EPOCHS_FINETUNE is reached.

---

## Chapter 5 — Methodology

**Length**: ~8-10 pages, the "how it was built" chapter. Every subsection below maps to one or two notebook cells — write this chapter directly from the code, not from memory.

### 5.1 Dataset: CASIA v2
- Describe: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset — CASIA v2, 2 classes, `Au` (authentic) and `Tp` (tampered) folders.
- Report your actual split numbers verbatim from Cell 3's output:
  - Train: 8,829 (Au=5,243, Tp=3,586)
  - Val: 1,892 (Au=1,124, Tp=768)
  - Test: 1,893 (Au=1,124, Tp=769)
  - Total ≈ 12,614 images. Split: 70/15/15, stratified by label, via two chained `train_test_split` calls (`test_size=0.15` then `0.15/0.85` on the remainder) — show the exact code, and note the class imbalance (≈59% Au / 41% Tp) that motivates the `WeightedRandomSampler` in 5.2.
- Include the class-balance bar chart your Cell 2b already produces (Train/Val/Test × Au/Tp counts).

### 5.2 Data loading and augmentation
- Describe the transform pipeline from Cell 3 (Cell 4 markdown): resize to 224×224, horizontal+vertical flip, ±10° rotation, color jitter (brightness/contrast/saturation 0.15/0.15/0.08), random grayscale (p=0.05), ImageNet normalization. Eval transform: resize + normalize only, no augmentation — state this explicitly, it matters for reproducibility.
- **Important honesty note for this section**: several of these augmentations (color jitter, grayscale conversion, JPEG-adjacent resampling from `Resize`) plausibly interact with the CASIA v2 compression-artifact bias mentioned in 1.5 — you don't need to resolve this, but a sentence acknowledging the interaction shows methodological awareness.
- `WeightedRandomSampler`: explain the inverse-class-frequency weighting formula used (`1/class_count` per sample), and forward-reference ablation B6 (sampler removed) as the direct empirical test of whether this matters.

### 5.3 Model architecture: ForensicHashNet
Walk through Cell 4/9 top to bottom, this is the core diagram of your thesis (draw an actual architecture figure — backbone → fusion MLP → two parallel heads):
- Backbone: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")`, 3-channel RGB input, `num_features`-dim output (1280 for EfficientNet-B0 — verify this number against your actual `feat_dim` print if you added one, otherwise state "EfficientNet-B0's native 1280-dim pooled feature").
- Fusion block: Linear(feat_dim→1024) → BatchNorm1d → GELU → Dropout(0.4).
- Hash head: Linear(1024→512) → BN → GELU → Dropout(0.3) → Linear(512→128) → BN → (tanh applied outside the Sequential, in `forward`).
- Classification head: single Linear(1024→2), branching directly off the **fused features**, not off the hash — state explicitly why this matters (classification gradient does not directly reshape the hash code; this is a deliberate design choice you should defend or at minimum flag, since it's also the exact thing ablation Prompt 6 removes/inverts by deriving classification from hash-to-center distance instead).
- Report actual measured numbers from your Cell 9 output: hash shape [B,128], logits shape [B,2], **5.92M trainable parameters** (this is small because the backbone is frozen at that point in the sanity-check print — clarify in text whether this count is pre- or post-unfreeze, since it will change once Phase 2 unfreezes the backbone; check this before writing the number down, don't just copy it blindly).

### 5.4 Loss function assembly
Direct code walkthrough of Cell 5's `ForensicHashLoss` and Cell 6's `compute_loss`, referencing back to the formulas already derived in Ch3/Ch4. Report the baseline hyperparameter table:

| Symbol | Value | Role |
|---|---|---|
| LAMBDA_CLS | 1.0 | classification weight |
| LAMBDA_CSQ | 1.5 | CSQ center-pull weight |
| LAMBDA_DIV | 0.3 | intra-class diversity weight |
| LAMBDA_QUANT | 0.5 | quantization weight |
| LAMBDA_BALAN | 0.5 | bit-balance weight |
| CSQ_SCALE | 2.0 | CSQ logistic sharpness |
| TARGET_INTRA_D | 32 (=HASH_BITS/4) | diversity margin, bits |
| HASH_BITS | 128 | code length |

### 5.5 Training protocol
- Two-phase schedule, exact epoch counts (5 frozen + up to 15 fine-tune), batch size 32, image size 224, seed 42, `cudnn.deterministic=True`.
- Hardware: state what you actually trained on — the notebook output shows **Tesla T4, 15.6GB VRAM** (Google Colab) for this baseline run; note if your later real-world-style experiments (if you replicate the reference thesis's GTX 1650 setup) use different hardware, and say so per-experiment, not just once — training wall-clock time is a legitimate thing to report per hardware config.
- Early stopping / checkpointing logic exactly as coded (AUC-improvement + loss-slack criterion, patience=6).

### 5.6 Evaluation protocol
- Classification metrics: accuracy, balanced accuracy, AUC-ROC, F1 (binary, positive class = tampered), confusion matrix, sensitivity/specificity — all computed exactly as in Cell 21.
- Hash-quality diagnostics: saturation, bit balance, degenerate-bit count, intra/inter-class Hamming distance mean/std, separability ratio, zero-distance collapse rate — as in Cell 11.
- Retrieval metrics: Precision@k for k ∈ {1,5,10,20,50,100,200,500}, mAP, precision-recall curve — as in Cells 35-36, using train set as gallery and test set as query.
- State plainly that all of these are computed once per trained model (no cross-validation across the *architecture* — only across *seeds*, in the E3 ablation), and that "noiseless observations" (single train run per config, no repeated-seed averaging except where explicitly tested) is an assumption you're making for compute-budget reasons — say this openly, it is exactly the same assumption the reference AutoML thesis states for its own experiments, so you have precedent to cite for the practice, but you should still name it as a limitation on statistical confidence.

---

## Chapter 6 — Experimental Results

**Length**: your biggest chapter, 15-20+ pages. This is where all 19 prompts.txt runs land. Structure it as: baseline first, then grouped ablations, each with a consistent sub-structure (config → what changed → results table → 2-3 sentence interpretation). Do NOT interpret results you have not actually run — leave `[TBD — insert run numbers]` placeholders until you have them, and remove this instruction line before submission.

### 6.1 Baseline results (A1: HASH_BITS=128, all defaults, SEED=42)
This is the only run with numbers in hand right now. Report verbatim from your notebook outputs:

**Training** (Phase 2, best epoch 15, early-stop not triggered before 15):
- Best val AUC: 0.7948 (epoch 15)

**Test set** (Cell 21):
| Metric | Value |
|---|---|
| Accuracy | 0.7200 |
| Balanced Accuracy | 0.7184 |
| AUC-ROC | 0.7867 |
| F1 (tampered) | 0.6732 |
| Sensitivity (TPR, tampered recall) | 0.7100 |
| Specificity (TNR, authentic recall) | 0.7269 |
| Confusion matrix | TP=546, TN=817, FP=307, FN=223 |

**Hash quality** (Cell 11):
| Metric | Value | Target/healthy range (from your own code's inline comments) |
|---|---|---|
| Saturation (mean \|h\|) | 0.6447 | >0.85 — **below target** |
| Degenerate bits | 0/128 | <5 — met |
| Bit balance | 0.1460 | <0.05 — **not met** |
| Intra-class Hamming (Au / Tp) mean | 54.2 / 55.2 | 30-60 — met |
| Intra-class Hamming (Au / Tp) std | 36.8 / 28.1 | <20 — **not met** |
| Inter-class Hamming mean | 68.6 | >60 (ideally >80) — marginal |
| Separability ratio | 1.24× | >1.5 — **not met** |
| Zero-distance collapse | 0.0 / 0.0 | <0.01 — met |

**Retrieval**: mAP = 0.7315 (train-as-gallery, test-as-query).

**Write an honest paragraph here**: your own baseline's diagnostic thresholds (which you set yourself in the code) are met on only 3 of 7 hash-quality metrics. This is a genuinely interesting, reportable finding — the hash is functional for classification and retrieval but is not "high quality" by the standards the codebase itself defines. This is exactly the kind of finding that should motivate your ablation study, not be swept under the rug — a discussion of *why* separability ratio and bit balance miss target (candidate causes: LAMBDA_CSQ/LAMBDA_BALAN weight balance, CSQ_SCALE too low, 15 epochs insufficient) sets up section 6.x hyperparameter results nicely.

### 6.2 Loss-component ablations (B-series)
Table structure, one row per run, columns = Accuracy, AUC, F1, mAP, plus the hash-quality metrics from 6.1's table (or a subset — pick the ones that move):

- **B2** — `LAMBDA_DIV=0` (Prompt 2): tests whether the intra-diversity term (your own contribution, Ch3.4) matters at all. Expect this to be the single most important ablation for your novelty claim — if hash-quality/separability barely change with LAMBDA_DIV=0, that undercuts the contribution; if they degrade noticeably, that's your strongest result. [TBD — run and report]
- **B5** — `LAMBDA_CSQ=0` (Prompt 7): hash head still outputs bits but gets no center-pull signal. Per your own prompt, flag which secondary metrics (separability ratio, inter-class Hamming) become meaningless once centers aren't being pulled toward — say this explicitly in your results table with a footnote rather than silently reporting numbers that don't mean what the column header implies.
- **B6** — no `WeightedRandomSampler`, uniform sampling instead (Prompt 9): tests whether class-balancing at the sampler level matters given the loss already includes `label_smoothing` but no explicit class weighting in the CE loss itself. Expect this to show up mainly in specificity/sensitivity balance rather than raw accuracy.
- **"No cls head" ablation** (Prompt 6): architecturally the biggest change — classification derived from nearest-CSQ-center distance on the hash vector instead of a learned softmax head. This changes your evaluation pipeline too (accuracy now computed from Hamming/cosine distance-to-center, not logits) — write a subsection explaining the changed evaluation logic before reporting numbers, so the comparison to the baseline table is apples-to-apples-with-a-caveat, not silently different.

### 6.3 Backbone comparison (C1: EfficientNet-B0 vs ResNet-50)
- State the parameter-count and FLOPs difference between the two backbones (look these up — EfficientNet-B0 ≈5.3M params, ResNet-50 ≈25.6M params — cite standard sources, don't estimate).
- Report whether the fusion-layer input dimension change (feat_dim from EfficientNet's 1280 to ResNet-50's 2048) was the only required change, or whether the Phase 2 optimizer's parameter-group structure needed adjustment (your own Prompt 5 explicitly asks you to flag this — answer it in the text, don't leave it open).
- Results table: same metric columns as 6.1.

### 6.4 Hyperparameter sensitivity (D-series)
This is naturally a set of small sweep tables/plots, one per hyperparameter, each anchored on the baseline as the middle point where possible:

- **D1 — HASH_BITS ∈ {32, 64, 128}** (Prompts 11-12): report accuracy/AUC/F1/mAP as a function of code length. This is the standard "rate-distortion" plot in hashing papers — shorter codes should cost some retrieval mAP for storage/speed gain; show whether that tradeoff actually appears in your data. **Explicitly note in your methods write-up (not just as a footnote)**: any PCA/t-SNE/Hamming-plot cell that hardcodes `128` instead of reading `HASH_BITS` must be fixed before these runs are trustworthy — this was flagged in your own Prompt 11; resolve it before generating this section's figures, and say in-text that you did.
- **D2 — LAMBDA_DIV ∈ {0, 0.3(baseline), 0.6}** (Prompt 3, plus B2 from 6.2): a 3-point sweep on the diversity weight. This directly complements B2 — together they tell you whether more diversity weight keeps helping past the baseline or whether 0.3 is already near-optimal/over-weighted.
- **D3 — TARGET_INTRA_D ∈ {16, 32(baseline), 64}** (Prompts 13-14): sweep the diversity *margin* (in bits) independently of its *weight*. Note this is a genuinely separate axis from D2 — D2 changes how hard the term is enforced, D3 changes what target it's enforcing toward — make sure your discussion doesn't conflate the two.
- **D4 — CSQ_SCALE ∈ {1.0, 2.0(baseline), 4.0}** (Prompts 15-16): sharper vs. softer CSQ logistic loss.
- **D5 — LR_BACKBONE ∈ {1e-5(baseline), 3e-5(mentioned as "original"), 5e-5}** (Prompt 8): note your own prompt states baseline is 1e-5 but references "the original 3e-5 run" as a third point — clarify in the thesis which of these three is actually your reported baseline config (your Cell 1 config shows `LR_BACKBONE = 1e-5`), and reconcile this discrepancy explicitly before writing the results table, otherwise your baseline chapter (6.1) and this sweep will contradict each other.
- **D6 — LAMBDA_CSQ ∈ {1.0, 1.5(baseline), 3.0}** (Prompts 17-18).

For each D-series sweep: one small table + one line/bar plot (metric vs. swept value), 2-4 sentences of interpretation. Do not write more than that per sweep — with 6 sweeps this section is already long; save deeper synthesis for 6.7.

### 6.5 Seed variance (E3)
- Rerun A1 exactly, SEED=123 instead of 42 (Prompt 4).
- **This section must include the reminder your own prompt asked for**: state explicitly what needs to be true about the data split for this to test seed variance rather than accidentally reusing the same split. Looking at your Cell 3 code, `train_test_split(..., random_state=SEED)` is called with the global `SEED` — meaning changing `SEED` **does** change the train/val/test split, not just weight initialization and augmentation randomness. This is worth a full paragraph: it means your "seed variance" experiment actually conflates two sources of variance (data split variance + training stochasticity variance), which is a legitimate methodological limitation to name. If you want to isolate training-stochasticity variance alone, you'd need to fix the split and vary only `torch.manual_seed`/`WeightedRandomSampler` seeding separately — decide whether to do this properly or report the conflated number with the caveat stated plainly.
- Report the spread between the two runs on all headline metrics as your estimate of result stability, and use it to sanity-check whether differences seen in the D-series sweeps (6.4) are larger than this seed-noise floor — a sweep result smaller than your two-seed spread should be reported as "within noise," not as a real effect.

### 6.6 Post-hoc analyses on the baseline checkpoint (Prompt 10)
No retraining — three analyses on the already-trained baseline model and its cached test-set hashes/logits:
1. **Threshold sweep**: Accuracy/F1/TPR/TNR vs. classification threshold (varying the 0.5 cutoff on softmax probability). Useful for showing whether the default threshold is actually optimal for your class-imbalanced test set, and for letting a reader pick an operating point matching their false-positive tolerance.
2. **Precision@k and mAP vs. k** for retrieval — you already have the machinery from Cell 35-36; this section formalizes it as a swept analysis rather than a single number.
3. **Bit-truncation retrieval analysis**: PCA-rank the 128 hash dimensions, keep only the top-K (K=16,32,64,128), redo retrieval. **Label this correctly per your own prompt's instruction**: this is a *post-hoc truncation* analysis on a model trained at 128 bits, not a true HASH_BITS sensitivity result (that's D1, which retrains from scratch at each bit length). State this distinction explicitly in the section heading and in the first sentence, and compare the truncation curve against the true D1 sweep curve in the same figure if possible — the gap between them is itself informative (it tells you how much of a shorter code's performance loss is recoverable by just training at that length vs. how much is lost to truncating a model that was never optimized for that length).

### 6.7 Alternative loss function: DeepCauchyHashLoss (Prompt 19)
- Cao et al., CVPR 2018, "Deep Cauchy Hashing for Hamming Space Retrieval." Explain the core difference from your CSQ-based loss: pairwise Cauchy-distribution-based similarity loss (not fixed centers) + Cauchy quantization loss, no fixed binary centers at all — this is a genuinely different hashing paradigm (pairwise/similarity-preserving vs. center-based), worth its own short theory subsection back in Ch3 if you have room, or a compact recap here.
- This comparison, if run, is your strongest "is CSQ+diversity actually the right choice for this problem" result — treat it as the capstone comparison of Chapter 6, after all the CSQ-internal ablations. [TBD — no run present yet; this is listed "DONE!" at the top of your prompts file only as a marker that prompt-writing is done, not that the run itself has been executed — verify before citing any numbers here]

### 6.8 Summary table and cross-experiment discussion
One big consolidated table, every run as a row (A1 baseline, B2/B5/B6/no-cls-head, C1, D1-D6 all values, E3, plus the DeepCauchy comparison), same metric columns throughout. This is the table an examiner will actually study — get the formatting right (probably needs to be landscape/rotated in the final document, or split into "classification metrics" and "hash-quality metrics" tables).
Close the chapter with 1-2 pages answering directly: which single change moved accuracy/AUC the most; which changes were noise (per 6.5's variance floor); did the intra-diversity contribution (B2/D2/D3) actually help; did backbone choice matter more than loss-hyperparameter tuning (C1 vs. D-series) — that comparison is a genuinely interesting thing to highlight since it speaks to where a practitioner's tuning budget is best spent.

---

## Chapter 7 — Discussion

**Length**: ~3-4 pages, prose only, no new numbers. Pull threads together, revisit the CASIA v2 caveat from 1.5 against your final numbers, discuss what the hash-quality metrics (saturation 0.64, bit balance 0.146, separability 1.24×) actually mean for real-world deployment (e.g., a hash that's only 1.24× more inter- than intra-class separated is a genuinely weak retrieval signal compared to typical CSQ results on category-level benchmarks like CIFAR/ImageNet reported in the original paper, which is worth naming as a domain-difficulty finding rather than an implementation failure — 2-class forensic hashing on visually heterogeneous "tampered" images is a harder problem than N-way category hashing on visually coherent classes). Discuss limitations honestly: single train/test split per config (no k-fold), compute-constrained epoch count (15), no cross-dataset generalization test, RGB-only (no forensic residual stream).

## Chapter 8 — Conclusion and Future Work

**Length**: ~2 pages. Restate objectives from 1.3, state which were met, state numerically what you achieved (best config's headline numbers), and list concrete future work: reintroduce SRM/noise-residual stream, cross-dataset evaluation (NIST16/Columbia/IMD2020), automatic loss-weight balancing instead of manual LAMBDA search, pixel-level localization extension, full DeepCauchy comparison if not completed in time.

---

## Cross-cutting writing notes (apply to every chapter)

1. **Never let an LLM invent a number.** Every metric in Chapter 6 must trace to an actual printed notebook output you paste in. If you ask an LLM to "fill in a plausible result" for an unrun ablation, that is fabricated data in an academic thesis — treat any such output as a placeholder to be deleted, not a draft to be lightly edited.
2. **Every equation in Ch3/Ch4 should be checked against the actual code**, not written from a textbook and assumed to match — your loss function has specific, checkable details (e.g., the `1e-8` epsilon in the diversity-loss denominator, the exact sign convention in `own_sim`/`other_sim`) that a generic "here's how CSQ works" paragraph from an LLM will get subtly wrong if not grounded in your Cell 5/11 source.
3. **Reuse the reference thesis's formatting conventions** (numbered equations, `Algorithm` boxes for anything iterative, figure captions that state the finding not just the axes) since it's from the same faculty and presumably the expected house style — but don't reuse its content structure mechanically; your chapter list above already diverges from it where your problem requires it (you have no black-box-optimization chapter, for instance, since you're not doing hyperparameter search over architectures the way that thesis is).
4. **Figures to actually generate**: dataset split bar chart (5.1), architecture diagram (5.3), training curves (already in Cell 8/8b, 5.5), confusion matrix + ROC (6.1, from Cell 9b), hash activation histogram + per-bit balance chart (6.1, from Cells 28-29), t-SNE and PCA scatter of hash codes (6.1, from Cells 30-31), intra/inter-class Hamming distance histogram (6.1, from Cell 32), hash code heatmap sample (6.1, from Cell 33), margin distribution (6.1, from Cell 34), retrieval PR curve + Precision@k curve (6.1/6.6, from Cells 35-36), all D-series sweep line plots (6.4), threshold sweep and bit-truncation plots (6.6, from Prompt 10 code once written).
