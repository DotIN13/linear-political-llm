# Option 2: Cultural and Social Cue Ablations

## Working Title
From Ties to Hats: Token-Scoring Ablations of Cultural and Social Cues in Political Visual Association

## Abstract
We propose a controlled framework for identifying subtle political associations in visual representations using token scoring and multi-level validation. The base method assigns political scores to generated tokens conditioned on images and aggregates them into calibrated image-level estimates. We then perform structured cue ablations that manipulate specific cultural and social attributes (tie color/style, clothing formality, hats including MAGA-like variants, accessories, and scene decor) while preserving identity and scene semantics. Validation combines synthetic intervention checks, paired-image statistical tests, and cross-model consistency analysis to ensure measured effects reflect cue-specific changes rather than confounders. We find that small appearance modifications can shift political token evidence in predictable directions, with heterogeneous magnitudes across models and cue types. Clothing-formality cues and symbolic accessories are consistently high-impact, while single-color edits are weaker unless coupled with context. These results show that token-level measurement can disentangle subtle cue effects and expose compositional political associations that are not visible in coarse output labels.

## Outline

### 1. Introduction
- Motivation: political associations emerge from everyday social cues, not only explicit slogans.
- Challenge: existing evaluations conflate multiple visual factors.
- Contribution: token-scoring ablation protocol for cue-level causal evidence.

### 2. Related Work
- Bias and steering in multimodal models.
- Counterfactual and intervention-based evaluation.
- Cultural semiotics and political signaling in visual media.

### 3. Method: Token Scoring + Validation Base
- Per-token political scoring under image-conditioned generation.
- Image-level aggregation and uncertainty decomposition.
- Validation stack:
  - Controlled synthetic probes.
  - Matched-pair significance tests.
  - Cross-model agreement and disagreement diagnostics.

### 4. Cue-Ablation Benchmark Design
- Cue families:
  - Apparel (tie, suit/casual wear).
  - Headwear (MAGA-like hat, neutral hat, no hat).
  - Accessories and backdrop markers.
- Editing protocol to preserve identity, pose, and scene semantics.
- Quality control and edit-faithfulness checks.

### 5. Experimental Protocol
- Single-factor and multi-factor ablations.
- Additive versus interaction-effect experiments.
- Prompt-invariant evaluation settings.
- Metrics: effect size, significance, calibration, and transfer.

### 6. Results
- Ranking of cue families by political effect size.
- Interaction heatmaps for combined cue edits.
- Model-family sensitivity and robustness trends.
- Case studies showing token-evidence redistribution under ablation.

### 7. Discussion
- What constitutes a high-risk subtle cue.
- Implications for auditing generative and retrieval pipelines.
- Practical guidance for model evaluation before deployment.

### 8. Limitations and Ethics
- Cultural locality and non-universality of cue semantics.
- Risks of overgeneralizing from U.S.-centric symbolism.
- Recommended safeguards for public-facing systems.

### 9. Conclusion
- Cue ablations plus token scoring offer a practical, interpretable path to measure subtle political associations in visual models.
