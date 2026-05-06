# Option 1: Implicit Political Iconography in the Wild

## Working Title
Token-Level Political Scoring Reveals Implicit Iconography in Uncurated Real-World Images

## Abstract
We study whether political meaning in visual content can be detected without explicit political symbols by grounding analysis in token-level scoring and strict validation. Our method learns a calibrated political token score over generated text tokens conditioned on images, then aggregates per-token evidence into image-level ideological attribution with uncertainty estimates. We validate the score in three stages: (1) controlled synthetic checks with known political cues, (2) matched-pair perturbation tests that isolate non-semantic factors, and (3) human-verified wild-image benchmarks collected from news and social media contexts. Using this base, we test a central hypothesis: implicit iconography (composition, color palettes, scene context, attire style, crowd structure) carries politically associated signal even when overt symbols are absent. Across models and datasets, token scoring identifies robust signal above chance and remains informative under distribution shift, but effect sizes vary by model family and are strongest when contextual cues co-occur. Our results provide an interpretable framework for measuring latent political visual associations in the wild, while emphasizing calibration limits and the need for paired validation to avoid over-claiming.

## Outline

### 1. Introduction
- Problem: political iconography is often implicit, diffuse, and hard to quantify.
- Gap: output-level labels do not reveal where political evidence appears in generation.
- Thesis: token scoring with rigorous validation can recover subtle political visual signal.

### 2. Related Work
- Political bias in LLM/VLM systems.
- Token-level interpretability and attribution.
- Visual framing and political communication studies.
- Dataset shift and robustness evaluation.

### 3. Method: Token Scoring + Validation Base
- Define per-token political score from conditioned generation.
- Aggregate token scores into image-level score with confidence intervals.
- Calibration: reliability curves, expected calibration error, threshold tuning.
- Validation suite:
  - Synthetic sanity checks.
  - Matched-pair perturbations.
  - Human-audited wild benchmark.

### 4. Wild Implicit Iconography Dataset
- Collection protocol from public media contexts.
- Exclusion of explicit symbols for core split.
- Annotation schema for implicit cues and uncertainty.
- Train/validation/test and out-of-domain splits.

### 5. Experiments
- Main task: identify implicit political iconography in wild images.
- Robustness: geography, event type, media source, and time period.
- Cross-model transfer of token-scoring classifiers.
- Error taxonomy by cue type and ambiguity.

### 6. Results
- Quantitative performance and calibration metrics.
- Paired validation outcomes and significance tests.
- Qualitative token-evidence traces showing interpretable cue pathways.
- Failure modes under weak-context images.

### 7. Analysis
- Which cue families dominate token evidence.
- Interaction effects between context and attire.
- Sensitivity to prompt templates and decoding settings.

### 8. Limitations and Ethics
- Risk of reinforcing stereotypes through over-interpretation.
- Dataset and cultural scope constraints.
- Guidance for careful deployment and human oversight.

### 9. Conclusion
- Token-scoring-based measurement can detect implicit iconography, but claims must remain calibrated and validation-driven.
