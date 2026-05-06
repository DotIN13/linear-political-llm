# Option 3: Political Profiling Bias Detection

## Working Title
Token Scoring for Political Profiling Bias: Detecting Demographic-Conditioned Ideological Assumptions in Vision-Language Models

## Abstract
We investigate political profiling bias in vision-language models through a unified token-scoring and validation framework. The method computes political scores at the token level from image-conditioned generations and aggregates them into calibrated ideological predictions. We use this base to test whether demographic attributes induce systematic political assumptions, such as over-attributing minority individuals to a particular party orientation. To isolate profiling effects, we construct matched demographic counterfactuals that hold scene, occupation, and context fixed while varying protected-attribute presentation, and we evaluate with paired statistical tests, calibration diagnostics, and subgroup fairness metrics. Results show that several models exhibit measurable demographic-conditioned political shifts even in politically neutral contexts, with disparities amplified by co-occurring socioeconomic and attire cues. Token-evidence analysis localizes where profiling enters generation, enabling targeted auditing and mitigation. Our findings suggest that political profiling is a distinct fairness failure mode in multimodal systems and that token-level validation offers a transparent mechanism for detection, quantification, and intervention assessment.

## Outline

### 1. Introduction
- Problem: models may infer political orientation from demographic presentation rather than evidence.
- Stakes: fairness, representation harm, and downstream decision support risks.
- Goal: detect and quantify political profiling bias with interpretable token evidence.

### 2. Related Work
- Demographic bias and stereotype propagation in ML.
- Fairness in multimodal generation systems.
- Calibration and subgroup reliability.

### 3. Method: Token Scoring + Validation Base
- Token-level political scoring and image-level aggregation.
- Reliability calibration and uncertainty reporting.
- Validation pipeline:
  - Controlled non-political baselines.
  - Demographic matched-counterfactual pairs.
  - Cross-model replication.

### 4. Dataset and Counterfactual Construction
- Demographic strata and protected-attribute representation.
- Neutral-context scenario design.
- Counterfactual generation preserving non-demographic factors.
- Annotation and audit checks for confound leakage.

### 5. Bias Evaluation Protocol
- Metrics:
  - Paired mean shift in political score.
  - Subgroup calibration gaps.
  - Equalized error-rate style diagnostics for ideological attribution.
- Hypothesis tests and multiple-comparison control.
- Intersectional analysis (demographic x attire x context).

### 6. Results
- Evidence of demographic-conditioned political score shifts.
- Subgroup disparities in uncertainty and calibration.
- Cases where minority presentation is over-associated with specific political labels.
- Cross-model consistency and divergence.

### 7. Mitigation and Re-Evaluation
- Prompt-level and decoding-level mitigation attempts.
- Data-balancing and counterfactual augmentation strategies.
- Post-mitigation re-test under the same paired protocol.

### 8. Limitations and Ethics
- Sensitivity of demographic labeling and representation boundaries.
- Harm-aware reporting and privacy-preserving release constraints.
- Caveats on normative interpretation across political contexts.

### 9. Conclusion
- Token-level political scoring provides a reproducible audit path for detecting and tracking political profiling bias in VLMs.
