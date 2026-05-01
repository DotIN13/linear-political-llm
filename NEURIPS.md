# Narrative Draft for NeurIPS Submission

## Working Title
Linear Political Directions in Language and Vision-Language Models: Probing, Steering, and Cross-Modal Validation

## One-Sentence Thesis
Political orientation in modern LLM and VLM representations is partially linear and intervention-ready: we can detect it with simple probes, steer it at inference time, and evaluate both capability and limits across text and image-conditioned settings.

## Abstract-Style Narrative (Draft)
Large language models and vision-language models often express politically inflected behavior, but it is unclear whether these behaviors reflect coherent internal structure or brittle prompt-level artifacts. We study whether political orientation is encoded as a linear direction in hidden activations, and whether this direction is controllable through lightweight inference-time interventions. Using U.S. ideological supervision (DW-NOMINATE) and synthetic behavioral data, we train ridge probes over selected activations, identify high-signal heads, and steer model outputs with controllable strength. We evaluate across text generation tasks, voting preference tasks, and image-conditioned political perception settings, including paired red-blue tie manipulations and demographic controls. We find strong, monotonic steerability in Llama-family models, weaker or unstable effects in Qwen-family models, and evidence that ideological and behavioral axes are related but not identical. Across modalities, matched-pair evaluations reveal measurable shifts under controlled visual interventions, while robustness analyses highlight architecture- and alignment-dependent limits. Our results support a practical middle ground between passive auditing and expensive fine-tuning: linear probes provide interpretable diagnostics and actionable controls, but their transfer is model-family specific and requires rigorous paired evaluation.

## The Core Story Arc

### Act I: Why this problem matters
Political bias and ideological framing are now practical deployment risks, not just philosophical concerns. Current work often reports output-level behavior without identifying whether models contain stable internal directions that can be measured and controlled. This creates a gap between diagnosis and intervention.

### Act II: What we test
We ask three linked questions:
1. Is there a linearly accessible ideological direction in model activations?
2. Can we steer this direction continuously at inference time?
3. Does this transfer beyond text-only settings into vision-language judgments under controlled image edits?

### Act III: How we test it
We train ridge probes on activation features supervised by ideological targets, select high-correlation components, and apply additive steering with coefficient alpha. We evaluate both language outputs and image-conditioned outputs, with matched-pair protocols designed to isolate a single manipulated factor.

### Act IV: What we find
Steering is reliable and often monotonic for Llama models, less reliable for Qwen models, and task-dependent in magnitude. Voting behavior is linearly probeable but not a trivial projection of the liberal-conservative axis. In VLM settings, paired visual manipulations produce significant directional shifts, but effect sizes vary by model and probe configuration.

### Act V: What this means
Linear control is real but conditional. It is useful for auditing and safety interventions, but it is not universal across architectures or alignment recipes. The right takeaway is not "we solved political bias," but "we can measure and partially control one important component with transparent tools."

## Introduction Narrative (Near-Ready Draft)
Politically loaded behavior in generative models is now routinely observed in both language-only and multimodal deployments. Yet most evaluations remain output-centric: they catalog behaviors but do not reveal whether models encode a coherent, controllable internal political axis. This distinction matters. If political behavior is only a surface artifact of prompt wording, intervention must happen through brittle prompt engineering. If, instead, behavior is partly supported by linear representational structure, then lightweight, interpretable controls may be feasible at inference time.

This paper investigates that second possibility. We test whether political orientation is linearly accessible in hidden activations, whether this representation can be steered continuously, and whether the same methodology remains informative in vision-language contexts where visual cues interact with political judgments.

Our approach is deliberately simple: ridge probes over activation features, head selection by correlation criteria, and additive activation steering with tunable strength. This simplicity is a design choice. We aim to measure what can be achieved without fine-tuning, architectural changes, or opaque reward-model loops.

Across text and image-conditioned experiments, we find consistent evidence of linearly accessible political structure, but with strong model-family effects. Llama-family models typically show robust steering-response curves; Qwen-family models often show attenuated or unstable responses. We also find that behavioral outcomes (for example, voting preference) are steerable but not reducible to a single ideological scalar, suggesting partially distinct latent dimensions.

These results contribute a practical and conceptual bridge: from descriptive bias auditing to intervention-oriented representation analysis. At the same time, they clarify limits of transfer and motivate paired, scenario-controlled multimodal evaluation as a default standard.

## Contributions (Claim-Safe Wording)
1. We present a unified probing-and-steering framework for political orientation that spans language and vision-language settings.
2. We provide evidence that political orientation is linearly accessible and controllable in several open-weight models, with measurable family-specific differences in steerability.
3. We introduce matched-pair multimodal evaluations that isolate visual political cues (for example, red-blue tie manipulations and accessory controls) to estimate directional effects with statistical testing.
4. We separate ideological and behavioral probing targets, showing that voting behavior can be linearly accessed while remaining only partially aligned with a liberal-conservative axis.
5. We document practical limits, including model-dependent failures and asymmetries, and outline reproducible evaluation protocols emphasizing correlation-based and paired-inference metrics.

## Section-by-Section Narrative Blueprint

### 1. Introduction
Lead with deployment relevance: political framing affects trust, fairness, and public discourse. Frame the technical gap as "auditing versus controllable representation understanding." End with your three research questions and a brief contribution list.

### 2. Related Work
Organize by four threads:
1. Bias and political behavior in LLMs.
2. Linear probing and representation geometry.
3. Activation steering and test-time control.
4. Multimodal bias and paired-image evaluation.
Position your work as the intersection: interpretable test-time control plus multimodal validation.

### 3. Method
Define notation clearly:
- h: activation feature vector.
- y: supervision target (ideology or behavior).
- f(h): ridge probe score.
- alpha: steering coefficient.
Explain head selection and why correlation is used for feature ranking. Keep this section mechanical and reproducible.

### 4. Experimental Setup
Split into language and vision-language subsections.
- Language: prompts, tasks, output parsing, metrics.
- VLM: paired generation protocol, scenario definitions, controlled factors.
Explicitly define one scenario unit as probe-prefix x model x probe-type x metric (your current notebook convention).

### 5. Results
Use the same order as your research questions.
1. Linear accessibility: probe correlations and stability.
2. Controllability: response curves over alpha, grouped by k and model.
3. Behavioral vs ideological axis: correlations and divergences.
4. Multimodal paired effects: per-scenario significance and effect sizes.
Keep each subsection anchored to a single claim sentence.

### 6. Analysis and Ablations
Highlight where the method fails or weakens:
- sensitivity to head subset k,
- model-family asymmetry,
- prompt-distribution dependence,
- potential RLHF interaction hypotheses.
Include at least one negative result table; NeurIPS reviewers value this.

### 7. Limitations and Ethics
Be explicit and proactive:
- U.S.-centric ideological axis,
- synthetic data and annotator/model bias,
- non-universality across model families,
- misuse risk of ideological steering.
Then state safeguards: transparency, paired testing, release constraints if needed.

### 8. Conclusion
Close with calibrated language: linear methods are useful diagnostics and partial controls, not complete solutions to political bias.

## Results Narrative Templates

### Template A: Main quantitative finding
Across [models/tasks], increasing alpha produces [monotonic/non-monotonic] shifts in [metric], with strongest effects in [model family]. This supports the claim that [target dimension] is intervention-accessible in activation space.

### Template B: Cross-model contrast
While [family 1] exhibits stable positive steering-response correlation, [family 2] shows attenuated or sign-inconsistent responses, indicating that representational linearity and controllability depend on architecture and alignment history.

### Template C: Multimodal paired inference
Under matched-pair visual controls, the mean score difference between red and blue conditions is [direction], with [test] indicating [significance/non-significance]. This suggests visual political cues influence downstream scoring even when non-target attributes are held fixed.

## Reviewer-Facing Positioning (Tone Guidance)
1. Be precise, not grandiose. Avoid "solves bias" language.
2. Treat negative/weak results as evidence about boundary conditions.
3. Emphasize reproducible protocols and explicit statistical tests.
4. Distinguish clearly between ideology detection, behavioral prediction, and normative alignment.

## Suggested Figure Storyline
1. Figure 1: End-to-end pipeline (probe training -> head selection -> steering -> evaluation).
2. Figure 2: Steering-response curves across alpha and k for each model.
3. Figure 3: Behavioral-vs-ideological correlation heatmap.
4. Figure 4: Multimodal paired red-blue effect sizes with confidence intervals.
5. Figure 5: Failure/instability cases (especially Qwen asymmetry).

## Statistical Reporting Language
For paired multimodal comparisons, report paired tests by default (paired t-test and Wilcoxon signed-rank where appropriate), effect sizes, and confidence intervals. For scenario-level summaries, report multiple-comparison control and avoid claims based solely on raw p-values.

## Final Paragraph You Can Reuse in the Paper
Our findings indicate that politically salient behavior in contemporary foundation models is neither fully opaque nor universally controllable. A meaningful component is linearly organized and can be steered with lightweight interventions, but transfer depends strongly on model family, task definition, and modality. This suggests a pragmatic research agenda: combine interpretable probing with paired, statistically grounded evaluations to map where control is reliable, where it fails, and how alignment procedures reshape the geometry of political representations.

## Practical Next Step
Convert this narrative into your paper by rewriting each section opener (first paragraph) using the corresponding blueprint above, then align every major claim with one table or figure to maintain NeurIPS-level claim-evidence discipline.
