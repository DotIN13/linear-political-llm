Dear Reviewers,

Thank you for the thoughtful feedback and we appreciate the opportunity to address the points raised.

**Scope and Generalizability**
We acknowledge the limitation regarding the focus on the U.S. liberal-conservative spectrum. Our intention was to begin with a well-defined and extensively studied ideological axis, allowing for clearer benchmarking and interpretability. We agree that extending the analysis to other ideological dimensions such as libertarian-authoritarian or global ideologies is a valuable direction. We have identified World Value Survey (WVS) as a starting point for generalizing our findings to a broader scope. We will clarify this as a limitation in the discussion and are actively exploring broader ideological representations in ongoing work.

**Model Choice**
We appreciate your notes about model diversity. While our experiments centered on Llama-2-7b for consistency and reproducibility, we agree that alignment procedures can significantly influence model behavior. We have replicated our work on Llama-2-7b, Llama-3.1-8b, Qwen-2.5-7b, and Qwen-2.5-14b. The results show that while the steering effects are strong on Llama family models, they are less effective with Qwen models, and especially the steering of a **voting dimension** we probed using another LLM-generated voting preference dataset. We will include the limitation in the discussion, provide the testing results on these models in the appendix, and continue to explore how these linear dimensions generalize to latest models, and why some models are less sensitive to steering (potentially due to RLHF).

**Voting Task Variance**

We acknowledge the concern regarding inconsistency in the voting task. We suspected that the liberal-conservative dimensions do not directly correlate, so we probed the voting dimension using another LLM-generated voting preference dataset containing 600 voting dialogues, an example looks like:

> USER: During the 2020 contest, who was your preferred candidate?
> ASSISTANT: After watching Biden console grieving military families I felt his empathy is what the nation needs so I choose him.

We also expanded the prompt set to 40 prompts per configuration of $k + \alpha$ and analyzed the Pearson correlation between $\alpha$ (steering strength) and the Trump-voting ratio. A higher correlation indicates stronger, more controllable ideological shifts.

The results are summarized in the table below:

| Model                     | k=8    | k=16  | k=32   | k=64   | k=96   |
| ------------------------- | ------ | ----- | ------ | ------ | ------ |
| **LLaMA-2–7B-chat**       | 0.734  | 0.545 | 0.781  | 0.977  | 0.999  |
| **LLaMA-3.1–8B-instruct** | 0.754  | 0.762 | 0.864  | 0.759  | 0.576  |
| **Qwen-2.5–7B-instruct**  | 0.560  | 0.713 | 0.434  | 0.312  | -0.127 |
| **Qwen-2.5–14B-instruct** | -0.597 | 0.240 | -0.761 | -0.480 | -0.184 |

These findings show:
* The voting behavior dimension is linearly accessible, especially in Llama models, and could be orthoganal to a certain extent to the liberal-conservative dimension. We are working to find a way to quantify the correlation of these two dimensions, ideological and behavioral.
* We also noticed inconsistent or even negative correlations in Qwen models, indicating that such ideological features may be encoded differently, perhaps due to RLHF, or less specialization of attention heads.

We will include these results in the voting task, extend the current section with a discussion of the correlation between ideological & behavioral dimensions, and also include a description of the new voting dataset and model outputs in the appendix to help readers understand the evaluation process. Additionally, we acknowledge the limitation of the current method on certain family of models in the discussion and limitation section.

**Figure Clarity**
Lastly, thank you for pointing out the issue with Figure 1. We have updated the figure with a legend for the colors to aid interpretation.

---


Dear Reviewer,

Thank you for your detailed and constructive feedback. We appreciate the opportunity to clarify our contributions and address the concerns raised.

**Ideological Dimensions**

We agree that focusing exclusively on the liberal–conservative axis is a limitation and appreciate your suggestion. Our initial goal was to anchor the study around a well-studied, interpretable axis, allowing clearer benchmarking and validation. However, as we experiment with the political reasoning tasks and especially the voting task, we are became particularly interested in how ideological and behavioral dimensions interact, that is, why ideological dimension does not directly influence voting behavior. To this end, in our revision we incorporated a *voting dimension probe* using a separate LLM-generated dataset to study behavioral alignment. The new probing dataset consists of 600 voting statements like follows:

> USER: During the 2020 contest, who was your preferred candidate?
> ASSISTANT: After watching Biden console grieving military families I felt his empathy is what the nation needs so I choose him.

We use the prefill activation to train probes on the voting dimension. We found that the voting behavior dimension is linearly accessible, especially in Llama models, and could be orthoganal to a certain extent to the original liberal-conservative dimension. We are working to find a way to quantify the correlation of these two dimensions, ideological and behavioral. The results of steering along this dimension are included in the voting task section. Additionally, we will expand our discussion section to include potential future directions such as probing other ideological spectrums—e.g., libertarian–authoritarian—using frameworks like the World Values Survey (WVS).

**Dataset Size and Source**

We acknowledge the limitations of using a synthetic dataset and thank you for pointing this out. In response, we have expanded the dataset from 120 to **500 entries**. To support transparency and reproducibility, we will publish the full dataset on **Hugging Face** and include a representative sample in the appendix. Each entry includes a name, political domain, and a corresponding ideological response. For example:

```
name,domain,response  
Raúl M. Grijalva,Abortion,"As a staunch advocate for reproductive rights, I firmly believe that every individual should have the autonomy to make their own healthcare decisions, including the right to access safe and legal abortion services."  
Jaime Herrera Beutler,Abortion,"As a pro-life advocate, I believe in protecting the sanctity of life while also supporting policies that provide women with the resources and support they need during challenging times."  
```

We agree that using GPT-4o-mini could introduce unknown biases, but since our goal in this substudy is to understand the shift of classification distribution across different $k$ and $\alpha$, these biases doesn't directly harm the soundness of the experiment. However, we agree that we should additionally include real-world bias-detection datasets such as the [Manifesto Database](https://manifesto-project.wzb.eu/) to strengthen our argument.

**Prompt Diversity and Experimental Rigor**

We really appreciate your comment regarding prompt variability. To address this, we expanded our prompt templates for the tasks at hand. For example, for the voting task, we increase the number of different prompts to 20. Examples include:

* “If you had to vote in the 2019 U.S. political context, would you choose Joe Biden or Donald Trump for president?”
* “In your opinion, who would be the better choice for the next U.S. president in 2020—Biden or Trump?”

## **Model Transparency**

Thank you for pointing out the lack of clarity regarding the models used. In our updated manuscript, we now explicitly state that we evaluated our pipeline on the following four models:

* **LLaMA-2–7B**
* **LLaMA-3.1–8B**
* **Qwen-2.5–7B**
* **Qwen-2.5–14B**

These models were selected to represent different architectures and alignment strategies. Comparative results across models are provided in both the main text and appendix.

## **Result Presentation and Figure 2 Variance**

We agree that the fluctuation in Figure 2 needs further discretion. To address this, we have rerun key experiments with multiple prompt samples and  measured the **Pearson correlation between the steering coefficient (α)** and the **outcome metric** (e.g., liberal-bias ratio for bias-detection, Trump voting ratio for voting tasks, and average liberal-leaning rated by LLM for rewriting tasks). This quantitative method provides a more stable and interpretable signal than raw classification counts.

## **Typo**

Thank you for catching the typo in line 529. This has been corrected in the revised manuscript.

We are grateful for your thoughtful suggestions. These revisions have strengthened our work considerably.

---


Dear Reviewer,

Thank you for your thoughtful and constructive feedback. We appreciate the opportunity to clarify several aspects of our methodology and share upcoming revisions and improvements based on your suggestions.

**Dataset Methodology and Transparency**

The bias-detection dataset referenced in lines 277–294 was generated using GPT-4o-mini and consists of entries labeled by politician name, domain (e.g., abortion, healthcare), and an ideological response. We agree that the current description was insufficient. To address this, we will **release the full dataset publicly on Hugging Face**, and include **a representative sample in the appendix** for clarity. Each entry of the dataset follows a format such as:

```
name,domain,response  
Raúl M. Grijalva,Abortion,"As a staunch advocate for reproductive rights..."  
Jaime Herrera Beutler,Abortion,"As a pro-life advocate, I believe in protecting the sanctity of life..."
```

We agree that using GPT-4o-mini could introduce unknown biases, but since our goal in this substudy is to understand the shift of classification distribution across different models and prompts, these biases don't directly harm the soundness of the experiment.

**Bias in Probe Training**

Thank you for raising the important point about potential bias in probe training. Our probe is not trained on GPT-generated text; instead, it is based on template-generated statements, following the method of [Kim et al., 2024](https://arxiv.org/abs/2503.02080). Specifically, the training prompt is:

> `USER: Generate a statement by [NAME], a politician in the United States.`
> `ASSISTANT: In 2019, [NAME] said that ...`

Probes are trained to map prefill activations to the **DW-NOMINATE first-dimension scores** of U.S. legislators. This approach minimizes linguistic artifacts from generation models and focuses more on ideologically structured representation.

**Use of Human Data**

We agree that including human-generated data would strengthen the evaluation. While we initially avoided datasets like the Manifesto Project due to documented annotation biases (e.g., [Bjelobaba 2024](https://gupea.ub.gu.se/handle/2077/83668), [Volkens et al. 2013](http://dx.doi.org/10.1111/1467-9248.12015)), we now recognize the value in comparing synthetic and real data.

In response, we will include evaluation with selected entries from the Manifesto Project as a **parallel experiment** in the bias detection section. This will allow us to more robustly assess generalization and better contextualize the learned ideological directions.

**Bias Neutralization Evaluation**

Thank you for pointing out the ambiguity in our current evaluation of rewriting. At the time of writing, we lacked a scalable quantitative metric for these rewrites. Since then, we have adopted a more systematic approach:

* We now perform quantitative evaluation by applying GPT-4o to score rewritten statements, measuring the correlation between steering coefficient $α$ and `liberal-leaning ratios` across varying $k$.
* While not perfect, GPT-4o achieves \~70% agreement with human judgment in our verification sample, and allows scalable analysis.
* We are also experimenting with a **"re-embedding" strategy**: applying the probe to the rewritten text and measuring how much the steering has shifted the average score of the probes, compared to that of the neutralized text written by the original model. This approach may provide a better understanding of the relative effectiveness of the steering.

We will clarify this evaluation protocol and include correlation reports in the updated manuscript.

**LoRA and Fine-tuning Comparison**

We agree that the distinction between inference-time steering and parameter updates (e.g., direct finetuning, LoRA) is critical to both the scientific and engineering contributions of this work. We appreciate the suggestion.

While our focus in this paper is on activation steering, we plan to conduct comparative experiments with LoRA fine-tuning if time permits. In particular, we are interested in whether LoRA models **overwrite**, **shade**, or **preserve** the original linear dimensions. This could offer deeper insight into whether finetuning (RLHF, LoRA) masks undesirable behaviors or helps models encode ideological reasoning more robustly.

We will include any experiments conducted in a new section before Discussion and frame it as a compelling avenue for future work in the Discussion section.

**Answers to Questions**

- 227: Yes, we use the first-dimension scores of DW-NOMINATE as the target for our probes.
- 316: We actually use 40 different prompts for the voting task. Additionally, in the revision, we calculate the correlation between $\alpha$ and Trump-voting ratio over these prompts to enhance the robustness of the experiment. We will clarify this in the text.
- 386: Attention heads with highest k pearson correlation between the training dataset DW-NOMINATE scores and the Ridge model outputs are selected. We will clarify this in 3. Methodology.
- 392: We will add a description of the bias detection dataset in 3.2.1, and include a sample in the appendix.
- 394: This is a typo, no labels were given for the bias detection dataset. We only observe the shift of the classification distribution across different $k$ and $\alpha$, but not the absolute performance on a dataset. We will revise this in the text.
- 529: This is indeed a bad citekey. We will add the correct citation for human confirmation bias in the revised manuscript.
- 592: It is a hypothesis that the asymetric steering is due to the model's RLHF training. We will clarify this in the text and include a discussion on how RLHF may affect the linear dimensions we probe.

Once again, thank you for the insightful feedback. These comments have significantly improved the direction of our work.
