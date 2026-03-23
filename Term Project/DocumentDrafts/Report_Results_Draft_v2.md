# Section 6: Result Discussion (Version 2 – 3-Class Sentiment Classification)

## Experimental Setup Change

Version 2 of the pipeline introduces a fundamental change to the classification task: the binary sentiment labelling scheme is replaced with a **three-class formulation**. Rather than discarding neutral reviews (ratings 5–6) and mapping the remainder to Positive/Negative, all 62,000 reviews are retained and assigned to one of three classes:

- **Negative** – ratings 1–3
- **Neutral** – ratings 4–6
- **Positive** – ratings 7–10

To account for the inherent class imbalance in this expanded label space, the LinearSVC is trained with `class_weight='balanced'`. Evaluation metrics are reported as **macro-averages** across all three classes, weighting each class equally regardless of size. All other experimental controls (80/20 stratified split, `random_state=42`, `CountVectorizer` with `min_df=5`, square-root discounting for Scenario 5) remain identical to Version 1.

---

## V2 Results

Based on the 3-class experimentation, the performance results are as follows:

| Scenario | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| **Scenario 1:** All stopwords and punctuation retained | 0.7198 | 0.6177 | 0.6023 | 0.6091 |
| **Scenario 2:** Stopwords removed | 0.6468 | 0.5649 | 0.5595 | 0.5620 |
| **Scenario 3:** Punctuation removed | 0.6583 | 0.5840 | 0.5759 | 0.5796 |
| **Scenario 4:** Both stopwords and punctuation removed | 0.6499 | 0.5697 | 0.5635 | 0.5664 |
| **Scenario 5:** Frequency Discounting (Square-root rule) | 0.7151 | 0.6171 | 0.6023 | 0.6090 |

---

## V1 vs V2 Comparison

The table below places both pipeline versions side-by-side to isolate the effect of moving from binary to 3-class classification.

| Scenario | V1 Accuracy | V2 Accuracy | V1 F1 | V2 F1 (Macro) |
|---|---|---|---|---|
| Scenario 1: All retained | 0.8957 | 0.7198 | 0.9382 | 0.6091 |
| Scenario 2: Stopwords removed | 0.8703 | 0.6468 | 0.9179 | 0.5620 |
| Scenario 3: Punctuation removed | 0.8794 | 0.6583 | 0.9237 | 0.5796 |
| Scenario 4: Both removed | 0.8719 | 0.6499 | 0.9189 | 0.5664 |
| Scenario 5: Freq. Discounting | **0.8990** | 0.7151 | **0.9400** | 0.6090 |

Across all five scenarios, the transition to three-class classification produces a consistent and substantial reduction in both accuracy (~17–23 percentage points) and macro F1 (~33 percentage points). This is expected behaviour: introducing the neutral class dramatically increases task difficulty, as the boundary between mildly positive/negative reviews and neutral ones is linguistically ambiguous by nature. Macro averaging also penalises models more heavily when minority classes (here, the neutral class) are misclassified.

---

## Analysis

### Preprocessing Impact is Consistent Across Both Versions

The **relative ranking of scenarios is preserved** between V1 and V2, reinforcing the robustness of the preprocessing findings:

- Scenario 1 (all features retained) remains the strongest baseline in both versions.
- Scenario 2 (stopwords removed) delivers the worst performance in both V1 and V2, supporting **H1**: removing stopwords degrades classification performance. Stopwords such as "not", "very", and "quite" carry disproportionate sentiment signal that static removal discards.
- Scenario 3 (punctuation removed) consistently outperforms Scenario 2, suggesting punctuation is less informationally critical than function words, though still contributory.
- Scenario 4 (both removed) produces the second-worst macro F1 in V2, slightly above Scenario 2, confirming that cumulative deletion is uniformly harmful.

### Frequency Discounting Advantage Diminishes in the 3-Class Setting

The most notable divergence between V1 and V2 concerns **Scenario 5 (frequency discounting)**. In V1, square-root discounting of the count matrix produced the single best result across all scenarios (Accuracy: 0.8990, F1: 0.9400), clearly outperforming the unprocessed baseline (S1) by 0.33% accuracy and 0.0018 F1. In V2, this advantage essentially disappears: Scenario 5 (Accuracy: 0.7151, F1: 0.6090) falls marginally below Scenario 1 (Accuracy: 0.7198, F1: 0.6091), a difference of only 0.47% in accuracy and a negligible 0.0001 in F1.

This finding is interpretable: in binary sentiment, the primary discriminative signal lies in the presence or absence of strongly valenced words, and reducing the dominance of high-frequency neutral terms via discounting allows those signals to stand out more clearly. In the 3-class setting, however, the neutral class introduces a regime where high-frequency function words and hedging language may themselves serve as important class discriminators. Discounting these frequencies moderately flattens the representation, offering no net gain.

### Class Imbalance and the Neutral Class

The introduction of the neutral category (ratings 4–6) creates a structurally harder problem. Movie reviews on IMDb tend toward polarisation — reviewers are more likely to write either strongly positive or strongly negative reviews than to express measured neutrality. As a result, the neutral class is likely the smallest and most linguistically heterogeneous, making it harder for a bag-of-words SVM to classify reliably. The `class_weight='balanced'` parameter compensates for this imbalance during training, but the low macro F1 values across all V2 scenarios indicate that the neutral class continues to drive down aggregate performance.

---

## Section 7: Conclusion (V2)

The three-class extension of the IMDb62 sentiment classification pipeline confirms and extends the core findings of V1 while surfacing important new dynamics introduced by the neutral category. The fundamental conclusion that **stopword and punctuation removal is consistently harmful to classification performance** holds firmly across both experimental designs: every deletion scenario produces lower accuracy and macro F1 than the full-feature baseline, with stopword removal alone causing the largest individual degradation (~7.3 percentage points in accuracy in V2).

The more nuanced finding concerns frequency discounting. In V1, discounting unambiguously outperformed deletion and the raw baseline alike. In V2, discounting continues to outperform all deletion variants but loses its edge over the unprocessed baseline (S1), with the two approaches producing essentially equivalent macro F1 scores (0.6091 vs 0.6090). This suggests that the benefit of square-root frequency normalisation is most pronounced in **binary, polarised classification settings**, and that its advantage may diminish as the label space expands to include less clearly valenced categories.

Taken together, the two pipeline versions empirically support the theoretical position that **frequency is not noise** and that deletion-based preprocessing heuristics are not justified by the evidence. At the same time, they reveal that the optimal frequency treatment strategy is task-dependent: discounting confers a clear advantage in binary sentiment, but the choice between discounting and raw counting becomes marginal under a three-class formulation. Future work should investigate per-class performance breakdowns and explore whether discounting selectively helps the Positive/Negative boundary while neutral classification remains the primary bottleneck.
