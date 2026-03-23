# Section 6: Result Discussion

Our experiments evaluate the classification accuracy, macro-averaged precision, recall, and F1-score across four dataset pre-processing configurations and a fifth frequency discounting approach. The testing isolated the Support Vector Classifier architecture (LinearSVC) and only modified feature text extraction variables. The dataset was evaluated on a three-class formulation (Negative, Neutral, Positive) with `class_weight='balanced'` to accommodate the inherent class imbalance of the IMDB62 dataset.

Based on the experimentation, the performance results are as follows:

| Scenario | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| **Scenario 1:** All stopwords and punctuation retained | 0.7198 | 0.6177 | 0.6023 | 0.6091 |
| **Scenario 2:** Stopwords removed | 0.6468 | 0.5649 | 0.5595 | 0.5620 |
| **Scenario 3:** Punctuation removed | 0.6583 | 0.5840 | 0.5759 | 0.5796 |
| **Scenario 4:** Both stopwords and punctuation removed | 0.6499 | 0.5697 | 0.5635 | 0.5664 |
| **Scenario 5:** Frequency Discounting (Square-root rule) | 0.7151 | 0.6171 | 0.6023 | 0.6090 |

### Analysis

The empirical results reveal that traditional deletion-based preprocessing significantly degrades multi-class sentiment classification performance. 

1. **Stopwords and Punctuation carry discriminative signal**: 
Removing stopwords independently (Scenario 2) yields the lowest overall scores (Accuracy: 64.68%, F1: 0.5620)—a steep decline from the baseline (Scenario 1) across all metrics. This validates our hypothesis that high-frequency function words (such as negations or intensifiers) are not mere noisy artifacts but carry vital sentiment cues. Similarly, removing punctuation (Scenario 3) also harms performance compared to the baseline, supporting literature emphasizing the affective load contained in exclamation marks and sequence boundaries. Cumulative deletion (Scenario 4) predictably underperforms the baseline, reaffirming that discarding these linguistic features strips the model of valuable discriminative context.

2. **Frequency Discounting parallels raw baseline performance**: 
Applying square-root frequency discounting to the document-term matrix (Scenario 5) produces results virtually identical to the unprocessed baseline (Accuracy: 71.51% vs 71.98%; F1: 0.6090 vs 0.6091). While discounting does not strictly outperform the baseline in this 3-class environment, it vastly outperforms any deletion strategy. When moving to a multi-class problem incorporating a complex Neutral category, raw frequencies and discounted frequencies maintain comparable syntactic boundaries required by the balanced linear SVM to distinguish nuanced sentiment.

# Section 7: Conclusion

The systematic evaluation of text preprocessing heuristics on the IMDB62 corpus strongly suggests that arbitrary removal of stopwords and punctuation is detrimental to sentiment classification. By mapping ratings into a robust three-class sentiment boundary framework and evaluating via a class-weighted Support Vector Machine, this study conclusively demonstrates that high-frequency linguistic features encode substantial stylistic and affective information. 

Specifically, models lacking stopwords or punctuation suffered sharp declines in both macro-averaged F1-scores and overarching accuracy. While principled frequency-weighting schemes (such as square-root discounting) did not unequivocally dominate the unprocessed raw-frequency baseline, they consistently and vastly outperformed all deletion methodologies. These findings advocate a paradigm shift in text preprocessing for sentiment analysis: practitioners should prioritize preserving topological and syntactical signals over aggressive dimensionality reduction, defaulting to retention or continuous frequency discounting rather than the reflexive excision of 'stopwords'.
