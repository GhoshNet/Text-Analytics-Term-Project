# Section 6: Result Discussion

Our experiments evaluate the classification accuracy, precision, recall, and F1-score across four dataset pre-processing configurations and a fifth frequency discounting approach. The testing isolated the Support Vector Classifier architecture (LinearSVC) and only modified feature text extraction variables. 

Based on the experimentation, the performance results are as follows:

| Scenario | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Scenario 1:** All stopwords and punctuation retained | 0.8957 | 0.9309 | 0.9456 | 0.9382 |
| **Scenario 2:** Stopwords removed | 0.8703 | 0.9175 | 0.9182 | 0.9179 |
| **Scenario 3:** Punctuation removed | 0.8794 | 0.9222 | 0.9252 | 0.9237 |
| **Scenario 4:** Both stopwords and punctuation removed | 0.8719 | 0.9180 | 0.9199 | 0.9189 |
| **Scenario 5:** Frequency Discounting (Square-root rule) | 0.8990 | 0.9341 | 0.9461 | 0.9400 |

### Analysis
The results reinforce that removing both stopwords and punctuation (Scenario 4) aggressively sacrifices critical semantic and syntactic signals necessary for sentiment interpretation, leading to the lowest relative performance metrics alongside simply stripping all stopwords (Scenario 2). Notably, removing simply punctuation (Scenario 3) reduced accuracy by over a consistent percentile point (~1.6%) from the standard unprocessed baseline model. 

However, we observed that by replacing the deletion of highly frequent function words with structural frequency discounting—specifically mapping a square root matrix transformation over the count vectors—our dataset's representation stabilized without outright neutralizing meaningful modifiers. Scenario 5 yielded the highest total classification accuracy (89.90%) and F1-Score (0.9400). 

# Section 7: Conclusion
The empirical evaluation of traditional text preprocessing heuristics within the scope of IMDB62 movie review sentiment classification demonstrates that stopwords and punctuation possess demonstrable descriptive value. Standard deletion techniques artificially handicap statistical performance by removing these contextual cues (such as affective punctuation boundaries or negations and intensifiers masked as stopwords). Instead, utilizing principled continuous frequency-weighting schemes—such as the square-root discounting methodology adopted in our evaluation—better bridges the dimensionality and informative signal tradeoff. Frequency discounting successfully scales extreme function words without deleting them, thus outperforming deletion and standard bag-of-words presence models in discriminative text analysis settings.
