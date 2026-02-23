# Text Analytics Term Project: Frequency Treatment vs. Deletion

This repository contains the implementation pipeline for the Text Analytics term project investigating the impact of frequency discounting versus traditional deletion heuristics (stopwords and punctuation removal) on sentiment classification performance.

## Project Overview (Technical Summary)

The objective of this pipeline is to empirically evaluate whether high-frequency function words and punctuation serve as noise to be deleted, or if they encode meaningful sentiment and structural information that should be retained and scaled. 

### Dataset & Preprocessing
The model is trained and evaluated on the **IMDb62** movie review corpus. 
- Original ratings (1–10) are binarized: `1-4` as Negative (0), and `7-10` as Positive (1). Neutral ratings (`5-6`) are filtered out to emphasize polarity.
- The text features are represented using `CountVectorizer` (Term Frequency).
- To test the impact of text preprocessing, we utilize four dataset variants:
  - `imdb62_A.csv`: Stopwords included, Punctuation included.
  - `imdb62_B.csv`: Stopwords removed, Punctuation included.
  - `imdb62_C.csv`: Stopwords included, Punctuation removed.
  - `imdb62_D.csv`: Stopwords removed, Punctuation removed.

### Experimental Design & Modeling
The classification architecture applies a `LinearSVC` (Support Vector Classifier). To isolate the impact of the feature representation, the model architecture is held constant across five distinct scenarios:
1. **Scenario 1**: Baseline raw counts using Dataset A (All retained).
2. **Scenario 2**: Raw counts using Dataset B (Stopwords removed).
3. **Scenario 3**: Raw counts using Dataset C (Punctuation removed).
4. **Scenario 4**: Raw counts using Dataset D (Both removed).
5. **Scenario 5**: Frequency Discounting. Using Dataset A, we apply an element-wise **square-root scaling** on the sparse term frequency matrix calculated by the `CountVectorizer`. 

This mathematical transformation dampens the overwhelming influence of extremely high-frequency terms without strictly deleting them, preserving syntactic and affective tokens (e.g., negations, repeated exclamation marks).

### Evaluation Metrics
The pipeline leverages an 80/20 train/test stratified split. Models are evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score

*Current findings observe that the Square-Root frequency discounting approach (Scenario 5) yields the highest classification accuracy and F1-score out of all scenarios, indicating that traditional deletion heuristics handicap statistical performance by removing these contextual cues.*

## Repository Structure

- `project_pipeline.py`: The main executable script encompassing data loading, preprocessing, model training, frequency discounting logic, and evaluation.
- `requirements.txt`: Project dependencies.
- `Dataset/`: Directory housing the preprocessed IMDb62 CSV variants. (Not included in VCS if ignored)
- `results_summary.csv`: Output tabular file tracking classification metrics per scenario.
- `Report_Results_Draft.md`: Drafted analysis of the results to be ingested into the final academic report.

## How to Run the Code

1. **Environment Setup**
   Ensure you have Python 3.8+ installed. It is recommended to use a Conda or virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Placement**
   Make sure the `Dataset/` folder is present in the root directory and contains the four dataset variants (`imdb62_A.csv`, `imdb62_B.csv`, `imdb62_C.csv`, `imdb62_D.csv`).

3. **Execution**
   Run the overarching machine learning pipeline script:
   ```bash
   python project_pipeline.py
   ```

4. **Outputs**
   Once the script finishes execution, it prints the evaluation metrics to the standard output and generates a consolidated `results_summary.csv` file within the current working directory.
