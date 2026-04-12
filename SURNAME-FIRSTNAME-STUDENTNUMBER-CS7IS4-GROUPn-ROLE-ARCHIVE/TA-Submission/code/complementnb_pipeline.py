import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB
import ast

from crossvalidation_pipeline import run_all_scenarios_cv, run_statistical_tests


# Paths are resolved relative to this file's location so the script works
# regardless of the working directory it is launched from.
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Dataset')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Warning: Issue parsing {file_path}: {e}. Retrying with c engine...")
        df = pd.read_csv(file_path, on_bad_lines='skip')

    # Map rating to sentiment: 1-3 -> 0 (Negative), 4-6 -> 1 (Neutral), 7-10 -> 2 (Positive)
    def map_sentiment(rating):
        if rating <= 3.0:
            return 0
        elif rating <= 6.0:
            return 1
        else:
            return 2

    df['sentiment'] = df['rating'].apply(map_sentiment)

    def parse_tokens(text):
        try:
            tokens = ast.literal_eval(text)
            if isinstance(tokens, list):
                return ' '.join(tokens)
            return str(text)
        except:
            return str(text)

    df['clean_text'] = df['content'].apply(parse_tokens)
    return df['clean_text'].values, df['sentiment'].values


def build_model():
    """
    Factory function — returns a fresh ComplementNB instance.

    ComplementNB is designed for imbalanced multi-class text classification.
    It corrects for class imbalance without requiring class_weight parameter.
    """
    return ComplementNB()


def main():
    # Load each dataset once
    print("Loading datasets...")
    data_A = load_data(os.path.join(BASE_DIR, 'imdb62_A.csv'))
    data_B = load_data(os.path.join(BASE_DIR, 'imdb62_B.csv'))
    data_C = load_data(os.path.join(BASE_DIR, 'imdb62_C.csv'))
    data_D = load_data(os.path.join(BASE_DIR, 'imdb62_D.csv'))

    scenarios = [
        {
            'name':        'Scenario 1: All retained',
            'X_text':      data_A[0],
            'y':           data_A[1],
            'discounting': False,
        },
        {
            'name':        'Scenario 2: Stopwords removed',
            'X_text':      data_B[0],
            'y':           data_B[1],
            'discounting': False,
        },
        {
            'name':        'Scenario 3: Punctuation removed',
            'X_text':      data_C[0],
            'y':           data_C[1],
            'discounting': False,
        },
        {
            'name':        'Scenario 4: Both removed',
            'X_text':      data_D[0],
            'y':           data_D[1],
            'discounting': False,
        },
        {
            'name':        'Scenario 5: Frequency Discounting',
            'X_text':      data_A[0],
            'y':           data_A[1],
            'discounting': True,
        },
    ]

    # 10-fold stratified CV across all scenarios
    cv_results, summary_df, fold_df = run_all_scenarios_cv(
        scenarios=scenarios,
        model_factory=build_model,
        n_splits=10,
        average='macro',   # macro-averaged for 3-class
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'complementnb_results_summary.csv'), index=False)
    fold_df.to_csv(os.path.join(RESULTS_DIR, 'complementnb_results_per_fold.csv'), index=False)

    print("\n=== CV Summary (mean ± std across 10 folds) ===")
    print(summary_df.to_string(index=False))

    # Statistical tests on F1 (primary metric for multi-class)
    friedman, pairwise_df = run_statistical_tests(cv_results, metric='f1')
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, 'complementnb_statistical_tests.csv'), index=False)

    print("\nResults saved to results/:")
    print("  complementnb_results_summary.csv   — mean/std summary per scenario")
    print("  complementnb_results_per_fold.csv  — per-fold raw values")
    print("  complementnb_statistical_tests.csv — Friedman + Wilcoxon/Holm pairwise tests")


if __name__ == "__main__":
    main()
