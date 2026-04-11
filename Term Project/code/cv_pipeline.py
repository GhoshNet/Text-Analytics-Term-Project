"""
cv_pipeline.py
==============
Shared, model-agnostic 10-fold cross-validation and statistical testing pipeline.

Usage (any sklearn-compatible classifier)
-----------------------------------------
from cv_pipeline import run_all_scenarios_cv, run_statistical_tests
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Define a *factory* — a zero-argument callable that returns a fresh model
model_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

# Build scenario list (X_text and y come from your load_data function)
scenarios = [
    {'name': 'Scenario 1: All retained', 'X_text': X1, 'y': y1, 'discounting': False},
    {'name': 'Scenario 5: Freq Discounting', 'X_text': X5, 'y': y5, 'discounting': True},
    ...
]

cv_results, summary_df = run_all_scenarios_cv(
    scenarios=scenarios,
    model_factory=model_factory,
    n_splits=10,
    average='macro',  # 'binary' for 2-class, 'macro' for multi-class
)

friedman, pairwise_df = run_statistical_tests(cv_results, metric='f1')
"""

import numpy as np
import pandas as pd
from itertools import combinations

from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_sqrt_discounting(X_sparse):
    """Apply element-wise sqrt to a sparse TF matrix (modifies a copy)."""
    X_sparse = X_sparse.copy()
    X_sparse.data = np.sqrt(X_sparse.data)
    return X_sparse


def _holm_correction(raw_p_values, labels):
    """
    Apply Holm step-down correction to a list of raw p-values.

    Parameters
    ----------
    raw_p_values : list of float
    labels       : list of str (one label per p-value)

    Returns
    -------
    list of (label, raw_p, holm_p, significant) tuples sorted by raw_p
    """
    m = len(raw_p_values)
    sorted_pairs = sorted(zip(raw_p_values, labels), key=lambda x: x[0])
    corrected = []
    prev_holm_p = 0.0

    for rank, (p_raw, label) in enumerate(sorted_pairs, start=1):
        p_holm = min(p_raw * (m - rank + 1), 1.0)
        p_holm = max(p_holm, prev_holm_p)   # enforce monotonicity
        prev_holm_p = p_holm
        corrected.append((label, p_raw, p_holm, p_holm < 0.05))

    return corrected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cv_for_scenario(
    X_text,
    y,
    model_factory,
    apply_discounting=False,
    n_splits=10,
    average='binary',
):
    """
    Run stratified k-fold CV for a single scenario.

    Parameters
    ----------
    X_text            : array-like of str  – pre-processed text documents
    y                 : array-like of int  – class labels
    model_factory     : callable() -> sklearn estimator
                        Must return a *new* (unfitted) model on every call.
                        Examples:
                          lambda: LinearSVC(random_state=42, max_iter=2000, dual=False)
                          lambda: RandomForestClassifier(n_estimators=100, random_state=42)
                          lambda: MultinomialNB()
    apply_discounting : bool  – apply sqrt transform to TF count matrix
    n_splits          : int   – number of folds (default 10)
    average           : str   – 'binary' for 2-class, 'macro' for multi-class

    Returns
    -------
    dict mapping metric name -> list of per-fold float values
      {'accuracy': [...], 'precision': [...], 'recall': [...], 'f1': [...]}
    """
    X_text = np.asarray(X_text)
    y      = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y), start=1):
        X_train_text = X_text[train_idx]
        X_test_text  = X_text[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Fit vectorizer on training fold only — no leakage
        vectorizer = CountVectorizer(min_df=5)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test  = vectorizer.transform(X_test_text)

        if apply_discounting:
            X_train = _apply_sqrt_discounting(X_train)
            X_test  = _apply_sqrt_discounting(X_test)

        # Fresh model instance per fold
        clf = model_factory()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['precision'].append(
            precision_score(y_test, y_pred, average=average, zero_division=0)
        )
        fold_metrics['recall'].append(
            recall_score(y_test, y_pred, average=average, zero_division=0)
        )
        fold_metrics['f1'].append(
            f1_score(y_test, y_pred, average=average, zero_division=0)
        )

        print(
            f"  Fold {fold_idx:2d}/{n_splits}: "
            f"Acc={fold_metrics['accuracy'][-1]:.4f}  "
            f"F1={fold_metrics['f1'][-1]:.4f}"
        )

    return fold_metrics


def run_all_scenarios_cv(scenarios, model_factory, n_splits=10, average='binary'):
    """
    Run CV for every scenario and collect per-fold results.

    Parameters
    ----------
    scenarios     : list of dicts, each with keys:
                      'name'        (str)  – scenario label
                      'X_text'      (array-like of str)  – text documents
                      'y'           (array-like of int)  – labels
                      'discounting' (bool) – apply sqrt frequency discounting
    model_factory : callable() -> fresh sklearn estimator  (see run_cv_for_scenario)
    n_splits      : int – CV folds (default 10)
    average       : 'binary' or 'macro'

    Returns
    -------
    cv_results : dict  {scenario_name -> per-fold metric dict}
    summary_df : pd.DataFrame  mean ± std per scenario across all metrics
    fold_df    : pd.DataFrame  long-format fold-level data for archiving
    """
    cv_results = {}

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"Running {n_splits}-fold CV: {sc['name']}")
        print(f"{'='*60}")

        fold_metrics = run_cv_for_scenario(
            X_text=sc['X_text'],
            y=sc['y'],
            model_factory=model_factory,
            apply_discounting=sc['discounting'],
            n_splits=n_splits,
            average=average,
        )
        cv_results[sc['name']] = fold_metrics

        print()
        for metric, values in fold_metrics.items():
            print(
                f"  {metric.capitalize():10s}: "
                f"mean={np.mean(values):.4f}  std={np.std(values):.4f}"
            )

    # Summary DataFrame (mean ± std)
    summary_rows = []
    for sc_name, metrics in cv_results.items():
        row = {'Scenario': sc_name}
        for metric, values in metrics.items():
            row[f'{metric}_mean'] = round(np.mean(values), 4)
            row[f'{metric}_std']  = round(np.std(values),  4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Long-format fold DataFrame (useful for archiving / plotting)
    fold_rows = []
    for sc_name, metrics in cv_results.items():
        for fold_idx in range(n_splits):
            fold_rows.append({
                'Scenario': sc_name,
                'Fold':      fold_idx + 1,
                'Accuracy':  metrics['accuracy'][fold_idx],
                'Precision': metrics['precision'][fold_idx],
                'Recall':    metrics['recall'][fold_idx],
                'F1':        metrics['f1'][fold_idx],
            })
    fold_df = pd.DataFrame(fold_rows)

    return cv_results, summary_df, fold_df


def run_statistical_tests(cv_results, metric='f1', alpha=0.05):
    """
    Friedman test followed by Holm-corrected Wilcoxon signed-rank pairwise tests.

    This is the standard approach from Demšar (2006) for comparing classifiers
    across multiple datasets / CV folds.

    Parameters
    ----------
    cv_results : dict  {scenario_name -> per-fold metric dict}
                 as returned by run_all_scenarios_cv
    metric     : str – which metric to test: 'accuracy', 'f1', 'precision', 'recall'
    alpha      : float – significance level (default 0.05)

    Returns
    -------
    friedman_result : dict  {'statistic': float, 'p_value': float}
    pairwise_df     : pd.DataFrame  with columns:
                        Comparison, Raw_p, Holm_p, Significant
    """
    scenario_names = list(cv_results.keys())
    fold_vectors   = [cv_results[sc][metric] for sc in scenario_names]

    # --- Friedman test ---
    stat, p_friedman = friedmanchisquare(*fold_vectors)
    friedman_result = {'statistic': round(stat, 4), 'p_value': round(p_friedman, 4)}

    print(f"\n{'='*60}")
    print(f"Friedman Test  (metric: {metric})")
    print(f"{'='*60}")
    print(f"  Chi-square = {stat:.4f},  p = {p_friedman:.4f}")
    if p_friedman < alpha:
        print(f"  Reject H\u2080: at least one scenario differs  (p < {alpha})")
    else:
        print(f"  Fail to reject H\u2080: no significant difference  (p \u2265 {alpha})")

    # --- Pairwise Wilcoxon tests ---
    pairs      = list(combinations(range(len(scenario_names)), 2))
    raw_p_list = []
    labels     = []

    for i, j in pairs:
        a, b = fold_vectors[i], fold_vectors[j]
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            # All differences are zero — identical distributions
            p = 1.0
        raw_p_list.append(p)
        labels.append(f"{scenario_names[i]}  vs  {scenario_names[j]}")

    corrected = _holm_correction(raw_p_list, labels)

    pairwise_rows = [
        {
            'Comparison':  comp,
            'Raw_p':       round(raw_p,  4),
            'Holm_p':      round(holm_p, 4),
            'Significant': '\u2714 Yes' if sig else '\u2718 No',
        }
        for comp, raw_p, holm_p, sig in corrected
    ]
    pairwise_df = pd.DataFrame(pairwise_rows)

    print(f"\nPairwise Wilcoxon + Holm Correction  (metric: {metric})")
    print(pairwise_df.to_string(index=False))

    return friedman_result, pairwise_df
