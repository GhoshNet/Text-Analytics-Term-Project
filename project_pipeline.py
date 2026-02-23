import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix
import ast

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Warning: Issue parsing {file_path}: {e}. Retrying with c engine...")
        df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # Map rating to sentiment: 1-4 -> 0 (Negative), 7-10 -> 1 (Positive)
    # Filter out neutral reviews (5-6)
    df = df[~df['rating'].isin([5.0, 6.0])]
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 7 else 0)
    
    # We assume 'content' holds the processed list of tokens (as a string repr of list)
    # CountVectorizer expects lists of strings or strings, so we can join them back to text.
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

def evaluate_model(y_test, y_pred, scenario_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"--- Results for {scenario_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    return {'Scenario': scenario_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

def get_top_features(vectorizer, clf, n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    
    top_pos_idx = coef.argsort()[-n:][::-1]
    top_neg_idx = coef.argsort()[:n]
    
    top_pos = feature_names[top_pos_idx]
    top_neg = feature_names[top_neg_idx]
    
    return top_pos, top_neg

def run_experiment(data_path, scenario_name, apply_discounting=False):
    X_text, y = load_data(data_path)
    
    # Split: 80% train, 20% test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature extraction (Raw Counts)
    vectorizer = CountVectorizer(min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    if apply_discounting:
        # Apply square root weighting to the frequency count matrix
        # Since it's a sparse matrix, we can apply sqrt to the data elements directly.
        X_train.data = np.sqrt(X_train.data)
        X_test.data = np.sqrt(X_test.data)
    
    # Train SVM
    clf = LinearSVC(random_state=42, max_iter=2000, dual=False)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, scenario_name)
    
    # Get top features
    top_pos, top_neg = get_top_features(vectorizer, clf)
    # print(f"Top Positive Features: {', '.join(top_pos)}")
    # print(f"Top Negative Features: {', '.join(top_neg)}\n")
    
    return metrics

def main():
    base_dir = '/Users/tanmay/Documents/TCD_Course_Material/TA/Dataset'
    scenarios = [
        {"name": "Scenario 1: All retained", "file": f"{base_dir}/imdb62_A.csv", "discounting": False},
        {"name": "Scenario 2: Stopwords removed", "file": f"{base_dir}/imdb62_B.csv", "discounting": False},
        {"name": "Scenario 3: Punctuation removed", "file": f"{base_dir}/imdb62_C.csv", "discounting": False},
        {"name": "Scenario 4: Both removed", "file": f"{base_dir}/imdb62_D.csv", "discounting": False},
        {"name": "Scenario 5: Frequency Discounting (Dataset A)", "file": f"{base_dir}/imdb62_A.csv", "discounting": True}
    ]
    
    results = []
    
    for sc in scenarios:
        metrics = run_experiment(sc['file'], sc['name'], apply_discounting=sc['discounting'])
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('results_summary.csv', index=False)
    print("Experiments completed. Summary saved to results_summary.csv.")
    print(df_results)

if __name__ == "__main__":
    main()
