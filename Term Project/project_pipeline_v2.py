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
    
    # Map rating to sentiment: 1-3 -> 0 (Negative), 4-6 -> 1 (Neutral), 7-10 -> 2 (Positive)
    def map_sentiment(rating):
        if rating <= 3.0:
            return 0
        elif rating <= 6.0:
            return 1
        else:
            return 2
            
    df['sentiment'] = df['rating'].apply(map_sentiment)
    
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
    # Using 'macro' average for multi-class classification metrics
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"--- Results for {scenario_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro):    {rec:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}\n")
    return {'Scenario': scenario_name, 'Accuracy': acc, 'Precision_Macro': prec, 'Recall_Macro': rec, 'F1_Macro': f1}

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
    
    # Train SVM with class_weight='balanced' for imbalanced multi-class data
    clf = LinearSVC(random_state=42, max_iter=2000, dual=False, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, scenario_name)
    
    return metrics

def main():
    base_dir = '/Users/tanmay/Documents/TCD_Course_Material/TA/Dataset'
    scenarios = [
        {"name": "Scenario 1: All retained v2", "file": f"{base_dir}/imdb62_A.csv", "discounting": False},
        {"name": "Scenario 2: Stopwords removed v2", "file": f"{base_dir}/imdb62_B.csv", "discounting": False},
        {"name": "Scenario 3: Punctuation removed v2", "file": f"{base_dir}/imdb62_C.csv", "discounting": False},
        {"name": "Scenario 4: Both removed v2", "file": f"{base_dir}/imdb62_D.csv", "discounting": False},
        {"name": "Scenario 5: Frequency Discounting v2 (Dataset A)", "file": f"{base_dir}/imdb62_A.csv", "discounting": True}
    ]
    
    results = []
    
    for sc in scenarios:
        metrics = run_experiment(sc['file'], sc['name'], apply_discounting=sc['discounting'])
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('results_summary_v2.csv', index=False)
    print("Experiments completed. Summary saved to results_summary_v2.csv.")
    print(df_results)

if __name__ == "__main__":
    main()
