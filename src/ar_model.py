import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def build_features(point_seq, lags=5):
    """
    Build autoregressive feature matrix from binary point sequence.
    Each row: [y(t-1), y(t-2), ..., y(t-lags)] -> y(t)
    """
    X, y = [], []
    for i in range(lags, len(point_seq)):
        X.append(point_seq[i-lags:i])
        y.append(point_seq[i])
    return np.array(X), np.array(y)

def baseline_model(point_seq):
    """
    Baseline: always predict server wins (majority class).
    P(server wins) = empirical mean.
    """
    p = np.mean(point_seq)
    y = np.array(point_seq)
    preds = np.full(len(y), p)
    labels = np.ones(len(y), dtype=int)  # always predict server wins
    acc = accuracy_score(y, labels)
    ll = log_loss(y, preds)
    return acc, ll

def ar_logistic_model(point_seq, lags=5):
    """
    Autoregressive logistic regression.
    Tests whether past point outcomes predict the next point.
    """
    X, y = build_features(point_seq, lags)
    if len(X) < 50:
        return None
    
    model = LogisticRegression(max_iter=1000)
    
    # cross-validated accuracy and log loss
    acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    ll_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')
    
    return {
        'acc_mean': acc_scores.mean(),
        'acc_std': acc_scores.std(),
        'll_mean': ll_scores.mean(),
        'll_std': ll_scores.std()
    }

def evaluate_across_matches(point_sequences, lags=5, sample=None):
    """
    Run baseline vs AR model across many matches.
    Primary test: does momentum (past outcomes) improve prediction?
    """
    import random
    random.seed(42)
    sequences = [s for s in point_sequences if len(s) >= 50]
    if sample is not None:
        sequences = random.sample(sequences, min(sample, len(sequences)))
    
    baseline_accs, baseline_lls = [], []
    ar_accs, ar_lls = [], []
    
    for seq in sequences:
        b_acc, b_ll = baseline_model(seq)
        baseline_accs.append(b_acc)
        baseline_lls.append(b_ll)
        
        result = ar_logistic_model(seq, lags)
        if result:
            ar_accs.append(result['acc_mean'])
            ar_lls.append(result['ll_mean'])
    
    print(f"Evaluated on {len(ar_accs)} matches\n")
    print("Primary metric: Log Loss (lower = better probability calibration)")
    print("Secondary metric: Accuracy (insensitive near 63% majority class)\n")
    print(f"{'Model':<25} {'Log Loss':>10} {'Accuracy':>10}")
    print(f"{'-'*45}")
    print(f"{'Baseline (majority)':<25} {np.mean(baseline_lls):>10.4f} {np.mean(baseline_accs):>10.4f}")
    print(f"{'AR Logistic (lag=5)':<25} {np.mean(ar_lls):>10.4f} {np.mean(ar_accs):>10.4f}")

    ll_delta = np.mean(ar_lls) - np.mean(baseline_lls)
    acc_delta = np.mean(ar_accs) - np.mean(baseline_accs)
    print(f"\nDelta log loss:  {ll_delta:+.4f}  {'(AR worse)' if ll_delta > 0 else '(AR better)'}")
    print(f"Delta accuracy:  {acc_delta:+.4f}  {'(AR worse)' if acc_delta < 0 else '(AR better)'}")
    print(f"\nConclusion: {'AR model improves over baseline' if ll_delta < 0 else 'AR model does NOT improve over baseline'}")

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from parse import load_all_matches

    print("Loading data...")
    df = load_all_matches()
    sequences = df['point_seq'].tolist()

    print("Evaluating baseline vs AR logistic model...\n")
    evaluate_across_matches(sequences, lags=5)