import numpy as np
from hmmlearn import hmm
import warnings
import random
import pickle

warnings.filterwarnings('ignore')

def fit_hmm(point_sequences, n_states=2, n_iter=100, n_init=5, min_length=100, max_matches=500):
    """
    Fit HMM on a random sample of matches for speed.
    """
    sequences = [s for s in point_sequences if len(s) >= min_length]
    random.seed(42)
    sequences = random.sample(sequences, min(max_matches, len(sequences)))
    print(f"Training on {len(sequences)} matches")
    
    X = np.concatenate(sequences).reshape(-1, 1)
    lengths = [len(seq) for seq in sequences]
    
    best_model = None
    best_score = -np.inf
    
    for seed in range(n_init):
        model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=seed,
            verbose=False
        )
        try:
            model.fit(X, lengths)
            score = model.score(X, lengths)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue
    
    print(f"Best log-likelihood: {best_score:.2f}")
    
    with open('models/hmm_2state.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    return best_model

def interpret_model(model, baseline=0.6357):
    print(f"Converged: {model.monitor_.converged}")
    print(f"\nEmission probabilities (P(server wins) per state):")
    for i in range(model.n_components):
        p = model.emissionprob_[i, 1]
        if p > baseline + 0.01:
            label = "HIGH momentum"
        elif p < baseline - 0.01:
            label = "LOW momentum"
        else:
            label = "NEUTRAL"
        print(f"  State {i} [{label}]: P(server wins) = {p:.4f}")
    
    print(f"\nTransition matrix:")
    for i, row in enumerate(model.transmat_):
        print(f"  From state {i}: {row}")
    
    print(f"\nAverage state duration (points):")
    for i in range(model.n_components):
        avg_duration = 1 / (1 - model.transmat_[i][i])
        print(f"  State {i}: {avg_duration:.1f} points")

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from parse import load_all_matches

    print("Loading data...")
    df = load_all_matches()
    sequences = df['point_seq'].tolist()

    print("Fitting 2-state HMM...")
    model = fit_hmm(sequences, n_states=2)
    interpret_model(model)
    
