import numpy as np
import pickle
import ruptures as rpt

def load_hmm():
    with open('models/hmm_2state.pkl', 'rb') as f:
        return pickle.load(f)

def smooth_sequence(point_seq, window=10):
    """
    Convert binary point sequence to rolling serve win rate.
    Reduces noise and gives HMM/BOCPD a smoother signal.
    """
    arr = np.array(point_seq, dtype=float)
    smoothed = np.convolve(arr, np.ones(window)/window, mode='valid')
    return smoothed

def hmm_state_sequence(model, point_seq, window=10):
    """
    Decode HMM states on raw binary sequence (consistent with how the model was trained).
    Returns smoothed series as a convenience for visualization.
    """
    X = np.array(point_seq).reshape(-1, 1)
    states = model.predict(X)
    smoothed = smooth_sequence(point_seq, window)
    return states, smoothed

def detect_changepoints(smoothed_seq, n_bkps=5, model="rbf"):
    """
    Change point detection using ruptures library.
    Uses kernel-based method (rbf) on smoothed serve win rate.
    """
    signal = np.array(smoothed_seq).reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=10, jump=1)
    algo.fit(signal)
    breakpoints = algo.predict(pen=np.log(len(signal)))
    # ruptures returns end indices, last one is always len(signal)
    breakpoints = [b for b in breakpoints if b < len(signal)]
    return breakpoints

def plot_momentum(match_row, window=10):
    """
    Print momentum arc for a single match.
    """
    seq = match_row['point_seq']
    smoothed = smooth_sequence(seq, window)
    breakpoints = detect_changepoints(smoothed)
    
    print(f"\nChange points detected at smoothed points: {breakpoints}")
    print(f"(Corresponds to raw points: {[b + window for b in breakpoints]})")
    print(f"\nMomentum segments:")
    prev = 0
    for bp in breakpoints + [len(smoothed)]:
        segment = smoothed[prev:bp]
        print(f"  Points {prev}-{bp}: avg serve win rate = {segment.mean():.3f}")
        prev = bp

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from parse import load_all_matches

    print("Loading data and model...")
    df = load_all_matches()
    model = load_hmm()

    match = df.iloc[1]
    print(f"\nAnalyzing: {match['server1']} vs {match['server2']}")
    print(f"Score: {match['score']}, Points: {match['match_length']}")

    seq = match['point_seq']
    smoothed = smooth_sequence(seq)
    states, _ = hmm_state_sequence(model, seq)
    print(f"HMM state counts: {np.bincount(states)}")

    plot_momentum(match)

    for idx in [0, 2, 3]:
        match = df.iloc[idx]
        print(f"\n{'='*50}")
        print(f"{match['server1']} vs {match['server2']} | {match['score']}")
        plot_momentum(match)