import numpy as np
from collections import defaultdict
from scipy import stats

def build_transition_matrix(point_sequences, order=1):
    """
    Build Markov transition matrix from point sequences.
    State: last `order` outcomes (e.g. order=1: (0,) or (1,))
    Transition: probability of next point being 1 (server wins)
    """
    counts = defaultdict(lambda: [0, 0])  # state -> [count_R_won, count_S_won]
    
    for seq in point_sequences:
        for i in range(order, len(seq)):
            state = tuple(seq[i - order:i])
            outcome = seq[i]
            counts[state][outcome] += 1
    
    # convert to probabilities
    transitions = {}
    for state, (r_wins, s_wins) in counts.items():
        total = r_wins + s_wins
        transitions[state] = {
            'p_server_wins': s_wins / total,
            'n': total
        }
    
    return transitions

def test_independence(point_sequences):
    """
    Compare first-order Markov P(server wins | last point) 
    against baseline P(server wins) to test if points are i.i.d.
    """
    all_points = [p for seq in point_sequences for p in seq]
    baseline = np.mean(all_points)
    
    transitions = build_transition_matrix(point_sequences, order=1)
    
    p_server_after_server = transitions[(1,)]['p_server_wins']
    p_server_after_returner = transitions[(0,)]['p_server_wins']
    
    print(f"Baseline P(server wins): {baseline:.4f}")
    print(f"P(server wins | server won last): {p_server_after_server:.4f}")
    print(f"P(server wins | returner won last): {p_server_after_returner:.4f}")
    print(f"\nDelta after server win: {p_server_after_server - baseline:+.4f}")
    print(f"Delta after returner win: {p_server_after_returner - baseline:+.4f}")
    print(f"\nTotal points analyzed: {len(all_points):,}")

def test_higher_order(point_sequences):
    """
    Compare first vs second order Markov to see if more history helps.
    """
    transitions_2 = build_transition_matrix(point_sequences, order=2)
    
    print("\nSecond-order Markov transitions:")
    for state in [(0,0), (0,1), (1,0), (1,1)]:
        label = {(0,0): 'RR', (0,1): 'RS', (1,0): 'SR', (1,1): 'SS'}[state]
        p = transitions_2[state]['p_server_wins']
        n = transitions_2[state]['n']
        print(f"After {label}: P(server wins) = {p:.4f}  (n={n:,})")

def chi_square_independence_test(point_sequences):
    """
    Chi-square test: are consecutive points independent?
    """
    # build contingency table: prev_outcome x curr_outcome
    contingency = np.zeros((2, 2), dtype=int)
    
    for seq in point_sequences:
        for i in range(1, len(seq)):
            prev = seq[i-1]
            curr = seq[i]
            contingency[prev][curr] += 1
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square test for independence:")
    print(f"Contingency table (rows=prev, cols=curr):")
    print(f"R_wins S_wins")
    print(f"prev R: {contingency[0][0]:>6} {contingency[0][1]:>6}")
    print(f"prev S: {contingency[1][0]:>6} {contingency[1][1]:>6}")
    print(f"\nChi2: {chi2:.2f}, p-value: {p_value:.6f}, dof: {dof}")
    if p_value < 0.05:
        print("Result: Points are NOT independent (reject null hypothesis)")
    else:
        print("Result: Points are independent (fail to reject null)")

def _print_transition_table(counts):
    total = counts.sum()
    baseline = counts[:, 1].sum() / total
    print(f"  Baseline P(server wins): {baseline:.4f}")
    for prev in [0, 1]:
        label = 'S' if prev == 1 else 'R'
        row_total = counts[prev].sum()
        if row_total > 0:
            p = counts[prev][1] / row_total
            delta = p - baseline
            print(f"  P(server wins | prev={label}): {p:.4f}  delta={delta:+.4f}  (n={row_total:,})")
    chi2, p_value, _, _ = stats.chi2_contingency(counts)
    print(f"  Chi2={chi2:.2f}, p={p_value:.6f} — {'NOT independent' if p_value < 0.05 else 'independent'}")

def test_within_vs_cross_game(game_sequences_list):
    """
    Split transition analysis by game boundary.
    Within-game: consecutive points in the same game.
    Cross-game: last point of game N -> first point of game N+1.
    If dependence vanishes at cross-game boundaries, the chi-square signal
    is structural (game format), not momentum.
    """
    within = np.zeros((2, 2), dtype=int)
    cross = np.zeros((2, 2), dtype=int)

    for match_games in game_sequences_list:
        for game in match_games:
            for i in range(1, len(game)):
                within[game[i-1]][game[i]] += 1
        for i in range(1, len(match_games)):
            prev = match_games[i-1][-1]
            curr = match_games[i][0]
            cross[prev][curr] += 1

    print("=== Within-game transitions ===")
    _print_transition_table(within)
    print("\n=== Cross-game transitions ===")
    _print_transition_table(cross)

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from parse import load_all_matches
    
    print("Loading data...")
    df = load_all_matches()
    
    print("Testing point independence...\n")
    test_independence(df['point_seq'].tolist())
    chi_square_independence_test(df['point_seq'].tolist())
    test_higher_order(df['point_seq'].tolist())

    print("\n\nWithin-game vs cross-game transition analysis:")
    test_within_vs_cross_game(df['point_games'].tolist())