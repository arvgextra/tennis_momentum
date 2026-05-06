import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from changepoint import smooth_sequence, detect_changepoints

def plot_match_momentum(match_row, window=15, save_path=None):
    """
    Plot momentum arc for a single match with:
    - Smoothed serve win rate
    - Change point markers
    - Set boundary lines
    - Segment shading by momentum level
    """
    seq = match_row['point_seq']
    smoothed = smooth_sequence(seq, window)
    breakpoints = detect_changepoints(smoothed)
    
    # set boundaries from pbp
    sets = match_row['pbp'].split('.')
    set_boundaries = []
    cumulative = 0
    for s in sets[:-1]:
        games = s.split(';')
        points_in_set = sum(len([c for c in g if c in 'SRAD']) for g in games)
        cumulative += points_in_set
        set_boundaries.append(cumulative - window)  # adjust for smoothing offset
    
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(smoothed))
    
    # shade segments by momentum level
    baseline = np.mean(smoothed)
    prev = 0
    for bp in breakpoints + [len(smoothed)]:
        segment = smoothed[prev:bp]
        avg = segment.mean()
        if avg > baseline + 0.08:
            color = '#2ecc71'
            alpha = 0.15
        elif avg < baseline - 0.08:
            color = '#e74c3c'
            alpha = 0.15
        else:
            color = '#95a5a6'
            alpha = 0.15
        ax.axvspan(prev, bp, alpha=alpha, color=color)
        prev = bp
    
    # smoothed line
    ax.plot(x, smoothed, color='#2c3e50', linewidth=1.8, zorder=3)
    
    # baseline
    ax.axhline(baseline, color='#7f8c8d', linewidth=0.8, 
               linestyle='--', alpha=0.6, label=f'Match avg ({baseline:.3f})')
    
    # change points
    for bp in breakpoints:
        ax.axvline(bp, color='#e67e22', linewidth=1.8, linestyle=':', alpha=1.0, zorder=4)
    
    # set boundaries
    for i, sb in enumerate(set_boundaries):
        ax.axvline(sb, color='#2c3e50', linewidth=1.5, alpha=0.5)
        ax.text(sb + 1, 0.93, f'Set {i+2}', fontsize=8, color='#2c3e50', alpha=0.7)
    
    # labels
    ax.set_xlabel('Point Number', fontsize=11)
    ax.set_ylabel('Serve Win Rate', fontsize=11)
    ax.set_title(
        f"{match_row['server1']} vs {match_row['server2']}  |  {match_row['score']}",
        fontsize=13, fontweight='bold'
    )
    
    # legend
    high = mpatches.Patch(color='#2ecc71', alpha=0.4, label='High momentum')
    low = mpatches.Patch(color='#e74c3c', alpha=0.4, label='Low momentum')
    neutral = mpatches.Patch(color='#95a5a6', alpha=0.3, label='Neutral')
    cp_line = plt.Line2D([0], [0], color='#e67e22', linestyle=':', label='Change point')
    ax.legend(handles=[high, neutral, low, cp_line], 
              loc='lower right', fontsize=9)
    
    ax.set_ylim(0.1, 1.0)
    ax.set_xlim(0, len(smoothed))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from parse import load_all_matches

    df = load_all_matches()
    
    # plot the Haase vs Cilic match - best narrative
    match = df.iloc[1]
    plot_match_momentum(match, save_path='outputs/haase_cilic_momentum.png')
    
    # plot a few more
    for idx in [0, 2, 3]:
        match = df.iloc[idx]
        name = f"outputs/match_{idx}_momentum.png"
        plot_match_momentum(match, save_path=name)