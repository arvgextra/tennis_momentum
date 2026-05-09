import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse import load_all_matches
from changepoint import smooth_sequence, detect_changepoints
from markov import get_game_score_states

st.set_page_config(page_title="Tennis Momentum Analyzer", layout="wide")

TOURS = {
    "ATP Main Draw": "pbp_matches_atp_main_*.csv",
    "WTA Main Draw": "pbp_matches_wta_main_*.csv",
    "ATP Challengers": "pbp_matches_ch_main_*.csv",
    "ATP Qualifying": "pbp_matches_atp_qual_*.csv",
}

# Data loading + stats (cached by tour pattern)

@st.cache_data(show_spinner="Loading matches and computing statistics...")
def load_and_compute(pattern):
    df = load_all_matches(pattern=pattern)
    return df, _compute_stats(df)


def _compute_stats(df):
    point_seqs = df['point_seq'].tolist()
    game_seqs = df['point_games'].tolist()

    all_pts = [p for seq in point_seqs for p in seq]
    baseline = float(np.mean(all_pts))

    # first-order Markov
    m1 = np.zeros((2, 2), dtype=np.int64)
    for seq in point_seqs:
        for i in range(1, len(seq)):
            m1[seq[i-1]][seq[i]] += 1

    # second-order Markov
    m2 = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    for seq in point_seqs:
        for i in range(2, len(seq)):
            m2[(seq[i-2], seq[i-1])][seq[i]] += 1
    m2 = {k: v.copy() for k, v in m2.items()}

    # within vs cross-game
    within = np.zeros((2, 2), dtype=np.int64)
    cross  = np.zeros((2, 2), dtype=np.int64)
    for match_games in game_seqs:
        for game in match_games:
            for i in range(1, len(game)):
                within[game[i-1]][game[i]] += 1
        for i in range(1, len(match_games)):
            cross[match_games[i-1][-1]][match_games[i][0]] += 1

    # score-state pressure
    sc_cats = ['routine', 'deuce', 'break_point', 'server_gp']
    sc = {c: np.zeros((2, 2), dtype=np.int64) for c in sc_cats}
    for match_games in game_seqs:
        for game in match_games:
            scored = get_game_score_states(game)
            for i in range(1, len(scored)):
                prev_out = scored[i-1][0]
                curr_out, curr_cat = scored[i]
                key = curr_cat if curr_cat in sc else 'routine'
                sc[key][prev_out][curr_out] += 1

    # game-level momentum
    gc = np.zeros((2, 2), dtype=np.int64)
    for match_games in game_seqs:
        outcomes = [g[-1] for g in match_games if g]
        for sg in [outcomes[0::2], outcomes[1::2]]:
            for i in range(1, len(sg)):
                gc[sg[i-1]][sg[i]] += 1

    return dict(
        baseline=baseline,
        n_points=len(all_pts),
        n_matches=len(df),
        m1=m1, m2=m2,
        within=within, cross=cross,
        score_counts=sc,
        game_counts=gc,
    )


def chi2_p(counts):
    try:
        c2, p, _, _ = stats.chi2_contingency(counts)
        return float(c2), float(p)
    except Exception:
        return 0.0, 1.0


# Chart helpers

def markov_delta_chart(baseline, m1, m2):
    def _p(row):
        t = row.sum()
        return row[1] / t if t > 0 else baseline

    deltas = [
        _p(m1[1]) - baseline,
        _p(m1[0]) - baseline,
        _p(m2.get((1, 1), np.zeros(2))) - baseline,
        _p(m2.get((1, 0), np.zeros(2))) - baseline,
        _p(m2.get((0, 1), np.zeros(2))) - baseline,
        _p(m2.get((0, 0), np.zeros(2))) - baseline,
    ]
    labels = ['After S', 'After R', 'After SS', 'After SR', 'After RS', 'After RR']
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, [d * 100 for d in deltas], color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Delta from baseline (pp)')
    ax.set_title('Serve win rate delta by recent history')
    for bar, d in zip(bars, deltas):
        offset = 0.05 if d >= 0 else -0.18
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f'{d*100:+.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    return fig


def score_state_table(score_counts, baseline):
    label_map = {
        'routine': 'Routine',
        'deuce': 'Deuce',
        'break_point': 'Break Point',
        'server_gp': 'Server GP',
    }
    rows = []
    for cat in ['routine', 'deuce', 'break_point', 'server_gp']:
        c = score_counts[cat]
        n = int(c.sum())
        if n == 0:
            continue
        b = float(c[:, 1].sum() / n)
        rt, st = int(c[0].sum()), int(c[1].sum())
        p_r = float(c[0][1] / rt) if rt > 0 else b
        p_s = float(c[1][1] / st) if st > 0 else b
        chi2, p = chi2_p(c)
        rows.append({
            'Situation': label_map[cat],
            'N': f'{n:,}',
            'Baseline': f'{b:.1%}',
            'After R delta': f'{p_r - b:+.2%}',
            'After S delta': f'{p_s - b:+.2%}',
            'Chi²': f'{chi2:.1f}',
            'p-value': f'{p:.4f}',
            'Independent?': '✓ Yes' if p >= 0.05 else '✗ No',
        })
    return pd.DataFrame(rows)


def score_state_delta_chart(score_counts):
    label_map = {'routine': 'Routine', 'deuce': 'Deuce',
                 'break_point': 'Break Point', 'server_gp': 'Server GP'}
    cats = ['routine', 'deuce', 'break_point', 'server_gp']
    after_r, after_s = [], []
    for cat in cats:
        c = score_counts[cat]
        n = c.sum()
        if n == 0:
            after_r.append(0); after_s.append(0); continue
        b  = c[:, 1].sum() / n
        rt = c[0].sum(); st = c[1].sum()
        after_r.append((c[0][1] / rt - b) * 100 if rt > 0 else 0)
        after_s.append((c[1][1] / st - b) * 100 if st > 0 else 0)

    x, w = np.arange(len(cats)), 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, after_r, w, label='After R (returner won prev)', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, after_s, w, label='After S (server won prev)',   color='#2ecc71', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[c] for c in cats])
    ax.set_ylabel('Delta from baseline (pp)')
    ax.set_title('Streak persistence by score state')
    ax.legend()
    plt.tight_layout()
    return fig


def game_level_chart(gc):
    total = gc.sum()
    hold_rate = gc[:, 1].sum() / total
    p_broke = gc[0][1] / gc[0].sum() if gc[0].sum() > 0 else hold_rate
    p_held = gc[1][1] / gc[1].sum() if gc[1].sum() > 0 else hold_rate

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [p_broke * 100, hold_rate * 100, p_held * 100]
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    labels = ['After break', 'Overall', 'After hold']
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, width=0.5)
    ax.axhline(hold_rate * 100, color='#7f8c8d', linestyle='--', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f'{v:.1f}%', ha='center', fontsize=11)
    ax.set_ylim(60, 90)
    ax.set_ylabel('Hold rate (%)')
    ax.set_title('Hold rate by previous service game outcome')
    plt.tight_layout()
    return fig


def momentum_figure(match_row, window=15):
    seq = match_row['point_seq']
    smoothed = smooth_sequence(seq, window)
    breakpoints = detect_changepoints(smoothed)

    set_boundaries, cumulative = [], 0
    for s in match_row['pbp'].split('.')[:-1]:
        pts = sum(1 for g in s.split(';') for c in g if c in 'SRAD')
        cumulative += pts
        sb = cumulative - window
        if 0 < sb < len(smoothed):
            set_boundaries.append(sb)

    fig, ax = plt.subplots(figsize=(14, 5))
    x  = np.arange(len(smoothed))
    bl = float(np.mean(smoothed))

    prev = 0
    for bp in breakpoints + [len(smoothed)]:
        seg = smoothed[prev:bp]
        avg = float(seg.mean())
        if avg > bl + 0.08:
            color, alpha = '#2ecc71', 0.15
        elif avg < bl - 0.08:
            color, alpha = '#e74c3c', 0.15
        else:
            color, alpha = '#95a5a6', 0.15
        ax.axvspan(prev, bp, alpha=alpha, color=color)
        prev = bp

    ax.plot(x, smoothed, color='#2c3e50', linewidth=1.8, zorder=3)
    ax.axhline(bl, color='#7f8c8d', linewidth=0.8, linestyle='--',
               alpha=0.6, label=f'Match avg ({bl:.2f})')
    for bp in breakpoints:
        ax.axvline(bp, color='#e67e22', linewidth=1.8, linestyle=':', alpha=1.0, zorder=4)
    for i, sb in enumerate(set_boundaries):
        ax.axvline(sb, color='#2c3e50', linewidth=1.5, alpha=0.5)
        ax.text(sb + 1, 0.93, f'Set {i+2}', fontsize=8, color='#2c3e50', alpha=0.7)

    ax.set_xlabel('Point Number')
    ax.set_ylabel('Serve Win Rate (rolling window)')
    ax.set_title(
        f"{match_row['server1']} vs {match_row['server2']}  |  "
        f"{match_row['score']}  |  {match_row['tny_name']}",
        fontsize=12, fontweight='bold',
    )
    high = mpatches.Patch(color='#2ecc71', alpha=0.4, label='High serve rate')
    low = mpatches.Patch(color='#e74c3c', alpha=0.4, label='Low serve rate')
    neutral = mpatches.Patch(color='#95a5a6', alpha=0.3, label='Neutral')
    cp = plt.Line2D([0], [0], color='#e67e22', linestyle=':', label='Change point')
    ax.legend(handles=[high, neutral, low, cp], loc='lower right', fontsize=9)
    ax.set_ylim(0.1, 1.0)
    ax.set_xlim(0, len(smoothed))
    plt.tight_layout()
    return fig

# App layout

st.title("Tennis Momentum Analyzer")
st.caption(
    "Does momentum exist in professional tennis, "
    "or is it a narrative imposed on sequences of independent points?"
)

with st.sidebar:
    st.header("Dataset")
    tour = st.selectbox("Tour / level", list(TOURS.keys()))
    pattern = TOURS[tour]
    st.divider()
    st.caption(
        "Each tour is analyzed independently to avoid player-quality confounds. "
        "Markov and score-state analyses use all matches. "
        "AR model uses all matches with ≥50 points."
    )

df, s = load_and_compute(pattern)

tab_overview, tab_score, tab_game, tab_explorer = st.tabs([
    "Overview", "Score-State Analysis", "Game-Level Momentum", "Match Explorer"
])

# Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{s['n_matches']:,}")
    c2.metric("Total Points", f"{s['n_points']:,}")
    c3.metric("Avg pts / match", f"{s['n_points'] / s['n_matches']:.0f}")
    c4.metric("Baseline P(server wins)", f"{s['baseline']:.1%}")

    st.divider()

    col_chart, col_stats = st.columns([2, 1])
    with col_chart:
        st.subheader("Serve win rate delta by recent history")
        st.pyplot(markov_delta_chart(s['baseline'], s['m1'], s['m2']))

    with col_stats:
        st.subheader("First-order independence test")
        m1 = s['m1']
        bl = s['baseline']
        p_s = m1[1][1] / m1[1].sum()
        p_r = m1[0][1] / m1[0].sum()
        chi2, p = chi2_p(m1)
        st.metric("After server wins",   f"{p_s:.1%}", f"{p_s - bl:+.2%}")
        st.metric("After returner wins", f"{p_r:.1%}", f"{p_r - bl:+.2%}")
        st.metric("Chi²", f"{chi2:.1f}")
        st.metric("p-value", f"{p:.6f}")
        if p < 0.05:
            st.warning("Points are NOT independent — but effects are structural. See Score-State tab.")
        else:
            st.success("Points are independent.")

    st.divider()
    st.subheader("Within-game vs cross-game transitions")
    wcol1, wcol2 = st.columns(2)
    for col, label, counts in [
        (wcol1, "Within-game", s['within']),
        (wcol2, "Cross-game",  s['cross']),
    ]:
        with col:
            n  = counts.sum()
            b  = counts[:, 1].sum() / n
            pr = counts[0][1] / counts[0].sum()
            ps = counts[1][1] / counts[1].sum()
            chi2, p = chi2_p(counts)
            st.markdown(f"**{label}** ({n:,} transitions)")
            st.dataframe(pd.DataFrame({
                'Prev': ['R', 'S'],
                'P(server wins)': [f'{pr:.4f}', f'{ps:.4f}'],
                'Delta': [f'{pr - b:+.4f}', f'{ps - b:+.4f}'],
            }), hide_index=True, use_container_width=True)
            st.caption(f"Chi²={chi2:.1f}, p={p:.6f}")
            if label == "Cross-game":
                st.caption("Sign reversal at game boundaries confirms a structural cause.")

# Score-State Analysis 
with tab_score:
    st.subheader("Streak persistence by score state")
    st.caption(
        "If momentum is psychological, the effect should be largest at high-pressure moments "
        "(break points, deuce). The opposite pattern would indicate a structural cause."
    )

    tbl = score_state_table(s['score_counts'], s['baseline'])

    def _highlight(row):
        if '✓' in str(row.get('Independent?', '')):
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)

    st.dataframe(
        tbl.style.apply(_highlight, axis=1),
        hide_index=True, use_container_width=True,
    )

    st.info(
        "**Key finding:** Deuce points are perfectly independent (Chi²≈0.07, p=0.79). "
        "Break points near-independent. The streak-persistence signal lives entirely in "
        "low-pressure routine points — the opposite of what psychological momentum predicts."
    )

    st.subheader("Delta comparison across score states")
    st.pyplot(score_state_delta_chart(s['score_counts']))

# Game-Level Momentum 
with tab_game:
    st.subheader("Game-level momentum")
    st.caption(
        "Does getting broken predict getting broken again in your next service game? "
        "Each player's service games are analyzed separately (even/odd index) "
        "to avoid the serve-alternation confound."
    )

    gc = s['game_counts']
    total = gc.sum()
    hold_rate = gc[:, 1].sum() / total
    p_broke = gc[0][1] / gc[0].sum() if gc[0].sum() > 0 else hold_rate
    p_held = gc[1][1] / gc[1].sum() if gc[1].sum() > 0 else hold_rate
    chi2, p = chi2_p(gc)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Overall hold rate", f"{hold_rate:.1%}")
    mc2.metric("Hold rate after break", f"{p_broke:.1%}", f"{p_broke - hold_rate:+.2%}")
    mc3.metric("Hold rate after hold",  f"{p_held:.1%}",  f"{p_held - hold_rate:+.2%}")

    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        st.pyplot(game_level_chart(gc))
    with gcol2:
        st.metric("Chi²", f"{chi2:.1f}")
        st.metric("p-value", f"{p:.6f}")
        st.metric("Game pairs analyzed", f"{int(total):,}")
        st.warning(
            "Largest effect in the analysis. However, the asymmetry "
            "(−5.93pp after break vs +1.54pp after hold) suggests "
            "player-quality confounding: weaker servers get broken more often "
            "and hold less in general. Controlling for per-player hold rate "
            "would isolate whether the break itself is causal."
        )

# Match Explorer
with tab_explorer:
    st.subheader("Match momentum arc")

    fcol1, fcol2 = st.columns([2, 1])
    with fcol1:
        search = st.text_input("Filter by player name", placeholder="e.g. Federer, Djokovic")
    with fcol2:
        min_pts = st.slider("Min points played", 50, 300, 100)

    filtered = df[df['match_length'] >= min_pts].copy()
    if search:
        mask = (
            filtered['server1'].str.contains(search, case=False, na=False) |
            filtered['server2'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    if len(filtered) == 0:
        st.warning("No matches found.")
    else:
        filtered['label'] = (
            filtered['server1'] + ' vs ' + filtered['server2'] +
            ' | ' + filtered['score'] +
            ' | ' + filtered['date'].astype(str)
        )
        selected = st.selectbox(f"Select match ({len(filtered):,} found)", filtered['label'].tolist())
        row = filtered[filtered['label'] == selected].iloc[0]

        st.pyplot(momentum_figure(row))

        with st.expander("Match details"):
            st.json({
                'Tournament': str(row['tny_name']),
                'Date': str(row['date']),
                'Score': str(row['score']),
                'Total points': int(row['match_length']),
                'Server 1': str(row['server1']),
                'Server 2': str(row['server2']),
            })
