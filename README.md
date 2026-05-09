# Tennis Match Momentum Analyzer

**Does momentum exist in professional tennis, or is it a narrative imposed on sequences of independent points?**

Statistical analysis of ~2M points across 13,050 ATP main draw matches (2011–2015) using Markov chains, hidden Markov models, autoregressive logistic regression, and change-point detection.

---

## Findings

The short answer: **momentum is structural, not psychological.**

- Points are statistically dependent (Chi²=726, p≈0), but the dependence is explained entirely by game format — not psychology.
- At **deuce**, consecutive points are perfectly independent (Chi²=0.07, p=0.79). At **break points**, near-independent. The streak-persistence signal lives only in low-pressure routine points — the opposite of what psychological momentum predicts.
- Cross-game transitions **reverse sign** relative to within-game transitions, confirming the dependence is driven by serve alternation, not carry-over momentum.
- An autoregressive logistic model with 5 lag features performs **worse** than always predicting "server wins" on both log loss and accuracy — past point outcomes have no predictive value.
- A 2-state HMM assigns **100% of points to the LOW state** on correct decoding. The HIGH state (P=0.772) was a training artifact and never appears in real sequences.
- Game-level analysis shows a large effect (−5.93pp hold rate after a break), but this is vulnerable to player-quality confounding and does not survive as a clean momentum claim.

---

## Data

[Jeff Sackmann's tennis_pointbypoint](https://github.com/JeffSackmann/tennis_pointbypoint) dataset. Raw `pbp` field encodes point outcomes as characters per game, games delimited by `;`, sets by `.`:

| Character | Meaning |
|---|---|
| `S` | Server wins (ace or winner) |
| `A` | Server wins (any other) |
| `R` | Returner wins |
| `D` | Double fault (returner wins) |

Data files are gitignored. Download from the Sackmann repo and place CSVs in `data/`.

Supported tours (selectable in dashboard):

| Pattern | Description |
|---|---|
| `pbp_matches_atp_main_*.csv` | ATP main draw (default) |
| `pbp_matches_wta_main_*.csv` | WTA main draw |
| `pbp_matches_ch_main_*.csv` | ATP Challengers |
| `pbp_matches_atp_qual_*.csv` | ATP Qualifying |

---

## Setup

```bash
pip install numpy pandas scipy scikit-learn hmmlearn ruptures matplotlib streamlit
```

Place Sackmann CSVs in `data/`. All scripts run from the project root.

---

## Usage

**Run all statistical analyses:**
```bash
python src/markov.py # Markov chains, chi-square, score-state, game-level
python src/ar_model.py # AR logistic regression vs baseline
python src/hmm.py # Train + interpret 2-state HMM
python src/changepoint.py # PELT change-point detection on rolling serve win rate
```

`hmm.py` must run before `changepoint.py` (saves `models/hmm_2state.pkl`).

**Launch the interactive dashboard:**
```bash
streamlit run src/dashboard.py
```

---

## Project Structure

```
src/
  parse.py # CSV loading, pbp string parsing, game-boundary preservation
  markov.py # First/second-order Markov analysis, chi-square tests, # within/cross-game split, score-state pressure, game-level momentum
  ar_model.py # Autoregressive logistic regression baseline comparison
  hmm.py # 2-state CategoricalHMM (hmmlearn), trained on raw binary sequences
  changepoint.py # PELT change-point detection (ruptures), momentum arc printing
  visualize.py # Matplotlib momentum arc plots with segment shading
  dashboard.py # Streamlit dashboard (all four analyses + match explorer)
models/
  hmm_2state.pkl # Saved trained HMM
data/ # gitignored — Sackmann CSVs go here
outputs/ # Plot outputs
```

---

## Methods

### Markov Analysis
First and second-order transition matrices over binary point sequences (1 = server wins, 0 = returner wins). Chi-square test for independence against the i.i.d. null hypothesis. Results stratified by:
- **Within-game vs cross-game** — tests whether dependence persists across game boundaries or is explained by serve alternation
- **Score state** — routine, deuce, break point, server game point — tests whether pressure amplifies or eliminates streak persistence

### AR Logistic Regression
Lag-5 autoregressive feature matrix fed to logistic regression with 5-fold cross-validation. Primary metric: log loss (sensitive to probability calibration near the 63% majority class). Compared against a majority-class baseline.

### Hidden Markov Model
2-state `CategoricalHMM` (hmmlearn) trained on raw binary point sequences from 500 matches. Best of 5 random initializations by log-likelihood. Viterbi decoding on held-out matches. Key finding: the HIGH state (P(server wins)=0.772) is never assigned by Viterbi on real sequences — it is a soft-assignment artifact of EM training.

### Change-Point Detection
PELT algorithm (ruptures library, RBF kernel) on rolling serve win rate (window=10). Penalty set to log(n) — the BIC-derived heuristic — to avoid overfitting breakpoints. Used for visualization only; not a momentum inference method.

---

## Key Results Summary

| Method | Result |
|---|---|
| Chi-square (all points) | NOT independent (Chi²=726) — structural, not psychological |
| Within-game transitions | NOT independent — streak persistence within a game |
| Cross-game transitions | NOT independent — but **sign reverses** (serve alternation) |
| Deuce points | **Independent** (Chi²=0.07, p=0.79) |
| Break points | Near-independent (Chi²=11.66 vs 478 for routine) |
| AR logistic (lag=5) | **Worse than baseline** on log loss and accuracy |
| 2-state HMM (Viterbi) | **100% LOW state** — HIGH state never assigned |
| Game-level hold rate | Significant effect, but player-quality confounded |
