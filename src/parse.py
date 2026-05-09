import pandas as pd
import glob

def parse_pbp(pbp_string):
    points = []
    sets = pbp_string.split('.')
    for s in sets:
        games = s.split(';')
        for game in games:
            for char in game:
                if char in ('S', 'A'):
                    points.append(1)
                elif char in ('R', 'D'):
                    points.append(0)
    return points

def parse_pbp_games(pbp_string):
    """Returns list of point sequences, one per game, preserving game boundaries."""
    games = []
    for s in pbp_string.split('.'):
        for game in s.split(';'):
            points = []
            for char in game:
                if char in ('S', 'A'):
                    points.append(1)
                elif char in ('R', 'D'):
                    points.append(0)
            if points:
                games.append(points)
    return games

def load_all_matches(data_dir='data/', pattern='pbp_matches_atp_main_*.csv'):
    files = glob.glob(f'{data_dir}{pattern}')
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = f
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['point_seq'] = combined['pbp'].apply(parse_pbp)
    combined['point_games'] = combined['pbp'].apply(parse_pbp_games)
    combined['match_length'] = combined['point_seq'].apply(len)
    # drop rows with empty sequences
    combined = combined[combined['match_length'] > 0].reset_index(drop=True)
    return combined

if __name__ == "__main__":
    df = load_all_matches()
    print(f"Total matches: {len(df)}")
    print(f"Total points: {df['match_length'].sum():,}")
    print(f"Avg points per match: {df['match_length'].mean():.0f}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample:\n{df[['server1', 'server2', 'score', 'match_length']].head()}")