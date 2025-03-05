import pandas as pd
import numpy as np
import os

def extract_player_ids(df):
    """
    Extract unique player IDs for home and away teams from column names.
    """
    home_players = [col[5:-2] for col in df.columns if col.startswith('home_') and col.endswith('_x')]
    away_players = [col[5:-2] for col in df.columns if col.startswith('away_') and col.endswith('_x')]
    return home_players, away_players

def detect_attacking_direction(df, home_players, away_players):
    """
    Determine attacking direction for home and away teams based on initial player positions.
    """
    df = df.copy()
    
    first_seconds = df[df.index < 500]

    home_x_avg = first_seconds[[f'home_{pid}_x' for pid in home_players]].mean(axis=1).mean()
    away_x_avg = first_seconds[[f'away_{pid}_x' for pid in away_players]].mean(axis=1).mean()

    home_attacks_right_first_half = home_x_avg < away_x_avg

    df['Home_Attacking_Right'] = np.where(df['IdPeriod'] % 2 == 1, home_attacks_right_first_half, ~home_attacks_right_first_half)
    df['Away_Attacking_Right'] = ~df['Home_Attacking_Right']

    return df

def calculate_ball_physics(df):
    """
    Compute ball movement features: speed, acceleration, direction.
    """
    df = df.copy()
    
    if 'ball_x_Home' in df.columns and 'ball_y_Home' in df.columns:
        df[['ball_x_Home', 'ball_y_Home']] *= 0.01  

        df['Ball_Distance_Moved'] = np.sqrt(df['ball_x_Home'].diff()**2 + df['ball_y_Home'].diff()**2)
        df['Ball_Speed'] = df['Ball_Distance_Moved'] * 10  
        df['Ball_Acceleration'] = df['Ball_Speed'].diff() * 10
        df['Ball_Direction_Degrees'] = np.degrees(np.arctan2(df['ball_y_Home'].diff(), df['ball_x_Home'].diff()))
    
    return df

def compute_player_ball_distance(df, players, team_prefix):
    """
    Compute each player's distance to the ball.
    """
    if 'ball_x_Home' in df.columns and 'ball_y_Home' in df.columns:
        for player_id in players:
            df[f"{team_prefix}_{player_id}_DistanceToBall"] = np.sqrt(
                (df[f"{team_prefix}_{player_id}_x"] * 0.01 - df['ball_x_Home'])**2 + 
                (df[f"{team_prefix}_{player_id}_y"] * 0.01 - df['ball_y_Home'])**2
            )
    return df

def calculate_possession(df, home_players, away_players):
    """
    Determine which player is closest to the ball and assign possession.
    """
    df = df.copy()
    
    if 'ball_x_Home' in df.columns and 'ball_y_Home' in df.columns:
        home_dist_cols = [f'home_{pid}_DistanceToBall' for pid in home_players]
        away_dist_cols = [f'away_{pid}_DistanceToBall' for pid in away_players]

        df['Home_ClosestPlayer'] = df[home_dist_cols].idxmin(axis=1).str.replace("_DistanceToBall", "", regex=True)
        df['Away_ClosestPlayer'] = df[away_dist_cols].idxmin(axis=1).str.replace("_DistanceToBall", "", regex=True)

        df['Home_MinDistanceToBall'] = df[home_dist_cols].min(axis=1)
        df['Away_MinDistanceToBall'] = df[away_dist_cols].min(axis=1)

        df['Possession'] = np.where(
            df['Home_MinDistanceToBall'] < df['Away_MinDistanceToBall'], 
            df['Home_ClosestPlayer'], 
            df['Away_ClosestPlayer']
        )

    return df

def standardize_player_ids(df, match_id, output_folder="player_mappings"):
    """
    Standardizes player IDs dynamically per match and saves mapping to a text file.
    """
    os.makedirs(output_folder, exist_ok=True)

    home_players = sorted(set(col[5:-2] for col in df.columns if col.startswith('home_') and col.endswith('_x')))
    away_players = sorted(set(col[5:-2] for col in df.columns if col.startswith('away_') and col.endswith('_x')))

    home_mapping = {original: str(i+1) for i, original in enumerate(home_players)}
    away_mapping = {original: str(i+1) for i, original in enumerate(away_players)}

    mapping_file = os.path.join(output_folder, f"{match_id}_player_mapping.txt")
    with open(mapping_file, "w") as f:
        f.write(f"Match ID: {match_id}\n\n")
        f.write("Home Players Mapping:\n")
        for original, new in home_mapping.items():
            f.write(f"{original} -> home_{new}\n")
        f.write("\nAway Players Mapping:\n")
        for original, new in away_mapping.items():
            f.write(f"{original} -> away_{new}\n")

    print(f" Saved player ID mapping for {match_id} at {mapping_file}")

    new_columns = {}

    for original, new in home_mapping.items():
        for suffix in ['x', 'y', 'Speed', 'Acceleration', 'DistanceToBall', 'Direction']:
            old_col = f'home_{original}_{suffix}'
            new_col = f'home_{new}_{suffix}'
            if old_col in df.columns:
                new_columns[old_col] = new_col

    for original, new in away_mapping.items():
        for suffix in ['x', 'y', 'Speed', 'Acceleration', 'DistanceToBall', 'Direction']:
            old_col = f'away_{original}_{suffix}'
            new_col = f'away_{new}_{suffix}'
            if old_col in df.columns:
                new_columns[old_col] = new_col

    df = df.rename(columns=new_columns)

    df['Match_ID'] = match_id  # Ensure Match_ID is added

    if 'Possession' in df.columns:
        df['Possession'] = df['Possession'].astype(str)
        for original, new in home_mapping.items():
            df['Possession'] = df['Possession'].replace(f'home_{original}', f'home_{new}')
        for original, new in away_mapping.items():
            df['Possession'] = df['Possession'].replace(f'away_{original}', f'away_{new}')

    return df

def process_all_matches(input_dir, output_dir):
    """
    Process all match files and save feature-engineered datasets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match_files = [f for f in os.listdir(input_dir) if f.startswith("match_") and f.endswith(".csv")]

    for match_file in match_files:
        input_file = os.path.join(input_dir, match_file)
        match_id = match_file.split('_')[1].split(' ')[0]  
        output_file = os.path.join(output_dir, f"match_{match_id}_Features.csv")

        print(f" Processing match_{match_id}...")

        df = pd.read_csv(input_file)

        df = standardize_player_ids(df, match_id)
        home_players, away_players = extract_player_ids(df)

        if 'ball_x_Home' in df.columns and 'ball_y_Home' in df.columns:
            df = detect_attacking_direction(df, home_players, away_players)
            df = calculate_ball_physics(df)
            df = compute_player_ball_distance(df, home_players, "home")
            df = compute_player_ball_distance(df, away_players, "away")
            df = calculate_possession(df, home_players, away_players)

        df.to_csv(output_file, index=False)
        print(f" Processed data saved to {output_file}")

if __name__ == "__main__":
    input_dir = "/project_ghent/Test/Assignment/Merged"
    output_dir = "/project_ghent/Test/Assignment/Features_final/TestCase/Data_w_Features"
    process_all_matches(input_dir, output_dir)
