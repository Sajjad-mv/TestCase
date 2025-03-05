import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def load_match_files(data_dir, match_id):
    """
    Load Home and Away CSV files for a specific match
    
    Args:
        data_dir (str): Data directory path
        match_id (str): Match identifier (e.g., 'match_1')
        
    Returns:
        tuple: (home_df, away_df) - DataFrames for home and away teams
    """
    # Define file paths
    home_file = os.path.join(data_dir, match_id, "Home.csv")
    away_file = os.path.join(data_dir, match_id, "Away.csv")
    
    # Check if files exist
    if not os.path.exists(home_file) or not os.path.exists(away_file):
        raise FileNotFoundError(f"Home or Away file not found for match {match_id}")
    
    # Load the CSV files
    home_df = pd.read_csv(home_file)
    away_df = pd.read_csv(away_file)
    
    print(f"Loaded match {match_id}:")
    print(f"  Home data shape: {home_df.shape}")
    print(f"  Away data shape: {away_df.shape}")
    
    # Check if ball data is present
    has_ball_home = 'ball_x' in home_df.columns and 'ball_y' in home_df.columns
    has_ball_away = 'ball_x' in away_df.columns and 'ball_y' in away_df.columns
    
    print(f"  Ball data present in home file: {has_ball_home}")
    print(f"  Ball data present in away file: {has_ball_away}")
    
    return home_df, away_df

# Function to get basic statistics about the data
def get_data_statistics(df, team_type='home'):
    """
    Calculate basic statistics for a dataframe
    
    Args:
        df (DataFrame): Input dataframe
        team_type (str): 'home' or 'away'
        
    Returns:
        dict: Dictionary of statistics
    """
    stats = {}
    
    # Get match periods
    stats['periods'] = df['IdPeriod'].unique().tolist()
    
    # Get time range
    stats['min_time'] = df['Time'].min()
    stats['max_time'] = df['Time'].max()
    
    # Get player columns
    player_columns = [col for col in df.columns if f'{team_type}_' in col and col.endswith('_x')]
    stats['num_players'] = len(player_columns)
    
    # Get coordinate ranges
    x_cols = [col for col in df.columns if col.endswith('_x') and 'ball' not in col]
    y_cols = [col for col in df.columns if col.endswith('_y') and 'ball' not in col]
    
    all_x_values = [df[col].dropna() for col in x_cols]
    all_y_values = [df[col].dropna() for col in y_cols]
    
    # Flatten list of series
    all_x_values = pd.concat(all_x_values)
    all_y_values = pd.concat(all_y_values)
    
    stats['x_min'] = all_x_values.min()
    stats['x_max'] = all_x_values.max()
    stats['y_min'] = all_y_values.min()
    stats['y_max'] = all_y_values.max()
    
    # Ball statistics if available
    if 'ball_x' in df.columns and 'ball_y' in df.columns:
        stats['ball_x_min'] = df['ball_x'].min()
        stats['ball_x_max'] = df['ball_x'].max()
        stats['ball_y_min'] = df['ball_y'].min()
        stats['ball_y_max'] = df['ball_y'].max()
    
    return stats

# Example usage
def inspect_match_data(data_dir, match_id):
    """
    Load and inspect data for a single match
    """
    home_df, away_df = load_match_files(data_dir, match_id)
    
    # Get basic statistics
    home_stats = get_data_statistics(home_df, 'home')
    away_stats = get_data_statistics(away_df, 'away')
    
    # Print coordinate ranges
    print("\nCoordinate ranges (in cm, centered at [0,0]):")
    print(f"  Home X: [{home_stats['x_min']:.1f}, {home_stats['x_max']:.1f}]")
    print(f"  Home Y: [{home_stats['y_min']:.1f}, {home_stats['y_max']:.1f}]")
    print(f"  Away X: [{away_stats['x_min']:.1f}, {away_stats['x_max']:.1f}]")
    print(f"  Away Y: [{away_stats['y_min']:.1f}, {away_stats['y_max']:.1f}]")
    
    # Print ball ranges if available
    if 'ball_x_min' in home_stats:
        print("\nBall coordinate ranges:")
        print(f"  Ball X: [{home_stats['ball_x_min']:.1f}, {home_stats['ball_x_max']:.1f}]")
        print(f"  Ball Y: [{home_stats['ball_y_min']:.1f}, {home_stats['ball_y_max']:.1f}]")
    
    # Print match periods
    print(f"\nMatch periods: {home_stats['periods']}")
    print(f"Time range: {home_stats['min_time']} to {home_stats['max_time']}")
    
    return home_df, away_df, home_stats, away_stats


# Main program section that runs
if __name__ == "__main__":
    data_dir = "/project_ghent/Test/Assignment"  
    
    # Inspect all matches
    match_ids = ["match_1", "match_2", "match_3", "match_4", "match_5"]
    
    for match_id in match_ids:
        print(f"\n{'='*50}")
        print(f"Inspecting {match_id}")
        print(f"{'='*50}")
        try:
            home_df, away_df, home_stats, away_stats = inspect_match_data(data_dir, match_id)
        except Exception as e:
            print(f"Error processing {match_id}: {e}")