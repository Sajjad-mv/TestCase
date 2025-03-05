# Football Tracking Data - Merging Home and Away Files 

import pandas as pd
import os

def merge_home_away(data_dir, match_ids, output_dir):
    """Merge Home and Away CSV files for each match using 'Time' and 'IdPeriod', ensuring correct alignment."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for match_id in match_ids:
        home_file = os.path.join(data_dir, match_id, 'Home_Cleaned.csv')
        away_file = os.path.join(data_dir, match_id, 'Away_Cleaned.csv')

        # Load CSV files
        home_df = pd.read_csv(home_file)
        away_df = pd.read_csv(away_file)

        # Ensure both files have required columns
        if 'Time' not in home_df.columns or 'Time' not in away_df.columns:
            raise ValueError("Missing 'Time' column in one of the files")
        if 'IdPeriod' not in home_df.columns or 'IdPeriod' not in away_df.columns:
            raise ValueError("Missing 'IdPeriod' column in one of the files")

        # Merge dataframes based on 'Time' and 'IdPeriod'
        merged_df = pd.merge(home_df, away_df, on=['Time', 'IdPeriod'], suffixes=('_Home', '_Away'), how='outer')

        # Drop duplicate columns (like MatchId and Team if present)
        for col in ['MatchId_Home', 'MatchId_Away', 'Team_Home', 'Team_Away']:
            if col in merged_df.columns:
                merged_df.drop(columns=[col], inplace=True)

        # Sort by IdPeriod and Time
        merged_df.sort_values(by=['IdPeriod', 'Time'], inplace=True)

        # Save merged dataframe
        output_file = os.path.join(output_dir, f'{match_id}_Merged.csv')
        merged_df.to_csv(output_file, index=False)
        print(f'Merged file saved: {output_file}')


if __name__ == '__main__':
    data_dir = '/project_ghent/Test/Assignment'
    match_ids = ['match_1', 'match_2', 'match_3', 'match_4', 'match_5']
    output_dir = '/project_ghent/Test/Assignment/Merged'

    merge_home_away(data_dir, match_ids, output_dir)
