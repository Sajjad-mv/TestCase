import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from torch.utils.data import DataLoader, TensorDataset
import time

def load_and_preprocess_data(csv_path, match_id=None):
    """Load and preprocess data with optional match_id labeling"""
    df = pd.read_csv(csv_path)
    
    # Add match_id if provided
    if match_id is not None:
        df['match_id'] = match_id
        
    # Sort by IdPeriod and Time within the same match
    df = df.sort_values(by=['IdPeriod', 'Time'])
    return df

def add_ball_columns_to_match5(df_match5):
    """Add empty ball location columns to match 5 dataframe"""
    # Check if ball columns already exist
    if 'ball_x_Home' in df_match5.columns and 'ball_y_Home' in df_match5.columns:
        return df_match5
    
    # Add empty ball columns
    df_match5['ball_x_Home'] = np.nan
    df_match5['ball_y_Home'] = np.nan
    
    # Add other ball columns if needed
    for col in ['Ball_Speed', 'Ball_Acceleration', 'Ball_Direction_Degrees']:
        if col not in df_match5.columns:
            df_match5[col] = np.nan
    
    return df_match5

def combine_all_matches(match_files, match5_file, n_frames_match5=11000):
    """
    Combine all matches into one dataframe sequentially and prepare for training
    
    Args:
        match_files: List of CSV file paths for matches 1-4
        match5_file: Path to match 5 CSV file
        n_frames_match5: Number of frames to use from match 5 (for test set)
    
    Returns:
        Combined dataframe, and separate test set from match 5
    """
    combined_df = pd.DataFrame()
    
    # Load and append matches 1-4 sequentially
    for i, file_path in enumerate(match_files):
        print(f"Loading match {i+1}: {file_path}")
        df = load_and_preprocess_data(file_path, match_id=i+1)
        
        # Reset IdPeriod to ensure sequential order across matches
        max_period = 0 if combined_df.empty else combined_df['IdPeriod'].max()
        df['IdPeriod'] = df['IdPeriod'] + max_period
        
        # Append to combined dataframe
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Load match 5 (first n_frames only)
    print(f"Loading first {n_frames_match5} frames from match 5: {match5_file}")
    df_match5 = load_and_preprocess_data(match5_file, match_id=5)
    df_match5 = df_match5.iloc[:n_frames_match5]
    
    # Reset IdPeriod for match 5 to ensure sequential order
    max_period = combined_df['IdPeriod'].max()
    df_match5['IdPeriod'] = df_match5['IdPeriod'] + max_period
    
    # Add empty ball columns to match 5
    df_match5 = add_ball_columns_to_match5(df_match5)
    
    # Separate match 5 for test set
    test_set = df_match5.copy()
    
    # Add match 5 to combined dataframe
    combined_df = pd.concat([combined_df, df_match5], ignore_index=True)
    
    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"Test set shape: {test_set.shape}")
    
    return combined_df, test_set


def extract_features_combined(df, is_test_set=False, saved_scalers=None, simple_features=True):
    """
    Extract and normalize features from the combined dataframe
    
    Args:
        df: Combined dataframe or test set
        is_test_set: Whether this is the test set (match 5)
        saved_scalers: Scalers from training data
        simple_features: If True, only use x, y coordinates (simpler model)
        
    Returns:
        Normalized node features, scalers (if not test set)
    """
    node_features = []
    scalers = [] if saved_scalers is None else saved_scalers
    
    # Extract home players
    for i in range(1, 15):
        if f'home_{i}_x' in df.columns and f'home_{i}_y' in df.columns:
            # For simple features, only extract x, y coordinates
            if simple_features:
                player_features = df[[f'home_{i}_x', f'home_{i}_y']]
            else:
                # Select base features for all players
                player_features = df[[f'home_{i}_x', f'home_{i}_y']]
                
                # Add distance to ball if available
                if f'home_{i}_DistanceToBall' in df.columns:
                    player_features[f'home_{i}_DistanceToBall'] = df[f'home_{i}_DistanceToBall']
                else:
                    # Create a placeholder column for missing DistanceToBall
                    player_features[f'home_{i}_DistanceToBall'] = 0
            
            # Handle normalization
            if is_test_set and saved_scalers:
                # Use saved scaler for test data
                scaler_idx = len(node_features)
                if scaler_idx < len(saved_scalers):
                    scaler = saved_scalers[scaler_idx]
                    normalized = pd.DataFrame(
                        scaler.transform(player_features.fillna(0)), 
                        columns=player_features.columns
                    )
                else:
                    # Fallback if scaler missing
                    normalized = player_features.fillna(0)
            else:
                # Create new scaler for training data
                scaler = StandardScaler()
                normalized = pd.DataFrame(
                    scaler.fit_transform(player_features.fillna(0)), 
                    columns=player_features.columns
                )
                scalers.append(scaler)
            
            node_features.append(normalized)
    
    # Extract away players (same pattern)
    for i in range(1, 15):
        if f'away_{i}_x' in df.columns and f'away_{i}_y' in df.columns:
            if simple_features:
                player_features = df[[f'away_{i}_x', f'away_{i}_y']]
            else:
                player_features = df[[f'away_{i}_x', f'away_{i}_y']]
                
                if f'away_{i}_DistanceToBall' in df.columns:
                    player_features[f'away_{i}_DistanceToBall'] = df[f'away_{i}_DistanceToBall']
                else:
                    player_features[f'away_{i}_DistanceToBall'] = 0
            
            # Handle normalization
            if is_test_set and saved_scalers:
                # Use saved scaler for test data
                scaler_idx = len(node_features)
                if scaler_idx < len(saved_scalers):
                    scaler = saved_scalers[scaler_idx]
                    normalized = pd.DataFrame(
                        scaler.transform(player_features.fillna(0)), 
                        columns=player_features.columns
                    )
                else:
                    # Fallback if scaler missing
                    normalized = player_features.fillna(0)
            else:
                # Create new scaler for training data
                scaler = StandardScaler()
                normalized = pd.DataFrame(
                    scaler.fit_transform(player_features.fillna(0)), 
                    columns=player_features.columns
                )
                scalers.append(scaler)
            
            node_features.append(normalized)
    
    # Extract ball features - only x, y for simple model
    ball_columns = ['ball_x_Home', 'ball_y_Home']
    
    if not simple_features:
        # Add additional ball columns for full model
        additional_ball_columns = ['Ball_Speed', 'Ball_Acceleration', 'Ball_Direction_Degrees']
        available_ball_columns = ball_columns + [col for col in additional_ball_columns if col in df.columns]
    else:
        available_ball_columns = ball_columns
    
    if all(col in df.columns for col in ball_columns):
        ball_features = df[available_ball_columns]
        
        # Handle normalization
        if is_test_set and saved_scalers:
            # Use saved scaler for test data
            scaler_idx = len(node_features)
            if scaler_idx < len(saved_scalers):
                scaler = saved_scalers[scaler_idx]
                
                # For test set, we may need to handle NaN values in ball features differently
                if df['match_id'].max() == 5:  # This is match 5
                    # For match 5 test set, fill NaN values with zeros
                    ball_features = ball_features.fillna(0)
                
                normalized = pd.DataFrame(
                    scaler.transform(ball_features.fillna(0)), 
                    columns=ball_features.columns
                )
            else:
                # Fallback if scaler missing
                normalized = ball_features.fillna(0)
        else:
            # Create new scaler for training data
            scaler = StandardScaler()
            normalized = pd.DataFrame(
                scaler.fit_transform(ball_features.fillna(0)), 
                columns=ball_features.columns
            )
            scalers.append(scaler)
            
            # Save ball position scaler separately for denormalization
            ball_pos_scaler = StandardScaler()
            ball_pos_scaler.fit(df[ball_columns].fillna(0))
            
            # Save mean and std for ball positions for later use
            with open("ball_pos_norm_params.txt", "w") as f:
                f.write(f"ball_x_mean: {ball_pos_scaler.mean_[0]}\n")
                f.write(f"ball_x_std: {ball_pos_scaler.scale_[0]}\n")
                f.write(f"ball_y_mean: {ball_pos_scaler.mean_[1]}\n")
                f.write(f"ball_y_std: {ball_pos_scaler.scale_[1]}\n")
            
            scalers.append(('ball_pos', ball_pos_scaler))
        
        node_features.append(normalized)
    
    if is_test_set:
        return node_features
    else:
        return node_features, scalers

def prepare_sequences_combined(node_features, seq_length=4, pred_length=1, is_test_set=False, match_id_col=None):
    """
    Create sequences for training/testing with combined data
    
    Args:
        node_features: List of dataframes with features for each node
        seq_length: Length of input sequence
        pred_length: How many steps ahead to predict
        is_test_set: Whether this is the test set (match 5)
        match_id_col: Optional column with match_id values for ensuring sequences don't cross match boundaries
    
    Returns:
        X: Input sequences
        y_ball_x, y_ball_y: Target values (if not test set)
    """
    sequences = []
    
    if not is_test_set:
        # For training: collect target values
        targets_ball_x = []
        targets_ball_y = []
    
    # Ensure all node features have the same length
    min_length = min([len(df) for df in node_features])
    
    # Get feature dimensions for each node
    feature_dims = [df.shape[1] for df in node_features]
    print(f"Feature dimensions per node: {feature_dims}")
    
    # Number of valid sequences
    if is_test_set:
        # For test set, we use all possible sequences
        num_sequences = min_length - seq_length + 1
    else:
        # For training, we reserve data for targets
        num_sequences = min_length - seq_length - pred_length + 1
    
    print(f"Creating {num_sequences} sequences with length {seq_length}")
    
    valid_start_indices = []
    
    # If match_id_col provided, ensure sequences don't cross match boundaries
    if match_id_col is not None:
        for i in range(num_sequences):
            end_idx = i + seq_length - 1
            if is_test_set:
                # For test set, all sequences are within match 5
                valid_start_indices.append(i)
            else:
                # For training, check if sequence crosses match boundaries
                if (match_id_col.iloc[i] == match_id_col.iloc[end_idx] and
                    match_id_col.iloc[end_idx] == match_id_col.iloc[end_idx + pred_length]):
                    valid_start_indices.append(i)
    else:
        # Without match_id_col, use all possible sequences
        valid_start_indices = list(range(num_sequences))
    
    print(f"Using {len(valid_start_indices)} valid sequences after checking match boundaries")
    
    for i in valid_start_indices:
        # Prepare input sequence
        seq_data = []
        for node_idx, node_df in enumerate(node_features):
            node_seq = node_df.iloc[i:i+seq_length].values
            seq_data.append(node_seq)
        
        # Store sequence
        sequences.append(seq_data)
        
        # Prepare target values (only for training)
        if not is_test_set:
            # Assuming ball is the last node and first two features are x, y
            ball_node_idx = len(node_features) - 1
            target_idx = i + seq_length + pred_length - 1
            
            # Get ball position at target time step
            ball_x = node_features[ball_node_idx].iloc[target_idx, 0]  # x-position
            ball_y = node_features[ball_node_idx].iloc[target_idx, 1]  # y-position
            
            targets_ball_x.append(ball_x)
            targets_ball_y.append(ball_y)
    
    # Format sequences for model input
    # [batch_size, seq_len, num_nodes, in_dim (features)]
    formatted_sequences = []
    
    # Standardize feature dimensions
    max_features = max(feature_dims)
    
    for seq in sequences:
        # Create padded array for sequence
        padded_seq = np.zeros((len(seq), max_features, seq_length))
        
        for node_idx, node_seq in enumerate(seq):
            # Fill available features
            num_features = node_seq.shape[1]
            # Transpose to get [node, feature, sequence]
            padded_seq[node_idx, :num_features, :] = node_seq.T
        
        # Transpose to get [sequence, node, feature]
        transposed = np.transpose(padded_seq, (2, 0, 1))
        formatted_sequences.append(transposed)
    
    X = np.array(formatted_sequences)
    # Final format: [batch_size, seq_len, num_nodes, in_dim]
    
    if not is_test_set:
        y_ball_x = np.array(targets_ball_x)
        y_ball_y = np.array(targets_ball_y)
        return X, y_ball_x, y_ball_y
    else:
        return X

def run_combined_training_pipeline(match_files, match5_file, n_frames_match5=11000, 
                               seq_length=4, pred_length=1, batch_size=32, epochs=10,
                               simple_features=True):
    """
    Run the complete training pipeline with combined data
    
    Args:
        match_files: List of CSV files for matches 1-4
        match5_file: CSV file for match 5
        n_frames_match5: Number of frames to use from match 5
        seq_length: Sequence length for input
        pred_length: Prediction horizon
        batch_size: Batch size for training
        epochs: Number of training epochs
        simple_features: If True, only use x, y coordinates (simpler model)
    
    Returns:
        Trained model, predictions for match 5, results dataframe
    """
    from model_training import initialize_mtgnn_model, run_football_prediction_pipeline
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Step 1: Combining match data...")
    combined_df, test_df = combine_all_matches(match_files, match5_file, n_frames_match5)
    
    print("\nStep 2: Extracting features from combined data...")
    print(f"Using {'simple' if simple_features else 'full'} feature set...")
    node_features, scalers = extract_features_combined(combined_df, simple_features=simple_features)
    
    # Save scalers for later use
    model_type = "simple" if simple_features else "full"
    scalers_file = f"combined_scalers_{model_type}.pt"
    torch.save(scalers, scalers_file)
    print(f"Saved {len(scalers)} scalers to {scalers_file}")
    
    print("\nStep 3: Preparing sequences for training...")
    X, y_ball_x, y_ball_y = prepare_sequences_combined(
        node_features, 
        seq_length=seq_length, 
        pred_length=pred_length,
        is_test_set=False,
        match_id_col=combined_df['match_id']
    )
    
    print("\nStep 4: Splitting data for training and validation...")
    # Split data excluding match 5 rows
    match5_mask = combined_df['match_id'] == 5
    non_match5_indices = (~match5_mask).cumsum() - 1
    max_idx = non_match5_indices.max()
    
    train_idx, val_idx = train_test_split(
        range(max_idx + 1),
        test_size=0.2,
        random_state=42
    )
    
    # Use the indices to split the data
    X_train, y_ball_x_train, y_ball_y_train = X[train_idx], y_ball_x[train_idx], y_ball_y[train_idx]
    X_val, y_ball_x_val, y_ball_y_val = X[val_idx], y_ball_x[val_idx], y_ball_y[val_idx]
    
    # Create a test set from validation data (required for run_football_prediction_pipeline format)
    X_val, X_test, y_ball_x_val, y_ball_x_test, y_ball_y_val, y_ball_y_test = train_test_split(
        X_val, y_ball_x_val, y_ball_y_val, test_size=0.5, random_state=42
    )
    
    # Format data for the pipeline function
    data_splits = (
        X_train, X_val, X_test, 
        y_ball_x_train, y_ball_x_val, y_ball_x_test,
        y_ball_y_train, y_ball_y_val, y_ball_y_test
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create adjacency matrix
    num_nodes = X_train.shape[2]
    adj_matrix = torch.ones(num_nodes, num_nodes)
    
    print("\nStep 5: Training model...")
    # Initialize and train model using the original functions
    _, in_dim, num_nodes, seq_length_actual = X_train.shape
    print(f"Input dimensions: batch_size={X_train.shape[0]}, in_dim={in_dim}, num_nodes={num_nodes}, seq_length={seq_length_actual}")
    
    # Use the existing function from model_training.py to train the model
    model, results, metrics = run_football_prediction_pipeline(
        data_splits,
        adj_matrix=adj_matrix,
        node_features=None,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        initial_model=None  # Start with a fresh model
    )
    
    # Save model with appropriate name
    model_file = f"final_combined_model_{model_type}.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Saved trained model to {model_file}")
    
    print("\nStep 6: Preparing test data (match 5)...")
    test_features = extract_features_combined(
        test_df, 
        is_test_set=True, 
        saved_scalers=scalers,
        simple_features=simple_features
    )
    
    X_test = prepare_sequences_combined(
        test_features,
        seq_length=seq_length,
        is_test_set=True
    )
    
    print(f"Match 5 test data shape: {X_test.shape}")
    
    print("\nStep 7: Generating predictions for match 5...")
    output_file = f"match5_ball_predictions_{model_type}.csv"
    norm_predictions, denorm_predictions, results_df = predict_test_set(
        model, 
        X_test, 
        original_df=test_df,
        saved_scalers=scalers,
        output_file=output_file
    )
    
    print(f"Generated predictions for {len(results_df)} time steps")
    print(f"Predictions saved to {output_file}")
    
    return model, results_df


def denormalize_predictions(predictions, saved_scalers=None):
    """
    Denormalize predictions back to original scale
    
    Args:
        predictions: Normalized predictions [batch_size, 2]
        saved_scalers: List of scalers from training
    
    Returns:
        Denormalized predictions
    """
    # Find ball position scaler
    ball_pos_scaler = None
    if saved_scalers:
        for item in saved_scalers:
            if isinstance(item, tuple) and item[0] == 'ball_pos':
                ball_pos_scaler = item[1]
                break
    
    if ball_pos_scaler:
        # Use scaler to denormalize
        return ball_pos_scaler.inverse_transform(predictions)
    else:
        # Fallback to parameters from file
        try:
            params = {}
            with open("ball_pos_norm_params.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    params[key] = float(value)
            
            # Apply denormalization
            x_mean = params.get('ball_x_mean', 0)
            x_std = params.get('ball_x_std', 1)
            y_mean = params.get('ball_y_mean', 0)
            y_std = params.get('ball_y_std', 1)
            
            denorm_predictions = predictions.copy()
            denorm_predictions[:, 0] = predictions[:, 0] * x_std + x_mean  # x
            denorm_predictions[:, 1] = predictions[:, 1] * y_std + y_mean  # y
            
            return denorm_predictions
        except Exception as e:
            print(f"Error using norm params file: {str(e)}")
            return predictions

def visualize_predictions(predictions_df, num_samples=5, output_dir="."):
    """
    Visualize sample ball position predictions
    
    Args:
        predictions_df: DataFrame with predictions
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(12, 10))
    
    # Set field dimensions
    field_width = 10500  # -5250 to 5250
    field_height = 6800  # -3400 to 3400
    
    # Draw field
    plt.plot([-field_width/2, field_width/2, field_width/2, -field_width/2, -field_width/2],
             [-field_height/2, -field_height/2, field_height/2, field_height/2, -field_height/2], 'g-')
    
    # Sample predictions to plot
    if len(predictions_df) <= num_samples:
        samples = range(len(predictions_df))
    else:
        # Take evenly spaced samples
        samples = np.linspace(0, len(predictions_df)-1, num_samples, dtype=int)
    
    # Plot each sample with a different color
    colors = ['r', 'b', 'm', 'c', 'y', 'k', 'orange', 'purple']
    for i, idx in enumerate(samples):
        color = colors[i % len(colors)]
        x = predictions_df.iloc[idx]['ball_x_Home']
        y = predictions_df.iloc[idx]['ball_y_Home']
        plt.scatter(x, y, color=color, s=100, label=f'Point {idx}')
    
    plt.title('Sample Ball Position Predictions')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"))
    print(f"Saved sample predictions visualization to {os.path.join(output_dir, 'sample_predictions.png')}")
    plt.close()

def predict_test_set(model, X_test, original_df=None, saved_scalers=None, output_file="match5_ball_predictions.csv"):
    """
    Generate predictions for test set
    
    Args:
        model: Trained model
        X_test: Test data
        original_df: Original test dataframe (for metadata)
        saved_scalers: List of scalers from training
        output_file: File name to save predictions
    
    Returns:
        Normalized predictions, denormalized predictions, results dataframe
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model dimensions from data
    batch_size, seq_length, num_nodes, in_dim = X_test.shape
    
    # Create adjacency matrix
    adj_matrix = torch.ones(num_nodes, num_nodes).to(device)
    
    # Create data loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Generate predictions
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            
            outputs = model(batch_x, A_tilde=adj_matrix)
            
            # Get predictions for ball node (last node)
            ball_outputs = outputs[:, :, -1, :].mean(dim=-1)  # [batch_size, out_dim]
            all_predictions.append(ball_outputs.cpu().numpy())
    
    # Combine predictions
    normalized_predictions = np.vstack(all_predictions)
    
    # Denormalize predictions
    denormalized_predictions = denormalize_predictions(normalized_predictions, saved_scalers)
    
    # Create results dataframe
    results_df = pd.DataFrame(denormalized_predictions, columns=['ball_x_Home', 'ball_y_Home'])
    
    # Add metadata if original dataframe provided
    if original_df is not None and len(results_df) <= len(original_df):
        # Add time information
        if 'Time' in original_df.columns:
            results_df['Time'] = original_df.iloc[:len(results_df)]['Time'].values
            
        # Add period information
        if 'IdPeriod' in original_df.columns:
            results_df['IdPeriod'] = original_df.iloc[:len(results_df)]['IdPeriod'].values
    
    # Save predictions to CSV
    results_df.to_csv(output_file, index=False)
    
    # Visualize sample predictions
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    visualize_predictions(results_df, output_dir=output_dir)
    
    return normalized_predictions, denormalized_predictions, results_df


def main():
    """Main function to run the combined training pipeline"""
    print("=== Football Ball Position Prediction with Combined Data ===")
    
    # Get inputs from user
    match_files_input = input("Enter comma-separated paths to match 1-4 CSV files: ").strip()
    match_files = [path.strip() for path in match_files_input.split(',')]
    
    match5_file = input("Enter path to match 5 CSV file: ").strip()
    
    # Choose model type
    model_type = input("Choose model type (1 for full features, 2 for simple x,y features only): ").strip()
    simple_features = model_type == "2"
    print(f"Selected model type: {'Simple (x,y only)' if simple_features else 'Full (all features)'}")
    
    # Get parameters with defaults
    n_frames_match5 = int(input("Enter number of frames to use from match 5 (default 11000): ") or "11000")
    seq_length = int(input("Enter sequence length (default 4): ") or "4")
    pred_length = int(input("Enter prediction length (default 1): ") or "1")
    batch_size = int(input("Enter batch size (default 32): ") or "32")
    epochs = int(input("Enter number of epochs (default 10): ") or "10")
    
    # Record start time
    start_time = time.time()
    
    # Run the pipeline
    model, results_df = run_combined_training_pipeline(
        match_files=match_files,
        match5_file=match5_file,
        n_frames_match5=n_frames_match5,
        seq_length=seq_length,
        pred_length=pred_length,
        batch_size=batch_size,
        epochs=epochs,
        simple_features=simple_features
    )
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print("\nTraining and prediction complete!")
    model_type_str = "simple" if simple_features else "full"
    print(f"Results saved to match5_ball_predictions_{model_type_str}.csv")


if __name__ == "__main__":
    main()
