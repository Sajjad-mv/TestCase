import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def verify_adjacency_matrix(adj_matrix, num_nodes):
    """Verify and fix adjacency matrix if needed"""
    if adj_matrix is None:
        print("Creating identity adjacency matrix")
        return torch.eye(num_nodes)
    
    # Force correct shape if needed
    if adj_matrix.shape != (num_nodes, num_nodes):
        print(f"WARNING: Adjacency matrix has incorrect shape: {adj_matrix.shape}, creating correct one")
        return torch.ones((num_nodes, num_nodes))
    
    return adj_matrix
# Import the MTGNN model
from mtgnn import MTGNN

# 1. Create a PyTorch dataset and dataloader
def create_dataloaders(X_train, X_val, X_test, y_ball_x_train, y_ball_x_val, y_ball_x_test, 
                      y_ball_y_train, y_ball_y_val, y_ball_y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for training, validation, and testing
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Combine ball_x and ball_y targets
    y_train_tensor = torch.FloatTensor(np.stack([y_ball_x_train, y_ball_y_train], axis=1))
    y_val_tensor = torch.FloatTensor(np.stack([y_ball_x_val, y_ball_y_val], axis=1))
    y_test_tensor = torch.FloatTensor(np.stack([y_ball_x_test, y_ball_y_test], axis=1))
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Helper function to fix kernel size issues
def fix_kernel_size_issue(model, seq_length):
    """Fix kernel size issues for short sequences in MTGNN model"""
    # Find and fix all problematic convolution layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.kernel_size[1] > seq_length:
                print(f"Fixing kernel in {name}: {module.kernel_size} -> (1, {min(seq_length, 3)})")
                new_conv = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=(1, min(seq_length, 3)),  # Use smaller kernel
                    bias=module.bias is not None
                ).to(module.weight.device)
                
                # Initialize with same scheme as original model
                if module.weight is not None:
                    nn.init.xavier_uniform_(new_conv.weight)
                if module.bias is not None and new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
                
                # Replace the module
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, new_conv)
                else:
                    setattr(model, child_name, new_conv)
    
    return model

# 2. Configure and initialize the MTGNN model

def initialize_mtgnn_model(num_nodes, in_dim, seq_length, device):
    """
    Initialize MTGNN model with safe configuration
    """
    # Import the safe model initializer
    from safe_mtgnn_init import create_safe_mtgnn_model
    
    # Create a safe model with smaller sequence length (to avoid kernel size issues)
    safe_seq_length = min(seq_length, 151)  # Limit to 4 to be safe
    if safe_seq_length < seq_length:
        print(f"WARNING: Reducing sequence length from {seq_length} to {safe_seq_length} to avoid kernel size issues")
    
    # Create the model
    model = create_safe_mtgnn_model(num_nodes, in_dim, safe_seq_length, device)
    
    return model
    
def train_mtgnn(model, train_loader, val_loader, adj_matrix=None, epochs=20, lr=0.001, device='cuda'):
    """
    Train the MTGNN model
    """
    # Ensure adjacency matrix is properly sized
    if adj_matrix is not None:
        # Use hardcoded value for num_nodes
        num_nodes = 29  # Fixed value for 29 nodes (28 players + 1 ball)
        if adj_matrix.shape != (num_nodes, num_nodes):
            print(f"Fixing adjacency matrix shape: {adj_matrix.shape} -> ({num_nodes}, {num_nodes})")
            if os.path.exists("static_adjacency.pt"):
                adj_matrix = torch.load("static_adjacency.pt").to(data.device)
            else:
                # Create identity matrix as fallback
                adj_matrix = torch.eye(num_nodes).to(data.device)
        # Move adjacency matrix to device if provided
    if adj_matrix is not None:
        # Verify adjacency matrix has correct shape
        adj_matrix = verify_adjacency_matrix(adj_matrix, 29)
    
        adj_matrix = adj_matrix.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                if adj_matrix is not None:
                    output = model(data, A_tilde=adj_matrix)
                else:
                    output = model(data)
                
                # Reshape output to match target - fixed version
                # Model output shape: [batch_size, out_dim, num_nodes, seq_len]
                # Target shape: [batch_size, out_dim]
                
                # 1. Keep only the ball node (last node)
                ball_output = output[:, :, -1, :]  # [batch_size, out_dim, seq_len]
                
                # 2. Average over sequence dimension
                ball_output = ball_output.mean(dim=-1)  # [batch_size, out_dim]
                
                # Calculate loss - now shapes match
                loss = criterion(ball_output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
            
            except RuntimeError as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                if "size mismatch" in str(e) or "kernel size" in str(e):
                    print("This is likely a kernel size issue. Fixing the model and continuing...")
                    seq_length = data.shape[-1]
                    model = fix_kernel_size_issue(model, seq_length)
                    continue
                else:
                    raise
        
        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                if adj_matrix is not None:
                    output = model(data, A_tilde=adj_matrix)
                else:
                    output = model(data)
                
                # Reshape output to match target - fixed version
                # Model output shape: [batch_size, out_dim, num_nodes, seq_len]
                # Target shape: [batch_size, out_dim]
                
                # 1. Keep only the ball node (last node)
                ball_output = output[:, :, -1, :]  # [batch_size, out_dim, seq_len]
                
                # 2. Average over sequence dimension
                ball_output = ball_output.mean(dim=-1)  # [batch_size, out_dim]
                
                # Calculate loss - now shapes match
                loss = criterion(ball_output, target)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            print(f'New best model saved with validation loss: {best_val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses

# 4. Evaluate the model
def evaluate_model(model, test_loader, adj_matrix=None, device='cuda'):
    """
    Evaluate the model on test data
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Ensure adjacency matrix is properly sized
    if adj_matrix is not None:
        # Use hardcoded value for num_nodes
        num_nodes = 29  # Fixed value for 29 nodes (28 players + 1 ball)
        if adj_matrix.shape != (num_nodes, num_nodes):
            print(f"Fixing adjacency matrix shape: {adj_matrix.shape} -> ({num_nodes}, {num_nodes})")
            if os.path.exists("static_adjacency.pt"):
                adj_matrix = torch.load("static_adjacency.pt").to(data.device)
            else:
                # Create identity matrix as fallback
                adj_matrix = torch.eye(num_nodes).to(data.device)
        # Move adjacency matrix to device if provided
    if adj_matrix is not None:
        adj_matrix = adj_matrix.to(device)
    
    test_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if adj_matrix is not None:
                output = model(data, A_tilde=adj_matrix)
            else:
                output = model(data)
            
            # Reshape output to match target
            output = output.squeeze(-1).transpose(1, 2)

            # Add debugging to see the actual shape
            print(f"Model output shape: {output.shape}")
    
            # Check dimensions and adjust indexing accordingly
            if output.dim() == 3:
                # If output is [batch_size, num_nodes, features]
                output = output[:, -1, :]  # Get the last node (ball)
            elif output.dim() == 4:
                # If output is [batch_size, seq_len, num_nodes, features]
                output = output[:, -1, :, :].mean(dim=-1)
            
            # Calculate loss
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Combine all batches
    all_preds = np.vstack([p for p in all_preds])
    all_targets = np.vstack([t for t in all_targets])

    np.save("predictions.npy", all_preds)
    np.save("targets.npy", all_targets)
    print("Predictions saved to predictions.npy")
    print("Targets saved to targets.npy")
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    
    print(f'Test Loss: {test_loss/len(test_loader):.6f}')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE: {mae:.6f}')
    
    return all_preds, all_targets, mse, rmse, mae

# 5. Visualize predictions
def visualize_predictions(predictions, targets, num_samples=5):
    """
    Visualize ball position predictions vs actual values
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    
    # If only one sample, make axes iterable
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(predictions))
        
        # Get predicted and actual x, y coordinates
        pred_x, pred_y = predictions[idx]
        true_x, true_y = targets[idx]
        
        # Plot
        axes[i].scatter(true_x, true_y, color='blue', label='Actual', s=50)
        axes[i].scatter(pred_x, pred_y, color='red', label='Predicted', s=50)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_title(f'Sample {i+1}: Ball Position Prediction')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('ball_position_predictions.png')
    plt.show()

# 6. Main function to run the entire pipeline
def run_football_prediction_pipeline(data_splits, adj_matrix=None, node_features=None, 
                                    batch_size=32, epochs=20, device='cuda', 
                                    initial_model=None):
    """
    Run the complete pipeline for football prediction with MTGNN
    """
    # Unpack data splits
    X_train, X_val, X_test, y_ball_x_train, y_ball_x_val, y_ball_x_test, y_ball_y_train, y_ball_y_val, y_ball_y_test = data_splits
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, 
        y_ball_x_train, y_ball_x_val, y_ball_x_test,
        y_ball_y_train, y_ball_y_val, y_ball_y_test,
        batch_size=batch_size
    )
    
    # Get model dimensions from data
    batch, in_dim, num_nodes, seq_length = X_train.shape
    print(f"Input shape: batch={batch}, in_dim={in_dim}, num_nodes={num_nodes}, seq_length={seq_length}")
    
    # Initialize model
    if initial_model is None:
        model = initialize_mtgnn_model(num_nodes, in_dim, seq_length, device)
    else:
        model = initial_model  # Use the provided model instead
    
    # Train model
    model, train_losses, val_losses = train_mtgnn(
        model, train_loader, val_loader, 
        adj_matrix=adj_matrix,
        epochs=epochs, 
        device=device
    )
    
    # Evaluate model
    predictions, targets, mse, rmse, mae = evaluate_model(model, test_loader, adj_matrix, device)
    
    # Visualize results
    visualize_predictions(predictions, targets)
    
    return model, (predictions, targets), (mse, rmse, mae)