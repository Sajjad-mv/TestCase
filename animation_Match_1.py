import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.widgets import Button
import os
import warnings
warnings.filterwarnings('ignore')

def create_ball_trajectory_animation():
    # Check if target_comparison.csv exists
    if os.path.exists('targets_comparison.csv'):
        # Load the comparison data
        comparison_df = pd.read_csv('targets_comparison.csv')
        print(f"Loaded comparison data: {comparison_df.shape} rows")
        
        # Check which columns are available
        has_original = 'original_x' in comparison_df.columns and 'original_y' in comparison_df.columns
        has_pred_denorm = 'pred_denorm_x' in comparison_df.columns and 'pred_denorm_y' in comparison_df.columns
        has_pred_norm = 'pred_norm_x' in comparison_df.columns and 'pred_norm_y' in comparison_df.columns
        has_target_denorm = 'target_denorm_x' in comparison_df.columns and 'target_denorm_y' in comparison_df.columns
        
        print(f"Available data: original={has_original}, pred_denorm={has_pred_denorm}, "
              f"pred_norm={has_pred_norm}, target_denorm={has_target_denorm}")
        
        # Handle missing data
        if not has_original:
            print("Warning: Original position data not found in comparison file.")
            return
            
        # Choose prediction data based on availability
        if has_pred_denorm:
            pred_x_col = 'pred_denorm_x'
            pred_y_col = 'pred_denorm_y'
        elif has_pred_norm:
            pred_x_col = 'pred_norm_x'
            pred_y_col = 'pred_norm_y'
        else:
            print("Warning: No prediction data found in comparison file.")
            pred_x_col = None
            pred_y_col = None
            
        # Fill NaN values with None for visualization purposes
        comparison_df = comparison_df.fillna(np.nan)
        
    else:
        print("target_comparison.csv not found. Looking for predictions.npy and original data...")
        
        # Try to load predictions and test_targets separately
        has_original = False
        pred_x_col = None
        pred_y_col = None
        
        if os.path.exists('predictions.npy') and os.path.exists('test_targets.pt'):
            try:
                import torch
                predictions = np.load('predictions.npy')
                y_ball_x_test, y_ball_y_test = torch.load('test_targets.pt')
                
                # Create a dataframe from these files
                comparison_df = pd.DataFrame({
                    'pred_norm_x': predictions[:, 0],
                    'pred_norm_y': predictions[:, 1],
                    'target_norm_x': y_ball_x_test,
                    'target_norm_y': y_ball_y_test
                })
                
                has_original = True
                pred_x_col = 'pred_norm_x'
                pred_y_col = 'pred_norm_y'
                comparison_df['original_x'] = y_ball_x_test  # Assuming targets are actual positions
                comparison_df['original_y'] = y_ball_y_test
                
                print(f"Created comparison data from separate files: {comparison_df.shape} rows")
                
            except Exception as e:
                print(f"Error loading prediction files: {str(e)}")
                return
        else:
            print("Required data files not found!")
            return
    
    # Check for ball_pos_norm_params.txt to denormalize if needed
    if os.path.exists('ball_pos_norm_params.txt') and not has_pred_denorm and has_pred_norm:
        try:
            params = {}
            with open('ball_pos_norm_params.txt', 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    params[key] = float(value)
            
            # Denormalize predictions
            comparison_df['pred_denorm_x'] = comparison_df[pred_x_col] * params.get('ball_x_std', 1) + params.get('ball_x_mean', 0)
            comparison_df['pred_denorm_y'] = comparison_df[pred_y_col] * params.get('ball_y_std', 1) + params.get('ball_y_mean', 0)
            
            pred_x_col = 'pred_denorm_x'
            pred_y_col = 'pred_denorm_y'
            has_pred_denorm = True
            
            print("Successfully denormalized predictions using ball_pos_norm_params.txt")
        except Exception as e:
            print(f"Error denormalizing data: {str(e)}")
    
    # At this point, we should have original positions and possibly predictions
    # Let's check the range of values to decide if we need to convert to meters
    if has_original:
        x_min, x_max = comparison_df['original_x'].min(), comparison_df['original_x'].max()
        y_min, y_max = comparison_df['original_y'].min(), comparison_df['original_y'].max()
        
        print(f"Original data range - X: {x_min:.2f} to {x_max:.2f}, Y: {y_min:.2f} to {y_max:.2f}")
        
        # If values are in centimeters (typical football pitch is ~100m), convert to meters
        scale_factor = 0.01 if x_max > 1000 or x_min < -1000 or y_max > 1000 or y_min < -1000 else 1
        
        if scale_factor != 1:
            comparison_df['original_x'] *= scale_factor
            comparison_df['original_y'] *= scale_factor
            
            if has_pred_denorm:
                comparison_df[pred_x_col] *= scale_factor
                comparison_df[pred_y_col] *= scale_factor
                
            print(f"Scaled data by factor {scale_factor} (converted to meters)")
    
    # Pitch dimensions (FIFA standard: 105m x 68m)
    pitch_length, pitch_width = 105, 68
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-pitch_length/2, pitch_length/2)
    ax.set_ylim(-pitch_width/2, pitch_width/2)
    ax.set_title("Ball Trajectory: Original vs. Predicted")
    
    # Function to draw the pitch
    def draw_pitch(ax):
        # Main field
        ax.add_patch(Rectangle((-pitch_length/2, -pitch_width/2), pitch_length, pitch_width,
                              linewidth=2, edgecolor="black", facecolor="green", alpha=0.3))
        
        # Center circle
        ax.add_patch(Circle((0, 0), 9.15, edgecolor="white", facecolor="none", linewidth=2))
        
        # Halfway line
        ax.axvline(0, color='white', linewidth=2)
        
        # Goal areas
        ax.add_patch(Rectangle((-pitch_length/2, -20.16), 16.5, 40.32, edgecolor="white", facecolor="none", linewidth=2))
        ax.add_patch(Rectangle((pitch_length/2-16.5, -20.16), 16.5, 40.32, edgecolor="white", facecolor="none", linewidth=2))
        
        # Penalty areas
        ax.add_patch(Rectangle((-pitch_length/2, -9.16), 5.5, 18.32, edgecolor="white", facecolor="none", linewidth=2))
        ax.add_patch(Rectangle((pitch_length/2-5.5, -9.16), 5.5, 18.32, edgecolor="white", facecolor="none", linewidth=2))
        
        # Center dot
        ax.add_patch(Circle((0, 0), 0.3, edgecolor="white", facecolor="white"))
        
        # Goals
        goal_width = 7.32
        ax.add_patch(Rectangle((-pitch_length/2-1, -goal_width/2), 1, goal_width, edgecolor="white", facecolor="none", linewidth=2))
        ax.add_patch(Rectangle((pitch_length/2, -goal_width/2), 1, goal_width, edgecolor="white", facecolor="none", linewidth=2))
    
    # Draw pitch
    draw_pitch(ax)
    
    # Convert dataframe to numpy arrays for faster access
    original_x = comparison_df['original_x'].to_numpy()
    original_y = comparison_df['original_y'].to_numpy()
    
    has_predictions = has_pred_denorm or has_pred_norm
    if has_predictions:
        predicted_x = comparison_df[pred_x_col].to_numpy()
        predicted_y = comparison_df[pred_y_col].to_numpy()
    else:
        predicted_x = np.zeros_like(original_x)
        predicted_y = np.zeros_like(original_y)
    
    # Initialize scatter plots for animation
    original_scatter = ax.scatter([], [], s=120, c='blue', label="Original Position", edgecolors='black', zorder=5)
    original_line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2, label="Original Path")
    
    if has_predictions:
        predicted_scatter = ax.scatter([], [], s=120, c='red', label="Predicted Position", edgecolors='black', zorder=5)
        predicted_line, = ax.plot([], [], 'r-', alpha=0.7, linewidth=2, label="Predicted Path")
    
    # Status text
    info_text = ax.text(0, pitch_width/2-3, "", fontsize=12, ha='center', va='top', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    trail_length = 20  # Number of past positions to show
    max_frames = len(comparison_df)
    
    # Animation control variables
    paused = False
    current_frame = 0
    frame_skip = 3  # Adjust based on dataset size
    
    # Function to update animation
    def update(frame):
        global current_frame
        if not paused:
            current_frame = frame
        
        # Calculate trail start (to avoid negative indexing)
        trail_start = max(0, current_frame - trail_length)
        
        # Update original ball position and trail
        original_scatter.set_offsets([original_x[current_frame], original_y[current_frame]])
        original_line.set_data(original_x[trail_start:current_frame+1], original_y[trail_start:current_frame+1])
        
        # Update prediction if available
        if has_predictions:
            # Check if current prediction is NaN
            if np.isnan(predicted_x[current_frame]) or np.isnan(predicted_y[current_frame]):
                predicted_scatter.set_offsets([np.nan, np.nan])
            else:
                predicted_scatter.set_offsets([predicted_x[current_frame], predicted_y[current_frame]])
            
            # Filter out NaN values from the trail
            valid_indices = ~np.isnan(predicted_x[trail_start:current_frame+1]) & ~np.isnan(predicted_y[trail_start:current_frame+1])
            pred_trail_x = predicted_x[trail_start:current_frame+1][valid_indices]
            pred_trail_y = predicted_y[trail_start:current_frame+1][valid_indices]
            predicted_line.set_data(pred_trail_x, pred_trail_y)
        
        # Update status text
        info_text.set_text(f"Frame: {current_frame}/{max_frames-1}")
        
        if has_predictions:
            return original_scatter, original_line, predicted_scatter, predicted_line, info_text
        else:
            return original_scatter, original_line, info_text
    
    # Play/Pause button function
    def toggle_pause(event):
        global paused
        paused = not paused
        play_pause_btn.label.set_text("Play" if paused else "Pause")
        play_pause_btn.canvas.draw_idle()
    
    # Step functions
    def step_forward(event):
        global current_frame
        current_frame = min(current_frame + 1, max_frames - 1)
        update(current_frame)
        fig.canvas.draw_idle()
    
    def step_backward(event):
        global current_frame
        current_frame = max(current_frame - 1, 0)
        update(current_frame)
        fig.canvas.draw_idle()
    
    def jump_forward(event):
        global current_frame
        current_frame = min(current_frame + 10, max_frames - 1)
        update(current_frame)
        fig.canvas.draw_idle()
    
    def jump_backward(event):
        global current_frame
        current_frame = max(current_frame - 10, 0)
        update(current_frame)
        fig.canvas.draw_idle()
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, max_frames, frame_skip),
        interval=50, blit=True, repeat=True
    )
    
    # Add control buttons
    ax_play_pause = plt.axes([0.45, 0.01, 0.1, 0.04])
    ax_step_back = plt.axes([0.3, 0.01, 0.1, 0.04])
    ax_step_fwd = plt.axes([0.6, 0.01, 0.1, 0.04])
    ax_jump_back = plt.axes([0.15, 0.01, 0.1, 0.04])
    ax_jump_fwd = plt.axes([0.75, 0.01, 0.1, 0.04])
    
    play_pause_btn = Button(ax_play_pause, "Pause")
    step_back_btn = Button(ax_step_back, "← Step")
    step_fwd_btn = Button(ax_step_fwd, "Step →")
    jump_back_btn = Button(ax_jump_back, "← Jump")
    jump_fwd_btn = Button(ax_jump_fwd, "Jump →")
    
    play_pause_btn.on_clicked(toggle_pause)
    step_back_btn.on_clicked(step_backward)
    step_fwd_btn.on_clicked(step_forward)
    jump_back_btn.on_clicked(jump_backward)
    jump_fwd_btn.on_clicked(jump_forward)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Save animation (MP4)
    try:
        save_path = "ball_trajectory_animation.mp4"
        # Lower framerate for smooth playback
        writer = animation.FFMpegWriter(fps=10, metadata={"title": "Ball Trajectory Animation"})
        ani.save(save_path, writer=writer)
        print(f"Animation saved as {save_path}")
    except Exception as e:
        print(f"Error saving animation: {str(e)}")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    return ani

if __name__ == "__main__":
    create_ball_trajectory_animation()