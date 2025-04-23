import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import torch as th
from env_g import TrainYardEnv

# Define proper colors for each train type
TRAIN_COLORS = {
    'red': '#e74c3c',      # Bright red
    'green': '#2ecc71',    # Bright green
    'blue': '#3498db',     # Bright blue
    'yellow': '#f1c40f',   # Bright yellow
    'black': '#34495e',    # Dark blue/black
    'purple': '#9b59b6',   # Purple
    'brown': '#a65628',    # Brown
    'pink': '#e84393',     # Pink
    'orange': '#e67e22',   # Orange
    'gray': '#95a5a6'      # Gray (for unknown)
}

# Create directories for logs and models
log_dir = "train_yard_logs_g/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir + "/models", exist_ok=True)
os.makedirs(log_dir + "/tensorboard", exist_ok=True)

# Create a gymnasium-compatible wrapper for our environment
class TrainYardGymWrapper(gym.Env):
    """
    Wrapper to make TrainYardEnv compatible with gymnasium (used by Stable-Baselines3).
    """
    def __init__(self, verbose=False):
        super(TrainYardGymWrapper, self).__init__()
        
        # Create the original environment
        self.env = TrainYardEnv(verbose=verbose)
        
        # Define action and observation spaces compatible with gymnasium
        entry_exit_tracks = list(self.env.entry_exit_tracks.keys())
        
        # Define action space as MultiDiscrete
        self.action_space = spaces.MultiDiscrete([
            len(entry_exit_tracks),  # entry_track index
            len(entry_exit_tracks),  # exit_track index
            60                       # wait_time (0-59)
        ])
        
        # Use the same observation space dimensions as the original environment
        obs_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.float32
        )
        
        # Initialize other attributes
        self.current_time = self.env.current_time
        self.track_status = self.env.track_status
        self.entry_exit_tracks = self.env.entry_exit_tracks
        self.parking_tracks = self.env.parking_tracks
        self.loading_tracks = self.env.loading_tracks
        self.timetable = self.env.timetable
        self.next_arrival_train = self.env.next_arrival_train
        self.train_status = self.env.train_status
        self.train_locations = self.env.train_locations
        self.completed_trains = self.env.completed_trains if hasattr(self.env, 'completed_trains') else 0
        self.delayed_departures = self.env.delayed_departures if hasattr(self.env, 'delayed_departures') else 0
        self.total_delay_minutes = self.env.total_delay_minutes if hasattr(self.env, 'total_delay_minutes') else 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        # Handle seed if provided (gymnasium API requirement)
        if seed is not None:
            np.random.seed(seed)
            
        # Reset the original environment
        obs = self.env.reset()
        
        # Update synchronized attributes
        self._sync_attributes()
        
        # In gymnasium, reset returns (observation, info)
        return obs, {}
    
    def step(self, action):
        """Take action in environment."""
        # Convert MultiDiscrete action to Dict action
        dict_action = {
            'entry_track': int(action[0]),
            'exit_track': int(action[1]),
            'wait_time': int(action[2])
        }
        
        # Call original environment step method
        obs, reward, done, info = self.env.step(dict_action)
        
        # Update synchronized attributes
        self._sync_attributes()
        
        # In gymnasium, step returns (observation, reward, terminated, truncated, info)
        # We'll use 'done' for both terminated and truncated for simplicity
        return obs, reward, done, False, info
    
    def _sync_attributes(self):
        """Synchronize key attributes from the wrapped environment."""
        self.current_time = self.env.current_time
        self.track_status = self.env.track_status
        self.next_arrival_train = self.env.next_arrival_train
        self.train_status = self.env.train_status
        self.train_locations = self.env.train_locations
        self.completed_trains = self.env.completed_trains if hasattr(self.env, 'completed_trains') else 0
        self.delayed_departures = self.env.delayed_departures if hasattr(self.env, 'delayed_departures') else 0
        self.total_delay_minutes = self.env.total_delay_minutes if hasattr(self.env, 'total_delay_minutes') else 0
    
    def _minutes_to_time_str(self, minutes):
        """Delegate to original env method."""
        return self.env._minutes_to_time_str(minutes)
    
    def _all_tracks(self):
        """Delegate to original env method if available or implement it."""
        if hasattr(self.env, '_all_tracks'):
            return self.env._all_tracks()
        return list(self.entry_exit_tracks) + list(self.parking_tracks) + list(self.loading_tracks)
    
    def close(self):
        """Close environment resources."""
        if hasattr(self.env, 'close'):
            self.env.close()

class TrainYardVisualizer:
    """Class to visualize the train yard environment using the user's preferred visualization."""
    
    def __init__(self, env, figsize=(14, 8)):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.track_occupancy = {}
        self.current_occupancy = {}
        self.occupancy_start = {}
        
        # Initialize track occupancy data
        for track in list(env.entry_exit_tracks) + list(env.loading_tracks) + list(env.parking_tracks):
            self.track_occupancy[track] = []
            self.current_occupancy[track] = None
            self.occupancy_start[track] = 0
        
        # Record initial occupancy
        for track, status in env.track_status.items():
            if status['train']:
                self.current_occupancy[track] = status['train']
                self.occupancy_start[track] = env.current_time
        
        # Create log file for track assignments
        self.log_file = open("agent_track_assignments.log", "w")
        self.log_file.write("AGENT TRAIN YARD TRACK ASSIGNMENTS LOG\n")
        self.log_file.write("===================================\n\n")
        
        # Log initial state
        for track, train in self.current_occupancy.items():
            if train:
                self.log_file.write(f"Initial state: {train} on track {track}\n")
    
    def update(self, prev_time=None):
        """Update track occupancy data and visualization."""
        # If prev_time not provided, use current time
        if prev_time is None:
            prev_time = self.env.current_time
        
        # Log current time
        self.log_file.write(f"\nTime: {self.env._minutes_to_time_str(self.env.current_time)}\n")
        
        # Check for track status changes
        for track, status in self.env.track_status.items():
            # If track state changed
            if status['train'] != self.current_occupancy[track]:
                # If previously occupied, record the period
                if self.current_occupancy[track]:
                    self.track_occupancy[track].append({
                        'train': self.current_occupancy[track],
                        'start': self.occupancy_start[track],
                        'end': prev_time
                    })
                    
                    # Log track departure
                    self.log_file.write(f"  {self.current_occupancy[track]} left track {track} at {self.env._minutes_to_time_str(prev_time)}\n")
                
                # Update current state
                self.current_occupancy[track] = status['train']
                if status['train']:
                    self.occupancy_start[track] = prev_time
                    # Log track arrival
                    self.log_file.write(f"  {status['train']} entered track {track} at {self.env._minutes_to_time_str(prev_time)}\n")
        
        # Create visualization
        self._create_visualization()
    
    def finalize(self):
        """Record final state and clean up."""
        # Record final occupancy
        for track, train in self.current_occupancy.items():
            if train:
                self.track_occupancy[track].append({
                    'train': train,
                    'start': self.occupancy_start[track],
                    'end': self.env.current_time
                })
        
        # Close log file
        self.log_file.write(f"\nEpisode complete. Final time: {self.env._minutes_to_time_str(self.env.current_time)}\n")
        self.log_file.close()
        
        # Filter out unreasonably long occupancy periods
        max_duration = 48 * 60  # 48 hours in minutes
        for track in self.track_occupancy:
            self.track_occupancy[track] = [
                period for period in self.track_occupancy[track]
                if period['end'] - period['start'] <= max_duration
            ]
        
        # Create final visualization
        self._create_visualization()
    
    def _create_visualization(self):
        """Create the visualization using the user's preferred approach."""
        # Clear the axis
        self.ax.clear()
        
        # Group tracks by type for better visualization
        entry_exit_tracks = sorted(list(self.env.entry_exit_tracks))
        loading_tracks = sorted(list(self.env.loading_tracks))
        parking_tracks = sorted(list(self.env.parking_tracks))
        
        # Calculate vertical positions with gaps between groups
        track_positions = {}
        current_pos = 0
        
        # Add entry/exit tracks
        for track in entry_exit_tracks:
            track_positions[track] = current_pos
            current_pos += 1
        
        # Add a gap
        current_pos += 0.5
        
        # Add loading tracks
        for track in loading_tracks:
            track_positions[track] = current_pos
            current_pos += 1
        
        # Add a gap
        current_pos += 0.5
        
        # Add parking tracks
        for track in parking_tracks:
            track_positions[track] = current_pos
            current_pos += 1
        
        # Find time range for x-axis
        all_periods = [period for track_periods in self.track_occupancy.values() for period in track_periods]
        if all_periods:
            min_time = min(period['start'] for period in all_periods)
            max_time = max(period['end'] for period in all_periods)
        else:
            min_time = 0
            max_time = 24 * 60  # 24 hours
        
        # Create datetime objects for x-axis
        base_date = datetime(2025, 1, 1)
        
        # Add background colors for track groups
        # Entry/Exit tracks background
        if entry_exit_tracks:
            entry_min = track_positions[entry_exit_tracks[0]] - 0.4
            entry_max = track_positions[entry_exit_tracks[-1]] + 0.4
            rect = patches.Rectangle(
                (base_date + timedelta(minutes=min_time), entry_min),
                timedelta(minutes=max_time - min_time + 60),
                entry_max - entry_min,
                linewidth=0,
                facecolor='#f8f9fa',
                zorder=0
            )
            self.ax.add_patch(rect)
            self.ax.text(
                base_date + timedelta(minutes=min_time - 10),
                (entry_min + entry_max) / 2,
                'Entry/Exit',
                va='center',
                ha='right',
                fontsize=10,
                fontweight='bold'
            )
        
        # Loading tracks background
        if loading_tracks:
            loading_min = track_positions[loading_tracks[0]] - 0.4
            loading_max = track_positions[loading_tracks[-1]] + 0.4
            rect = patches.Rectangle(
                (base_date + timedelta(minutes=min_time), loading_min),
                timedelta(minutes=max_time - min_time + 60),
                loading_max - loading_min,
                linewidth=0,
                facecolor='#e9ecef',
                zorder=0
            )
            self.ax.add_patch(rect)
            self.ax.text(
                base_date + timedelta(minutes=min_time - 10),
                (loading_min + loading_max) / 2,
                'Loading',
                va='center',
                ha='right',
                fontsize=10,
                fontweight='bold'
            )
        
        # Parking tracks background
        if parking_tracks:
            parking_min = track_positions[parking_tracks[0]] - 0.4
            parking_max = track_positions[parking_tracks[-1]] + 0.4
            rect = patches.Rectangle(
                (base_date + timedelta(minutes=min_time), parking_min),
                timedelta(minutes=max_time - min_time + 60),
                parking_max - parking_min,
                linewidth=0,
                facecolor='#f8f9fa',
                zorder=0
            )
            self.ax.add_patch(rect)
            self.ax.text(
                base_date + timedelta(minutes=min_time - 10),
                (parking_min + parking_max) / 2,
                'Parking',
                va='center',
                ha='right',
                fontsize=10,
                fontweight='bold'
            )
        
        # Store used train colors for legend
        used_train_colors = {}
        
        # Plot occupancy periods
        for track, periods in self.track_occupancy.items():
            if track in track_positions:  # Only plot tracks in our layout
                for period in periods:
                    train = period['train']
                    base_train = train.split('a')[0].split('b')[0]
                    
                    # Get train type from train ID for color assignment
                    train_type = 'gray'  # Default color
                    for prefix in TRAIN_COLORS.keys():
                        if base_train.startswith(prefix):
                            train_type = prefix
                            break
                    
                    # Get color and store for legend
                    color = TRAIN_COLORS[train_type]
                    used_train_colors[base_train] = color
                    
                    # Convert times to datetime for plotting
                    start_dt = base_date + timedelta(minutes=period['start'])
                    end_dt = base_date + timedelta(minutes=period['end'])
                    
                    rect = patches.Rectangle(
                        (start_dt, track_positions[track] - 0.4),
                        end_dt - start_dt,
                        0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=color,
                        alpha=0.7,
                        zorder=1
                    )
                    self.ax.add_patch(rect)
                    
                    # Only add labels to sufficiently wide boxes
                    if period['end'] - period['start'] > 30:  # Only label if period > 30 minutes
                        midpoint = start_dt + (end_dt - start_dt) / 2
                        # Choose text color based on background brightness
                        text_color = 'white' if train_type in ['black', 'blue', 'purple', 'brown'] else 'black'
                        self.ax.text(midpoint, track_positions[track], train, 
                               ha='center', va='center', fontsize=7, color=text_color,
                               zorder=2)
        
        # Format axes
        self.ax.set_title('Train Yard Schedule (Agent)', fontsize=14)
        self.ax.set_xlabel('Time', fontsize=12)
        self.ax.set_ylabel('Tracks', fontsize=12)
        
        # Set time limits
        self.ax.set_xlim(
            base_date + timedelta(minutes=min_time - 30),
            base_date + timedelta(minutes=max_time + 60)
        )
        
        # Set y-ticks to track names
        y_ticks = []
        y_labels = []
        for track, pos in track_positions.items():
            y_ticks.append(pos)
            
            # For entry/exit tracks, add length information
            if track in self.env.entry_exit_tracks:
                track_length = self.env.entry_exit_tracks[track]
                y_labels.append(f"{track} ({track_length})")
            # For loading tracks, add length information
            elif track in self.env.loading_tracks:
                track_length = self.env.loading_tracks[track]
                y_labels.append(f"{track} ({track_length})")
            # For parking tracks, add length information
            elif track in self.env.parking_tracks:
                track_length = self.env.parking_tracks[track]
                y_labels.append(f"{track} ({track_length})")
            else:
                y_labels.append(track)
        
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(y_labels, fontsize=8)
        
        # Format x-axis as time
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add horizontal grid lines
        for pos in y_ticks:
            self.ax.axhline(y=pos, color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
        
        # Add vertical grid lines
        self.ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
        # Add legend for train types
        handles = []
        labels = []
        for train_id, color in sorted(used_train_colors.items()):
            handles.append(patches.Patch(color=color, label=train_id))
            labels.append(train_id)
        
        self.ax.legend(handles, labels, loc='upper right', fontsize=8)
        
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
        self.fig.canvas.draw()
    
    def save(self, filename='agent_train_schedule.png'):
        """Save the current visualization to a file."""
        self.fig.savefig(filename, dpi=200)
        print(f"Visualization saved to {filename}")

def make_env(rank, seed=0):
    """
    Create a function that returns a wrapped environment
    """
    def _init():
        # Use our gymnasium-compatible wrapper
        env = TrainYardGymWrapper(verbose=False)
        # In newer gymnasium versions, seeding happens in reset(), not with seed()
        env = Monitor(env, log_dir + f"/monitor_{rank}")
        return env
    set_random_seed(seed)
    return _init

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train_agent(total_timesteps=100000, seed=0):
    """
    Train a PPO agent on the train yard environment
    """
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # Create vectorized environment with proper seeding
    env = DummyVecEnv([make_env(i, seed+i) for i in range(1)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Define callbacks for evaluation and checkpoints
    eval_env = DummyVecEnv([make_env(0, seed+1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "/models/best",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir + "/models/checkpoints/",
        name_prefix="train_yard_model",
        save_replay_buffer=True,
        verbose=1
    )
    
    # Set up learning rate schedule
    lr_schedule = linear_schedule(3e-4)
    
    # Define policy network architecture
    policy_kwargs = {
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
        'activation_fn': th.nn.ReLU
    }
    
    # Create PPO model with sophisticated configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=2048,         # Number of steps to collect before updating
        batch_size=64,        # Minibatch size for updates
        n_epochs=10,          # Number of epochs for each update
        gamma=0.99,           # Discount factor
        gae_lambda=0.95,      # GAE parameter
        clip_range=0.2,       # PPO clip parameter
        ent_coef=0.01,        # Entropy coefficient to encourage exploration
        vf_coef=0.5,          # Value function coefficient
        max_grad_norm=0.5,    # Max gradient norm for gradient clipping
        tensorboard_log=log_dir + "/tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed
    )
    
    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10
    )
    total_time = time.time() - start_time
    
    # Save the final model and environment normalization
    model.save(log_dir + "/models/final_ppo_model")
    env.save(log_dir + "/models/final_vec_normalize.pkl")
    
    print(f"Training completed in {total_time/3600:.2f} hours")
    
    # Evaluate the final model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, env

def visualize_agent(model_path, env_path, steps_per_update=5):
    """
    Visualize a trained agent using our visualization tools
    """
    print(f"Visualizing agent with model: {model_path}")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create and wrap the environment
    env = TrainYardGymWrapper(verbose=True)
    
    # Load the normalization statistics
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(env_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # Run one episode with the trained agent
    reset_result = vec_env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    done = False
    
    # Initialize visualizer
    visualizer = TrainYardVisualizer(env)
    plt.ion()
    
    step_count = 0
    total_reward = 0
    prev_time = env.current_time
    
    print("Starting episode...")
    while not done:
        # Get action from the trained policy
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action - handle different gym API versions
        step_result = vec_env.step(action)
        
        # Check if we have 4 values (old gym API) or 5 values (new gymnasium API)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
            reward = rewards[0]
            done = terminated[0] or truncated[0]
        else:
            obs, rewards, dones, info = step_result
            reward = rewards[0]
            done = dones[0]
        
        total_reward += reward
        step_count += 1
        
        # Print action details - fix numpy array handling
        if env.next_arrival_train:
            track_list = list(env.entry_exit_tracks.keys())
            dict_action = {
                            'entry_track': action.item(0), 
                            'exit_track': action.item(1),
                            'wait_time': action.item(2)
                        }
            print(f"\nARRIVING TRAIN: {env.next_arrival_train}")
            print(f"ACTION: Entry track: {track_list[dict_action['entry_track']]}, "
                  f"Exit track: {track_list[dict_action['exit_track']]}, "
                  f"Wait time: {dict_action['wait_time']} minutes")
        
        # Update visualization periodically
        if step_count % steps_per_update == 0:
            visualizer.update(prev_time)
            plt.pause(0.1)
            print(f"Step {step_count} - Time: {env._minutes_to_time_str(env.current_time)}, " 
                  f"Reward so far: {total_reward:.2f}")
        
        prev_time = env.current_time
    
    # Final update and save visualization
    visualizer.finalize()
    visualizer.save("trained_agent_schedule_g.png")
    
    print(f"\nEpisode complete!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final time: {env._minutes_to_time_str(env.current_time)}")
    print(f"Trains completed: {env.completed_trains}/{len(env.timetable)}")
    print(f"Delayed departures: {env.delayed_departures}")
    print(f"Total delay minutes: {env.total_delay_minutes}")
    
    plt.ioff()
    plt.show()

def create_training_plan():
    """Create and print a training plan with hyperparameter tuning"""
    print("\n===== TRAIN YARD AGENT TRAINING PLAN =====\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Yard RL Agent Training')
    parser.add_argument('--mode', type=str, default='plan', 
                        choices=['plan', 'train', 'visualize'],
                        help='Mode to run (plan, train, visualize)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of timesteps for training')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for visualization')
    parser.add_argument('--env', type=str, default=None,
                       help='Path to environment normalization for visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'plan' or args.mode == 'train':
        # Always show the training plan first
        create_training_plan()
    
    if args.mode == 'train':
        # Start training
        model, env = train_agent(total_timesteps=args.timesteps)
        print("\nTraining complete. To visualize learning progress, run:")
        print(f"tensorboard --logdir={log_dir}/tensorboard")
        
    elif args.mode == 'visualize':
        # Use default paths if not specified
        if args.model is None:
            args.model = log_dir + "/models/best/best_model.zip"
        if args.env is None:
            args.env = log_dir + "/models/final_vec_normalize.pkl"
            
        # Visualize the trained agent
        visualize_agent(args.model, args.env)