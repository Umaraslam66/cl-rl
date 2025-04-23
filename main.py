import numpy as np
import matplotlib.pyplot as plt
from env_g import TrainYardEnv
import time

def test_environment():
    """
    Simple test function to verify that the TrainYardEnv is working as expected
    by running a single train through the yard.
    """
    print("Initializing train yard environment...")
    env = TrainYardEnv(verbose=True)  # Enable verbose logging
    
    # Reset the environment to start fresh
    print("Resetting environment...")
    observation = env.reset()
    
    # Set up visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.ion()  # Enable interactive mode
    
    # Keep track of events for later analysis
    track_assignments = {}
    train_movements = []
    
    # Run the environment for a limited number of steps
    # We'll focus on just one train to keep it simple
    max_steps = 100
    target_train = "red1"  # We'll track this specific train
    
    print(f"Starting simulation, tracking train '{target_train}'...")
    
    for step in range(max_steps):
        # If we have a next arrival train, assign it to tracks
        if env.next_arrival_train:
            train_id = env.next_arrival_train
            print(f"\nStep {step}: Assigning tracks for arriving train: {train_id}")
            
            # Get available entry and exit tracks
            entry_exit_tracks = list(env.entry_exit_tracks.keys())
            
            # Find the train in the timetable to get its info
            train_info = next((t for t in env.timetable if t['train'] == train_id), None)
            
            if train_info:
                # Find suitable tracks based on train length
                train_length = train_info['length']
                suitable_entries = [t for t in entry_exit_tracks 
                                   if env.entry_exit_tracks[t] >= train_length]
                suitable_exits = [t for t in entry_exit_tracks 
                                 if env.entry_exit_tracks[t] >= train_length]
                
                if suitable_entries and suitable_exits:
                    # Choose first suitable track for simplicity
                    entry_track = suitable_entries[0]
                    exit_track = suitable_exits[-1]  # Choose a different one
                    wait_time = 5  # Arbitrary wait time in minutes
                    
                    # Record the assignment
                    track_assignments[train_id] = {
                        'entry': entry_track,
                        'exit': exit_track,
                        'wait': wait_time
                    }
                    
                    # Convert to action indices
                    entry_idx = entry_exit_tracks.index(entry_track)
                    exit_idx = entry_exit_tracks.index(exit_track)
                    
                    action = {
                        'entry_track': entry_idx,
                        'exit_track': exit_idx,
                        'wait_time': wait_time
                    }
                    
                    print(f"Assigned tracks for {train_id}:")
                    print(f"  Entry track: {entry_track}")
                    print(f"  Exit track: {exit_track}")
                    print(f"  Wait time: {wait_time} minutes")
                else:
                    print(f"No suitable tracks found for train {train_id} (length: {train_length})")
                    # Use default action
                    action = {
                        'entry_track': 0,
                        'exit_track': 0,
                        'wait_time': 0
                    }
            else:
                print(f"Warning: Could not find train {train_id} in timetable")
                # Use default action
                action = {
                    'entry_track': 0,
                    'exit_track': 0,
                    'wait_time': 0
                }
        else:
            # No train to assign, use a default action
            action = {
                'entry_track': 0,
                'exit_track': 0,
                'wait_time': 0
            }
        
        # Take a step in the environment
        observation, reward, done, info = env.step(action)
        
        # Track train states
        for train_id, state in env.train_status.items():
            if train_id == target_train or train_id.startswith(f"{target_train}"):
                location = env.train_locations.get(train_id, "unknown")
                train_movements.append({
                    'step': step,
                    'train': train_id,
                    'state': state,
                    'location': location,
                    'time': env._minutes_to_time_str(env.current_time)
                })
                print(f"Train {train_id} is in state '{state}' at {location} ({env._minutes_to_time_str(env.current_time)})")
        
        # Visualize current environment state
        ax.clear()
        
        # Plot tracks with occupancy
        track_y = {}
        y_pos = 0
        
        # Group tracks by type
        entry_exit_tracks = sorted(list(env.entry_exit_tracks.keys()))
        loading_tracks = sorted(list(env.loading_tracks.keys()))
        parking_tracks = sorted(list(env.parking_tracks.keys()))
        
        # Add entry/exit tracks
        for track in entry_exit_tracks:
            track_y[track] = y_pos
            y_pos += 1
        
        # Add loading tracks
        y_pos += 0.5  # Gap
        for track in loading_tracks:
            track_y[track] = y_pos
            y_pos += 1
        
        # Add parking tracks
        y_pos += 0.5  # Gap
        for track in parking_tracks:
            track_y[track] = y_pos
            y_pos += 1
        
        # Plot tracks
        for track, y in track_y.items():
            # Draw track
            ax.plot([0, 10], [y, y], 'k-', linewidth=2)
            
            # Check if track is occupied
            status = env.track_status[track]
            if status['train']:
                # Draw train on track
                ax.plot([5], [y], 'ro', markersize=10)
                ax.text(5.2, y, status['train'], fontsize=8, va='center')
        
        # Set labels
        ax.set_title(f"Train Yard State - Step {step}, Time: {env._minutes_to_time_str(env.current_time)}")
        ax.set_yticks(list(track_y.values()))
        ax.set_yticklabels(list(track_y.keys()))
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, y_pos + 1)
        ax.set_xticks([])
        
        # Add info text
        info_text = f"Completed Trains: {env.completed_trains}\n"
        info_text += f"Delayed Departures: {env.delayed_departures}\n"
        info_text += f"Total Delay Minutes: {env.total_delay_minutes}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        if done:
            print(f"\nSimulation completed after {step+1} steps!")
            break
    
    # Print final stats
    print("\n===== Simulation Complete =====")
    print(f"Final Time: {env._minutes_to_time_str(env.current_time)}")
    print(f"Completed Trains: {env.completed_trains}/{len(env.timetable)}")
    print(f"Delayed Departures: {env.delayed_departures}")
    print(f"Total Delay Minutes: {env.total_delay_minutes}")
    
    # Print the movement history for our target train
    print(f"\n===== Movement History for {target_train} =====")
    for movement in train_movements:
        print(f"{movement['time']}: {movement['train']} - {movement['state']} at {movement['location']}")
    
    # Close the plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_environment()