# # import numpy as np
# from env_c import TrainYardEnv

# def simple_policy(env, obs):
#     """A simple policy that tries to minimize track usage."""
#     # Find least used tracks for entry and exit
#     entry_usage = {}
#     exit_usage = {}
    
#     for track in env.entry_exit_tracks:
#         entry_usage[track] = env.track_status[track]['occupied_until']
#         exit_usage[track] = env.track_status[track]['occupied_until']
    
#     # Sort by usage (least used first)
#     entry_tracks = sorted(entry_usage.keys(), key=lambda x: entry_usage[x])
#     exit_tracks = sorted(exit_usage.keys(), key=lambda x: exit_usage[x])
    
#     # Get track indices
#     track_list = list(env.entry_exit_tracks.keys())
#     entry_idx = track_list.index(entry_tracks[0])
#     exit_idx = track_list.index(exit_tracks[0])
    
#     # Check if entry track has enough length for the arriving train
#     if env.next_arrival_train:
#         train_info = next(t for t in env.timetable if t['train'] == env.next_arrival_train)
#         train_length = train_info['length']
        
#         # Find first suitable track with enough length
#         for track in entry_tracks:
#             if env.entry_exit_tracks[track] >= train_length:
#                 entry_idx = track_list.index(track)
#                 break
    
#     return {
#         'entry_track': entry_idx,
#         'exit_track': exit_idx,
#         'wait_time': 0  # No wait time for simplicity
#     }

# def print_track_status(env, current_time):
#     """Print the current status of all tracks."""
#     print("\n=== TRACK STATUS AT", env._minutes_to_time_str(current_time), "===")
    
#     # Group tracks by type
#     track_types = {
#         "Entry/Exit Tracks": env.entry_exit_tracks.keys(),
#         "Parking Tracks": env.parking_tracks.keys(),
#         "Loading Tracks": env.loading_tracks.keys()
#     }
    
#     for track_type, tracks in track_types.items():
#         print(f"\n{track_type}:")
#         for track in sorted(tracks):
#             status = env.track_status[track]
#             if status['occupied_until'] > current_time:
#                 train = status['train'] or "Unknown"
#                 free_time = env._minutes_to_time_str(status['occupied_until'])
#                 print(f"  {track} (length: {env.entry_exit_tracks.get(track) or env.parking_tracks.get(track) or env.loading_tracks.get(track)}): Occupied by {train} until {free_time}")
#             else:
#                 print(f"  {track} (length: {env.entry_exit_tracks.get(track) or env.parking_tracks.get(track) or env.loading_tracks.get(track)}): Free")

# def log_train_event(events, train_id, time, action, location=None):
#     """Log a train event for later tracking."""
#     if train_id not in events:
#         events[train_id] = []
#     events[train_id].append({
#         'time': time,
#         'action': action,
#         'location': location
#     })

# def print_train_history(env, events, train_id):
#     """Print the complete history of a train."""
#     if train_id not in events:
#         print(f"No events logged for train {train_id}")
#         return
    
#     print(f"\n=== HISTORY OF TRAIN {train_id} ===")
#     for event in events[train_id]:
#         time_str = env._minutes_to_time_str(event['time'])
#         if event['location']:
#             print(f"{time_str}: {event['action']} at {event['location']}")
#         else:
#             print(f"{time_str}: {event['action']}")

# def test_environment_with_tracking():
#     # Initialize environment
#     env = TrainYardEnv()
#     obs = env.reset()
    
#     # Set trains to track (select a few trains of different types)
#     tracked_trains = ['red1', 'green1', 'yellow1', 'black1']
    
#     # Dictionary to store train events
#     train_events = {}
    
#     # Dictionary to store previous train statuses for change detection
#     prev_statuses = {}
    
#     done = False
#     total_reward = 0
#     step_count = 0
#     max_steps = 200  # Reduced for clarity in output
#     last_time = 0
    
#     # Run episode with policy actions
#     while not done and step_count < max_steps:
#         # Use policy to get action
#         action = simple_policy(env, obs)
        
#         # Take step
#         prev_time = env.current_time
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#         step_count += 1
        
#         # Log events for arrival/entry/departure
#         if env.next_arrival_train and env.next_arrival_train in tracked_trains:
#             log_train_event(train_events, env.next_arrival_train, env.current_time, "Arrived")
        
#         # Check for status changes in tracked trains
#         for train_id in list(env.train_status.keys()):
#             base_id = train_id.split('a')[0].split('b')[0]  # Get base train ID without a/b suffix
            
#             if base_id in tracked_trains:
#                 current_status = env.train_status[train_id]
#                 prev_status = prev_statuses.get(train_id)
                
#                 if prev_status != current_status:
#                     # Status has changed, log the event
#                     location = env.train_locations.get(train_id, "unknown")
#                     log_train_event(train_events, train_id, env.current_time, 
#                                    f"Status changed from {prev_status or 'new'} to {current_status}", location)
#                     prev_statuses[train_id] = current_status
        
#         # Print status updates
#         if step_count % 20 == 0 or done:
#             print(f"\nStep {step_count} - Time: {info['time_str']}, " 
#                   f"Tracks used: {info['tracks_used']}, "
#                   f"Trains completed: {info['trains_completed']}/{len(env.timetable)}")
        
#         # Print track status every hour of simulation time
#         current_hour = env.current_time // 60
#         last_hour = last_time // 60
#         if current_hour > last_hour:
#             print_track_status(env, env.current_time)
#         last_time = env.current_time
    
#     if step_count >= max_steps:
#         print("\nWARNING: Maximum steps reached, simulation may not have completed")
    
#     print(f"\nEpisode complete. Total reward: {total_reward}")
#     print(f"Total steps taken: {step_count}")
#     print(f"Final time: {info['time_str']}")
    
#     # Print histories for tracked trains
#     for train_id in tracked_trains:
#         print_train_history(env, train_events, train_id)
#         # Also print histories for split trains
#         print_train_history(env, train_events, f"{train_id}a")
#         print_train_history(env, train_events, f"{train_id}b")
    
#     # Show final track status
#     print_track_status(env, env.current_time)
    
#     # Display final state of all trains
#     print("\nFinal train status:")
#     for train, status in sorted(env.train_status.items()):
#         print(f"Train {train}: {status}")

# if __name__ == "__main__":
#     test_environment_with_tracking()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from env_c import TrainYardEnv

def simple_policy(env, obs):
    """A simple policy that tries to minimize track usage."""
    track_list = list(env.entry_exit_tracks.keys())
    entry_tracks = sorted(env.entry_exit_tracks.keys(), 
                         key=lambda x: env.track_status[x]['occupied_until'])
    exit_tracks = sorted(env.entry_exit_tracks.keys(),
                        key=lambda x: env.track_status[x]['occupied_until'])
    
    entry_idx = track_list.index(entry_tracks[0])
    exit_idx = track_list.index(exit_tracks[0])
    
    # Select appropriate entry track based on train length
    if env.next_arrival_train:
        train_info = next(t for t in env.timetable if t['train'] == env.next_arrival_train)
        for track in entry_tracks:
            if env.entry_exit_tracks[track] >= train_info['length']:
                entry_idx = track_list.index(track)
                break
    
    return {
        'entry_track': entry_idx,
        'exit_track': exit_idx,
        'wait_time': 0
    }

def test_with_visualization():
    env = TrainYardEnv()
    obs = env.reset()
    
    # Keep track of occupancy directly
    track_occupancy = {}
    for track in list(env.entry_exit_tracks) + list(env.parking_tracks) + list(env.loading_tracks):
        track_occupancy[track] = []
    
    # Track current occupancy
    current_occupancy = {t: None for t in track_occupancy.keys()}
    occupancy_start = {t: 0 for t in track_occupancy.keys()}
    
    done = False
    step_count = 0
    max_steps = 200
    
    # Record initial state
    for track, status in env.track_status.items():
        if status['train']:
            current_occupancy[track] = status['train']
            occupancy_start[track] = env.current_time
    
    while not done and step_count < max_steps:
        prev_time = env.current_time
        
        # Take step
        action = simple_policy(env, obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Check for track status changes
        for track, status in env.track_status.items():
            # If track state changed
            if status['train'] != current_occupancy[track]:
                # If previously occupied, record the period
                if current_occupancy[track]:
                    track_occupancy[track].append({
                        'train': current_occupancy[track],
                        'start': occupancy_start[track],
                        'end': prev_time
                    })
                
                # Update current state
                current_occupancy[track] = status['train']
                if status['train']:
                    occupancy_start[track] = prev_time
        
        if step_count % 50 == 0:
            print(f"Step {step_count} - Time: {info['time_str']}")
    
    # Record final state
    for track, train in current_occupancy.items():
        if train:
            track_occupancy[track].append({
                'train': train,
                'start': occupancy_start[track],
                'end': env.current_time
            })
    
    print(f"Episode complete. Steps: {step_count}")
    
    # Filter out any unreasonably long occupancy periods
    max_duration = 48 * 60  # 48 hours in minutes
    for track in track_occupancy:
        track_occupancy[track] = [
            period for period in track_occupancy[track]
            if period['end'] - period['start'] <= max_duration
        ]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up colors for trains
    train_colors = {}
    train_prefixes = ['red', 'green', 'blue', 'yellow', 'black', 'purple', 'brown', 'pink']
    
    # Create track positions for y-axis
    tracks = sorted(list(env.entry_exit_tracks)) + sorted(list(env.parking_tracks)) + sorted(list(env.loading_tracks))
    track_positions = {track: i for i, track in enumerate(tracks)}
    
    # Find time range for x-axis
    all_periods = [period for track_periods in track_occupancy.values() for period in track_periods]
    if all_periods:
        min_time = min(period['start'] for period in all_periods)
        max_time = max(period['end'] for period in all_periods)
    else:
        min_time = 0
        max_time = 24 * 60  # 24 hours
    
    # Create datetime objects for x-axis
    base_date = datetime(2025, 1, 1)
    
    # Plot occupancy periods
    for track, periods in track_occupancy.items():
        for period in periods:
            train = period['train']
            base_train = train.split('a')[0].split('b')[0]
            
            # Set consistent colors by train type
            if base_train not in train_colors:
                for i, prefix in enumerate(train_prefixes):
                    if base_train.startswith(prefix):
                        hue = i / len(train_prefixes)
                        train_colors[base_train] = hsv_to_rgb((hue, 0.8, 0.9))
                        break
                if base_train not in train_colors:
                    hue = hash(base_train) % 1000 / 1000
                    train_colors[base_train] = hsv_to_rgb((hue, 0.8, 0.9))
            
            # Convert times to datetime for plotting
            start_dt = base_date + timedelta(minutes=period['start'])
            end_dt = base_date + timedelta(minutes=period['end'])
            
            rect = patches.Rectangle(
                (start_dt, track_positions[track] - 0.4),
                end_dt - start_dt,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=train_colors[base_train],
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Only add labels to sufficiently wide boxes
            if period['end'] - period['start'] > 30:  # Only label if period > 30 minutes
                midpoint = start_dt + (end_dt - start_dt) / 2
                ax.text(midpoint, track_positions[track], train, 
                       ha='center', va='center', fontsize=7)
    
    # Format axes
    ax.set_title('Train Yard Schedule')
    ax.set_xlabel('Time')
    ax.set_ylabel('Tracks')
    
    # Set time limits
    ax.set_xlim(
        base_date + timedelta(minutes=min_time),
        base_date + timedelta(minutes=max_time + 60)
    )
    
    # Set y-ticks to track names
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels(tracks, fontsize=8)
    
    # Format x-axis as time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.xticks(rotation=45)
    
    # Add legend for train types
    handles = []
    labels = []
    for train_id, color in sorted(train_colors.items()):
        handles.append(patches.Patch(color=color, label=train_id))
        labels.append(train_id)
    
    ax.legend(handles, labels, loc='upper right')
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('train_schedule.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    test_with_visualization()