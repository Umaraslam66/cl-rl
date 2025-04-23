import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from env_c import TrainYardEnv

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
        train_length = train_info['length']
        
        # Print debugging info
        print(f"\nARRIVING TRAIN: {env.next_arrival_train} (length: {train_length})")
        print(f"Available entry tracks (sorted by availability):")
        
        for track in entry_tracks:
            track_length = env.entry_exit_tracks[track]
            occupied_until = env.track_status[track]['occupied_until']
            occupied_str = env._minutes_to_time_str(occupied_until) if occupied_until > env.current_time else "FREE"
            suitable = "✓" if track_length >= train_length else "✗"
            
            print(f"  {track} (length: {track_length}): {occupied_str} {suitable}")
            
            if track_length >= train_length:
                entry_idx = track_list.index(track)
                print(f"  → Selected entry track: {track}")
                break
    
    action = {
        'entry_track': entry_idx,
        'exit_track': exit_idx,
        'wait_time': 0
    }
    
    # Print the selected tracks
    print(f"ACTION: Entry track: {track_list[entry_idx]}, Exit track: {track_list[exit_idx]}")
    
    return action

def test_with_visualization():
    env = TrainYardEnv()
    obs = env.reset()
    
    # Create a log file for track assignments
    log_file = open("track_assignments.log", "w")
    log_file.write("TRAIN YARD TRACK ASSIGNMENTS LOG\n")
    log_file.write("================================\n\n")
    
    # Keep track of occupancy directly
    track_occupancy = {}
    for track in list(env.entry_exit_tracks) + list(env.parking_tracks) + list(env.loading_tracks):
        track_occupancy[track] = []
    
    # Track current occupancy
    current_occupancy = {t: None for t in track_occupancy.keys()}
    occupancy_start = {t: 0 for t in track_occupancy.keys()}
    
    # Print initial track information
    print("\nTRACK INFORMATION:")
    print("Entry/Exit tracks:")
    for track, length in env.entry_exit_tracks.items():
        print(f"  {track}: {length} units")
    
    print("\nLoading tracks:")
    for track, length in env.loading_tracks.items():
        print(f"  {track}: {length} units")
    
    print("\nParking tracks:")
    for track, length in env.parking_tracks.items():
        print(f"  {track}: {length} units")
    
    done = False
    step_count = 0
    max_steps = 2000
    
    # Record initial state
    for track, status in env.track_status.items():
        if status['train']:
            current_occupancy[track] = status['train']
            occupancy_start[track] = env.current_time
            log_file.write(f"Initial state: {status['train']} on track {track}\n")
    
    while not done and step_count < max_steps:
        prev_time = env.current_time
        
        # Take step
        action = simple_policy(env, obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Log time
        log_file.write(f"\nTime: {info['time_str']} (Step {step_count})\n")
        
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
                    
                    # Log track departure
                    log_file.write(f"  {current_occupancy[track]} left track {track} at {env._minutes_to_time_str(prev_time)}\n")
                
                # Update current state
                current_occupancy[track] = status['train']
                if status['train']:
                    occupancy_start[track] = prev_time
                    # Log track arrival
                    log_file.write(f"  {status['train']} entered track {track} at {env._minutes_to_time_str(prev_time)}\n")
        
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
    log_file.write(f"\nEpisode complete. Final time: {env._minutes_to_time_str(env.current_time)}\n")
    log_file.close()
    
    # Filter out any unreasonably long occupancy periods
    max_duration = 48 * 60  # 48 hours in minutes
    for track in track_occupancy:
        track_occupancy[track] = [
            period for period in track_occupancy[track]
            if period['end'] - period['start'] <= max_duration
        ]
    
    # Create visualization with improved track grouping
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group tracks by type for better visualization
    entry_exit_tracks = sorted(list(env.entry_exit_tracks))
    loading_tracks = sorted(list(env.loading_tracks))
    parking_tracks = sorted(list(env.parking_tracks))
    
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
    all_periods = [period for track_periods in track_occupancy.values() for period in track_periods]
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
        ax.add_patch(rect)
        ax.text(
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
        ax.add_patch(rect)
        ax.text(
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
        ax.add_patch(rect)
        ax.text(
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
    for track, periods in track_occupancy.items():
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
                ax.add_patch(rect)
                
                # Only add labels to sufficiently wide boxes
                if period['end'] - period['start'] > 30:  # Only label if period > 30 minutes
                    midpoint = start_dt + (end_dt - start_dt) / 2
                    # Choose text color based on background brightness
                    text_color = 'white' if train_type in ['black', 'blue', 'purple', 'brown'] else 'black'
                    ax.text(midpoint, track_positions[track], train, 
                           ha='center', va='center', fontsize=7, color=text_color,
                           zorder=2)
    
    # Format axes
    ax.set_title('Train Yard Schedule', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Tracks', fontsize=12)
    
    # Set time limits
    ax.set_xlim(
        base_date + timedelta(minutes=min_time - 30),
        base_date + timedelta(minutes=max_time + 60)
    )
    
    # Set y-ticks to track names
    y_ticks = []
    y_labels = []
    for track, pos in track_positions.items():
        y_ticks.append(pos)
        
        # For entry/exit tracks, add length information
        if track in env.entry_exit_tracks:
            track_length = env.entry_exit_tracks[track]
            y_labels.append(f"{track} ({track_length})")
        # For loading tracks, add length information
        elif track in env.loading_tracks:
            track_length = env.loading_tracks[track]
            y_labels.append(f"{track} ({track_length})")
        # For parking tracks, add length information
        elif track in env.parking_tracks:
            track_length = env.parking_tracks[track]
            y_labels.append(f"{track} ({track_length})")
        else:
            y_labels.append(track)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Format x-axis as time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add horizontal grid lines
    for pos in y_ticks:
        ax.axhline(y=pos, color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    
    # Add vertical grid lines
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Add legend for train types
    handles = []
    labels = []
    for train_id, color in sorted(used_train_colors.items()):
        handles.append(patches.Patch(color=color, label=train_id))
        labels.append(train_id)
    
    ax.legend(handles, labels, loc='upper right', fontsize=8)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
    plt.savefig('train_schedule_diagnostic.png', dpi=200)
    plt.show()
    
    # Print where to find the log file
    print(f"Track assignment log has been written to 'track_assignments.log'")

if __name__ == "__main__":
    test_with_visualization()