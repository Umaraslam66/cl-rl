# import numpy as np
from env_c import TrainYardEnv

def simple_policy(env, obs):
    """A simple policy that tries to minimize track usage."""
    # Find least used tracks for entry and exit
    entry_usage = {}
    exit_usage = {}
    
    for track in env.entry_exit_tracks:
        entry_usage[track] = env.track_status[track]['occupied_until']
        exit_usage[track] = env.track_status[track]['occupied_until']
    
    # Sort by usage (least used first)
    entry_tracks = sorted(entry_usage.keys(), key=lambda x: entry_usage[x])
    exit_tracks = sorted(exit_usage.keys(), key=lambda x: exit_usage[x])
    
    # Get track indices
    track_list = list(env.entry_exit_tracks.keys())
    entry_idx = track_list.index(entry_tracks[0])
    exit_idx = track_list.index(exit_tracks[0])
    
    # Check if entry track has enough length for the arriving train
    if env.next_arrival_train:
        train_info = next(t for t in env.timetable if t['train'] == env.next_arrival_train)
        train_length = train_info['length']
        
        # Find first suitable track with enough length
        for track in entry_tracks:
            if env.entry_exit_tracks[track] >= train_length:
                entry_idx = track_list.index(track)
                break
    
    return {
        'entry_track': entry_idx,
        'exit_track': exit_idx,
        'wait_time': 0  # No wait time for simplicity
    }

def print_track_status(env, current_time):
    """Print the current status of all tracks."""
    print("\n=== TRACK STATUS AT", env._minutes_to_time_str(current_time), "===")
    
    # Group tracks by type
    track_types = {
        "Entry/Exit Tracks": env.entry_exit_tracks.keys(),
        "Parking Tracks": env.parking_tracks.keys(),
        "Loading Tracks": env.loading_tracks.keys()
    }
    
    for track_type, tracks in track_types.items():
        print(f"\n{track_type}:")
        for track in sorted(tracks):
            status = env.track_status[track]
            if status['occupied_until'] > current_time:
                train = status['train'] or "Unknown"
                free_time = env._minutes_to_time_str(status['occupied_until'])
                print(f"  {track} (length: {env.entry_exit_tracks.get(track) or env.parking_tracks.get(track) or env.loading_tracks.get(track)}): Occupied by {train} until {free_time}")
            else:
                print(f"  {track} (length: {env.entry_exit_tracks.get(track) or env.parking_tracks.get(track) or env.loading_tracks.get(track)}): Free")

def log_train_event(events, train_id, time, action, location=None):
    """Log a train event for later tracking."""
    if train_id not in events:
        events[train_id] = []
    events[train_id].append({
        'time': time,
        'action': action,
        'location': location
    })

def print_train_history(env, events, train_id):
    """Print the complete history of a train."""
    if train_id not in events:
        print(f"No events logged for train {train_id}")
        return
    
    print(f"\n=== HISTORY OF TRAIN {train_id} ===")
    for event in events[train_id]:
        time_str = env._minutes_to_time_str(event['time'])
        if event['location']:
            print(f"{time_str}: {event['action']} at {event['location']}")
        else:
            print(f"{time_str}: {event['action']}")

def test_environment_with_tracking():
    # Initialize environment
    env = TrainYardEnv()
    obs = env.reset()
    
    # Set trains to track (select a few trains of different types)
    tracked_trains = ['red1', 'green1', 'yellow1', 'black1']
    
    # Dictionary to store train events
    train_events = {}
    
    # Dictionary to store previous train statuses for change detection
    prev_statuses = {}
    
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 200  # Reduced for clarity in output
    last_time = 0
    
    # Run episode with policy actions
    while not done and step_count < max_steps:
        # Use policy to get action
        action = simple_policy(env, obs)
        
        # Take step
        prev_time = env.current_time
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Log events for arrival/entry/departure
        if env.next_arrival_train and env.next_arrival_train in tracked_trains:
            log_train_event(train_events, env.next_arrival_train, env.current_time, "Arrived")
        
        # Check for status changes in tracked trains
        for train_id in list(env.train_status.keys()):
            base_id = train_id.split('a')[0].split('b')[0]  # Get base train ID without a/b suffix
            
            if base_id in tracked_trains:
                current_status = env.train_status[train_id]
                prev_status = prev_statuses.get(train_id)
                
                if prev_status != current_status:
                    # Status has changed, log the event
                    location = env.train_locations.get(train_id, "unknown")
                    log_train_event(train_events, train_id, env.current_time, 
                                   f"Status changed from {prev_status or 'new'} to {current_status}", location)
                    prev_statuses[train_id] = current_status
        
        # Print status updates
        if step_count % 20 == 0 or done:
            print(f"\nStep {step_count} - Time: {info['time_str']}, " 
                  f"Tracks used: {info['tracks_used']}, "
                  f"Trains completed: {info['trains_completed']}/{len(env.timetable)}")
        
        # Print track status every hour of simulation time
        current_hour = env.current_time // 60
        last_hour = last_time // 60
        if current_hour > last_hour:
            print_track_status(env, env.current_time)
        last_time = env.current_time
    
    if step_count >= max_steps:
        print("\nWARNING: Maximum steps reached, simulation may not have completed")
    
    print(f"\nEpisode complete. Total reward: {total_reward}")
    print(f"Total steps taken: {step_count}")
    print(f"Final time: {info['time_str']}")
    
    # Print histories for tracked trains
    for train_id in tracked_trains:
        print_train_history(env, train_events, train_id)
        # Also print histories for split trains
        print_train_history(env, train_events, f"{train_id}a")
        print_train_history(env, train_events, f"{train_id}b")
    
    # Show final track status
    print_track_status(env, env.current_time)
    
    # Display final state of all trains
    print("\nFinal train status:")
    for train, status in sorted(env.train_status.items()):
        print(f"Train {train}: {status}")

if __name__ == "__main__":
    test_environment_with_tracking()
