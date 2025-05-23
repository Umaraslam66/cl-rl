import gym
import numpy as np
from gym import spaces
import heapq
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainYardEnv")

class TrainYardEnv(gym.Env):
    """
    Gym environment for train yard operations with train splitting, loading, and coupling.
    This improved version includes better state management, error handling, and realistic constraints.
    """
    
    MOVEMENT_TYPES = ['to_entry', 'to_loading', 'to_parking', 'to_exit', 'misc']
    TRAIN_STATES = [
        'scheduled', 'arriving', 'splitting', 'split_complete', 
        'waiting_at_entry', 'moving_to_loading', 'loading', 'loading_complete',
        'moving_to_exit', 'waiting_at_exit', 'coupling', 'coupled', 'departed'
    ]
    
    # Define valid state transitions
    VALID_TRANSITIONS = {
        'scheduled': ['arriving'],
        'arriving': ['splitting'],
        'splitting': ['split_complete'],
        'split_complete': ['waiting_at_entry'],
        'waiting_at_entry': ['moving_to_loading'],
        'moving_to_loading': ['loading'],
        'loading': ['loading_complete'],
        'loading_complete': ['moving_to_exit'],
        'moving_to_exit': ['waiting_at_exit'],
        'waiting_at_exit': ['coupling'],
        'coupling': ['coupled'],
        'coupled': ['departed']
    }
    
    def __init__(self, verbose=False):
        super(TrainYardEnv, self).__init__()
        
        # Configure logging based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Load track and timetable data
        self.entry_exit_tracks = self._parse_tracks("32:750,33:750,34:750,35:750,36:670,m3:450,m4:450,m5:550,m6:600,m7:750,m8:750,m9:750,m10:750,m11:900,m12:850,m13:800")
        self.parking_tracks = self._parse_tracks("51:665,52:665,53:741,54:746,55:752")
        self.loading_tracks = self._parse_tracks("56A:571,56B:161,57:787,54:746,71:303,41:487,42:487,43:313,44:313,4x:313,29:242,30:237,20:500")
        self.timetable = self._load_timetable()
        
        # Track connections - simplified model of which tracks can connect to others
        self._initialize_track_connections()
        
        # Movement parameters - more realistic timing model
        self.movement_times = {
            'entry_to_loading': 15,    # Minutes to move from entry to loading
            'loading_to_exit': 15,     # Minutes to move from loading to exit
            'coupling_time': 15,       # Minutes to couple train halves
            'splitting_time': 15,      # Minutes to split a train
            'min_headway': 10          # Minimum headway between trains
        }
        
        # State tracking
        self.current_time = 0
        self.track_status = {t: {'occupied_until': 0, 'train': None, 'reserved_for': None} 
                            for t in self._all_tracks()}
        self.trains = {}
        self.train_locations = {}
        self.train_status = {}
        self.event_queue = []
        self.next_arrival_train = None
        self.max_departure_time = max([t['departure'] for t in self.timetable])
        
        # Headway tracking for all movement types
        self.last_movement_time = {movement_type: 0 for movement_type in self.MOVEMENT_TYPES}
        
        # Performance tracking
        self.delayed_departures = 0
        self.total_delay_minutes = 0
        self.completed_trains = 0
        
        # Action and observation spaces
        entry_exit_tracks = list(self.entry_exit_tracks.keys())
        self.action_space = spaces.Dict({
            'entry_track': spaces.Discrete(len(entry_exit_tracks)),
            'exit_track': spaces.Discrete(len(entry_exit_tracks)),
            'wait_time': spaces.Discrete(60)  # Wait up to 60 minutes
        })
        
        # Enhanced observation includes all active trains
        self._initialize_observation_space()
        
        # Initialize statistics and performance metrics
        self.stats = defaultdict(int)
        
        # Initialize event queue with train arrivals
        for train in self.timetable:
            heapq.heappush(self.event_queue, (train['arrival'], 'train_arrival', train['train']))
            # Initialize train state
            self.train_status[train['train']] = 'scheduled'
    
    def _all_tracks(self) -> List[str]:
        """Get a list of all tracks in the yard."""
        return list(self.entry_exit_tracks) + list(self.parking_tracks) + list(self.loading_tracks)
    
    def _initialize_track_connections(self):
        """Initialize a simplified model of track connections."""
        # This is a simplified model - in a real implementation, this would be based on actual yard topology
        self.track_connections = {}
        
        # All entry/exit tracks can connect to all loading tracks
        for entry_track in self.entry_exit_tracks:
            self.track_connections[entry_track] = set(self.loading_tracks)
            
        # All loading tracks can connect to all entry/exit tracks
        for loading_track in self.loading_tracks:
            self.track_connections[loading_track] = set(self.entry_exit_tracks)
            
        # All parking tracks can connect to all other track types
        for parking_track in self.parking_tracks:
            self.track_connections[parking_track] = set(self.entry_exit_tracks) | set(self.loading_tracks)
            
        logger.debug(f"Initialized track connections for {len(self.track_connections)} tracks")
    
    def _initialize_observation_space(self):
        """Initialize the observation space with enhanced dimensions."""
        # Track count
        track_count = len(self._all_tracks())
        
        # Obs includes: current time, track statuses, upcoming trains (5), and active trains (up to 10)
        obs_dim = (
            1 +                 # Current time
            track_count * 3 +   # Tracks (occupied, time until free, reserved)
            5 * 5 +             # 5 upcoming trains with 5 features each
            10 * 6              # Up to 10 active trains with 6 features each
        )
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        logger.debug(f"Initialized observation space with dimension {obs_dim}")
    
    def _load_timetable(self):
        """Load train timetable with arrival, departure, loading info."""
        data = [
            ["red1", "4:45", "11:30", 120, 620, "43-44"],
            ["red2", "12:45", "19:00", 120, 620, "43-44"],
            ["red3", "19:00", "1 06:00", 120, 620, "43-44"],
            ["green1", "1:30", "11:15", 120, 620, "4x"],
            ["green2", "10:30", "19:00", 120, 620, "4x"],
            ["green3", "18:30", "1 03:00", 120, 620, "4x"],
            ["purple1", "5:45", "19:50", 90, 550, "71"],
            ["black1", "8:45", "15:45", 105, 630, "57-71"],
            ["black2", "21:15", "1 03:45", 105, 630, "57-71"],
            ["yellow1", "6:15", "12:00", 120, 630, "41-42"],
            ["yellow2", "16:15", "22:00", 120, 630, "41-42"],
            ["yellow3", "22:30", "1 06:30", 120, 630, "41-42"],
            ["brown1", "5:15", "17:00", 90, 550, "20"],
            ["pink1", "3:45", "1 04:00", 70, 550, "29-30"],
            ["blue1", "15:15", "1 03:00", 180, 550, "57"]
        ]
        
        timetable = []
        for train, arrival, departure, load_time, length, load_track in data:
            arrival_mins = self._time_to_minutes(arrival)
            departure_mins = self._time_to_minutes(departure)
            load_tracks = load_track.split("-") if "-" in load_track else [load_track]
            
            timetable.append({
                'train': train,
                'arrival': arrival_mins,
                'departure': departure_mins,
                'load_time': load_time,
                'length': length,
                'load_tracks': load_tracks
            })
        
        return sorted(timetable, key=lambda x: x['arrival'])
    
    def _time_to_minutes(self, time_str):
        """Convert time string (e.g., '4:45' or '1 06:00') to minutes."""
        if " " in time_str:  # Format like "1 06:00" (next day)
            day, time = time_str.split(" ")
            day_offset = int(day) * 24 * 60
        else:
            day_offset = 0
            time = time_str
        
        hours, minutes = map(int, time.split(":"))
        return day_offset + hours * 60 + minutes
    
    def _minutes_to_time_str(self, minutes):
        """Convert minutes to time string format (e.g., 'Day 1 06:00')."""
        days = minutes // (24 * 60)
        hours = (minutes % (24 * 60)) // 60
        mins = minutes % 60
        
        if days > 0:
            return f"Day {days} {hours:02d}:{mins:02d}"
        else:
            return f"{hours:02d}:{mins:02d}"
    
    def _parse_tracks(self, tracks_str):
        """Parse track string to dictionary of {track_id: length}."""
        return {t.split(":")[0]: int(t.split(":")[1]) for t in tracks_str.replace('"', '').replace("{", "").replace("}", "").split(",")}
    
    def _can_move_between_tracks(self, from_track: str, to_track: str) -> bool:
        """Check if movement between tracks is physically possible."""
        if from_track not in self.track_connections:
            logger.warning(f"Track {from_track} not in track connections map")
            return False
            
        return to_track in self.track_connections[from_track]
    
    def _is_track_suitable(self, track: str, train_length: int) -> bool:
        """Check if a track can accommodate a train of the given length."""
        # Determine which track dictionary to check
        if track in self.entry_exit_tracks:
            return self.entry_exit_tracks[track] >= train_length
        elif track in self.parking_tracks:
            return self.parking_tracks[track] >= train_length
        elif track in self.loading_tracks:
            return self.loading_tracks[track] >= train_length
        else:
            logger.warning(f"Unknown track: {track}")
            return False
    
    def _get_track_length(self, track: str) -> int:
        """Get the length of a track."""
        if track in self.entry_exit_tracks:
            return self.entry_exit_tracks[track]
        elif track in self.parking_tracks:
            return self.parking_tracks[track]
        elif track in self.loading_tracks:
            return self.loading_tracks[track]
        else:
            logger.warning(f"Unknown track for length query: {track}")
            return 0
    
    def _validate_state_transition(self, train_id: str, new_state: str) -> bool:
        """Validate if a state transition is allowed with improved flexibility."""
        current_state = self.train_status.get(train_id)
        
        # Allow transition to the same state to prevent blocking
        if current_state == new_state:
            return True
            
        # If train doesn't exist yet, any valid initial state is fine
        if current_state is None:
            return new_state in ['scheduled', 'arriving']
        
        # Special case for parent trains when halves reach the exit track
        if (new_state == 'coupling' and 
            (current_state == 'split_complete' or current_state == 'waiting_at_entry' or 
            current_state == 'coupled')):  # Allow recoupling attempts
            return True
            
        # Check if transition is valid
        if current_state in self.VALID_TRANSITIONS:
            if new_state in self.VALID_TRANSITIONS[current_state]:
                return True
                
        # Recovery path for stuck trains
        if (current_state == 'split_complete' and new_state == 'coupled') or \
        (current_state == 'coupling' and new_state == 'coupled') or \
        (current_state == 'coupled' and new_state == 'departed'):
            logger.warning(f"Allowing special transition for stuck train: {current_state} -> {new_state}")
            return True
                
        logger.warning(f"Invalid state transition for {train_id}: {current_state} -> {new_state}")
        return False
    
    def _update_train_state(self, train_id: str, new_state: str) -> bool:
        """Update train state with validation."""
        if self._validate_state_transition(train_id, new_state):
            old_state = self.train_status.get(train_id)
            self.train_status[train_id] = new_state
            logger.debug(f"Train {train_id} state changed: {old_state} -> {new_state}")
            return True
        return False
    
    def _apply_headway_constraint(self, base_time: int, movement_type: str) -> int:
        """Apply headway constraint to movement timing."""
        return max(base_time, self.last_movement_time[movement_type] + self.movement_times['min_headway'])
    
    def _reserve_track(self, track: str, train_id: str, until_time: int) -> bool:
        """Reserve a track for a train until a specified time."""
        # Check if track is available
        if self.track_status[track]['occupied_until'] > self.current_time:
            logger.warning(f"Cannot reserve track {track} for {train_id} - already occupied")
            return False
            
        # Check if track is already reserved
        if self.track_status[track]['reserved_for'] is not None:
            logger.warning(f"Cannot reserve track {track} for {train_id} - already reserved")
            return False
            
        self.track_status[track]['reserved_for'] = train_id
        logger.debug(f"Reserved track {track} for {train_id} until {self._minutes_to_time_str(until_time)}")
        return True
    
    def _release_reservation(self, track: str, train_id: str) -> None:
        """Release a track reservation."""
        if self.track_status[track]['reserved_for'] == train_id:
            self.track_status[track]['reserved_for'] = None
            logger.debug(f"Released reservation of track {track} for {train_id}")
    
    def _occupy_track(self, track: str, train_id: str, until_time: int) -> None:
        """Mark a track as occupied by a train."""
        self.track_status[track]['train'] = train_id
        self.track_status[track]['occupied_until'] = until_time
        self.track_status[track]['reserved_for'] = None  # Clear reservation
        self.train_locations[train_id] = track
        logger.debug(f"Track {track} occupied by {train_id} until {self._minutes_to_time_str(until_time)}")
    
    def _release_track(self, track: str) -> None:
        """Mark a track as free."""
        train_id = self.track_status[track]['train']
        if train_id and train_id in self.train_locations and self.train_locations[train_id] == track:
            del self.train_locations[train_id]
            
        self.track_status[track]['train'] = None
        logger.debug(f"Track {track} released")
    
    def _check_for_deadlocks(self) -> bool:
        """Simple deadlock detection - check for trains waiting with no progress possible."""
        # Count trains in waiting states
        waiting_trains = 0
        for train_id, state in self.train_status.items():
            if state in ['waiting_at_entry', 'loading_complete']:
                waiting_trains += 1
                
        # If all active trains are waiting and event queue is empty, we have a deadlock
        if waiting_trains > 0 and waiting_trains == len(self.train_status) and not self.event_queue:
            logger.warning(f"Potential deadlock detected: {waiting_trains} trains waiting with no progress")
            return True
        return False
    
    def _cleanup_completed_train_data(self, train_id: str) -> None:
        """Clean up data for completed trains to avoid memory buildup."""
        # Keep the train status (for reporting) but clean up other data
        base_id = train_id.split('a')[0].split('b')[0]
        
        # Only clean when the main train and its halves are all departed
        train_ids = [base_id, f"{base_id}a", f"{base_id}b"]
        all_departed = all(self.train_status.get(tid) == 'departed' for tid in train_ids if tid in self.train_status)
        
        if all_departed:
            # Clean location and other operational data
            for tid in train_ids:
                if tid in self.train_locations:
                    del self.train_locations[tid]
            
            # Keep a minimal record of the train for reporting
            logger.debug(f"Cleaned up data for completed train {base_id}")
    
    def step(self, action):
        """Take action and advance simulation until next decision point."""
        import time as py_time  # Import time module for timeout

        # Process action (selecting tracks and wait time)
        track_list = list(self.entry_exit_tracks.keys())
        entry_track = track_list[action['entry_track']]
        exit_track = track_list[action['exit_track']]
        wait_time = action['wait_time']
        
        # Add timeout protection
        start_time = py_time.time()
        timeout = 10.0  # 10 seconds max per step
        
        # Reset action-specific statistics
        self.stats['invalid_track_assignments'] = 0
        
        # Assign tracks to next arriving train
        if self.next_arrival_train:
            train_info = next((t for t in self.timetable if t['train'] == self.next_arrival_train), None)
            
            if train_info is None:
                logger.error(f"Cannot find train info for {self.next_arrival_train}")
                self.next_arrival_train = None
            else:
                self.trains[self.next_arrival_train] = {
                    'info': train_info,
                    'entry_track': entry_track,
                    'exit_track': exit_track,
                    'wait_time': wait_time
                }
                
                # Validate the track assignment
                valid_assignment = True
                
                # Check if entry track is long enough
                if not self._is_track_suitable(entry_track, train_info['length']):
                    logger.warning(f"Entry track {entry_track} too short for {self.next_arrival_train}")
                    self.stats['invalid_track_assignments'] += 1
                    valid_assignment = False
                
                # Check if exit track is long enough
                if not self._is_track_suitable(exit_track, train_info['length']):
                    logger.warning(f"Exit track {exit_track} too short for {self.next_arrival_train}")
                    self.stats['invalid_track_assignments'] += 1
                    valid_assignment = False
                    
                # Schedule train entry if valid assignment
                if valid_assignment:
                    # Try to reserve the entry track
                    if self._reserve_track(entry_track, self.next_arrival_train, self.current_time + 60):
                        heapq.heappush(self.event_queue, (self.current_time, 'train_entry', self.next_arrival_train))
                        logger.info(f"Scheduled entry of {self.next_arrival_train} on track {entry_track}")
                    else:
                        # If can't reserve, try again later
                        heapq.heappush(self.event_queue, (self.current_time + 10, 'train_arrival', self.next_arrival_train))
                        logger.info(f"Rescheduled arrival of {self.next_arrival_train} in 10 minutes")
                else:
                    # If invalid assignment, try again later
                    heapq.heappush(self.event_queue, (self.current_time + 10, 'train_arrival', self.next_arrival_train))
                    logger.info(f"Rescheduled arrival of {self.next_arrival_train} in 10 minutes due to invalid track assignment")
                
                self.next_arrival_train = None
        
        # Process events until next decision needed with timeout protection
        done = False
        try:
            # Process events with timeout monitoring
            done = self._process_events()
            
            # Check for timeout
            elapsed = py_time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Step execution timeout reached after {elapsed:.1f}s. Terminating episode.")
                done = True
        except Exception as e:
            logger.error(f"Unexpected error in step execution: {e}")
            done = True
        
        # Check for deadlocks
        if not done and self._check_for_deadlocks():
            logger.warning("Deadlock detected - simulation will terminate")
            done = True
        
        # Calculate enhanced reward
        reward = self._calculate_reward()
        
        # Prepare info dictionary with detailed stats
        info = {
            'current_time': self.current_time,
            'time_str': self._minutes_to_time_str(self.current_time),
            'tracks_used': sum(1 for status in self.track_status.values() 
                            if status['occupied_until'] > self.current_time),
            'trains_completed': self.completed_trains,
            'delayed_departures': self.delayed_departures,
            'total_delay_minutes': self.total_delay_minutes,
            'invalid_track_assignments': self.stats['invalid_track_assignments'],
            'execution_time': py_time.time() - start_time
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate a more comprehensive reward."""
        # Base reward components
        used_tracks = sum(1 for status in self.track_status.values() 
                         if status['occupied_until'] > self.current_time)
        
        # Track efficiency component (penalize unused capacity)
        total_track_length = sum(length for length in 
                              list(self.entry_exit_tracks.values()) + 
                              list(self.parking_tracks.values()) + 
                              list(self.loading_tracks.values()))
        
        used_length = sum(self._get_track_length(track) for track, status in self.track_status.items() 
                        if status['occupied_until'] > self.current_time)
        
        efficiency_score = used_length / total_track_length if total_track_length > 0 else 0
        
        # Completion reward
        completion_reward = self.completed_trains * 10
        
        # Delay penalty
        delay_penalty = self.total_delay_minutes * 0.1
        
        # Combine components
        reward = -used_tracks + completion_reward - delay_penalty + (efficiency_score * 5)
        
        return reward
    
    def _process_events(self):
        """Process events until next decision required or simulation completed."""
        # Safety check - if no events and no arriving train, we're done
        if not self.event_queue and not self.next_arrival_train:
            # Make sure all trains have been processed
            base_trains = [train for train in self.train_status.keys() 
                        if not (train.endswith('a') or train.endswith('b'))]
            
            if all(self.train_status.get(train) == 'departed' for train in base_trains):
                logger.info("All trains have departed - simulation complete")
                return True
            else:
                logger.warning("No events but trains not departed - possible deadlock")
                return True
        
        max_process_iterations = 1000  # Safety to prevent infinite loops
        iterations = 0
        
        # Track if any meaningful progress is being made
        initial_event_count = len(self.event_queue)
        last_progress_time = self.current_time
        stalled_events = set()
        
        while self.event_queue and not self.next_arrival_train and iterations < max_process_iterations:
            iterations += 1
            
            # Get next event
            time, event_type, train_id = heapq.heappop(self.event_queue)
            
            # Check for repeated events indicating a deadlock
            event_key = f"{event_type}_{train_id}"
            if event_key in stalled_events and time <= last_progress_time + 5:
                logger.warning(f"Repeated event detected: {event_type} for {train_id}")
                if iterations > 100:  # Only force progress after giving it a chance to resolve
                    logger.warning(f"Forcing time advancement to break potential deadlock")
                    self.current_time = time + 30  # Force time forward
                    # Handle any trains that might be stuck
                    for tid, status in self.train_status.items():
                        if status in ['coupling', 'loading_complete', 'waiting_at_exit']:
                            logger.warning(f"Forcing train {tid} to next state to break deadlock")
                            if status == 'coupling':
                                self.train_status[tid] = 'coupled'
                            elif status == 'loading_complete':
                                self.train_status[tid] = 'moving_to_exit'
                            elif status == 'waiting_at_exit':
                                # Find the base train ID
                                base_id = tid.split('a')[0].split('b')[0]
                                if base_id in self.train_status and self.train_status[base_id] != 'departed':
                                    self.train_status[base_id] = 'coupled'
                    break
            else:
                stalled_events.add(event_key)
            
            # Update current time
            if time > self.current_time:
                self.current_time = time
                last_progress_time = time
                stalled_events.clear()  # Reset stalled events when time advances
                logger.debug(f"Time advanced to {self._minutes_to_time_str(time)}")
            
            # Process event based on type
            try:
                if event_type == 'train_arrival':
                    self._handle_train_arrival(train_id, time)
                elif event_type == 'train_entry':
                    self._handle_train_entry(train_id, time)
                elif event_type == 'train_split':
                    self._handle_train_split(train_id, time)
                elif event_type == 'half_to_loading':
                    self._handle_half_to_loading(train_id, time)
                elif event_type == 'loading_complete':
                    self._handle_loading_complete(train_id, time)
                elif event_type == 'half_to_exit':
                    self._handle_half_to_exit(train_id, time)
                elif event_type == 'couple_train':
                    self._handle_couple_train(train_id, time)
                elif event_type == 'train_departure':
                    self._handle_train_departure(train_id, time)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
            except Exception as e:
                logger.error(f"Error processing event {event_type} for train {train_id}: {e}")
                # Continue with next event rather than crashing
        
        # If we've reached max iterations, log an error
        if iterations >= max_process_iterations:
            logger.warning(f"Max iterations reached in event processing. Current time: {self._minutes_to_time_str(self.current_time)}")
            # Force completion to avoid infinite loops
            return True
        
        # Check if simulation should end (all trains departed or current time > max departure + buffer)
        if self.current_time > self.max_departure_time + 120:  # 2 hour buffer
            logger.info(f"Simulation end time reached: {self._minutes_to_time_str(self.current_time)}")
            return True
                
        return False
    
    def _handle_train_arrival(self, train_id: str, time: int) -> None:
        """Handle train arrival event."""
        if not self._update_train_state(train_id, 'arriving'):
            logger.warning(f"Failed to update state for arriving train {train_id}")
            
        self.next_arrival_train = train_id
        logger.info(f"Train {train_id} arriving at {self._minutes_to_time_str(time)}")
    
    def _handle_train_entry(self, train_id: str, time: int) -> None:
        """Handle train entry into the yard."""
        # Validate train exists
        if train_id not in self.trains:
            logger.error(f"Train {train_id} not in trains dictionary")
            return
            
        # Get assigned entry track
        entry_track = self.trains[train_id]['entry_track']
        
        # Check if entry track is free and reserved for this train
        if (self.track_status[entry_track]['occupied_until'] <= time and 
            self.track_status[entry_track]['reserved_for'] == train_id):
            
            # Update train state
            if not self._update_train_state(train_id, 'splitting'):
                logger.warning(f"Failed to update state for splitting train {train_id}")
                
            # Release reservation and occupy track
            self._release_reservation(entry_track, train_id)
            
            # Calculate split completion time
            split_time = time + self.movement_times['splitting_time']
            
            # Occupy track until split complete
            self._occupy_track(entry_track, train_id, split_time)
            
            # Update movement timing for headway
            self.last_movement_time['to_entry'] = time
            
            # Schedule split completion
            heapq.heappush(self.event_queue, (split_time, 'train_split', train_id))
            logger.info(f"Train {train_id} entered yard at track {entry_track}, will split by {self._minutes_to_time_str(split_time)}")
        else:
            # Track not available, reschedule entry
            new_time = time + 10
            heapq.heappush(self.event_queue, (new_time, 'train_entry', train_id))
            logger.info(f"Rescheduled entry of {train_id} to {self._minutes_to_time_str(new_time)} - track {entry_track} not available")
    
    def _handle_train_split(self, train_id: str, time: int) -> None:
        """Handle train splitting with headway constraints."""
        # Update state to show split is complete
        if not self._update_train_state(train_id, 'split_complete'):
            logger.warning(f"Failed to update state for split completion of train {train_id}")
        
        # Get train information
        train_info = self.trains[train_id]['info']
        entry_track = self.trains[train_id]['entry_track']
        load_tracks = train_info['load_tracks']
        wait_time = self.trains[train_id]['wait_time']
        
        # Update entry track occupation time to include wait time
        self.track_status[entry_track]['occupied_until'] = time + wait_time
        
        # Create train halves with proper state transition
        if not self._update_train_state(f"{train_id}a", 'waiting_at_entry'):
            self.train_status[f"{train_id}a"] = 'waiting_at_entry'
            
        if not self._update_train_state(f"{train_id}b", 'waiting_at_entry'):
            self.train_status[f"{train_id}b"] = 'waiting_at_entry'
        
        # First half to loading track if available
        if load_tracks:
            first_load = load_tracks[0]
            
            # Check if loading track is suitable
            if self._is_track_suitable(first_load, train_info['length'] // 2):
                # Check if loading track is free and connection exists
                if (self.track_status[first_load]['occupied_until'] <= time and 
                    self._can_move_between_tracks(entry_track, first_load)):
                    
                    # Apply headway constraint
                    movement_time = self._apply_headway_constraint(time + wait_time, 'to_loading')
                    
                    # Update train state
                    if not self._update_train_state(f"{train_id}a", 'moving_to_loading'):
                        logger.warning(f"Failed to update state for train half {train_id}a moving to loading")
                    
                    # Calculate loading completion time
                    load_completion = movement_time + self.movement_times['entry_to_loading'] + train_info['load_time']
                    
                    # Occupy loading track
                    self._occupy_track(first_load, f"{train_id}a", load_completion)
                    
                    # Schedule half movement to loading
                    heapq.heappush(self.event_queue, 
                                  (movement_time + self.movement_times['entry_to_loading'], 
                                   'half_to_loading', f"{train_id}a"))
                    
                    # Update last movement time
                    self.last_movement_time['to_loading'] = movement_time + self.movement_times['entry_to_loading']
                    
                    logger.info(f"Scheduled train half {train_id}a to move to loading track {first_load} at {self._minutes_to_time_str(movement_time)}")
        
        logger.info(f"Train {train_id} split complete at {self._minutes_to_time_str(time)}")
    
    def _handle_half_to_loading(self, half_id: str, time: int) -> None:
        """Handle half train arriving at loading track."""
        base_id = half_id[:-1]  # Remove 'a' or 'b'
        
        # Find loading track
        loading_track = self.train_locations.get(half_id)
        
        if loading_track is None:
            logger.error(f"Cannot find location for train half {half_id}")
            return
        
        # Update train state
        if not self._update_train_state(half_id, 'loading'):
            logger.warning(f"Failed to update state for train half {half_id} at loading")
        
        # Schedule loading completion
        load_time = self.trains[base_id]['info']['load_time']
        heapq.heappush(self.event_queue, (time + load_time, 'loading_complete', half_id))
        
        logger.info(f"Train half {half_id} arrived at loading track {loading_track}, will complete loading at {self._minutes_to_time_str(time + load_time)}")
        
        # Check if this was the first half and schedule second half if needed
        if half_id.endswith('a'):
            second_half = f"{base_id}b"
            
            # Only proceed if second half is waiting and first half is at its loading track
            if self.train_status.get(second_half) == 'waiting_at_entry':
                train_info = self.trains[base_id]['info']
                entry_track = self.trains[base_id]['entry_track']
                
                # If there's a second loading track defined
                if len(train_info['load_tracks']) > 1:
                    second_load = train_info['load_tracks'][1]
                    
                    # Check if it's suitable and available
                    if (self._is_track_suitable(second_load, train_info['length'] // 2) and
                        self.track_status[second_load]['occupied_until'] <= time and
                        self._can_move_between_tracks(entry_track, second_load)):
                        
                        # Apply headway constraint (minimum 10 min after first half arrived)
                        movement_time = self._apply_headway_constraint(time + 10, 'to_loading')
                        
                        # Update train state
                        if not self._update_train_state(second_half, 'moving_to_loading'):
                            logger.warning(f"Failed to update state for train half {second_half} moving to loading")
                        
                        # Calculate loading completion time
                        load_completion = movement_time + self.movement_times['entry_to_loading'] + train_info['load_time']
                        
                        # Occupy loading track
                        self._occupy_track(second_load, second_half, load_completion)
                        
                        # Schedule half movement
                        heapq.heappush(self.event_queue, 
                                      (movement_time + self.movement_times['entry_to_loading'], 
                                       'half_to_loading', second_half))
                        
                        # Update last movement time
                        self.last_movement_time['to_loading'] = movement_time + self.movement_times['entry_to_loading']
                        
                        logger.info(f"Scheduled train half {second_half} to move to loading track {second_load} at {self._minutes_to_time_str(movement_time)}")
    
    def _handle_loading_complete(self, half_id: str, time: int) -> None:
        """Handle half train completing loading with headway constraints."""
        base_id = half_id[:-1]  # Remove 'a' or 'b'
        loading_track = self.train_locations.get(half_id)
        exit_track = self.trains[base_id]['exit_track']
        
        if loading_track is None:
            logger.error(f"Cannot find location for train half {half_id}")
            return
        
        # Prevent redundant state transitions
        if self.train_status.get(half_id) != 'loading_complete':
            if not self._update_train_state(half_id, 'loading_complete'):
                logger.warning(f"Failed to update state for train half {half_id} loading complete - forcing state")
                self.train_status[half_id] = 'loading_complete'
        
        # Check if exit track is available or will be available soon
        current_occupant = self.track_status[exit_track]['train']
        time_available = self.track_status[exit_track]['occupied_until']
        
        # Check for near-term availability (within 30 minutes)
        if time_available <= time + 30 and self._can_move_between_tracks(loading_track, exit_track):
            # Apply headway constraint
            movement_time = self._apply_headway_constraint(max(time, time_available), 'to_exit')
            
            # Update train state
            if not self._update_train_state(half_id, 'moving_to_exit'):
                logger.warning(f"Failed to update state for train half {half_id} moving to exit - forcing state")
                self.train_status[half_id] = 'moving_to_exit'
            
            # Release loading track
            self.track_status[loading_track]['occupied_until'] = movement_time
            self.track_status[loading_track]['train'] = None
            
            # Occupy exit track
            exit_arrival = movement_time + self.movement_times['loading_to_exit']
            self._occupy_track(exit_track, half_id, exit_arrival + 60)  # Add buffer time
            
            # Schedule arrival at exit
            heapq.heappush(self.event_queue, (exit_arrival, 'half_to_exit', half_id))
            
            # Update last movement time
            self.last_movement_time['to_exit'] = exit_arrival
            
            logger.info(f"Train half {half_id} completed loading, moving to exit track {exit_track}, arriving at {self._minutes_to_time_str(exit_arrival)}")
            
            # If second half is done, schedule it to move if appropriate
            if half_id.endswith('b'):
                # Check if first half is at exit
                first_half = f"{base_id}a"
                if self.train_status.get(first_half) == 'waiting_at_exit':
                    # Both halves will be at exit, schedule coupling
                    heapq.heappush(self.event_queue, (exit_arrival, 'couple_train', base_id))
                    logger.info(f"Both halves will be at exit, coupling scheduled for {self._minutes_to_time_str(exit_arrival)}")
        else:
            # Exit track not available soon, reschedule check
            detail = f"occupied by {current_occupant} until {self._minutes_to_time_str(time_available)}" if current_occupant else "not available"
            logger.info(f"Rescheduled exit movement for {half_id} - exit track {exit_track} {detail}")
            heapq.heappush(self.event_queue, (time + 10, 'loading_complete', half_id))
    
    def _handle_half_to_exit(self, half_id: str, time: int) -> None:
        """Handle half train arriving at exit track with improved state management."""
        base_id = half_id[:-1]  # Remove 'a' or 'b'
        exit_track = self.trains[base_id]['exit_track']
        
        # Update location and state
        self.train_locations[half_id] = exit_track
        
        if not self._update_train_state(half_id, 'waiting_at_exit'):
            logger.warning(f"Failed to update state for train half {half_id} waiting at exit - forcing state")
            self.train_status[half_id] = 'waiting_at_exit'
                
        logger.info(f"Train half {half_id} arrived at exit track {exit_track}")
        
        # Check if both halves are at exit
        other_half = f"{base_id}{'b' if half_id.endswith('a') else 'a'}"
        other_half_ready = (
            self.train_status.get(other_half) == 'waiting_at_exit' and
            self.train_locations.get(other_half) == exit_track
        )
        
        if other_half_ready:
            logger.info(f"Both halves at exit, coupling scheduled for train {base_id}")
            heapq.heappush(self.event_queue, (time, 'couple_train', base_id))
    
    def _handle_couple_train(self, train_id: str, time: int) -> None:
        """Handle coupling of train halves with improved state checking."""
        exit_track = self.trains[train_id]['exit_track']
        half_a = f"{train_id}a"
        half_b = f"{train_id}b"
        
        # Extended verification for halves
        halves_at_exit = (
            self.train_locations.get(half_a) == exit_track and
            self.train_locations.get(half_b) == exit_track and
            self.train_status.get(half_a) == 'waiting_at_exit' and
            self.train_status.get(half_b) == 'waiting_at_exit'
        )
        
        # Verify both halves are at exit with proper status
        if not halves_at_exit:
            # More detailed logging to diagnose the issue
            logger.error(f"Cannot couple train {train_id} - halves not both at exit or in correct state")
            logger.error(f"Half A location: {self.train_locations.get(half_a)}, state: {self.train_status.get(half_a)}")
            logger.error(f"Half B location: {self.train_locations.get(half_b)}, state: {self.train_status.get(half_b)}")
            
            # Recovery - check if halves exist and fix their states if needed
            for half_id in [half_a, half_b]:
                if half_id in self.train_locations and self.train_locations[half_id] == exit_track:
                    if self.train_status.get(half_id) != 'waiting_at_exit':
                        logger.warning(f"Fixing state for {half_id} to 'waiting_at_exit'")
                        self.train_status[half_id] = 'waiting_at_exit'
            
            # Check if parent train is in the 'departed' state, if so, reset it
            if self.train_status.get(train_id) == 'departed':
                logger.warning(f"Resetting state for {train_id} from 'departed' to 'split_complete'")
                self.train_status[train_id] = 'split_complete'
                
            # Recheck after recovery attempt to see if we can proceed
            halves_at_exit = (
                self.train_locations.get(half_a) == exit_track and
                self.train_locations.get(half_b) == exit_track and
                self.train_status.get(half_a) == 'waiting_at_exit' and
                self.train_status.get(half_b) == 'waiting_at_exit'
            )
            
            if not halves_at_exit:
                # Still can't proceed, so we need to reschedule
                logger.info(f"Rescheduling coupling attempt for {train_id} in 10 minutes")
                heapq.heappush(self.event_queue, (time + 10, 'couple_train', train_id))
                return
        
        # Allow coupling even if the train is already in "coupled" state
        valid_parent_states = ['split_complete', 'waiting_at_entry', 'coupled', None]
        if self.train_status.get(train_id) not in valid_parent_states:
            logger.warning(f"Train {train_id} in invalid state for coupling: {self.train_status.get(train_id)}")
            # Force to a valid state
            self.train_status[train_id] = 'split_complete'
        
        # Now try to update train state
        if not self._update_train_state(train_id, 'coupling'):
            logger.warning(f"Failed to update state for train {train_id} coupling - forcing state")
            self.train_status[train_id] = 'coupling'
        
        # Coupling takes time
        coupling_complete = time + self.movement_times['coupling_time']
        
        # Update track occupation
        self.track_status[exit_track]['occupied_until'] = coupling_complete
        self.track_status[exit_track]['train'] = train_id
        
        # Update train states when coupling complete
        if not self._update_train_state(train_id, 'coupled'):
            logger.warning(f"Failed to update state for train {train_id} coupled - forcing state")
            self.train_status[train_id] = 'coupled'
        
        # Schedule departure
        scheduled_departure = self.trains[train_id]['info']['departure']
        actual_departure = max(coupling_complete, scheduled_departure)
        
        if actual_departure > scheduled_departure:
            delay = actual_departure - scheduled_departure
            self.delayed_departures += 1
            self.total_delay_minutes += delay
            logger.warning(f"Train {train_id} will depart {delay} minutes late")
        
        heapq.heappush(self.event_queue, (actual_departure, 'train_departure', train_id))
        logger.info(f"Train {train_id} coupled at track {exit_track}, departure scheduled for {self._minutes_to_time_str(actual_departure)}")

    
    def _handle_train_departure(self, train_id: str, time: int) -> None:
        """Handle train departure with more robust state checking."""
        exit_track = self.trains[train_id]['exit_track']
        
        # More flexible state check for departure
        if self.train_status.get(train_id) not in ['coupled', 'departed']:
            logger.error(f"Cannot depart train {train_id} - not properly coupled (state: {self.train_status.get(train_id)})")
            
            # Force state if needed for recovery
            if train_id in self.train_locations and self.train_locations[train_id] == exit_track:
                logger.warning(f"Forcing state for {train_id} to 'coupled' for recovery")
                self.train_status[train_id] = 'coupled'
            else:
                # Reschedule departure
                logger.info(f"Rescheduling departure for {train_id} in 10 minutes")
                heapq.heappush(self.event_queue, (time + 10, 'train_departure', train_id))
                return
        
        # Skip if already departed
        if self.train_status.get(train_id) == 'departed':
            logger.warning(f"Train {train_id} already departed, skipping departure event")
            return
        
        # Update train state
        if not self._update_train_state(train_id, 'departed'):
            logger.warning(f"Failed to update state for train {train_id} departing - forcing state")
            self.train_status[train_id] = 'departed'
        
        # Release exit track
        self._release_track(exit_track)
        
        # Increment completed trains counter
        self.completed_trains += 1
        
        # Clean up train data
        self._cleanup_completed_train_data(train_id)
        
        logger.info(f"Train {train_id} departed at {self._minutes_to_time_str(time)}")
    
    def _get_observation(self):
        """Generate enhanced observation vector."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Current time (normalized over 48 hours)
        obs[0] = self.current_time / (48 * 60)
        
        idx = 1
        # Track status (3 features per track: occupied, time until free, reserved)
        all_tracks = self._all_tracks()
        for track in all_tracks:
            status = self.track_status[track]
            # Occupied?
            obs[idx] = 1 if status['occupied_until'] > self.current_time else 0
            idx += 1
            
            # Time until free (normalized over 24 hours)
            time_until_free = max(0, status['occupied_until'] - self.current_time)
            obs[idx] = min(1, time_until_free / (24 * 60))
            idx += 1
            
            # Reserved?
            obs[idx] = 1 if status['reserved_for'] is not None else 0
            idx += 1
        
        # Upcoming trains (next 5)
        upcoming = [t for t in self.timetable if t['arrival'] > self.current_time][:5]
        for i in range(5):
            if i < len(upcoming):
                train = upcoming[i]
                # Time to arrival (normalized over 24 hours)
                obs[idx] = min(1, (train['arrival'] - self.current_time) / (24 * 60))
                # Length (normalized over 1000 units)
                obs[idx+1] = train['length'] / 1000
                # Load time (normalized over 4 hours)
                obs[idx+2] = train['load_time'] / 240
                # Number of load tracks (normalized over 2)
                obs[idx+3] = len(train['load_tracks']) / 2
                # Time to departure (normalized over 24 hours)
                obs[idx+4] = min(1, (train['departure'] - self.current_time) / (24 * 60))
            idx += 5
        
        # Active trains (up to 10) - exclude departed trains and include train halves
        active_trains = [(tid, state) for tid, state in self.train_status.items() 
                        if state != 'departed' and state != 'scheduled'][:10]
        
        for i in range(10):
            if i < len(active_trains):
                train_id, state = active_trains[i]
                
                # Get train info
                base_id = train_id.split('a')[0].split('b')[0]
                is_half = 'a' in train_id or 'b' in train_id
                train_info = self.trains.get(base_id, {}).get('info', {})
                
                # State encoding (one-hot for 12 possible states)
                state_idx = self.TRAIN_STATES.index(state) if state in self.TRAIN_STATES else 0
                state_val = state_idx / len(self.TRAIN_STATES)
                obs[idx] = state_val
                
                # Is train half?
                obs[idx+1] = 1 if is_half else 0
                
                # Location encoding
                location = self.train_locations.get(train_id)
                track_type = 0  # Unknown
                if location in self.entry_exit_tracks:
                    track_type = 1
                elif location in self.parking_tracks:
                    track_type = 2
                elif location in self.loading_tracks:
                    track_type = 3
                obs[idx+2] = track_type / 3
                
                # Departure time relative to current (normalized over 24 hours)
                if 'departure' in train_info:
                    time_to_departure = max(0, train_info['departure'] - self.current_time)
                    obs[idx+3] = min(1, time_to_departure / (24 * 60))
                
                # Length (normalized over 1000 units)
                if 'length' in train_info:
                    length = train_info['length']
                    if is_half:
                        length = length // 2
                    obs[idx+4] = length / 1000
                
                # Loading progress (if in loading state)
                if state == 'loading' and location and 'load_time' in train_info:
                    occupied_time = max(0, self.track_status[location]['occupied_until'] - self.current_time)
                    loading_progress = 1 - (occupied_time / train_info['load_time'])
                    obs[idx+5] = max(0, min(1, loading_progress))
            
            idx += 6
        
        return obs
    
    def reset(self):
        """Reset environment to initial state."""
        # Reset time and event tracking
        self.current_time = 0
        self.track_status = {t: {'occupied_until': 0, 'train': None, 'reserved_for': None} 
                            for t in self._all_tracks()}
        self.trains = {}
        self.train_locations = {}
        self.train_status = {}
        self.event_queue = []
        self.next_arrival_train = None
        
        # Reset performance metrics
        self.delayed_departures = 0
        self.total_delay_minutes = 0
        self.completed_trains = 0
        self.stats = defaultdict(int)
        
        # Reset headway tracking
        self.last_movement_time = {movement_type: 0 for movement_type in self.MOVEMENT_TYPES}
        
        # Initialize event queue with train arrivals
        for train in self.timetable:
            heapq.heappush(self.event_queue, (train['arrival'], 'train_arrival', train['train']))
            # Initialize train state
            self.train_status[train['train']] = 'scheduled'
        
        logger.info("Environment reset")
        return self._get_observation()