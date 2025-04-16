import gym
import numpy as np
from gym import spaces
import heapq
#https://github.com/Umaraslam66/cl-rl
class TrainYardEnv(gym.Env):
    """Gym environment for train yard operations with train splitting, loading, and coupling."""
    
    def __init__(self):
        super(TrainYardEnv, self).__init__()
        
        # Load track and timetable data
        self.entry_exit_tracks = self._parse_tracks("32:750,33:750,34:750,35:750,36:670,m3:450,m4:450,m5:550,m6:600,m7:750,m8:750,m9:750,m10:750,m11:900,m12:850,m13:800")
        self.parking_tracks = self._parse_tracks("51:665,52:665,53:741,54:746,55:752")
        self.loading_tracks = self._parse_tracks("56A:571,56B:161,57:787,54:746,71:303,41:487,42:487,43:313,44:313,4x:313,29:242,30:237,20:500")
        self.timetable = self._load_timetable()
        
        # State tracking
        self.current_time = 0
        self.track_status = {t: {'occupied_until': 0, 'train': None} for t in list(self.entry_exit_tracks) + list(self.parking_tracks) + list(self.loading_tracks)}
        self.trains = {}
        self.train_locations = {}
        self.train_status = {}
        self.event_queue = []
        self.next_arrival_train = None
        self.max_departure_time = max([t['departure'] for t in self.timetable])
        
        # Action and observation spaces
        entry_exit_tracks = list(self.entry_exit_tracks.keys())
        self.action_space = spaces.Dict({
            'entry_track': spaces.Discrete(len(entry_exit_tracks)),
            'exit_track': spaces.Discrete(len(entry_exit_tracks)),
            'wait_time': spaces.Discrete(60)  # Wait up to 60 minutes
        })
        
        # Observation includes current time, track status, and upcoming trains
        obs_dim = 1 + len(self.track_status)*2 + 5*5  # Current time + tracks + 5 upcoming trains with 5 features each
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize event queue with train arrivals
        for train in self.timetable:
            heapq.heappush(self.event_queue, (train['arrival'], 'train_arrival', train['train']))
    
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
    
    def step(self, action):
        """Take action and advance simulation until next decision point."""
        # Process action (selecting tracks and wait time)
        track_list = list(self.entry_exit_tracks.keys())
        entry_track = track_list[action['entry_track']]
        exit_track = track_list[action['exit_track']]
        wait_time = action['wait_time']
        
        # Assign tracks to next arriving train
        if self.next_arrival_train:
            train_info = next(t for t in self.timetable if t['train'] == self.next_arrival_train)
            self.trains[self.next_arrival_train] = {
                'info': train_info,
                'entry_track': entry_track,
                'exit_track': exit_track,
                'wait_time': wait_time
            }
            
            # Check if track is long enough and schedule train entry
            if self.entry_exit_tracks[entry_track] >= train_info['length']:
                heapq.heappush(self.event_queue, (self.current_time, 'train_entry', self.next_arrival_train))
            
            self.next_arrival_train = None
        
        # Process events until next decision needed
        done = self._process_events()
        
        # Calculate reward (minimize track usage)
        used_tracks = sum(1 for status in self.track_status.values() 
                         if status['occupied_until'] > self.current_time)
        completed_trains = sum(1 for status in self.train_status.values() 
                              if status == 'departed')
        
        reward = -used_tracks + (completed_trains * 10)
        
        return self._get_observation(), reward, done, {
            'current_time': self.current_time,
            'time_str': self._minutes_to_time_str(self.current_time),
            'tracks_used': used_tracks,
            'trains_completed': completed_trains
        }
    
    def _process_events(self):
        """Process events until next decision required."""
        # Safety check - if no events and no arriving train, we're done
        if not self.event_queue and not self.next_arrival_train:
            # Make sure all trains have been processed
            if all(status == 'departed' for train, status in self.train_status.items() 
                   if not (train.endswith('a') or train.endswith('b'))):
                return True
        
        max_process_iterations = 1000  # Safety to prevent infinite loops
        iterations = 0
        
        while self.event_queue and not self.next_arrival_train and iterations < max_process_iterations:
            iterations += 1
            time, event_type, train_id = heapq.heappop(self.event_queue)
            self.current_time = time
            
            if event_type == 'train_arrival':
                self.train_status[train_id] = 'arriving'
                self.next_arrival_train = train_id
                return False
            
            elif event_type == 'train_entry':
                # Train enters selected track and begins splitting
                entry_track = self.trains[train_id]['entry_track']
                self.track_status[entry_track]['occupied_until'] = time + 15  # Split time
                self.track_status[entry_track]['train'] = train_id
                self.train_locations[train_id] = entry_track
                self.train_status[train_id] = 'splitting'
                heapq.heappush(self.event_queue, (time + 15, 'train_split', train_id))
            
            elif event_type == 'train_split':
                # Train is split into two halves
                self._handle_train_split(train_id, time)
            
            elif event_type == 'half_to_loading':
                # Half train arrives at loading track
                self._handle_half_to_loading(train_id, time)
            
            elif event_type == 'loading_complete':
                # Half train completes loading
                self._handle_loading_complete(train_id, time)
            
            elif event_type == 'half_to_exit':
                # Half train arrives at exit track
                self._handle_half_to_exit(train_id, time)
            
            elif event_type == 'couple_train':
                # Couple train halves together
                self._handle_couple_train(train_id, time)
            
            elif event_type == 'train_departure':
                # Train departs
                self._handle_train_departure(train_id, time)
        
        # If we've reached max iterations, log an error
        if iterations >= max_process_iterations:
            print(f"WARNING: Max iterations reached in event processing. Current time: {self._minutes_to_time_str(self.current_time)}")
            # Force completion to avoid infinite loops
            return True
        
        # Check if simulation is done (all trains departed or current time > max departure)
        if self.current_time > self.max_departure_time + 60:  # Add buffer
            return True
            
        return False
    
    def _handle_train_split(self, train_id, time):
        """Handle train splitting into two halves."""
        train_info = self.trains[train_id]['info']
        load_tracks = train_info['load_tracks']
        wait_time = self.trains[train_id]['wait_time']
        
        # Update entry track
        entry_track = self.trains[train_id]['entry_track']
        self.track_status[entry_track]['occupied_until'] = time + wait_time
        
        # Create train halves
        self.train_status[f"{train_id}a"] = 'waiting_at_entry'
        self.train_status[f"{train_id}b"] = 'waiting_at_entry'
        
        # First half to loading track
        if load_tracks:
            first_load = load_tracks[0]
            if self.track_status[first_load]['occupied_until'] <= time:
                self.train_status[f"{train_id}a"] = 'moving_to_loading'
                self.track_status[first_load]['occupied_until'] = time + 15 + train_info['load_time']
                self.track_status[first_load]['train'] = f"{train_id}a"
                heapq.heappush(self.event_queue, (time + 15, 'half_to_loading', f"{train_id}a"))
        
        # Second half to loading or wait
        if len(load_tracks) >= 2:
            second_load = load_tracks[1]
            if self.track_status[second_load]['occupied_until'] <= time:
                self.train_status[f"{train_id}b"] = 'moving_to_loading'
                self.track_status[second_load]['occupied_until'] = time + 15 + train_info['load_time']
                self.track_status[second_load]['train'] = f"{train_id}b"
                heapq.heappush(self.event_queue, (time + 15, 'half_to_loading', f"{train_id}b"))
    
    def _handle_half_to_loading(self, half_id, time):
        """Handle half train arriving at loading track."""
        base_id = half_id[:-1]  # Remove 'a' or 'b'
        
        # Find loading track
        loading_track = next((t for t, s in self.track_status.items() 
                             if s['train'] == half_id), None)
        
        if loading_track:
            self.train_locations[half_id] = loading_track
            self.train_status[half_id] = 'loading'
            
            # Schedule loading completion
            load_time = self.trains[base_id]['info']['load_time']
            heapq.heappush(self.event_queue, (time + load_time, 'loading_complete', half_id))
    
    def _handle_loading_complete(self, half_id, time):
        """Handle half train completing loading."""
        base_id = half_id[:-1]
        exit_track = self.trains[base_id]['exit_track']
        
        # Find current loading track
        loading_track = self.train_locations[half_id]
        
        # Move to exit if available
        if self.track_status[exit_track]['occupied_until'] <= time:
            self.track_status[loading_track]['occupied_until'] = time
            self.track_status[loading_track]['train'] = None
            self.track_status[exit_track]['occupied_until'] = time + 15
            self.track_status[exit_track]['train'] = half_id
            
            self.train_status[half_id] = 'moving_to_exit'
            heapq.heappush(self.event_queue, (time + 15, 'half_to_exit', half_id))
            
            # If first half is done, second half can start loading
            if half_id.endswith('a') and self.train_status.get(f"{base_id}b") == 'waiting_at_entry':
                second_half = f"{base_id}b"
                train_info = self.trains[base_id]['info']
                
                if len(train_info['load_tracks']) > 0:
                    loading_track = train_info['load_tracks'][0]  # Use first available
                    
                    self.train_status[second_half] = 'moving_to_loading'
                    self.track_status[loading_track]['occupied_until'] = time + 15 + train_info['load_time']
                    self.track_status[loading_track]['train'] = second_half
                    heapq.heappush(self.event_queue, (time + 15, 'half_to_loading', second_half))
    
    def _handle_half_to_exit(self, half_id, time):
        """Handle half train arriving at exit track."""
        base_id = half_id[:-1]
        
        self.train_locations[half_id] = self.trains[base_id]['exit_track']
        self.train_status[half_id] = 'waiting_at_exit'
        
        # Check if both halves are at exit
        other_half = f"{base_id}{'b' if half_id.endswith('a') else 'a'}"
        if self.train_status.get(other_half) == 'waiting_at_exit':
            heapq.heappush(self.event_queue, (time, 'couple_train', base_id))
    
    def _handle_couple_train(self, train_id, time):
        """Handle coupling of train halves."""
        exit_track = self.trains[train_id]['exit_track']
        
        # Coupling takes 15 minutes
        self.track_status[exit_track]['occupied_until'] = time + 15
        self.track_status[exit_track]['train'] = train_id
        
        self.train_status[train_id] = 'coupled'
        # Keep the half train statuses for debugging
        
        # Schedule departure
        departure_time = self.trains[train_id]['info']['departure']
        heapq.heappush(self.event_queue, (max(time + 15, departure_time), 'train_departure', train_id))
    
    def _handle_train_departure(self, train_id, time):
        """Handle train departure."""
        exit_track = self.trains[train_id]['exit_track']
        
        # Release exit track
        self.track_status[exit_track]['occupied_until'] = time
        self.track_status[exit_track]['train'] = None
        
        self.train_status[train_id] = 'departed'
    
    def _get_observation(self):
        """Generate observation vector."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Current time (normalized)
        obs[0] = self.current_time / (48 * 60)  # Two days max
        
        idx = 1
        # Track status
        for track, status in self.track_status.items():
            obs[idx] = 1 if status['occupied_until'] > self.current_time else 0  # Occupied?
            idx += 1
            # Time until free (normalized)
            time_until_free = max(0, status['occupied_until'] - self.current_time)
            obs[idx] = min(1, time_until_free / (24 * 60))
            idx += 1
        
        # Upcoming trains (next 5)
        upcoming = [t for t in self.timetable if t['arrival'] > self.current_time][:5]
        for i in range(5):
            if i < len(upcoming):
                train = upcoming[i]
                obs[idx] = min(1, (train['arrival'] - self.current_time) / (24 * 60))  # Time to arrival
                obs[idx+1] = train['length'] / 1000  # Length
                obs[idx+2] = train['load_time'] / 240  # Load time
                obs[idx+3] = len(train['load_tracks']) / 2  # Number of load tracks
                obs[idx+4] = min(1, (train['departure'] - self.current_time) / (24 * 60))  # Time to departure
            idx += 5
        
        return obs
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_time = 0
        self.track_status = {t: {'occupied_until': 0, 'train': None} 
                            for t in list(self.entry_exit_tracks) + list(self.parking_tracks) + list(self.loading_tracks)}
        self.trains = {}
        self.train_locations = {}
        self.train_status = {}
        self.event_queue = []
        self.next_arrival_train = None
        
        # Initialize event queue with train arrivals
        for train in self.timetable:
            heapq.heappush(self.event_queue, (train['arrival'], 'train_arrival', train['train']))
        
        return self._get_observation()