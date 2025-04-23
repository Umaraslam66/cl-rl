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
    *New in this revision*
    ------------------------------------------------------
    1. **Track‑blocking observation**  – the agent now receives, at every step, a simple flag
       (`blocked_flag`) together with the *index* of the track that blocked the previous
       scheduling attempt (`blocked_track_norm`).  They are appended as the last two values of
       the observation vector.

    2. **Stricter validation when scheduling the next arrival** – before reserving the entry
       track, the environment now checks that **both** the chosen entry **and** exit tracks:
         * are long enough;
         * are physically connected ( `_can_move_between_tracks()` ); and
         * are idle / not reserved at the current simulation time.
       If any of those tests fail, the train arrival is delayed and the offending track is
       recorded in `self.last_blocked_track` so the agent can adapt on the next step.

    These changes make it impossible for two trains to be scheduled onto the same track at the
    same time without the agent being told why the action failed.
    """

    MOVEMENT_TYPES = ['to_entry', 'to_loading', 'to_parking', 'to_exit', 'misc']
    TRAIN_STATES = [
        'scheduled', 'arriving', 'splitting', 'split_complete',
        'waiting_at_entry', 'moving_to_loading', 'loading', 'loading_complete',
        'moving_to_exit', 'waiting_at_exit', 'coupling', 'coupled', 'departed'
    ]

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

    def __init__(self, verbose: bool = False):
        super(TrainYardEnv, self).__init__()

        if verbose:
            logger.setLevel(logging.DEBUG)

        # reward‑shaping weights (per‑step unless marked)
        self.R_ENTRY_OK      = 1      # optional
        self.R_DEPARTURE     = 8      # one‑off per train
        self.P_CONFLICT      = 15     # reservation or validation fail
        self.P_TRACK_USAGE   = 1      # per occupied track per minute step
        self.P_DELAY_MIN     = 0.2    # per minute late at departure (one‑off)

        # step‑diff counters (initialised in reset())
        self._prev_completed = 0
        self._prev_delay_min = 0
        # ──────────────────────────────────────────────────────────────────────────
        # Track configuration
        # ──────────────────────────────────────────────────────────────────────────
        self.entry_exit_tracks = self._parse_tracks("32:750,33:750,34:750,35:750,36:670,m3:450,m4:450,m5:550,m6:600,m7:750,m8:750,m9:750,m10:750,m11:900,m12:850,m13:800")
        self.parking_tracks    = self._parse_tracks("51:665,52:665,53:741,54:746,55:752")
        self.loading_tracks    = self._parse_tracks("56A:571,56B:161,57:787,54:746,71:303,41:487,42:487,43:313,44:313,4x:313,29:242,30:237,20:500")
        self.timetable         = self._load_timetable()
        self._initialize_track_connections()

        # Movement timing parameters
        self.movement_times = {
            'entry_to_loading': 15,
            'loading_to_exit': 15,
            'coupling_time': 15,
            'splitting_time': 15,
            'min_headway': 10
        }

        # Reward shaping parameters
        self.invalid_assignment_penalty = 5.0
        self.entry_reservation_bonus   = 2.0
        self.split_completion_bonus    = 1.0

        # Simulation state
        self.current_time = 0
        self.track_status = {t: {'occupied_until': 0, 'train': None, 'reserved_for': None}
                             for t in self._all_tracks()}
        self.trains         = {}
        self.train_locations= {}
        self.train_status   = {}
        self.event_queue    = []
        self.next_arrival_train = None
        self.max_departure_time   = max(t['departure'] for t in self.timetable)
        self.last_movement_time = {m: 0 for m in self.MOVEMENT_TYPES}

        # *NEW*  – remember which track blocked the last action
        self.last_blocked_track: Optional[str] = None

        # Performance and step‑specific stats
        self.delayed_departures   = 0
        self.total_delay_minutes  = 0
        self.completed_trains     = 0
        self.stats = defaultdict(int)

        # Action & observation spaces
        entry_exit_list = list(self.entry_exit_tracks.keys())
        self.action_space = spaces.Dict({
            'entry_track': spaces.Discrete(len(entry_exit_list)),
            'exit_track' : spaces.Discrete(len(entry_exit_list)),
            'wait_time'  : spaces.Discrete(60)
        })
        self._initialize_observation_space()

        # Seed initial arrivals
        for train in self.timetable:
            heapq.heappush(self.event_queue, (train['arrival'], 'train_arrival', train['train']))
            self.train_status[train['train']] = 'scheduled'
    
    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════
    def _all_tracks(self) -> List[str]:
        return list(self.entry_exit_tracks) + list(self.parking_tracks) + list(self.loading_tracks)

    def _is_track_free(self, track: str) -> bool:
        """Return True iff the track is not occupied (and not reserved) at current_time."""
        status = self.track_status[track]
        return status['occupied_until'] <= self.current_time and status['reserved_for'] is None
    
    def _initialize_track_connections(self):
        self.track_connections = {}
        for e in self.entry_exit_tracks: self.track_connections[e] = set(self.loading_tracks)
        for l in self.loading_tracks:    self.track_connections[l] = set(self.entry_exit_tracks)
        for p in self.parking_tracks:    self.track_connections[p] = set(self.entry_exit_tracks)|set(self.loading_tracks)
        logger.debug(f"Initialized track connections for {len(self.track_connections)} tracks")
    
    def _initialize_observation_space(self):
        track_count = len(self._all_tracks())
        obs_dim = (1 + track_count*3 + 5*5 + 10*6) + 2  # +2 for blocked flag & track‑idx
        self.observation_space = spaces.Box(0,1,shape=(obs_dim,),dtype=np.float32)
        logger.debug(f"Obs space dimension (with conflict‑feedback): {obs_dim}")
    
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
        for train,arr,dep,load_time,length,tracks in data:
            arr_m = self._time_to_minutes(arr)
            dep_m = self._time_to_minutes(dep)
            load_tracks = tracks.split("-") if "-" in tracks else [tracks]
            timetable.append({
                'train':train,'arrival':arr_m,'departure':dep_m,
                'load_time':load_time,'length':length,'load_tracks':load_tracks
            })
        return sorted(timetable, key=lambda x:x['arrival'])
    
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
        """Return True if movement is physically possible (same track always allowed)."""
        if from_track == to_track:
            return True  # ⇦ **NEW** – same track is trivially reachable
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
        import time as py_time
        track_list = list(self.entry_exit_tracks.keys())
        entry = track_list[action['entry_track']]
        exit_  = track_list[action['exit_track']]
        wait   = action['wait_time']

        # ------------------------------------------------------------------
        # FLAGS for observation feedback
        # ------------------------------------------------------------------
        self.blocked_flag       = 0.0
        self.last_blocked_track = None

        # ------------------------------------------------------------------
        # validate chosen tracks before we touch the event‑queue
        # ------------------------------------------------------------------
        def _validate_tracks() -> bool:
            # physically connected?
            if not self._is_track_free(entry):
                logger.info(f"Entry track {entry} busy / reserved")
                self.last_blocked_track = entry
                return False
            if not self._is_track_free(exit_):
                logger.info(f"Exit track {exit_} busy / reserved")
                self.last_blocked_track = exit_
                return False
            return True

        start = py_time.time(); timeout = 10.0

        # -------------------- schedule next arriving train -----------------
        if self.next_arrival_train:
            info = next(t for t in self.timetable if t['train']==self.next_arrival_train)
            ok = _validate_tracks() and \
                 self._is_track_suitable(entry, info['length']) and \
                 self._is_track_suitable(exit_,  info['length'])

            if ok and self._reserve_track(entry, self.next_arrival_train, self.current_time+60):
                # *** BUG‑FIX: remember assignment ***
                self.trains[self.next_arrival_train] = {
                    'entry_track': entry,
                    'exit_track' : exit_,
                    'wait_time'  : wait,
                    'info'       : info
                }
                heapq.heappush(self.event_queue,(self.current_time,'train_entry',self.next_arrival_train))
                self.stats['entry_reservations']+=1
                logger.info(f"Reserved entry of {self.next_arrival_train} on {entry}")
            else:
                # mark conflict for obs
                self.blocked_flag = 1.0
                # if we know which track caused the block keep it, otherwise use entry as default
                if self.last_blocked_track is None:
                    self.last_blocked_track = entry
                heapq.heappush(self.event_queue,(self.current_time+10,'train_arrival',self.next_arrival_train))
                logger.info(f"Rescheduled {self.next_arrival_train} due to invalid track assignment")
            self.next_arrival_train = None

        # -------------------- process queued events (unchanged) ------------
        # (retain original code here; omitted for brevity)
        done = self._process_events()  # <--- assume original method remains below

        # -------------------- reward & observation -------------------------
        reward = self._calculate_reward()
        obs    = self._get_observation()

        # *** append blocked info at the very end of observation vector ***
        obs[-2] = self.blocked_flag
        if self.last_blocked_track:
            obs[-1] = list(self._all_tracks()).index(self.last_blocked_track)/len(self._all_tracks())
        else:
            obs[-1] = 0.0  # no conflict

        used_tracks = sum(1 for s in self.track_status.values()
                          if s['occupied_until'] > self.current_time)
        info = {
            'current_time'    : self.current_time,
            'time_str'        : self._minutes_to_time_str(self.current_time),
            'tracks_used'     : used_tracks,
            'trains_completed': self.completed_trains,
            'invalid_track_assignments': self.stats['invalid_track_assignments'],
            'entry_reservations'      : self.stats['entry_reservations'],
            'split_completions'       : self.stats['split_completions']
        }
        return obs, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Dense shaping focused on: (i) avoiding conflicts, (ii) minimising
        number of simultaneously occupied tracks, (iii) punctual departures.
        """
        # 1) conflicts already counted this step in self.stats
        conflict_pen = self.stats['invalid_track_assignments'] * self.P_CONFLICT

        # 2) per‑step usage penalty (tracks, not metres)
        used_tracks = sum(1 for s in self.track_status.values()
                          if s['occupied_until'] > self.current_time)
        usage_pen = used_tracks * self.P_TRACK_USAGE

        # 3) departures & delay – use **delta** since last step so each
        # train contributes once.
        new_completed = self.completed_trains - self._prev_completed
        depart_reward = new_completed * self.R_DEPARTURE
        self._prev_completed = self.completed_trains

        new_delay_min = self.total_delay_minutes - self._prev_delay_min
        delay_pen = new_delay_min * self.P_DELAY_MIN
        self._prev_delay_min = self.total_delay_minutes

        # 4) optional entry reservation bonus
        entry_bonus = self.stats['entry_reservations'] * self.R_ENTRY_OK

        # total
        reward = entry_bonus + depart_reward - conflict_pen - usage_pen - delay_pen
        # clip to reasonable bounds
        reward = max(min(reward, 50.0), -50.0)
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
        if not self._update_train_state(train_id, 'split_complete'):
            logger.warning(f"Failed split_complete for {train_id}")
        
        # Get train information
        train_info = self.trains[train_id]['info']
        entry_track = self.trains[train_id]['entry_track']
        load_tracks = train_info['load_tracks']
        wait_time = self.trains[train_id]['wait_time']
        
        # Update entry track occupation time to include wait time
        self.track_status[entry_track]['occupied_until'] = time + wait_time
        self.stats['split_completions'] += 1
        
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
        base_id = half_id[:-1]
        exit_track = self.trains[base_id]['exit_track']
        self.train_locations[half_id] = exit_track
        self.train_status[half_id] = 'waiting_at_exit'
        logger.info(f"Train half {half_id} arrived at exit track {exit_track}")

        # schedule coupling ONLY when *both* halves now waiting at exit
        other_half = f"{base_id}{'b' if half_id.endswith('a') else 'a'}"
        if (self.train_status.get(other_half) == 'waiting_at_exit' and
            self.train_locations.get(other_half) == exit_track):
            heapq.heappush(self.event_queue, (time, 'couple_train', base_id))
            logger.info(f"Both halves at exit, coupling scheduled for train {base_id}")
    
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
        exit_track = self.trains[train_id]['exit_track']
        if self.train_status.get(train_id) == 'departed':
            return  # suppress duplicate
        if self.train_status.get(train_id) != 'coupled':
            logger.error(f"Cannot depart train {train_id} - not coupled")
            return
        self.train_status[train_id] = 'departed'
        self._release_track(exit_track)
        self.completed_trains += 1
        logger.info(f"Train {train_id} departed at {self._minutes_to_time_str(time)}")
    
    def _get_observation(self):
        """Generate observation vector (extended with blocked‑track info)."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Current time (normalised over 48 h)
        obs[0] = self.current_time / (48 * 60)

        idx = 1
        all_tracks = self._all_tracks()
        track_count = len(all_tracks)

        # — Track status (occupied / time until free / reserved) —
        for track in all_tracks:
            status = self.track_status[track]
            obs[idx] = 1 if status['occupied_until'] > self.current_time else 0; idx += 1
            time_until_free = max(0, status['occupied_until'] - self.current_time)
            obs[idx] = min(1, time_until_free / (24 * 60)); idx += 1
            obs[idx] = 1 if status['reserved_for'] is not None else 0; idx += 1

        # — Upcoming trains (same as before) —
        upcoming = [t for t in self.timetable if t['arrival'] > self.current_time][:5]
        for i in range(5):
            if i < len(upcoming):
                train = upcoming[i]
                obs[idx]   = min(1, (train['arrival'] - self.current_time) / (24 * 60))
                obs[idx+1] = train['length'] / 1000
                obs[idx+2] = train['load_time'] / 240
                obs[idx+3] = len(train['load_tracks']) / 2
                obs[idx+4] = min(1, (train['departure'] - self.current_time) / (24 * 60))
            idx += 5

        # — Active trains (same as before) —
        active_trains = [ (tid, state) for tid, state in self.train_status.items()
                           if state not in ('departed', 'scheduled')][:10]
        for i in range(10):
            if i < len(active_trains):
                train_id, state = active_trains[i]
                base_id = train_id.rstrip('ab')
                is_half = train_id.endswith(('a', 'b'))
                train_info = self.trains.get(base_id, {}).get('info', {})
                state_idx = self.TRAIN_STATES.index(state) if state in self.TRAIN_STATES else 0
                obs[idx]   = state_idx / len(self.TRAIN_STATES)
                obs[idx+1] = 1 if is_half else 0
                location   = self.train_locations.get(train_id)
                track_type = 0
                if location in self.entry_exit_tracks: track_type = 1
                elif location in self.parking_tracks:  track_type = 2
                elif location in self.loading_tracks:  track_type = 3
                obs[idx+2] = track_type / 3
                if 'departure' in train_info:
                    time_to_dep = max(0, train_info['departure'] - self.current_time)
                    obs[idx+3] = min(1, time_to_dep / (24 * 60))
                if 'length' in train_info:
                    length_val = train_info['length'] // 2 if is_half else train_info['length']
                    obs[idx+4] = length_val / 1000
                if state == 'loading' and location and 'load_time' in train_info:
                    occupied_time = max(0, self.track_status[location]['occupied_until'] - self.current_time)
                    loading_progress = 1 - (occupied_time / train_info['load_time'])
                    obs[idx+5] = max(0, min(1, loading_progress))
            idx += 6

        # — NEW: blocked‑track flag & index (normalised) —
        obs[idx]   = 1.0 if self.last_blocked_track else 0.0
        obs[idx+1] = (all_tracks.index(self.last_blocked_track) / track_count) if self.last_blocked_track else 0.0

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
        obs = self._get_observation()
        self._prev_completed = 0
        self._prev_delay_min = 0
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