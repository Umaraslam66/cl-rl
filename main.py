"""Quick sanity‑check script for the updated TrainYard environment.

This intentionally keeps things minimal:
1. Resets the environment
2. Steps a few times with deliberately conflicting
   actions (trying to use the *same* entry/exit track for sequential trains)
3. Prints out observation slices and info dict so you can verify that the
   `blocked_flag` and `blocked_track_norm` elements behave as expected.

Run with:  python test_env.py
"""

import numpy as np
from env_g import TrainYardEnv

# Helper to pretty‑print observation details ---------------------------------

# def decode_blocked_info(obs):
#     """Return (blocked_flag, blocked_track_index) decoded from observation."""
#     blocked_flag = obs[0]  # after env rewrite this is at the *end* so adapt if needed
#     blocked_track_norm = obs[1]
#     return blocked_flag, blocked_track_norm


# def main():
#     env = TrainYardEnv(verbose=False)
#     obs = env.reset()
#     print("Environment reset – starting simulation\n")

#     # Artificial agent: always demands track‑0 for both entry & exit
#     # to maximise likelihood of a clash when consecutive trains arrive.
#     fixed_track = 0  # index into entry_exit_tracks list
#     done = False
#     step_no = 0

#     while not done and step_no < 20:
#         action = {
#             "entry_track": fixed_track,
#             "exit_track": fixed_track,
#             "wait_time": 0
#         }
#         obs, reward, done, info = env.step(action)
#         bflag, btrack = decode_blocked_info(obs[-2:])  # last two elements
#         print(f"Step {step_no:02d}: time={info['time_str']}, reward={reward:.2f}, "
#               f"blocked={bool(bflag)}, norm_track={btrack:.2f}")
#         step_no += 1

#     print("\nSimulation finished – completed trains:", info.get("trains_completed"))


# if __name__ == "__main__":
#     main()



"""test_env.py – sanity test for TrainYardEnv
------------------------------------------------
* Cycles through **all** entry tracks instead of always picking index 0.
* Uses the next track (mod N) as exit so entry ≠ exit.
* Alternates `wait_time` 0 / 5 min to exercise that parameter.
* Prints blocked‑flag info that lives in the **last two** obs slots.
Run with:
    python test_env.py
"""
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def decode_blocked_info(obs):
    """Return (flag, norm_idx) from the last two obs slots."""
    if len(obs) < 2:
        return False, 0.0
    return bool(obs[-2]), obs[-1]


def choose_action(step_no: int, n_tracks: int):
    """Round‑robin entry; next track as exit; alternate wait time."""
    entry = step_no % n_tracks
    exit_ = (entry + 1) % n_tracks
    wait  = 5 if step_no % 2 else 0
    return {"entry_track": entry, "exit_track": exit_, "wait_time": wait}

# ------------------------------------------------- main loop ---------------

def main(max_steps: int = 40):
    env = TrainYardEnv(verbose=False)
    obs = env.reset()
    n_tracks = len(env.entry_exit_tracks)

    print("Environment reset – starting simulation\n")

    for step in range(max_steps):
        action = choose_action(step, n_tracks)
        obs, reward, done, info = env.step(action)

        blocked_flag, blocked_norm = decode_blocked_info(obs)
        print(
            f"Step {step:02d}: time={info.get('time_str','?')}, "
            f"reward={reward:.2f}, blocked={blocked_flag}, blocked_norm={blocked_norm:.2f}, "
            f"completed={info.get('trains_completed','?')}, tracks_used={info.get('tracks_used','?')}"
        )

        if done:
            print("\nSimulation finished – completed trains:", info.get('trains_completed','?'))
            break

    env.close()


if __name__ == "__main__":
    main()

