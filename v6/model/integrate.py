
import torch
import numpy as np
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from PPO import PPOAgent


CHECKPOINT_PATH = "../checkpoints/v6_model_checkpoint_1020.pth"
MODEL_INPUTS = 20          
NUM_PHASES = 6

YELLOW_DURATION = 10       
PEDESTRIAN_DURATION = 30   
MAX_GREEN_DURATION = 60   
MIN_GREEN_DURATION = 10    

PHASE_ORDER = ["southbound", "brudger", "northbound", "estanislao", "cityhall", "pedestrian"]

PHASE_TO_EDGES = {
    0: ["southbound_entrance"],
    1: ["brudger_entrance"],
    2: ["northbound_entrance"],
    3: ["estanislao_entrance"],
    4: ["cityhall_entrance"],
    5: []
}

# follow array order since its like that in the input layer
# [northbound, southbound, cityhall, brudger, estanislao]
#


def get_vehicle_counts():
    """
    Return a list of vehicle counts for each entering edge
    Order: northbound, southbound, cityhall, brudger, estanislao
    """
    raise NotImplementedError

def get_max_waiting_times():
    """
    Return a list of maximum waiting times (in seconds) for each entering edge
    Same order as get_vehicle_counts()
    """
    raise NotImplementedError

def get_ev_presence():
    """
    Return a list of booleans (or 0/1) indicating whether an emergency vehicle
    is present on each entering edge. Same order as above
    e.g [1,0,0,0,0] this means ev in northbound
    """
    raise NotImplementedError

def get_pedestrian_data():
    """
    Return a tuple (pedestrian_exists: bool, max_pedestrian_wait_time: float)
    like (1,30) peds exist and waiting for 30s
    """
    raise NotImplementedError()

def get_weather():
    """
    Return weather scalar: -1 (clear/dry), 0 (cloudy), 1 (rainy)
    """
    raise NotImplementedError()

def get_day_norm():
    """
    Return day of week normalized to [0,1] (0=Monday, 1=Sunday)
    to get this just divide the day to 7
    """
    raise NotImplementedError()

def get_time_of_day():
    """
    Return time of day normalized to [0,1] (0=midnight, 1=11:59:59 PM)
    same thing just divide the current time in seconds to /24 * 60 * 60 to normalize
    """
    raise NotImplementedError()

def set_traffic_light_phase(phase_index, duration):
    """
    Send command to physical traffic light controller
    phase_index: 0-5 check phase order
    duration: green time in seconds
    """
    raise NotImplementedError()

def set_traffic_light_yellow():
    """Activate yellow light on current phase"""
    raise NotImplementedError()

# this jst represent the worst case time like stadnar traffic light
MAX_CAP_TIME = (MAX_GREEN_DURATION * 5) + (YELLOW_DURATION * 6) + PEDESTRIAN_DURATION

def build_state_vector():

    counts = get_vehicle_counts()
    waits = get_max_waiting_times()
    evs = get_ev_presence()
    ped_exists, ped_wait = get_pedestrian_data()
    weather = get_weather()
    day_norm = get_day_norm()
    time_of_day = get_time_of_day()

    vector = []
    for i in range(5):
        vector.append(counts[i])
        vector.append(waits[i] / MAX_CAP_TIME)
        vector.append(int(evs[i]))

    vector.append(int(ped_exists))
    vector.append(ped_wait / MAX_CAP_TIME)
    vector.append(weather)
    vector.append(day_norm)
    vector.append(time_of_day)

    return np.array(vector, dtype=np.float32)


def get_ev_edge():
    """
    Return the edge ID where an EV is present, or None.
    Edge IDs must match PHASE_TO_EDGES keys.
    """
    evs = get_ev_presence()
    edge_names = ["northbound_entrance", "southbound_entrance", "cityhall_entrance",
                  "brudger_entrance", "estanislao_entrance"]
    for i, ev in enumerate(evs):
        if ev:
            return edge_names[i]
    return None


def main():
    agent = PPOAgent(num_inputs=MODEL_INPUTS, num_phases=NUM_PHASES,
                     lr_actor=0.0003, lr_critic=0.001, gamma=0.99)
    agent.load(CHECKPOINT_PATH)

    current_phase = 0

    while True:
        state = build_state_vector()

        ev_edge = get_ev_edge()

        if ev_edge is not None:
            chosen_phase = None
            for ph, edges in PHASE_TO_EDGES.items():
                if ev_edge in edges:
                    chosen_phase = ph
                    break
            if chosen_phase is None:
                chosen_phase = current_phase
            _, time_percentage = agent.get_deterministic_action(state)
            is_override = True
        else:
            chosen_phase, time_percentage = agent.get_deterministic_action(state)
            is_override = False

        if chosen_phase == 5:
            actual_seconds = PEDESTRIAN_DURATION
        else:
            actual_seconds = int(MIN_GREEN_DURATION +
                                 time_percentage * (MAX_GREEN_DURATION - MIN_GREEN_DURATION))

        print(f"Phase: {PHASE_ORDER[chosen_phase]}, Green: {actual_seconds}s, Override: {is_override}")

        if chosen_phase != current_phase:
            set_traffic_light_yellow()
            time.sleep(YELLOW_DURATION)
            current_phase = chosen_phase

        set_traffic_light_phase(chosen_phase, actual_seconds)
        time.sleep(actual_seconds)


if __name__ == "__main__":
    main()