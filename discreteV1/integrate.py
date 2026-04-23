import time
import numpy as np
from stable_baselines3 import PPO

CHECKPOINT_PATH = "ppo_traffic_final.zip"

PHASES = ["southbound", "brudger", "northbound", "estanislao", "cityhall", "pedestrian"]
DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

MAX_GREEN_DURATION = 60
YELLOW_DURATION = 10
PEDESTRIAN_DURATION = 30
MAX_CAP_TIME = (
    (MAX_GREEN_DURATION * 5) +    
    (YELLOW_DURATION * 6) +      
    PEDESTRIAN_DURATION
)



MAX_EDGE_CAPACITY = {
    "northbound_entrance": 30,
    "southbound_entrance": 30,
    "cityhall_exit": 5,
    "brudger_exit": 15,
    "estanislao_exit": 15
}


def get_lane_data(lane_name):
    """
    Reads hardware sensors (cameras/induction loops) for a specific lane.
    Returns: (vehicle_count: int, max_waiting_time_seconds: float)
    """
    raise NotImplementedError()

def get_pedestrian_data():
    """
    Reads crosswalk button states or pedestrian cameras.
    Returns: (pedestrian_waiting: bool, max_waiting_time_seconds: float)
    """
    raise NotImplementedError()

def get_weather():
    """
    Reads a Weather API or local environmental sensor.
    Returns float: -1.0 (Clear), 0.0 (Cloudy), 1.0 (Rainy)
    """
    raise NotImplementedError()

def set_traffic_light_yellow():
    """
    Hardware command: Set the currently active phase to YELLOW.
    """
    raise NotImplementedError()

def set_traffic_light_phase(phase_index):
    """
    Hardware command: Set the target phase to GREEN, all others to RED.
    phase_index: 0=southbound, 1=brudger, 2=northbound, 3=estanislao, 4=cityhall, 5=pedestrian
    """
    raise NotImplementedError()


def build_state_vector():
    """
    Constructs the exact 13-element observation array expected by the trained SB3 model.
    """
    vector = []
    
    edge_order = ["northbound", "southbound", "cityhall", "brudger", "estanislao"]
    
    for edge in edge_order:
        count, wait = get_lane_data(edge)
        
        normalized_count = (count / MAX_EDGE_CAPACITY[edge])
        normalized_wait = (wait / MAX_CAP_TIME)
        
        vector.append(normalized_count)
        vector.append(normalized_wait)
        
    ped_exists, ped_wait = get_pedestrian_data()
    vector.append(1.0 if ped_exists else 0.0)
    vector.append(min(1.0, ped_wait / MAX_CAP_TIME))
    
    weather_val = get_weather()
    vector.append(float(weather_val))
    
    return np.array(vector, dtype=np.float32)



def main():
    print(f"Loading trained AI model from {CHECKPOINT_PATH}...")
    model = PPO.load(CHECKPOINT_PATH)
    print("Model loaded successfully. Starting hardware control loop.")

    current_phase = 2
    set_traffic_light_phase(current_phase)

    while True:
        try:
            state = build_state_vector()

            action, _states = model.predict(state, deterministic=True)
            
            phase_idx, duration_idx = action
            chosen_phase_name = PHASES[int(phase_idx)]
            
            if chosen_phase_name == "pedestrian":
                actual_seconds = PEDESTRIAN_DURATION
            else:
                actual_seconds = DURATIONS[int(duration_idx)]

            print(f"AI Decision: {chosen_phase_name.upper()} for {actual_seconds} seconds.")

            if phase_idx != current_phase:
                print(f"Switching to YELLOW for {YELLOW_DURATION}s")
                set_traffic_light_yellow()
                time.sleep(YELLOW_DURATION)
                current_phase = int(phase_idx)

            print(f"Switching {chosen_phase_name.upper()} to GREEN")
            set_traffic_light_phase(current_phase)
            
            time.sleep(actual_seconds)
            
        except KeyboardInterrupt:
            print("\nShutting down AI traffic controller.")
            break
        except Exception as e:
            print(f"CRITICAL SYSTEM ERROR: {e}")
            print("Falling back to fixed-timer emergency mode...")
            time.sleep(10) 

if __name__ == "__main__":
    main()