import traci
import numpy as np

TL_ID = "main_tl"  

ENTERING_EDGES = [
    "northbound_entrance",
    "southbound_entrance",
    "cityhall_entrance",
    "brudger_entrance",
    "estanislao_entrance"
]

DETECTORS = [
    "northbound_detector_2",
    "northbound_detector_1",
    "southbound_detector_2",
    "southbound_detector_1",
    "cityhall_detector",
    "brudger_detector",
    "estanislao_detector"
]

PHASES = {
    "southbound": 0,
    "brudger":    1,
    "northbound": 2,
    "estanislao": 3,
    "cityhall":   4,
    "pedestrian": 5,
}

NUM_AGENT_PHASES = 6

EDGE_DETECTORS = {
    "northbound_entrance": ["northbound_detector_2", "northbound_detector_1"],
    "southbound_entrance": ["southbound_detector_2", "southbound_detector_1"],
    "cityhall_entrance":   ["cityhall_detector"],
    "brudger_entrance":    ["brudger_detector"],
    "estanislao_entrance": ["estanislao_detector"],
}

YELLOW_DURATION = 10
RED_DURATION = 3
PEDESTRIAN_DURATION = 30
MAX_GREEN_DURATION = 60   
MIN_GREEN_DURATION = 10

MAX_CAP_TIME = (
    (MAX_GREEN_DURATION * 5) +    
    (YELLOW_DURATION * 6) +      
    (RED_DURATION * 6) + 
    PEDESTRIAN_DURATION
)


    
def get_lane_state(edge_id):
    detectors = EDGE_DETECTORS[edge_id]
    
    car_ids = []
    for det in detectors:
        car_ids += traci.lanearea.getLastStepVehicleIDs(det)
    
    car_count = len(car_ids)
    max_waiting_time = 0.0
    ev_exists = False
    
    for car_id in car_ids:
        wait = traci.vehicle.getWaitingTime(car_id)
        if wait > max_waiting_time:
            max_waiting_time = wait
        if traci.vehicle.getVehicleClass(car_id) == "emergency":
            ev_exists = True
    
    return car_count, max_waiting_time, ev_exists
    
    
def pedestrian_exists_and_max_waiting_time():
    """
    Check if there are pedestrians waiting at the crosswalk and get the maximum waiting time.
    Returns:
        tuple: A tuple containing a boolean indicating whether there are pedestrians waiting at the crosswalk and the maximum waiting time of pedestrians.    
    """
    max_waiting_time = 0
    for ped_id in traci.person.getIDList():
        if traci.person.getWaitingTime(ped_id) > 0:
            max_waiting_time = max(max_waiting_time, traci.person.getWaitingTime(ped_id))

    if max_waiting_time > 0:
        return True, max_waiting_time
    return False, 0


def turn_current_phase_yellow():
    """
    Turn the current traffic light phase to yellow.
    """
    curr = traci.trafficlight.getRedYellowGreenState(TL_ID)
    yellow_state = curr.replace('G','y').replace('g','y')
    traci.trafficlight.setRedYellowGreenState(TL_ID, yellow_state)
    
    for _ in range(int(YELLOW_DURATION)):
        traci.simulationStep()
        if traci.simulation.getMinExpectedNumber() == 0:
            break


def turn_all_red():
    """
    Turn all traffic light phases to red.
    """
    curr = traci.trafficlight.getRedYellowGreenState(TL_ID)
    all_red = curr.replace('G','r').replace('g','r').replace('Y','r').replace('y','r')
    traci.trafficlight.setRedYellowGreenState(TL_ID, all_red)
    
    for _ in range(int(RED_DURATION)):
        traci.simulationStep()
        if traci.simulation.getMinExpectedNumber() == 0:
            break

def turn_phase_green(phase_id, duration, step):
    """
    Turn a specific traffic light phase to green.

    Args:
        phase_id (str): The ID of the traffic light phase to turn green.
    """
    traci.trafficlight.setProgram(TL_ID, "0")
    traci.trafficlight.setPhase(TL_ID, phase_id)
    traci.trafficlight.setPhaseDuration(TL_ID, 10000)

    for _ in range(int(duration)):
        traci.simulationStep()
        step += 1
        if traci.simulation.getMinExpectedNumber() == 0:
            break
    return step

def get_state():
    """
    Get the current state of the traffic environment.
    ([car_count], [max_waiting_time], [ev_exists], [pedestrian_exists])
    *car count: A list of the number of cars in each lane.
    *max waiting time: A list of the maximum waiting time of cars in each lane.
    *ev exists: A boolean indicating whether there are electric vehicles in the lane.
    *pedestrian exists: 1 or 0 indicating whether there are pedestrians waiting at the crosswalk. simulating a button
    Returns:
        dict: A dictionary containing the current state of the traffic environment.
    """
    state = {}
    for edge in ENTERING_EDGES:
        car_count, max_waiting_time, ev_exists = get_lane_state(edge)
        state[f"{edge}_car_count"] = car_count
        state[f"{edge}_max_waiting_time"] = max_waiting_time
        state[f"{edge}_ev_exists"] = ev_exists

    pedestrian_exists, pedestrian_max_waiting_time = pedestrian_exists_and_max_waiting_time()
    state["pedestrian_exists"] = pedestrian_exists
    state["pedestrian_max_waiting_time"] = pedestrian_max_waiting_time

    return state
    
def get_state_vector(weather=-1, day=0, time_of_day=0):

    vector = []
    for edge in ENTERING_EDGES:
        count, wait, ev = get_lane_state(edge)
        vector.append(count)
        vector.append(wait / MAX_CAP_TIME)
        vector.append(int(ev))          
    
    ped_exists, ped_wait = pedestrian_exists_and_max_waiting_time()
    vector.append(int(ped_exists))
    vector.append(ped_wait / MAX_CAP_TIME)
    
    vector.append(weather)
    vector.append(day / 6.0)     
    vector.append(time_of_day)   
    
    return np.array(vector, dtype=np.float32)