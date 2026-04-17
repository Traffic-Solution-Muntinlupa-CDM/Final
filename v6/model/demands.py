import traci
import random
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from sumoFiles.sumoUtils import ENTERING_EDGES

VEHICLE_TYPES = ["car", "car", "car", "truck"]  

ROUTES_PER_EDGE = {
    "northbound_entrance": ["northbound_thru", "northbound_est", "northbound_brudger", "northbound_city"],
    "southbound_entrance": ["southbound_thru", "southbound_est", "southbound_brudger", "southbound_city"],
    "cityhall_entrance":   ["cityhall_north", "cityhall_south", "cityhall_est", "cityhall_brudger"],
    "brudger_entrance":    ["brudger_north", "brudger_south", "brudger_city", "brudger_est"],
    "estanislao_entrance": ["estanislao_north", "estanislao_south", "estanislao_city", "estanislao_brudger"],
}

SCENARIOS = [
    {"name": "low_traffic",  "cars_per_step": 0.003, "weather": -1, "ev_prob": 0.0001, "ped_prob": 0.0005},
    {"name": "normal",       "cars_per_step": 0.7, "weather": -1, "ev_prob": 0.02, "ped_prob": 0.10},
    {"name": "peak_hour",    "cars_per_step": 1.5, "weather": -1, "ev_prob": 0.03, "ped_prob": 0.20},
    {"name": "rainy_normal", "cars_per_step": 0.5, "weather":  1, "ev_prob": 0.01, "ped_prob": 0.08},
    {"name": "rainy_peak",   "cars_per_step": 1.0, "weather":  1, "ev_prob": 0.02, "ped_prob": 0.15},
    {"name": "cloudy_normal","cars_per_step": 0.6, "weather":  0, "ev_prob": 0.02, "ped_prob": 0.10},
]

_vehicle_counter = 0

def spawn_vehicle(edge_id, vtype="car"):
    global _vehicle_counter
    route = random.choice(ROUTES_PER_EDGE[edge_id])
    vid = f"{vtype}_{edge_id}_{_vehicle_counter}"
    _vehicle_counter += 1
    try:
        traci.vehicle.add(vid, route, typeID=vtype)
    except traci.TraCIException:
        pass
    return vid

def spawn_ev(edge_id):
    global _vehicle_counter
    route = random.choice(ROUTES_PER_EDGE[edge_id])
    vid = f"ev_{edge_id}_{_vehicle_counter}"
    _vehicle_counter += 1
    try:
        traci.vehicle.add(vid, route, typeID="emergency")
    except traci.TraCIException:
        pass
    return vid



def apply_weather(weather_scalar):
    speeds = {-1: 13.9, 0: 11.0, 1: 8.0}
    speed = speeds[weather_scalar]
    traci.vehicletype.setMaxSpeed("car", speed)
    traci.vehicletype.setMaxSpeed("truck", speed * 0.6)
    traci.vehicletype.setMaxSpeed("emergency", min(16.0, speed * 1.1))


def spawn_step(scenario):
    for edge in ENTERING_EDGES:
        if random.random() < scenario["cars_per_step"]:
            vtype = random.choice(VEHICLE_TYPES)
            spawn_vehicle(edge, vtype)
        if random.random() < scenario["ev_prob"]:
            spawn_ev(edge)

    if random.random() < scenario["ped_prob"]:
        spawn_pedestrian()

_ped_counter = 0

PED_WALK_STAGES = {
    "ped_route_up":   ["brudger_exit", "cityhall_entrance"],
    "ped_route_down": ["brudger_entrance", "estanislao_exit"],
}

def spawn_pedestrian():
    global _ped_counter
    ped_id = f"ped_{_ped_counter}"
    _ped_counter += 1
    route_name = random.choice(list(PED_WALK_STAGES.keys()))
    edges = PED_WALK_STAGES[route_name]
    try:
        traci.person.add(ped_id, edges[0], 0.0)
        traci.person.appendWalkingStage(ped_id, edges, 0.0)
    except traci.TraCIException:
        pass

def reset_episode(episode_idx=None):
    if episode_idx is not None:
        scenario = SCENARIOS[episode_idx % len(SCENARIOS)]
    else:
        scenario = random.choice(SCENARIOS)
    apply_weather(scenario["weather"])
    return scenario