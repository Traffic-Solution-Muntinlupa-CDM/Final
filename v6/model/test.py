import traci
import torch
import csv
import os
import numpy as np

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from PPO import PPOAgent
from sumoFiles.sumoUtils import (
    TL_ID, ENTERING_EDGES, PHASES, EDGE_DETECTORS,
    get_state_vector, turn_current_phase_yellow, turn_all_red,
    YELLOW_DURATION, RED_DURATION, PEDESTRIAN_DURATION,
    MAX_GREEN_DURATION, MIN_GREEN_DURATION
)
from demands import spawn_step, reset_episode, SCENARIOS, apply_weather
from train import get_reward

CHECKPOINT_PATH = "../checkpoints/v6_model_checkpoint_1020.pth"
NUM_EVAL_EPISODES_PER_SCENARIO = 1
MAX_STEPS = 3600
USE_GUI = True

PHASE_ORDER = ["southbound", "brudger", "northbound", "estanislao", "cityhall", "pedestrian"]

PHASE_TO_EDGES = {
    0: ["southbound_entrance"],
    1: ["brudger_entrance"],
    2: ["northbound_entrance"],
    3: ["estanislao_entrance"],
    4: ["cityhall_entrance"],
    5: []
}

def log_decision(step, state, chosen_phase, time_percentage, actual_seconds, decision_writer):
    """
    Log full state (20 features) + action.
    """
    decision_writer.writerow([
        step, chosen_phase, f"{time_percentage:.3f}", actual_seconds,
        *[f"{state[i]:.3f}" for i in range(20)]
    ])

def run_evaluation(scenario, agent, episode_idx):
    """
    Run one evaluation episode for a given scenario.
    Returns dict of metrics.
    """
    weather = scenario["weather"]
    day_of_week = episode_idx % 7

    log_dir = "../logs/decisions"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/decisions_{scenario['name']}_{episode_idx}.csv"
    decision_log = open(log_filename, 'w', newline='')
    decision_writer = csv.writer(decision_log)
    decision_writer.writerow([
    "step", "chosen_phase", "time_pct", "actual_seconds",
    "north_count", "north_wait_norm", "north_ev",
    "south_count", "south_wait_norm", "south_ev",
    "cityhall_count", "cityhall_wait_norm", "cityhall_ev",
    "brudger_count", "brudger_wait_norm", "brudger_ev",
    "estanislao_count", "estanislao_wait_norm", "estanislao_ev",
    "ped_exists", "ped_wait_norm",
    "weather", "day_norm", "time_of_day"
])

    if USE_GUI:
        sumo_cmd = ["sumo-gui", "-c", "../sumoFiles/network.sumocfg",
                    "--start", "--quit-on-end", "--time-to-teleport", "-1",
                    "--pedestrian.striping.jamtime", "-1"]
    else:
        sumo_cmd = ["sumo", "-c", "../sumoFiles/network.sumocfg",
                    "--start", "--quit-on-end", "--time-to-teleport", "-1",
                    "--pedestrian.striping.jamtime", "-1"]
    traci.start(sumo_cmd)
    apply_weather(weather)

    sim_time = 0.0
    time_of_day = (sim_time % 86400) / 86400.0
    state = get_state_vector(weather, day_of_week/6.0, time_of_day)
    current_phase = 0
    episode_reward = 0.0
    step = 0
    decision_count = 0
    total_queue = 0.0
    episode_throughput = 0
    phase_counts = {i: 0 for i in range(6)}

    total_waiting_time = 0.0
    vehicle_count_processed = 0

    while step < MAX_STEPS:
        ev_edge = None
        for edge in ENTERING_EDGES:
            detectors = EDGE_DETECTORS.get(edge, [])
            for det in detectors:
                for veh_id in traci.lanearea.getLastStepVehicleIDs(det):
                    if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                        ev_edge = edge
                        break
                if ev_edge:
                    break
            if ev_edge:
                break

        if ev_edge is not None:
            chosen_phase = None
            for ph, edges in PHASE_TO_EDGES.items():
                if ev_edge in edges:
                    chosen_phase = ph
                    break
            if chosen_phase is None:
                chosen_phase = current_phase
            time_percentage = 1.0
            is_override = True
        else:
            chosen_phase, time_percentage = agent.get_deterministic_action(state)
            is_override = False

        if chosen_phase == 5:
            actual_seconds = PEDESTRIAN_DURATION
        else:
            actual_seconds = int(MIN_GREEN_DURATION +
                                 time_percentage * (MAX_GREEN_DURATION - MIN_GREEN_DURATION))

        if chosen_phase != current_phase:
            turn_current_phase_yellow()
            turn_all_red()
            step += YELLOW_DURATION + RED_DURATION
            episode_throughput += traci.simulation.getArrivedNumber()
            current_phase = chosen_phase

        traci.trafficlight.setProgram(TL_ID, "0")
        phase_index = PHASES[PHASE_ORDER[chosen_phase]]
        traci.trafficlight.setPhase(TL_ID, phase_index)
        traci.trafficlight.setPhaseDuration(TL_ID, 10000)

        green_steps = 0
        for _ in range(actual_seconds):
            spawn_step(scenario)
            traci.simulationStep()
            step += 1
            green_steps += 1
            episode_throughput += traci.simulation.getArrivedNumber()

            for veh_id in traci.vehicle.getIDList():
                total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
                vehicle_count_processed += 1

            if is_override:
                ev_still_present = False
                for edge in ENTERING_EDGES:
                    detectors = EDGE_DETECTORS.get(edge, [])
                    for det in detectors:
                        for veh_id in traci.lanearea.getLastStepVehicleIDs(det):
                            if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                                ev_still_present = True
                                break
                        if ev_still_present:
                            break
                    if ev_still_present:
                        break
                if not ev_still_present and green_steps > 5:
                    break
            else:
                ev_appeared = False
                for edge in ENTERING_EDGES:
                    detectors = EDGE_DETECTORS.get(edge, [])
                    for det in detectors:
                        for veh_id in traci.lanearea.getLastStepVehicleIDs(det):
                            if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                                ev_appeared = True
                                break
                        if ev_appeared:
                            break
                    if ev_appeared:
                        break
                if ev_appeared:
                    break

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        sim_time = traci.simulation.getTime()
        time_of_day = (sim_time % 86400) / 86400.0
        next_state = get_state_vector(weather, day_of_week/6.0, time_of_day)
        reward = get_reward(next_state, state, chosen_phase, actual_seconds, green_steps)

        log_decision(step, state, chosen_phase, time_percentage, actual_seconds, decision_writer)

        episode_reward += reward
        decision_count += 1
        phase_counts[chosen_phase] += 1

        avg_queue_step = np.mean([next_state[i*3+1] for i in range(len(ENTERING_EDGES))])
        total_queue += avg_queue_step

        state = next_state

    traci.close()
    decision_log.close()

    avg_wait_time = total_waiting_time / max(1, vehicle_count_processed) if vehicle_count_processed > 0 else 0
    avg_queue_episode = total_queue / max(1, decision_count)
    avg_reward_episode = episode_reward / max(1, decision_count)

    return {
        "scenario": scenario["name"],
        "episode": episode_idx,
        "steps": step,
        "decisions": decision_count,
        "avg_reward": avg_reward_episode,
        "avg_queue": avg_queue_episode,
        "throughput": episode_throughput,
        "avg_wait_time_sec": avg_wait_time,
        "phase_counts": phase_counts
    }

def main():
    agent = PPOAgent(num_inputs=20, num_phases=6, lr_actor=0.0003, lr_critic=0.001, gamma=0.99)
    agent.load(CHECKPOINT_PATH)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    results = []
    for scenario in SCENARIOS:
        print(f"\nEvaluating scenario: {scenario['name']}")
        for ep in range(NUM_EVAL_EPISODES_PER_SCENARIO):
            print(f"  Episode {ep+1}/{NUM_EVAL_EPISODES_PER_SCENARIO} ...")
            metrics = run_evaluation(scenario, agent, ep)
            results.append(metrics)
            print(f"    Reward: {metrics['avg_reward']:.3f}, Queue: {metrics['avg_queue']:.3f}, "
                  f"Throughput: {metrics['throughput']}, WaitTime: {metrics['avg_wait_time_sec']:.1f}s")

    csv_path = "../logs/evaluation_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "episode", "steps", "decisions", "avg_reward",
                         "avg_queue", "throughput", "avg_wait_time_sec", "phase0", "phase1",
                         "phase2", "phase3", "phase4", "phase5"])
        for r in results:
            writer.writerow([
                r["scenario"], r["episode"], r["steps"], r["decisions"],
                round(r["avg_reward"], 3), round(r["avg_queue"], 3),
                r["throughput"], round(r["avg_wait_time_sec"], 1),
                r["phase_counts"].get(0, 0), r["phase_counts"].get(1, 0),
                r["phase_counts"].get(2, 0), r["phase_counts"].get(3, 0),
                r["phase_counts"].get(4, 0), r["phase_counts"].get(5, 0)
            ])
    print(f"\nResults saved to {csv_path}")
    print(f"Per‑decision logs saved in {os.path.abspath('../logs/decisions')}")

if __name__ == "__main__":
    main()