import traci
import torch
import random
import csv
import os
import numpy as np
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from PPO import PPOAgent

from sumoFiles.sumoUtils import (
    TL_ID, ENTERING_EDGES, PHASES,
    EDGE_DETECTORS,
    get_state_vector,
    turn_current_phase_yellow, turn_all_red,
    YELLOW_DURATION, RED_DURATION,
    PEDESTRIAN_DURATION, MAX_GREEN_DURATION, MIN_GREEN_DURATION
)
from demands import spawn_step, reset_episode, SCENARIOS, apply_weather


PHASE_TO_EDGES = {
    0: ["southbound_entrance"],
    1: ["brudger_entrance"],
    2: ["northbound_entrance"],
    3: ["estanislao_entrance"],
    4: ["cityhall_entrance"],
    5: []  
}

PHASE_ORDER = ["southbound", "brudger", "northbound", "estanislao", "cityhall", "pedestrian"]


EDGE_TO_IDX = {edge: i for i, edge in enumerate(ENTERING_EDGES)}

def get_reward(next_state_vec, prev_state_vec, chosen_phase, actual_seconds, step_count):
    """
    Args:
        next_state_vec: state after green phase
        prev_state_vec: state before green phase
        chosen_phase: phase index that was active
        actual_seconds: planned green duration (not always fully used if EV broke early)
        step_count: actual simulation steps executed during this green (green_steps)
    """
    reward = 0.0

    served_edges = PHASE_TO_EDGES.get(chosen_phase, [])

    total_cleared_urgency = 0.0
    total_cars_cleared = 0.0

    for edge in served_edges:
        idx = EDGE_TO_IDX[edge]

        prev_wait = prev_state_vec[idx*3 + 1]
        prev_count = prev_state_vec[idx*3 + 0]
        new_count = next_state_vec[idx*3 + 0]

        cleared = max(prev_count - new_count, 0.0)
        total_cars_cleared += cleared

        edge_weight = 1.5 if edge == "cityhall_entrance" else 1.0
        urgency = (prev_wait ** 1.8) * edge_weight  

        total_cleared_urgency += cleared * urgency

    if total_cars_cleared > 0.5:
        reward += 6.0 * (total_cleared_urgency / total_cars_cleared)
    else:
        reward -= 0.3 * actual_seconds

    for edge in ENTERING_EDGES:
        idx = EDGE_TO_IDX[edge]

        wait_norm = next_state_vec[idx*3 + 1]
        count = next_state_vec[idx*3 + 0]

        weight = 2.0 if edge == "cityhall_entrance" else 1.0

        reward -= 0.25 * (wait_norm ** 1.5) * count * weight

    ped_exists = next_state_vec[15]
    ped_wait_norm = next_state_vec[16]

    if ped_exists > 0.5:
        reward -= ped_wait_norm * 1.2

        if chosen_phase == 5:
            prev_ped_wait = prev_state_vec[16]
            reduction = max(prev_ped_wait - ped_wait_norm, 0.0)
            reward += reduction * 4.0

    if chosen_phase == 5 and ped_exists < 0.5:
        reward -= 2.0

    if total_cars_cleared > 0.5:
        reward += 0.2 * step_count  

    return reward / 5.0

def get_curriculum_scenario(episode, performance=None):
    if episode < 150:
        return reset_episode(0)   
    elif episode < 300:
        if random.random() < 0.7:
            return reset_episode(0) 
        else:
            return reset_episode(1)  
    elif episode < 500:
        r = random.random()
        if r < 0.5:
            return reset_episode(0)
        elif r < 0.8:
            return reset_episode(1)
        else:
            return reset_episode(2)
    else:
        return reset_episode()

def train():
    print("Starting PPO Training with 6 phases, dynamic spawning, weather/time inputs...")
    agent = PPOAgent(
    num_inputs=20,
    num_phases=6,
    lr_actor=0.0003,
    lr_critic=0.001,
    gamma=0.99
    )
    agent.K_epochs = 10

    MAX_EPISODES = 5000
    csv_filename = "../logs/training_log.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "scenario", "steps", "decisions", "avg_reward", "avg_queue", "throughput"])
            

    for episode in range(MAX_EPISODES):
        sumo_cmd = [
            "sumo", "-c", "../sumoFiles/network.sumocfg",
            "--start", "--quit-on-end", "--time-to-teleport", "-1",
            "--pedestrian.striping.jamtime", "-1"
        ]
        traci.start(sumo_cmd)
        scenario = get_curriculum_scenario(episode)
        weather = scenario["weather"]
        day_of_week = episode % 7

        print(f"Episode {episode+1} | Scenario: {scenario['name']} | Weather: {weather}")


        sim_time = 0.0
        time_of_day = (sim_time % 86400) / 86400.0   
        state = get_state_vector(weather, day_of_week/6.0, time_of_day)
        current_phase = 0   
        episode_reward = 0.0
        step = 0
        decision_count = 0
        total_episode_queue = 0.0
        episode_throughput = 0

        MAX_STEPS = 3600 

        while step < MAX_STEPS:
            ev_edge = None
            for edge in ENTERING_EDGES:
                detectors = EDGE_DETECTORS.get(edge, [])
                for det in detectors:
                    for veh_id in traci.lanearea.getLastStepVehicleIDs(det):
                        if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                            ev_edge = edge
                            break
                    if ev_edge: break
                if ev_edge: break

            if ev_edge is not None:
                chosen_phase = None
                for ph, edges in PHASE_TO_EDGES.items():
                    if ev_edge in edges:
                        chosen_phase = ph
                        break
                if chosen_phase is None:
                    chosen_phase = current_phase
                is_override = True
                time_percentage = 1.0
            else:
                chosen_phase, time_percentage = agent.select_action(state)
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
            traci.trafficlight.setPhase(TL_ID, PHASES[PHASE_ORDER[chosen_phase]])
            traci.trafficlight.setPhaseDuration(TL_ID, 10000)  
            green_steps = 0
            for _ in range(actual_seconds):
                spawn_step(scenario)

                traci.simulationStep()
                step += 1
                green_steps += 1
                episode_throughput += traci.simulation.getArrivedNumber()

            
                if is_override:
                    ev_still_present = False
                    for edge in ENTERING_EDGES:
                        detectors = EDGE_DETECTORS.get(edge, [])
                        for det in detectors:
                            for veh_id in traci.lanearea.getLastStepVehicleIDs(det):
                                if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                                    ev_still_present = True
                                    break
                            if ev_still_present: break
                        if ev_still_present: break
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
                            if ev_appeared: break
                        if ev_appeared: break
                    if ev_appeared:
                        break

                if traci.simulation.getMinExpectedNumber() == 0:
                    break

            sim_time = traci.simulation.getTime()
            time_of_day = (sim_time % 86400) / 86400.0
            next_state = get_state_vector(weather, day_of_week/6.0, time_of_day)
            reward = get_reward(next_state, state, chosen_phase, actual_seconds, green_steps)
            episode_reward += reward
            decision_count += 1

            avg_queue = np.mean([next_state[i*3+1] for i in range(len(ENTERING_EDGES))])
            total_episode_queue += avg_queue

            if not is_override:
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(traci.simulation.getMinExpectedNumber() == 0)

            state = next_state

            if len(agent.buffer.rewards) >= 200:
                agent.update()
                agent.buffer.clear()

        traci.close()

        avg_reward = episode_reward / max(1, decision_count)
        avg_queue = total_episode_queue / max(1, decision_count)
        print(f"Episode {episode+1} | Steps: {step} | Decisions: {decision_count} | "
              f"Avg Reward: {avg_reward:.3f} | Avg Queue: {avg_queue:.2f} | "
              f"Throughput: {episode_throughput}")

        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode+1, scenario['name'], step, decision_count,
                             round(avg_reward, 3), round(avg_queue, 3), episode_throughput])

        if (episode + 1) % 10 == 0:
            agent.save(f"../checkpoints/v6_model_checkpoint_{episode+1}.pth")

    agent.save("../checkpoints/v6_model_final.pth")
    print("Training Complete")

if __name__ == "__main__":
    train()