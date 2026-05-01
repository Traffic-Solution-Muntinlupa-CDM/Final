import os
import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO

# ─── Config ───────────────────────────────────────────────
PHASES = ["southbound", "brudger", "northbound", "estanislao", "cityhall", "pedestrian"]
DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

PEDESTRIAN_DURATION = 30

MAX_EDGE_CAPACITY = {
    "northbound_entrance": 30,
    "southbound_entrance": 30,
    "cityhall_exit":        5,
    "brudger_exit":        15,
    "estanislao_exit":     15,
}

ENTERING_EDGES = [
    "northbound_entrance",
    "southbound_entrance",
    "cityhall_exit",
    "brudger_exit",
    "estanislao_exit",
]

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ─── Load models once ─────────────────────────────────────
_agent = PPO.load(os.path.join(os.path.dirname(__file__), "discreteV1/ppo_traffic_final.zip"))
_yolo  = YOLO(os.path.join(os.path.dirname(__file__), "yolov8n.pt"))


# ─── Public API ───────────────────────────────────────────
def INFER(nb, sb, ch, bg, es, ped_exist=0, ped_wait=0, weather=-1):
    """
    Run the traffic model with raw vehicle counts.

    Args:
        nb  (int): northbound vehicle count
        sb  (int): southbound vehicle count
        ch  (int): cityhall vehicle count
        bg  (int): brudger vehicle count
        es  (int): estanislao vehicle count
        ped_exist (int): pedestrian present (0/1), default 0
        ped_wait  (int): pedestrian wait time, default 0
        weather   (int): weather condition, default -1

    Returns:
        dict: { "phase": str, "green_time": int }
    """
    vehicle_counts = [nb, sb, ch, bg, es]

    normalized = [
        count / MAX_EDGE_CAPACITY[edge]
        for count, edge in zip(vehicle_counts, ENTERING_EDGES)
    ]

    state = [
        normalized[0], 0,    # northbound count and wait
        normalized[1], 0,    # southbound count and wait
        normalized[2], 0,    # cityhall count and wait
        normalized[3], 0,    # brudger count and wait
        normalized[4], 0,    # estanislao count and wait
        ped_exist, ped_wait, # ped exist, ped wait
        weather,             # weather
    ]

    action, _ = _agent.predict(state, deterministic=True)
    phase_idx, duration_idx = action
    phase = PHASES[int(phase_idx)]
    green_time = PEDESTRIAN_DURATION if phase == "pedestrian" else DURATIONS[int(duration_idx)]

    return {
        "phase": phase,
        "green_time": green_time,
    }


def count_cars(image_path, show=False, conf_threshold=0.5):
    """
    Count traffic vehicles (car, motorcycle, bus, truck) in an image using YOLO.

    Args:
        image_path (str): path to image
        show (bool): show detection window
        conf_threshold (float): confidence threshold

    Returns:
        int: number of vehicles detected
    """
    img = cv2.imread(image_path)
    results = _yolo(img)[0]
    vehicle_count = 0

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        if conf >= conf_threshold and cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = VEHICLE_CLASSES[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vehicle_count += 1

    filename = os.path.basename(image_path)
    cv2.putText(img, f"{filename} | Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if show:
        cv2.imshow("YOLO Vehicle Detection", img)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break
            if cv2.getWindowProperty("YOLO Vehicle Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    return vehicle_count