import sys
import os
import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO

TEST_MODE = False  # True = TestCases, False = Actual

ROOT = os.path.abspath(".")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "discreteV1"))

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

NUM_PHASES = 6

agent = PPO.load("./discreteV1/ppo_traffic_final.zip")

yolo = YOLO("yolov8n.pt")
CAR_CLASS_ID = 2


def count_cars(image_path):
    img = cv2.imread(image_path)
    results = yolo(img)[0]

    car_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == CAR_CLASS_ID and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"car {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

            car_count += 1

    # overlay filename + count
    filename = os.path.basename(image_path)
    text = f"{filename} | Cars: {car_count}"

    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)

    # show window ONLY in test mode
    if TEST_MODE:
        cv2.imshow("YOLO Car Detection", img)

        # wait for click / key
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                break

            if cv2.getWindowProperty("YOLO Car Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    return car_count


def process_case(case_name, vehicle_counts):

    # order of state
    ENTERING_EDGES = [
    "northbound_entrance",
    "southbound_entrance",
    "cityhall_exit",
    "brudger_exit",
    "estanislao_exit"
    ]
    normalized_counts = [ (count / MAX_EDGE_CAPACITY[edge]) for count, edge in zip(vehicle_counts, ENTERING_EDGES)]

    state = [
        normalized_counts[0], 0, # northbound count and wait
        normalized_counts[1], 0, # southbound count and wait
        normalized_counts[2], 0, # cityhall count and wait
        normalized_counts[3], 0, # brudger count and wait
        normalized_counts[4], 0, # estanislao count and wait
        0, 0, # ped exist, ped wait
        -1 # weather
    ]

    action, _states = agent.predict(state, deterministic=True)

    phase_idx, duration_idx = action
    phase = PHASES[int(phase_idx)]
    
    if phase == "pedestrian":
        time = PEDESTRIAN_DURATION
    else:
        time = DURATIONS[int(duration_idx)]

    print("\n==============================")
    print("Mode:", "TEST" if TEST_MODE else "ACTUAL")
    print("Case:", case_name)
    print("Vehicle Counts:", vehicle_counts)
    print("Phase:", phase)
    print("Green Time:", time)



if TEST_MODE:

    base_folder = "TestCases"

    for case_name in sorted(os.listdir(base_folder)):
        case_folder = os.path.join(base_folder, case_name)

        if not os.path.isdir(case_folder):
            continue

        vehicle_counts = [
            count_cars(os.path.join(case_folder, "nb.jpg")),
            count_cars(os.path.join(case_folder, "sb.jpg")),
            count_cars(os.path.join(case_folder, "ch.jpg")),
            count_cars(os.path.join(case_folder, "bg.jpg")),
            count_cars(os.path.join(case_folder, "es.jpg")),
        ]

        process_case(case_name, vehicle_counts)

else:

    base_folder = "Actual"

    vehicle_counts = [
        count_cars(os.path.join(base_folder, "nb.jpg")),
        count_cars(os.path.join(base_folder, "sb.jpg")),
        count_cars(os.path.join(base_folder, "ch.jpg")),
        count_cars(os.path.join(base_folder, "bg.jpg")),
        count_cars(os.path.join(base_folder, "es.jpg")),
    ]

    process_case("ACTUAL", vehicle_counts)
