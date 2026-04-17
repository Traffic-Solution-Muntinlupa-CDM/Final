import sys
import os
import cv2
from ultralytics import YOLO

TEST_MODE = False  # True = TestCases, False = Actual

ROOT = os.path.abspath(".")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "v6", "model"))

from PPO import PPOAgent


MODEL_INPUTS = 20
NUM_PHASES = 6

agent = PPOAgent(
    num_inputs=MODEL_INPUTS,
    num_phases=NUM_PHASES,
    lr_actor=0.0003,
    lr_critic=0.001,
    gamma=0.99
)

agent.load("v6/checkpoints/v6_model_checkpoint_1020.pth")
agent.policy_old.eval()

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

    state = [
        vehicle_counts,
        [0, 0, 0, 0, 0],  # max waiting time
        [0, 0, 0, 0, 0],  # emergency vehicles
        0,                # pedestrian request
        0,                # pedestrian waiting time
        0,                # weather
        0,                # day of week
        0                 # time of day
    ]

    state = state[0] + state[1] + state[2] + [
        state[3], state[4], state[5], state[6], state[7]
    ]

    phase, time = agent.get_deterministic_action(state)

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