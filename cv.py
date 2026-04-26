from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def count_cars(image_path, conf_threshold=0.5):
    """
    Counts traffic vehicles (car, motorcycle, bus, truck) using YOLOv8.

    Args:
        image_path (str): path to image
        conf_threshold (float): confidence threshold

    Returns:
        int: number of vehicles detected
    """
    img = cv2.imread(image_path)
    results = model(img)[0]

    vehicle_count = 0

    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf >= conf_threshold and cls in VEHICLE_CLASSES:
            vehicle_count += 1

    return vehicle_count