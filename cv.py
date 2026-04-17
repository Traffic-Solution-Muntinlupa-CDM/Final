from ultralytics import YOLO
import cv2

# Load YOLOv8 model (COCO pretrained already includes cars)
model = YOLO("yolov8n.pt")

def count_cars(image_path, conf_threshold=0.5):
    """
    Counts cars in an image using YOLOv8 COCO model.

    Args:
        image_path (str): path to image
        conf_threshold (float): confidence threshold

    Returns:
        int: number of cars detected
    """

    img = cv2.imread(image_path)
    results = model(img)[0]

    car_count = 0

    # COCO class for car = 2
    CAR_CLASS_ID = 2

    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf >= conf_threshold and cls == CAR_CLASS_ID:
            car_count += 1

    return car_count

