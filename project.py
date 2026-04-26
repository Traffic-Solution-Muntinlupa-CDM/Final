import cv2
from model import INFER, count_cars

CAM_INDEX = 0
CH_IMAGE_PATH = "Actual/ch.jpg"

DEFAULT_COUNTS = {
    "nb": 5,
    "sb": 5,
    "bg": 3,
    "es": 3,
}


def capture_cityhall():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("CITYHALL — Press SPACE to capture | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Cityhall Camera", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Cancelled")

        if k == ord(' '):
            cv2.imwrite(CH_IMAGE_PATH, frame)
            print(f"Saved: {CH_IMAGE_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()


def run():
    capture_cityhall()

    ch = count_cars(CH_IMAGE_PATH)
    print(f"Cars detected (cityhall): {ch}")

    result = INFER(
        nb=DEFAULT_COUNTS["nb"],
        sb=DEFAULT_COUNTS["sb"],
        ch=ch,
        bg=DEFAULT_COUNTS["bg"],
        es=DEFAULT_COUNTS["es"],
    )

    print("\n==============================")
    print("Phase     :", result["phase"])
    print("Green Time:", result["green_time"], "seconds")
    print("==============================")


if __name__ == "__main__":
    run()