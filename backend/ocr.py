import cv2
import pytesseract


def extract_text_from_video(video_path: str, interval_seconds: int = 2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_interval = int(fps * interval_seconds)

    results = []
    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps

            text = pytesseract.image_to_string(frame).strip()

            if text:
                results.append({
                    "text": text,
                    "timestamp": round(timestamp, 2)
                })

        frame_number += 1

    cap.release()

    return results