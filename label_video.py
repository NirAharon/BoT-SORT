from pathlib import Path
from sys import argv

from tqdm import tqdm
from yolov7.utils.plots import plot_one_box
import cv2


txt_file = Path(argv[1])
vid_file = txt_file.with_suffix(".mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

if vid_file.exists():
    cap = cv2.VideoCapture(str(vid_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(
        str(vid_file.with_name(vid_file.stem + "-output.mp4")), fourcc, fps, (w, h)
    )

    frame_i = 0
    detection_i = 0
    with txt_file.open("r", encoding="utf8") as detection_file:
        detections = detection_file.readlines()

    pbar = tqdm(total=f_count)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        while detection_i < len(detections):
            detection_info = detections[detection_i].split(",")

            if int(detection_info[0]) > frame_i:
                break

            box = list(map(lambda x: round(float(x)), detection_info[2:6]))
            x = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

            tid = detection_info[1]
            label = f"{tid}"
            plot_one_box(x, frame, [28, 68, 244], tid)

            detection_i += 1

        frame_i += 1
        pbar.update(1)
        out.write(frame)

    pbar.close()
    cap.release()
    out.release()
