# import cv2
# from rich.console import Console
# from rich import print
# import os


# class VideoStream:
#     def __init__(self, streams):
#         self.streams = streams
#         self.caps = [cv2.VideoCapture(stream) for stream in streams]
#         self.num_cameras = len(streams)

#     def get_next_frame(self, limit=10):
#         frame_count = 0
#         camera_index = 0

#         while frame_count < limit:
#             cap = self.caps[camera_index]

#             # Capture frame-by-frame
#             ret, frame = cap.read()
#             # If frame is not received correctly, exit
#             if not ret:
#                 print("Can't receive frame (stream end?). Exiting ...")
#                 break

#             yield frame

#             frame_count += 1
#             camera_index = (camera_index + 1) % self.num_cameras

#     def __del__(self):
#         for cap in self.caps:
#             cap.release()


# if __name__ == "__main__":
#     num_cameras = 4
#     streams = [
#         f"rtsp://admin:dataeazeDZ!@192.168.0.{102 + x}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
#         for x in range(num_cameras)
#     ]
#     print(streams)

#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)

#     vs = VideoStream(streams)

#     for i, frame in enumerate(vs.get_next_frame()):
#         fname = os.path.join(output_dir, f"frame_{i:06d}.png")
#         cv2.imwrite(fname, frame)


import cv2
from rich.console import Console
from rich import print
import os


class RoundRobinVideoStream:
    def __init__(self, streams, limit=4000):
        self.streams = streams
        self.caps = [cv2.VideoCapture(stream) for stream in streams]
        self.num_cameras = len(streams)
        self.limit = limit
        self.frame_count = 0

    def get_next_frame(self, camera_index):
      
        cap = self.caps[camera_index]

        ret, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        # If frame is not received correctly, exit
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None, None, None, None

        return frame, width, height, fps


    def __del__(self):
        for cap in self.caps:
            cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        return  self.next()

    def next(self):

        if self.frame_count < self.limit:
            frames = []
            widths = []
            heights = []
            fps_vals = []
            for i in range(self.num_cameras):
                frame, width, height, fps = self.get_next_frame(i)
                frames.append(frame)
                widths.append(width)
                heights.append(height)
                fps_vals.append(fps)
            self.frame_count += 1
            return frames, widths, heights, fps_vals
        else:
            raise StopIteration
   
    def get_num_cameras(self):
        return self.num_cameras
        


if __name__ == "__main__":
    num_cameras = 2
    streams = [
        f"rtsp://admin:dataeazeDZ!@192.168.0.{102 + x}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
        for x in range(num_cameras)
    ]
    print(streams)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    vs = iter(RoundRobinVideoStream(streams))
    # vs = iter(RoundRobinVideoStream(streams))

    # for i, frames in enumerate(next(vs)):
    #     #fname = os.path.join(output_dir, f"camera_{:02d}__frame_{:06d}.png")
    #     #cv2.imwrite(fname, frame)
    #     print(frames)

    num_cameras = vs.get_num_cameras()

    for i, frames in enumerate(vs):
        print(len(frames), frames[0].shape)
        for c in range(num_cameras):
            fname = os.path.join(output_dir, "camera_{:02d}__frame_{:06d}.png".format(c, i))
            cv2.imwrite(fname, frames[c])