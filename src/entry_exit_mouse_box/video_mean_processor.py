import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import tifffile
import time


class VideoMeanProcessor(object):

    def __init__(self, video_path, shape):
        # Absolute path of the video file.
        self.video_path = video_path
        # Capture object to read the video file.
        self.video      = cv2.VideoCapture(video_path)
        # Lock to read the file from the workers.
        self.lock       = threading.Lock()
        # Shape of the video frames.
        self.shape      = shape
        # Buffer in which the mean is stored.
        self.buffer     = np.zeros(self.shape, np.float32)
        # Total number of frames in the video.
        self.ttl_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self, frames_count=32):
        frames = []
        with self.lock: 
            for _ in range(frames_count):
                ret, frame = self.video.read()
                if not ret:
                    break
                frames.append(frame)
        return frames

    def process_frames(self, frames):
        acc = np.zeros(self.shape)
        for i in range(len(frames)):
            t = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / float(self.ttl_frames)
            acc += t

        with self.lock: 
            self.buffer += acc

    def worker(self):
        while True:
            frames = self.read_frames()
            if not frames:
                break  
            self.process_frames(frames)

    def start_processing(self, num_workers=16):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]
            for future in futures:
                future.result()
        return self.release_resources()

    def release_resources(self):
        self.video.release()
        return self.buffer.astype(np.uint8)
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from qtpy.QtCore import QThread, QObject, QTimer, Qt, Signal, Slot
from PyQt5.QtCore import pyqtSignal


class QtWorkerVMP(QObject):

    bg_ready = pyqtSignal(np.ndarray, str)

    def __init__(self, video_path, shape):
        super().__init__()
        self.video_path = video_path
        self.shape      = shape

    def run(self):
        vmp = VideoMeanProcessor(self.video_path, self.shape)
        ref = vmp.start_processing()
        self.bg_ready.emit(ref, self.video_path)


if __name__ == "__main__":
    directory = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/"
    name      = "WIN_20210830_11_11_50_Pro.mp4"
    full_path = os.path.join(directory, name)
    processor = VideoMeanProcessor(full_path, (480, 640))

    start_time = time.time()
    m = processor.start_processing()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The code block took {elapsed_time} seconds to execute.")

    tifffile.imwrite("/home/benedetti/Desktop/mean-test.tif", m)
