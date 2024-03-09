
import cv2
import threading
import os
import tifffile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from queue import Queue


class MaskFromBackground(object):
    def __init__(self, input_video_path, output_video_path, ref, tr=75, st=10, frame_count=64):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.frame_count = frame_count
        self.reference = ref
        self.video = cv2.VideoCapture(input_video_path)
        self.ttl_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.processed_frames = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref.shape[1], ref.shape[0]), isColor=False)
        self.threshold = tr
        self.expected_index = 0
        self.current_index = 0
        self.total = int(self.ttl_frames/frame_count)+1
        self.start = st
        self.reader_pos = 0

    def read_frames(self):
        frames = []
        p1 = self.reader_pos
        p2 = p1 + self.frame_count
        self.reader_pos = p2
        for _ in range(self.frame_count):
            ret, frame = self.video.read()
            if not ret:
                break
            frames.append(frame)
        return (p1, p2), frames

    def process_frames(self, batch, frames):
        buffer_out = []
        for i, frame in enumerate(frames):
            if batch[0] + i < self.start:
                diff = np.zeros(self.reference.shape[0:2], bool)
            else:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                diff = np.abs(processed_frame - self.reference) > self.threshold
            buffer_out.append(diff)
        return buffer_out

    def worker(self):
        while True:
            frames = None
            launch_index = 0

            with self.lock:
                batch, frames = self.read_frames()
                launch_index = self.current_index
                self.current_index += 1
            
            if not frames:
                break

            processed_frames = self.process_frames(batch, frames)

            with self.condition:
                self.processed_frames[launch_index] = processed_frames
                while self.processed_frames.get(self.expected_index):
                        print(f"{self.expected_index+1}/{self.total}")
                        self.add_frames_to_video(self.processed_frames.pop(self.expected_index))
                        self.expected_index += 1
                self.condition.notify_all()

    def add_frames_to_video(self, frames):
        for frame in frames:
            self.video_out.write(np.uint8(frame * 255))

    def start_processing(self, num_workers=16):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]
            for future in futures:
                future.result()

    def release_resources(self):
        self.video.release()
        self.video_out.release()
        


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from qtpy.QtCore import QThread, QObject, QTimer, Qt, Signal, Slot
from PyQt5.QtCore import pyqtSignal


class QtWorkerMFV(QObject):

    mask_ready = pyqtSignal(str)

    def __init__(self, in_path, out_path, ref, t, s):
        super().__init__()
        self.in_path   = in_path
        self.out_path  = out_path
        self.ref       = ref
        self.threshold = t
        self.start     = s

    def run(self):
        mfb = MaskFromBackground(self.in_path, self.out_path, self.ref, self.threshold, self.start)
        mfb.start_processing()
        self.mask_ready.emit(self.out_path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    directory = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/"
    name      = "WIN_20210830_11_11_50_Pro.mp4"
    full_path = os.path.join(directory, name)
    out_path  = os.path.join(directory, "test-mask-2.avi")
    ref = tifffile.imread("/home/benedetti/Desktop/mean-test.tif")
    
    start_time = time.time()
    processor = MaskFromBackground(full_path, out_path, ref)
    processor.start_processing()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The code block took {elapsed_time} seconds to execute.")

