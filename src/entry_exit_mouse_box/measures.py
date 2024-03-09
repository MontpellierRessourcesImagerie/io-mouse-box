from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import numpy as np
import time
from skimage.measure import label as cc_labeling
from skimage.measure import regionprops
from skimage.morphology import erosion, square


def calculate_maximal_length(mask):
    # If the only value is False, we can return 0
    if not np.any(mask):
        return 0
    
    # Find contours in the mask
    eroded_mask = erosion(mask, square(1))
    contours = mask - eroded_mask
    
    # In case there are multiple objects, you might want to choose the largest one or handle each separately.
    # Here, we just take the largest contour for simplicity.
    
    # Calculate the convex hull of the contour
    hull = cv2.convexHull(contours.astype(np.uint8) * 255)
    
    # Calculate the maximal length (Feret diameter)
    maximal_length = 0
    p1, p2 = None, None
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            dist = np.linalg.norm(hull[i] - hull[j])
            if dist > maximal_length:
                maximal_length = dist
                p1, p2 = hull[i], hull[j]
    
    return maximal_length


class MiceVisibilityProcessor(object):
    """
    This class processes an array indicating for each box (designated by the labels in 'areas') if a mouse is inside or not.
    The process is realized on several threads.
    The input video is was saved as a mask (0=BG, 255=FG) but it requires thresholding anyway due to compression.
    The areas are a grayscale image with one value per box (0=BG).
    To process the presence of a mouse, we use the length of the ellipse fitted to the mouse's label.
    It requires the input image to be calibrated.
    The process doesn't start from the frame 0 but from the frame 'start'.
    We don't need a control structure to write in the buffer as the threads are not writing in the same place.
    """
    def __init__(self, mask_path, areas, start, ml=20):
        self.video_path = mask_path
        self.video = cv2.VideoCapture(mask_path)
        self.lock = threading.Lock()
        self.start = start
        self.regions = areas
        self.ttl_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vals = np.unique(areas)
        self.n_boxes = len(self.vals) if 0 in self.vals else len(self.vals) - 1
        self.buffer = np.zeros((self.n_boxes, self.ttl_frames), bool)
        self.reader_pos = 0
        self.min_length = ml

    def read_frames(self, frames_count=32):
        frames = []
        p1 = self.reader_pos
        p2 = p1 + frames_count
        self.reader_pos = p2
        with self.lock: 
            for _ in range(frames_count):
                ret, frame = self.video.read()
                if not ret:
                    break
                frames.append(frame)
        return (p1, p2), frames

    def process_frames(self, batch, frames):
        for i, frame in enumerate(frames):
            if i + batch[0] < self.start:
                continue
            for box in self.vals:
                if box == 0:
                    continue
                # Clearing outside the box of interest
                mask = ~(self.regions == box)
                working_copy = frame.copy()
                working_copy[mask] = 0
                
                # Fitting an ellipse to the mouse's mask
                length = calculate_maximal_length(mask)
                self.buffer[box - 1, i + batch[0]] = length > self.min_length

    def worker(self):
        while True:
            batch, frames = self.read_frames()
            if not frames:
                break  
            self.process_frames(batch, frames)

    def start_processing(self, num_workers=16):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]
            for future in futures:
                future.result()
        return self.release_resources()

    def release_resources(self):
        self.video.release()
        return self.buffer
    

if __name__ == "__main__":
    mask_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/mask-WIN_20210830_11_11_50_Pro.avi"
    start     = 1189
    areas_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/areas-WIN_20210830_11_11_50_Pro.tif"
    scale = 0.13153348
    unit = "cm"
    
    start = time.time()
    mvp = MiceVisibilityProcessor(mask_path, areas_path, start)
    mvp.start_processing()
    duration = time.time() - start
    print(f"Processing took {duration:.2f} seconds.")