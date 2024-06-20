from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import numpy as np
import time
import os
from skimage.measure import label as cc_labeling
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass
from skimage.morphology import erosion, square
from entry_exit_mouse_box.utils import smooth_path_2d


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
    Calculates an array indicating for each box (designated by the labels in 'areas') if a mouse is inside or not.
    The process is realized on several threads.
    The input video was saved as a mask (0=BG, 255=FG) but it requires thresholding anyway due to compression.
    The areas are a grayscale image with one value per box (0=BG).
    To process the presence of a mouse, we use the length of the ellipse fitted to the mouse's label.
    It requires the input image to be calibrated.
    The process doesn't start from the frame 0 but from the frame 'start'.
    We don't need a control structure to write in the buffer as the threads are not writing in the same place.
    """
    def __init__(self, mask_path, areas, ma, start, duration):
        self.video_path   = mask_path
        self.duration     = int(duration)
        self.video        = cv2.VideoCapture(mask_path)
        self.lock         = threading.Lock()
        self.regions      = areas
        self.ttl_frames   = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.video.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {self.ttl_frames}")
        print(f"FPS: {self.fps}")
        self.vals         = set([int(i) for i in np.unique(areas) if int(i) != 0])
        print(f"Boxes: {self.vals}")
        self.n_boxes      = len(self.vals)
        self.reader_pos   = 0
        self.min_area     = ma
        self.start        = start
        print(f"Starters: {start}")

        self.visibility   = np.zeros((self.n_boxes, self.ttl_frames), np.int8)
        self.centroids    = np.zeros((self.ttl_frames, self.n_boxes, 2), float)
        self.in_out_count = np.zeros((self.n_boxes, 1), np.uint16)
        self.sessions     = None

        self.centroids.fill(-1.0)

    def read_frames(self, frames_count=32):
        with self.lock:
            frames = []
            p1 = self.reader_pos
            p2 = p1 + frames_count
            self.reader_pos = p2 

            for _ in range(frames_count):
                ret, frame = self.video.read()
                if not ret:
                    break
                frames.append(frame)
            
            return (p1, p2), frames


    def process_visibility_pos(self, batch, frames):
        for i, frame in enumerate(frames):
            mask  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127
            mask  = mask.astype(np.float32)
            mask *= self.regions
            mask  = mask.astype(np.uint8)

            vals, count = np.unique(mask, return_counts=True)
            values = {}
            # Bundle values in dict
            for v, c in zip(vals, count):
                if v == 0:
                    continue
                else:
                    values[v] = c
            # Adding missing values
            for box in self.vals:
                if values.get(box, None) is None:
                    values[box] = 0
            # Setting visibility
            for v, c in values.items():
                if batch[0] + i < self.start[v]:
                    self.visibility[v-1, batch[0] + i] = -1
                    continue
                if c < self.min_area: # Too small to be considered
                    self.visibility[v-1, batch[0] + i] = 0
                else:
                    self.visibility[v-1, batch[0] + i] = 1
                    self.centroids[batch[0] + i, v-1] = center_of_mass(mask == v)

    def worker(self):
        while True:
            batch, frames = self.read_frames()
            if not frames:
                break
            self.process_visibility_pos(batch, frames)


    # This function takes the array containing the centroid for each box and smooth the path
    # It skips the moments where the mouse is hidden, represented by (-1, -1)
    def smooth_centroids(self):
        for box in range(self.n_boxes):
            path = self.centroids[:, box]
            smoothed_path = []
            for i, p in enumerate(path):
                if p[0] >= 0.0:
                    smoothed_path.append(p)
            smoothed_path = np.array(smoothed_path)
            smoothed_path = smooth_path_2d(smoothed_path)
            for i, p in enumerate(smoothed_path):
                self.centroids[i, box] = p


    def fix_visibility(self):
        min_session = np.ceil(self.fps)
        for box in range(self.n_boxes):
            swaps = []
            end_of_video = min(self.start[box+1] + self.duration, self.ttl_frames-1)
            for f in range(self.ttl_frames):
                if self.visibility[box, f] == -1:
                    continue
                if f >= end_of_video:
                    self.visibility[box, f] = -2
                    self.centroids[f, box]  = (-1.0, -1.0)
                    continue
                # State transition, the next session starts at (f+1)
                if (self.visibility[box, f] != self.visibility[box, f+1]) or (f == end_of_video-1):
                    # It's the first transition, we don't care about the duration of the session
                    if len(swaps) == 0:
                        swaps.append(f+1)
                        continue
                    # We are in an unstable state.
                    if (f + 1 - swaps[-1] < min_session):
                        swaps.append(f + (1 if (f < end_of_video-1) else 2))
                    else:
                        for i in range(swaps[0], swaps[-1]):
                            self.visibility[box, i] = 0
                            self.centroids[i, box]  = (-1.0, -1.0)
                        swaps = [f+1]


    def start_processing(self, num_workers=os.cpu_count()):
        print("(1/3) Processing visibility...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]
            for future in futures:
                future.result()
        print("(2/3) Processing number of in/out...")
        self.fix_visibility()
        self.process_n_in_out()
        # self.smooth_centroids()
        print("(3/3) Processing sessions time and distance...")
        self.process_sessions()

        return self.release_resources()


    def release_resources(self):
        self.video.release()
        return self.visibility
    

    def process_n_in_out(self):
        for f in range(self.ttl_frames-1):
            for box in range(self.n_boxes):
                if self.visibility[box, f] < 0:
                    continue
                if self.visibility[box, f] != self.visibility[box, f+1]:
                    self.in_out_count[box] += 1


    def process_sessions(self):
        """
        We call 'session' the span of time during which the mouse is hidden or visible.
        For each box, a video is a succession of sessions, alternating between hidden and visible.
        During a session, the mouse is either hidden or visible.
        A session is defined by a duration (in seconds) and a distance (in pixels).
        """
        shape = (self.n_boxes, np.max(self.in_out_count)+1, 4)
        # session[box, session_index] = (frame_start, duration (s), duration (f), distance)
        self.sessions = np.zeros(shape, float)
        self.sessions.fill(np.nan)
        # (starting frame of the session, visibility for this session, session index, session distance)
        state = [(0, None, 0, 0.0) for _ in range(self.n_boxes)]
        eov = []
        for box in range(self.n_boxes):
            eov.append(False)

        for f in range(self.ttl_frames):
            for box in range(self.n_boxes):
                if self.visibility[box, f] < 0:
                    if self.visibility[box, f] == -2:
                        if eov[box]:
                            continue
                        else:
                            eov[box] = True
                    else:
                        continue
                if state[box][1] is None:
                    # Very first session for this box.
                    state[box] = (f, self.visibility[box, f], 0, 0.0)
                    continue
                # When we change of visibility status (or if we reached the last frame), we end the current session.
                if (self.visibility[box, f] != state[box][1]) or (eov[box]):
                    v = self.visibility[box, f]
                    n_frames = f - state[box][0]
                    duration = float(n_frames) / float(self.fps)
                    row = box
                    col = state[box][2]
                    self.sessions[row, col] = (float(state[box][0]), duration, float(n_frames), state[box][3])
                    state[box] = (f, v, col+1, 0.0)
                else:
                    # We accumulate the distance.
                    if self.centroids[max(0, f-1), box][0] >= 0.0:
                        state[box] = (state[box][0], state[box][1], state[box][2], state[box][3] + np.linalg.norm(self.centroids[f, box] - self.centroids[f-1, box]))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from qtpy.QtCore import QThread, QObject, QTimer, Qt, Signal, Slot
from PyQt5.QtCore import pyqtSignal


class QtWorkerMVP(QObject):

    measures_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, mask_path, areas, ma, start, duration):
        super().__init__()
        self.mask_path = mask_path
        self.areas     = areas
        self.min_area  = ma
        self.start     = start
        self.duration  = duration

    def run(self):
        mvp = MiceVisibilityProcessor(self.mask_path, self.areas, self.min_area, self.start, self.duration)
        mvp.start_processing()

        visibility = mvp.visibility
        in_out_count = mvp.in_out_count
        sessions = mvp.sessions
        centroids = mvp.centroids

        self.measures_ready.emit(visibility, in_out_count, sessions, centroids)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    

if __name__ == "__main__":
    import tifffile

    mask_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/mask-WIN_20210830_11_11_50_Pro.avi"
    start_f   = {
        1: 1189, 
        2: 1189, 
        3: 1189
    }
    areas_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/areas-WIN_20210830_11_11_50_Pro.tif"
    areas = tifffile.imread(areas_path)
    scale = 0.13153348
    unit = "cm"

    start = time.time()
    mvp = MiceVisibilityProcessor(mask_path, areas, 80, start_f)
    mvp.start_processing()
    duration = time.time() - start
    print(f"Processing took {duration:.2f} seconds.")

    sessions = mvp.sessions
    np.savetxt("/home/benedetti/Desktop/sessions.csv", sessions[0], delimiter=",", fmt="%.3f")
    