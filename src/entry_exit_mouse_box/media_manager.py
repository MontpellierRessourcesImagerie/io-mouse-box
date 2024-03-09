import cv2
import os


def properties_match(p1, p2):
    if p1['total_frames'] != p2['total_frames']:
        return False
    
    if abs(p1['fps'] - p2['fps']) > 0.2:
        return False
    
    if (p1['width'] != p2['width']) or (p1['height'] != p2['height']):
        return False
    
    return True


class MediaManager:
    def __init__(self, viewer):
        self.sources       = [] # (file_path, capture_instance, layer_name, process_function, image_category)
        self.properties    = [] # Available keys: total_frames, fps, width, height
        self.current_frame = -1 # True index, not the displayed index (starts at 0)
        self.viewer        = viewer # Instance of the Napari viewer
        self.logger        = None

    def __del__(self):
        self.release()

    def set_logger(self, logger):
        self.logger = logger

    def release(self):
        for source in self.sources:
            _, capture, _, _, _ = source
            capture.release()
        self.sources.clear()
        self.properties.clear()
        self.current_frame = -1

    def get_source_by_index(self, index):
        if index < 0 or index >= len(self.sources):
            raise ValueError("ERROR: Index out of range.")
        return self.sources[index]
    
    def get_source_by_name(self, name):
        for source in self.sources:
            _, _, layer_name, _, _ = source
            if layer_name == name:
                return source
        return None
    
    def get_n_frames(self):
        if len(self.properties) == 0:
            raise ValueError("ERROR: No media opened.")
        return self.properties[0]['total_frames']
    
    def get_fps(self):
        if len(self.properties) == 0:
            raise ValueError("ERROR: No media opened.")
        return self.properties[0]['fps']
    
    def get_width(self):
        if len(self.properties) == 0:
            raise ValueError("ERROR: No media opened.")
        return self.properties[0]['width']
    
    def get_height(self):
        if len(self.properties) == 0:
            raise ValueError("ERROR: No media opened.")
        return self.properties[0]['height']
    
    def get_n_sources(self):
        return len(self.sources)
    
    def release_source(self, index):
        if index < 0 or index >= len(self.sources):
            raise ValueError("ERROR: Index out of range.")
        _, capture, _, _, _ = self.sources[index]
        capture.release()
        self.sources.pop(index)
        self.properties.pop(index)
        if index == 0:
            self.current_frame = -1
        return None

    def add_source(self, file_path, target_layer, img_type, process=None):
        """
        Add a new video source to the manager.

        Args:
            file_path   : Path of the video file to be opened.
            target_layer: Name of the layer to which each frame of the video will be loaded.
            img_type    : Type of the image to be loaded ('image' or 'labels').
            process     : Function to be applied to each frame of the video before displaying it.

        Raises:
            FileNotFoundError: If the file is not found at the specified path.
            IOError          : If the file cannot be opened.
            ValueError       : If the properties of the video do not match the properties of the media already opened.

        Returns:
            The properties of the last video added under the form of a dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ERROR: File not found at {file_path}")
        
        # Checking that the target name is not already in use. If it is, release the source before adding the new one.
        for idx, source in enumerate(self.sources):
            if source[2] == target_layer:
                self.release_source(idx)

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            raise IOError("ERROR: Failed to open video file.")

        properties = {
            'total_frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps'         : capture.get(cv2.CAP_PROP_FPS),
            'width'       : int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height'      : int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

        # Checking that the file is not already opened. If it is, return the known properties.
        for idx, source in enumerate(self.sources):
            if source[0] == file_path:
                print("The file is already opened.")
                return properties[idx]

        # Checking that the properties are compatible with the media already opened.
        if len(self.properties) > 0 and not properties_match(properties, self.properties[0]):
            self.logger.error("The properties of the video do not match the properties of the media already opened.")
            self.logger.error(f"Video properties: {properties}")
            self.logger.error(f"Media properties: {self.properties[0]}")
            raise ValueError("ERROR: The properties of the video do not match the properties of the media already opened.")

        self.sources.append((file_path, capture, target_layer, process, img_type))
        self.properties.append(properties)

        if self.current_frame == -1:
            self.current_frame = 0

        # Set the correct frame index and extract the given frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = capture.read()

        if process is not None:
            frame = process(frame)
        if not ret:
            raise IOError("ERROR: Failed to read frame.")

        if target_layer in self.viewer.layers:
            layer = self.viewer.layers[target_layer]
            layer.data = frame
        else:
            if img_type == "image":
                layer = self.viewer.add_image(
                    frame, 
                    name=target_layer
                )
            elif img_type == "labels":
                layer = self.viewer.add_labels(
                    frame, 
                    name=target_layer,
                    blending="additive"
                )
            else:
                raise ValueError("ERROR: The image type is not recognized.")
        
        return properties


    def set_frame(self, frame_target):
        if len(self.sources) == 0:
            raise ValueError("ERROR: No media opened.")
        
        frame_number = int(frame_target)
        frame_number = max(0, frame_number)
        frame_number = min(frame_number, self.properties[0]['total_frames']-1)

        if frame_number == self.current_frame:
            return self.current_frame

        self.current_frame = frame_number

        for source in self.sources:
            _, capture, target_layer, process, img_type = source
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = capture.read()

            if process is not None:
                frame = process(frame)
            if not ret:
                print("ERROR: Failed to read frame.")
                return None

            if target_layer in self.viewer.layers:
                layer = self.viewer.layers[target_layer]
                layer.data = frame
            else:
                if img_type == "image":
                    self.viewer.add_image(
                        frame, 
                        name=target_layer
                    )
                elif img_type == "labels":
                    self.viewer.add_labels(
                        frame, 
                        name=target_layer
                    )
                else:
                    raise ValueError("ERROR: The image type is not recognized.")
    
    def get_video_properties(self):
        return self.properties

    def get_current_frame_number(self):
        return self.current_frame