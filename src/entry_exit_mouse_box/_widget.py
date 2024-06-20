from typing import TYPE_CHECKING

import os
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLineEdit,
                             QSpinBox, QTableWidget, QTableWidgetItem, QColorDialog, QComboBox,
                             QGroupBox, QLabel, QHeaderView, QFileDialog, QFrame, QCheckBox)

import cv2
from PyQt5.QtGui import QFont, QColor, QDoubleValidator
from qtpy.QtCore import QThread, Qt, Signal, Slot

import tifffile
import json
import numpy as np
from skimage.draw import polygon
from napari.utils.notifications import show_info, show_error, show_warning
from napari.utils import progress
from shapely.geometry import Polygon
import tempfile
from datetime import datetime

# if TYPE_CHECKING:
import napari
from entry_exit_mouse_box.media_manager import MediaManager
from entry_exit_mouse_box.video_mean_processor import QtWorkerVMP
from entry_exit_mouse_box.mask_from_video import QtWorkerMFV
from entry_exit_mouse_box.measures import QtWorkerMVP
from entry_exit_mouse_box.utils import setup_logger, smooth_path_2d, apply_lut
from entry_exit_mouse_box.results_table import SessionsResultsTable, FrameWiseResultsTable


BG_REF_LAYER      = "bg_ref"
MICE_LABELS_LAYER = "mice_labels"
AREAS_LAYER       = "areas"
MEDIA_LAYER       = "media"
TS_PREVIEW_LAYER  = "threshold_preview" 
CENTOIDS_LAYER    = "centroids"
PATH_LAYER        = "path"
FONT              = QFont()
FONT.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")


class MouseInOutWidget(QWidget):

    background_ready = Signal()
    tracking_ready   = Signal()
    measures_ready   = Signal()

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        # List containing the names of layers containing the boxes. (the polygons)
        self.boxes  = []
        # Instance of Napari viewer
        self.viewer = napari_viewer
        # Object managing the media sources (reading and synchronizing the frames, the masks, the labels, ...)
        self.mm     = MediaManager(self.viewer)
        # Dictionary containing the frame at which we start the measures for each box.
        # The key is the index of the row in the table, the value is the frame index.
        self.start  = {}
        # Instance of the logger
        self.logger = None
        # Should path be hidden.
        self.paths_hidden = False
        # Table containing the visibility status. [nBoxes, totalFrames] -> uint8
        # | -1: Mouse not present yet.
        # |  0: Mouse not visible.
        # |  1: Mouse visible.
        self.visibility   = None
        # Table containing the number of times the mouse entered and exited the box. [nBoxes] -> uint32
        self.in_out_count = None
        # Table containing the sessions. [nBoxes, nSessions, 4] -> float
        # | [0] Frame at which the session starts.
        # | [1] Duration of the session in seconds.
        # | [2] Duration of the session in frames.
        # | [3] Distance traveled by the mouse during this session.
        self.sessions     = None
        # Table containing the centroids of the mice. [totalFrames, nBoxes, 2] -> float
        self.centroids    = None
        # Lists of results tables.
        self.sessions_results = []
        self.frames_results   = []
        # Path of the directory containing the temporary files.
        self.temp_dir = None
        # Calibration of the image.
        self.calibration = None
        self.create_temp_dir()

        self.switch_log_file(os.path.join(self.temp_dir, datetime.now().strftime("%Y-%m-%dT%H%M")+".log"))
        self.mm.set_logger(self.logger)
        self.viewer.layers.events.inserted.connect(self.update_calibration)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.add_video_control_ui()
        self.experiment_duration_ui()
        self.add_media_control_ui()
        self.add_calibration_ui()
        self.add_tracking_ui()

    def add_video_control_ui(self):
        group_box = QGroupBox("Video Control")
        layout    = QVBoxLayout(group_box)

        # Clear state button.
        self.clear_state_button = QPushButton("✨ Clear state", self)
        self.clear_state_button.setFont(FONT)
        self.clear_state_button.clicked.connect(self.clear_state)
        layout.addWidget(self.clear_state_button)

        # Vertical spacing
        spacer = QWidget()
        spacer.setFixedSize(0, 20)  # Width, Height
        layout.addWidget(spacer)

        # Button to select the video.
        self.file_button = QPushButton("📁 Select File", self)
        self.file_button.setFont(FONT)
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Buttons 'backward' and 'forward'
        self.backward_button = QPushButton("⏮️ Backward", self)
        self.backward_button.setFont(FONT)
        self.backward_button.clicked.connect(self.jump_backward)

        self.forward_button = QPushButton("Forward ⏭️", self)
        self.forward_button.setFont(FONT)
        self.forward_button.clicked.connect(self.jump_forward)
        
        nav_layout_2 = QHBoxLayout()
        nav_layout_2.addWidget(self.backward_button)
        nav_layout_2.addWidget(self.forward_button)
        layout.addLayout(nav_layout_2)

        # Slider
        self.slider = QSlider(self)
        self.slider.setValue(-1)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setOrientation(Qt.Horizontal)
        layout.addWidget(self.slider)

        # Slot d'entier
        self.frame_input = QSpinBox(self)
        self.frame_input.setValue(-1)
        self.frame_input.valueChanged.connect(self.on_spinbox_change)
        self.frame_input.setPrefix("Frame: ")
        layout.addWidget(self.frame_input)

        # Vertical spacing
        spacer = QWidget()
        spacer.setFixedSize(0, 20)  # Width, Height
        layout.addWidget(spacer)

        # Video name
        self.video_name = QLabel("<b>---</b>", self)
        self.video_name.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.video_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_name)

        # Time in seconds
        self.info_label = QLabel("0 sec", self)
        self.info_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # Video properties
        self.properties_display = QLabel("0x0 (0 FPS) ↦ 0s", self)
        self.properties_display.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.properties_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.properties_display)

        self.layout.addWidget(group_box)

    def experiment_duration_ui(self):
        # Group box
        self.groupBox = QGroupBox("Experiment duration")
        groupBoxLayout = QVBoxLayout()

        # Spin boxes and labels layout
        entryLayout = QHBoxLayout()
        
        # Spin boxes and labels
        self.minutesSpin = QSpinBox()
        self.minutesSpin.setRange(1, 50)
        self.minutesSpin.setSuffix(' min')
        self.minutesSpin.setValue(10)
        
        self.secondsSpin = QSpinBox()
        self.secondsSpin.setRange(0, 59)
        self.secondsSpin.setSuffix(' sec')

        # Adding widgets to the entry layout
        entryLayout.addWidget(self.minutesSpin)
        entryLayout.addWidget(self.secondsSpin)

        # Add entry layout to group box layout
        groupBoxLayout.addLayout(entryLayout)

        # Set group box layout
        self.groupBox.setLayout(groupBoxLayout)
        
        # Add group box to main layout
        self.layout.addWidget(self.groupBox)

    def duration_to_frames(self):
        minutes = self.minutesSpin.value()
        seconds = self.secondsSpin.value()
        fps = self.mm.get_fps()
        return int((minutes * 60 + seconds) * fps)

    def add_media_control_ui(self):
        group_box = QGroupBox("Box Control")
        layout = QVBoxLayout(group_box)

        # Boutons add et pop
        btn_layout = QHBoxLayout()
        self.add_box_button = QPushButton("🔵 Add Box", self)
        self.add_box_button.setFont(FONT)
        self.add_box_button.clicked.connect(self.add_row)
        self.pop_box_button = QPushButton("❌ Pop Box", self)
        self.pop_box_button.setFont(FONT)
        self.pop_box_button.clicked.connect(self.remove_row)

        btn_layout.addWidget(self.add_box_button)
        btn_layout.addWidget(self.pop_box_button)
        
        layout.addLayout(btn_layout)

        # Tableau
        cols = ["Color", "Name", "Start"]
        self.table = QTableWidget(0, len(cols), self)
        self.table.setHorizontalHeaderLabels(cols)
        # self.table.horizontalHeader().setStretchLastSection(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)

        self.layout.addWidget(group_box)


    def update_calibration(self, event):
        if self.calibration is None:
            return
        pixelSize, unit = self.calibration
        for layer in self.viewer.layers:
            layer.scale = (pixelSize, pixelSize)
        self.viewer.scale_bar.unit = unit
        self.viewer.scale_bar.visible = True

    
    def add_calibration_ui(self):
        self.calibrationGroup = QGroupBox("Calibration")
        self.calibrationLayout = QVBoxLayout()

        nav_layout = QHBoxLayout()

        # Create QLineEdit for float input
        self.calibInput = QLineEdit()
        float_validator = QDoubleValidator()
        float_validator.setNotation(QDoubleValidator.StandardNotation)
        self.calibInput.setValidator(float_validator)
        nav_layout.addWidget(self.calibInput)

        # Create QComboBox for unit selection
        self.unitSelector = QComboBox()
        units = ["mm", "cm", "dm", "m"]  # Define the units from nanometers to meters
        self.unitSelector.addItems(units)
        nav_layout.addWidget(self.unitSelector)

        # Add the nav_layout to the calibration layout
        self.calibrationLayout.addLayout(nav_layout)

        # Apply calibration button
        self.calibrationButton = QPushButton("📏 Apply calibration")
        self.calibrationButton.setFont(FONT)
        self.calibrationButton.clicked.connect(self.apply_calibration)
        self.calibrationLayout.addWidget(self.calibrationButton)

        self.calibrationGroup.setLayout(self.calibrationLayout)
        self.layout.addWidget(self.calibrationGroup)


    def add_tracking_ui(self):
        group_box = QGroupBox("Tracking")
        layout = QVBoxLayout(group_box)

        size_layout = QHBoxLayout()

        self.set_min_area_button = QPushButton("📏 Set area", self)
        self.set_min_area_button.setFont(FONT)
        self.set_min_area_button.clicked.connect(self.set_min_area)
        size_layout.addWidget(self.set_min_area_button)

        self.min_area = QSpinBox(self)
        self.min_area.setValue(0)
        self.min_area.setMaximum(1000000)
        size_layout.addWidget(self.min_area)

        layout.addLayout(size_layout)

        self.extract_button = QPushButton("♻️ Clear background", self)
        self.extract_button.setFont(FONT)
        self.extract_button.clicked.connect(self.start_extract_background)
        layout.addWidget(self.extract_button)

        track_layout = QHBoxLayout()

        self.track_button = QPushButton("🐀 Launch tracking", self)
        self.track_button.setFont(FONT)
        self.track_button.clicked.connect(self.launch_tracking)
        track_layout.addWidget(self.track_button)

        self.threshold = QSpinBox(self)
        self.threshold.setValue(60)
        self.threshold.setMaximum(255)
        self.threshold.valueChanged.connect(self.on_threshold_update)
        track_layout.addWidget(self.threshold)

        layout.addLayout(track_layout)

        measure_layout = QHBoxLayout()

        self.extract_measures_button = QPushButton("📐 Measure", self)
        self.extract_measures_button.setFont(FONT)
        self.extract_measures_button.clicked.connect(self.extract_measures)
        measure_layout.addWidget(self.extract_measures_button)

        # Checkbox to hide and stop refreshing the paths:
        self.no_path_checkbox = QCheckBox("Hide paths", self)
        self.no_path_checkbox.stateChanged.connect(self.hide_paths)
        measure_layout.addWidget(self.no_path_checkbox)

        layout.addLayout(measure_layout)

        # Button to export the results.
        self.export_button = QPushButton("📤 Create results tables", self)
        self.export_button.setFont(FONT)
        self.export_button.clicked.connect(self.export_results)
        layout.addWidget(self.export_button)

        self.layout.addWidget(group_box)

    
    def apply_calibration(self):
        print("Applying calibration...")
        # Checking that there is an active layer and that is a shape layer containing a unique line.
        if not self.viewer.layers.selection.active:
            return
        source = self.viewer.layers.selection.active
        if 'shape_type' not in dir(source):
            return
        shape_types = source.shape_type
        if len(shape_types) != 1:
            return
        if shape_types[0] != 'line':
            return
        p0, p1 = source.data[0]
        p0 = p0[-2:]
        p1 = p1[-2:]
        distance = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) # in pixels
        length = float(self.calibInput.text().replace(",", "."))
        pixelSize = length / distance
        unit = self.unitSelector.currentText()
        self.set_calibration(pixelSize, unit)
        name = source.name
        del self.viewer.layers[name]
        if len(self.viewer.layers) > 0:
            self.viewer.layers.selection.active = self.viewer.layers[-1]


    def set_calibration(self, pixelSize, unit):
        self.calibration = (pixelSize, unit)
        for layer in self.viewer.layers:
            layer.scale = (pixelSize, pixelSize)
        self.viewer.scale_bar.unit = unit
        self.viewer.scale_bar.visible = True
        print(f"Calibration applied: {pixelSize:.2f} {unit} per pixel.")


    def hide_paths(self, state):
        if PATH_LAYER in self.viewer.layers:
            self.viewer.layers[PATH_LAYER].visible = (state == Qt.Unchecked)
            if (state == Qt.Unchecked):
                self.update_boxes()

    def set_min_area(self):
        """
        Uses a polygon drawn by the user over the head of a mouse to define the minimal area to consider that a mouse is present.
        When summoned, this function must find a shape layer as the active layer, and extract the area of the polygon.
        The shape layer is then deleted.
        The value extracted is not stored, it is written in the spinbox.
        """
        # Checking required ressources (media and shape layer).
        if (not self.mm.active) or (self.mm.get_n_sources() == 0):
            show_error("No media opened.")
            self.logger.error("No media opened.")
            return
        
        if self.viewer.layers.selection.active is None:
            show_error("No layer selected.")
            self.logger.error("No layer selected.")
            return
        
        # The active layer must be the shape layer, containing a polygon around the head of a mouse.
        active_layer = self.viewer.layers.selection.active
        active_layer.scale = self.viewer.layers[MEDIA_LAYER].scale

        # Checking that the active layer is a shape layer.
        if 'add_lines' not in dir(active_layer):
            show_error("A shape layer is expected.")
            self.logger.error("A shape layer is expected.")
            return
        
        # Checking that the shape layer is not empty.
        if len(active_layer.data) == 0:
            show_error("No data in the layer.")
            self.logger.error("No data in the layer.")
            return
        
        shape = active_layer.data[-1]
        if len(shape) <= 2:
            show_error("The shape must be a polygon.")
            self.logger.error("The shape must be a polygon.")
            return
        
        poly = Polygon(shape)
        area_pixels = int(poly.area) # in number of pixels
        self.min_area.setValue(area_pixels)

        del self.viewer.layers[active_layer.name]
        self.viewer.layers.selection.active = self.viewer.layers[MEDIA_LAYER]

    def on_threshold_update(self, value):
        """
        Creates a preview layer showing what the mask would be for the current frame for a given threshold.
        Updates are made only when the threshold is edited.
        The background reference is required.
        The produced layer is named 'threshold_preview' and is temporary, it will be discarded.
        This function is the callback for the threshold spinbox.

        Args:
            value: int - The threshold value to use for the preview.
        """
        if BG_REF_LAYER not in self.viewer.layers:
            show_warning("Couldn't find the background reference.")
            self.logger.warning("Couldn't find the background reference. Layers list: " + str([l.name for l in self.viewer.layers]))
            return
        
        bg_ref = self.viewer.layers[BG_REF_LAYER].data
        img    = self.viewer.layers[MEDIA_LAYER].data
        diff   = np.abs(img.astype(np.float32) - bg_ref) > value

        if TS_PREVIEW_LAYER in self.viewer.layers:
            self.viewer.layers[TS_PREVIEW_LAYER].data = diff
        else:
            self.viewer.add_image(
                diff, 
                name=TS_PREVIEW_LAYER,
                blending="translucent",
                opacity=0.8
            )

    def toggle_inputs(self, t):
        """
        Used to disable the inputs (buttons and text fields) when a long process is running.

        Args:
            t: bool - True to enable the inputs, False to disable them.
        """
        self.file_button.setEnabled(t)
        self.backward_button.setEnabled(t)
        self.forward_button.setEnabled(t)
        self.slider.setEnabled(t)
        self.frame_input.setEnabled(t)

        self.add_box_button.setEnabled(t)
        self.pop_box_button.setEnabled(t)
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, 0).setEnabled(t)
            if t:
                self.table.item(row, 1).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            else:
                item = self.table.item(row, 1)
                item.setFlags(item.flags() & ~(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable))
            self.table.cellWidget(row, 2).setEnabled(t)

        self.clear_state_button.setEnabled(t)
        self.extract_measures_button.setEnabled(t)
        self.track_button.setEnabled(t)
        self.extract_button.setEnabled(t)
        self.threshold.setEnabled(t)
        self.set_min_area_button.setEnabled(t)
        self.min_area.setEnabled(t)
        self.no_path_checkbox.setEnabled(t)
        self.export_button.setEnabled(t)
        self.calibInput.setEnabled(t)
        self.calibInput.setEnabled(t)
        self.unitSelector.setEnabled(t)
        self.calibrationButton.setEnabled(t)
        self.minutesSpin.setEnabled(t)
        self.secondsSpin.setEnabled(t)

    def switch_log_file(self, new_file_name):
        """
        Creates a new log file when we switch to a new experiment video.
        """
        if self.logger:
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
                
        self.logger = setup_logger(new_file_name)

    def make_start_frame(self, row, button):
        """
        Sets the frame at which we start measures from the current frame being displayed.
        The saved index is the one displayed on the screen. (The real one + 1).
        A source is required before we can set the start frame.
        """
        if self.mm.get_n_sources() == 0:
            return
        self.start[row+1] = self.mm.current_frame
        if self.mm.current_frame + self.duration_to_frames() > self.mm.get_n_frames():
            show_warning("The duration of the experiment exceeds the total number of frames.")
            self.logger.warning("The duration of the experiment exceeds the total number of frames.")
        button.setText(f"▸ {self.start[row+1]+1}")
        name = self.boxes[row]
        self.viewer.layers[name].editable = False

    def on_table_item_changed(self, item):
        """
        Called when the user modifies an existing line of the table containing the colors and names of the boxes.
        The new name is processed here before it is provided by the user.
        Both the layer's name and the box's name are updated.
        """
        if self.boxes[item.row()] not in self.viewer.layers: # The line is being added
            return
        
        new_name = item.text()
        if len(new_name) <= 1:
            self.logger.warning("The name of the box is too short.")
            new_name = f"Box-{len(self.boxes)}"

        self.logger.info(f"Box {item.row()} renamed to: {new_name}")
        self.viewer.layers[self.boxes[item.row()]].name = new_name
        self.boxes[item.row()] = new_name
    
    def select_color(self, row, button, k=None):
        if k is None:
            color = QColorDialog.getColor()
        else:
            color = k
        
        if color.isValid():
            button.setStyleSheet(f'background-color: {color.name()};')
            button.setText(color.name())
            nb = len(self.viewer.layers[self.boxes[row]].edge_color)
            self.viewer.layers[self.boxes[row]].edge_color = [color.name() for i in range(nb)]
            self.viewer.layers[self.boxes[row]].current_edge_color = color.name()

    def add_row(self):
        if self.mm.get_n_sources() == 0:
            show_error("No media opened.")
            self.logger.error("No media opened.")
            return
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        self.boxes.append(f"Box-{row_count}")
        # Adding color picker to the table
        color_button = QPushButton('Pick a color')
        color_button.clicked.connect(lambda: self.select_color(row_count, color_button))
        self.table.setCellWidget(row_count, 0, color_button)
        # Adding the name slot to the table
        item = QTableWidgetItem(f"Box-{row_count}")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.table.setItem(row_count, 1, item)
        # Adding the start frame slot to the table
        start_button = QPushButton("▸ --")
        start_button.clicked.connect(lambda: self.make_start_frame(row_count, start_button))
        self.table.setCellWidget(row_count, 2, start_button)
        # Adding the layer containing the shape
        n = f"Box-{row_count}"
        self.viewer.add_shapes([], name=n, shape_type='polygon', opacity=0.8, edge_width=2, edge_color='#aaaaaaff', face_color='#00000000')
        self.viewer.layers[n].mode = "add_rectangle"
        self.logger.info(f"Box {row_count} added.")

    def remove_row(self):
        row_count = self.table.rowCount()
        if row_count > 0:
            self.table.removeRow(row_count - 1)
            del(self.viewer.layers[self.boxes.pop()])
            self.logger.info(f"Box {row_count-1} removed.")
            del(self.start[row_count-1])

    def extract_measures(self):
        self.toggle_inputs(False)
        show_info("Extracting measures...")
        self.logger.info("Extracting measures...")
        self.pbr = progress(total=0)
        self.pbr.set_description("Extracting measures...")
        self.thread = QThread()

        mask_path = self.mm.get_source_by_name(MICE_LABELS_LAYER)[0]
        areas = self.viewer.layers[AREAS_LAYER].data
        ma = self.min_area.value()
        start = self.start

        self.mvp = QtWorkerMVP(mask_path, areas, ma, start, self.duration_to_frames())
        self.mvp.moveToThread(self.thread)
        self.mvp.measures_ready.connect(self.terminate_measures)
        self.thread.started.connect(self.mvp.run)
        self.thread.start()

    def terminate_measures(self, visibility, in_out_count, sessions, centroids):
        self.logger.info("Measures extracted.")
        show_info("Measures extracted!")
        self.pbr.close()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.toggle_inputs(True)

        self.visibility   = visibility
        np.save(os.path.join(self.temp_dir, "visibility.npy"), visibility)
        self.in_out_count = np.squeeze(in_out_count)
        np.save(os.path.join(self.temp_dir, "in_out_count.npy"), in_out_count)
        self.sessions     = sessions
        np.save(os.path.join(self.temp_dir, "sessions.npy"), sessions)
        self.centroids    = centroids
        np.save(os.path.join(self.temp_dir, "centroids.npy"), centroids)
        self.update_boxes()
        self.measures_ready.emit()

    def export_results(self):
        colors = np.array([self.viewer.layers[n].edge_color[0]*255 for n in self.boxes])

        # Centroids, visibility, name of boxes
        fwrt = FrameWiseResultsTable((
            colors, 
            self.centroids, 
            self.visibility, 
            self.boxes
        ))
        fwrt.set_exp_name("frames-" + os.path.basename(self.mm.get_source_by_name(MEDIA_LAYER)[0]))
        fwrt.show()
        self.frames_results.append(fwrt)

        sessions = self.calibrate_results()
        unit = self.viewer.scale_bar.unit
        unit = "px" if unit is None else str(unit)
        srt = SessionsResultsTable((
            colors, 
            self.in_out_count, 
            sessions, 
            self.boxes, 
            self.visibility,
            unit
        ))
        srt.set_exp_name("sessions-" + os.path.basename(self.mm.get_source_by_name(MEDIA_LAYER)[0]))
        srt.show()
        self.sessions_results.append(srt)

    def calibrate_results(self):
        """
        Applies the calibration to the distance traveled by the mice, stored in the sessions table.
        """
        # Checking if the image is calibrated.
        sessions = np.copy(self.sessions)
        if self.viewer.scale_bar.unit is None:
            show_warning("No calibration found, exporting distances in pixels.")
            self.logger.warning("No calibration found, exporting distances in pixels.")
            return sessions
        
        # Get the size of a pixel.
        pixel_size = self.viewer.layers[MEDIA_LAYER].scale[0]
        show_info(f"Calibration found: XY={pixel_size} {self.viewer.scale_bar.unit}")
        for box in range(len(self.boxes)):
            for session in range(np.max(self.in_out_count)+1):
                sessions[box, session, 3] *= pixel_size
        
        return sessions
    
    def create_temp_dir(self, root=None):
        if root is None:
            self.temp_dir = tempfile.gettempdir()
            return
        
        parent = os.path.dirname(root)
        if not os.path.isdir(parent):
            show_warning(f"Parent directory '{parent}' does not exist. Working in `tmp` directory.")
            self.temp_dir = tempfile.gettempdir()
            return
        
        if not os.path.isdir(root):
            os.mkdir(root)
        else:
            show_warning(f"Directory '{root}' already exists. All its content will be deleted.")
            for f in os.listdir(root):
                os.remove(os.path.join(root, f))
        
        self.temp_dir = root

    def update_boxes(self):
        # Ajouter un layer sur lequel on écrit la durée de la session.
        # Changer la couleur des boxes en fonction de la visibilité.
        if self.visibility is None:
            return
        
        self.update_face_color()
        self.update_centroids()
        self.update_mice_path()

    def update_face_color(self):
        for lbl, b_name in enumerate(self.boxes):
            if b_name not in self.viewer.layers:
                continue
            layer = self.viewer.layers[b_name]
            if self.visibility[lbl, self.mm.current_frame] == 1:
                layer.face_color = '#29b0ff50'
            elif self.visibility[lbl, self.mm.current_frame] == 0:
                layer.face_color = '#ff292950'
            elif self.visibility[lbl, self.mm.current_frame] < 0:
                layer.face_color = '#aaaaaaaa'
            else:
                layer.face_color = '#000000ff'
        
    def update_centroids(self):
        frame_points = self.centroids[self.mm.current_frame]
        if CENTOIDS_LAYER in self.viewer.layers:
            self.viewer.layers[CENTOIDS_LAYER].data = frame_points
        else:
            self.viewer.add_points(
                frame_points, 
                name=CENTOIDS_LAYER, 
                face_color='red', 
                size=6
            )
        layer = self.viewer.layers[CENTOIDS_LAYER]
        layer.edge_color = '#00000000'
        colors = []
        for box in range(len(self.boxes)):
            colors.append("#ff0000ff" if self.visibility[box, self.mm.current_frame] == 1 else "#00000000")
        layer.face_color = colors

    def update_mice_path(self):
        if self.paths_hidden:
            return
        
        full_path = []
        for lbl, b_name in enumerate(self.boxes):
            if b_name not in self.viewer.layers:
                continue
            if self.visibility[lbl, self.mm.current_frame] <= 0:
                continue
            # Finding the session index for this box.
            # A session starts at the first frame where the visibility is 1.
            session_index = 0
            while self.sessions[lbl, session_index][0] <= self.mm.current_frame:
                session_index += 1
                if session_index >= len(self.sessions[lbl]):
                    break
            # Reading the duration in seconds and in frames.
            session_index -= 1
            duration_f  = int(self.sessions[lbl, session_index][2])
            frame_start = int(self.sessions[lbl, session_index][0])
            path = smooth_path_2d(self.centroids[frame_start:frame_start+duration_f, lbl][::int(self.mm.get_fps()/6)])
            if len(path) < 3:
                continue
            full_path.append(path)

        if PATH_LAYER in self.viewer.layers:
            layer = self.viewer.layers[PATH_LAYER]
            layer.data = full_path
            l = len(full_path)
            if l > 0:
                layer.shape_type = ['path' for _ in range(l)]
        else:
            self.viewer.add_shapes(
                full_path, 
                name=PATH_LAYER, 
                shape_type='path', 
                opacity=0.8, 
                edge_width=1, 
                edge_color='#ff0000ff', 
                face_color='#00000000'
            )

    def clear_state(self):
        """
        Resets the state of the widget to its initial state.
        """
        self.mm.release()
        self.viewer.layers.clear()
        self.boxes = []
        self.slider.setMinimum(-1)
        self.slider.setValue(-1)
        self.slider.setMaximum(-1)
        self.frame_input.setMinimum(-1)
        self.frame_input.setValue(-1)
        self.frame_input.setMaximum(-1)
        self.video_name.setText(f"<b>---</b>")
        
        self.table.setRowCount(0)
        self.start = {}
        self.update_playback_info()
        
        for f_r in self.frames_results:
            f_r.close()
        for s_r in self.sessions_results:
            s_r.close()
        self.frames_results = []
        self.sessions_results = []
        
        self.visibility = None
        self.in_out_count = None
        self.sessions = None
        self.centroids = None

        self.thread = None
        self.pbr = None
        self.mvp = None
        self.mfv = None
        self.vmp = None

        self.logger.info("Widget cleared.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a video")
        if not file_path: 
            print("No file selected.")
            return
        self.set_media(file_path)
        
    def set_media(self, file_path):
        
        def bgr2rgb(frame):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        properties = self.mm.add_source(file_path, MEDIA_LAYER, "image", bgr2rgb)

        if properties is None:
            print("Failed to open the file: " + file_path)
            return

        self.slider.setMinimum(1)
        self.frame_input.setMinimum(1)
        self.slider.setMaximum(properties['total_frames'])
        self.frame_input.setMaximum(properties['total_frames'])
        self.set_frame(0)
        self.update_playback_info()
        self.video_name.setText(f"<b>{os.path.basename(file_path)}</b>")
        
        self.logger.info(f"File opened: '{file_path}'")
        self.logger.info("Properties: " + str(properties))

        self.create_temp_dir(os.path.join(
            os.path.dirname(file_path), 
            ".".join(os.path.basename(file_path).split(".")[:-1]) + ".tmp"
        ))
        return 0
    
    def update_playback_info(self):
        try:
            w, h, fps, nf = self.mm.get_width(), self.mm.get_height(), self.mm.get_fps(), self.mm.get_n_frames()
        except:
            w, h, fps, nf = 0, 0, 0.0, 0
        
        t = self.mm.current_frame / fps if nf > 0 else 0
        d = (nf / fps) if nf > 0 else 0

        self.info_label.setText(f"{round(t, 2)} sec")
        self.properties_display.setText(f"{w}x{h} ({round(fps, 2)} FPS) ↦ {round(d, 2)}s")
    
    def set_frame(self, n):
        if not self.mm.active:
            return
        self.mm.set_frame(n)
        self.slider.setValue(n+1)
        self.frame_input.setValue(n+1)
        self.update_playback_info()
        self.update_boxes()

    def jump_backward(self):
        f = self.mm.current_frame-25
        self.set_frame(f)

    def jump_forward(self):
        f = self.mm.current_frame+25
        self.set_frame(f)

    def on_slider_change(self, value):
        if int(self.frame_input.value()) != int(value):
            self.set_frame(int(value)-1)

    def on_spinbox_change(self, value):
        if int(self.slider.value()) != int(value):
            self.set_frame(int(value)-1)

    def start_extract_background(self):
        src = self.mm.get_source_by_name(MEDIA_LAYER)
        src_path = src[0]

        self.toggle_inputs(False)
        show_info("Extracting background...")
        self.logger.info("Extracting background...")
        self.pbr = progress(total=0)
        self.pbr.set_description("Extracting background...")
        self.thread = QThread()
        self.vmp = QtWorkerVMP(src_path, (self.mm.get_height(), self.mm.get_width()))
        self.vmp.moveToThread(self.thread)
        self.vmp.bg_ready.connect(self.terminate_extract_background)
        self.thread.started.connect(self.vmp.run)
        self.thread.start()
    
    def terminate_extract_background(self, ref, src_path):
        if BG_REF_LAYER in self.viewer.layers:
            self.viewer.layers[BG_REF_LAYER].data = ref
        else:
            self.viewer.add_image(
                ref, 
                name=BG_REF_LAYER, 
                visible=False
            )

        tifffile.imwrite(os.path.join(self.temp_dir, "bg-ref.tif"), ref)

        self.logger.info("Background extracted.")
        self.logger.info(f"Background reference saved in '{self.temp_dir}'")
        show_info("Background extracted!")
        self.pbr.close()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.toggle_inputs(True)
        self.background_ready.emit()
        
    def extract_classes(self):
        classes = {}

        for row in range(self.table.rowCount()):
            color_button_widget = self.table.cellWidget(row, 0)
            if color_button_widget:
                color = color_button_widget.palette().button().color()
                rgb = (color.red(), color.green(), color.blue(), 255)
            else:
                rgb = (0, 0, 0, 255)

            class_name_item = self.table.item(row, 1)
            class_name = class_name_item.text() if class_name_item is not None else ""
            classes[class_name] = rgb
            
        return classes
    
    def build_labels_from_polygon(self):
        canvas_shape = (self.mm.get_height(), self.mm.get_width())
        label_areas  = np.zeros(canvas_shape, np.uint8)
        
        for lbl, b_name in enumerate(self.boxes):
            shape_layer = self.viewer.layers[b_name]
            if len(shape_layer.data) <= 0:
                print(f"No data in the layer: {b_name}")
                continue
            np.save(os.path.join(self.temp_dir, f"box-{b_name.replace(' ', '_')}.npy"), shape_layer.data[0])
            shape = shape_layer.data[0]
            rr, cc = polygon(shape[:, 0], shape[:, 1], canvas_shape)
            label_areas[rr, cc] = lbl+1

        # Build a path to export the labels
        tifffile.imwrite(os.path.join(self.temp_dir, "labeled-areas.tif"), label_areas)
        self.logger.info(f"Labels (regions) saved in '{self.temp_dir}'")

        if AREAS_LAYER in self.viewer.layers:
            self.viewer.layers[AREAS_LAYER].data = label_areas
        else:
            self.viewer.add_labels(
                label_areas, 
                name=AREAS_LAYER, 
                visible=False
            )
    
    def launch_mask_processing(self):
        # Acquiring the background reference.
        if BG_REF_LAYER not in self.viewer.layers:
            print("Couldn't find the background reference. Abort.")
            return
        bg_ref = self.viewer.layers[BG_REF_LAYER].data

        # Checking that all the start frames are set.
        if len(self.start) != len(self.boxes):
            print("Not all boxes have a start frame set.")
            show_error("Not all boxes have a start frame set.")
            return

        # Building the output path
        if self.mm.get_n_sources() == 0:
            print("No media opened.")
            return
        
        media_path = self.mm.get_source_by_name(MEDIA_LAYER)[0]
        file_name = "mask.avi"
        output_path = os.path.join(self.temp_dir, file_name)

        # Removing threshold previewer
        if TS_PREVIEW_LAYER in self.viewer.layers:
            del self.viewer.layers[TS_PREVIEW_LAYER]

        # Launching the worker...
        self.toggle_inputs(False)
        show_info("Building mice labels...")
        self.logger.info("Building mice labels with threshold at: " + str(self.threshold.value()))
        self.logger.info("Building mice labels from frame: " + str(self.start))
        self.pbr = progress(total=0)
        self.pbr.set_description("Building mice labels...")
        self.thread = QThread()
        self.mfv = QtWorkerMFV(
            media_path, 
            output_path, 
            bg_ref,
            self.threshold.value(),
            self.start,
            self.viewer.layers[AREAS_LAYER].data
        )
        self.mfv.moveToThread(self.thread)
        self.mfv.mask_ready.connect(self.terminate_mask_processing)
        self.thread.started.connect(self.mfv.run)
        self.thread.start()
        

    def terminate_mask_processing(self, mask_path):

        def bgr2rgb_tr(frame):
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127
            canvas = np.zeros(mask.shape, np.uint8)
            canvas[mask] = self.viewer.layers[AREAS_LAYER].data[mask]
            return canvas
        
        self.mm.add_source(mask_path, MICE_LABELS_LAYER, "labels", bgr2rgb_tr)
        apply_lut(
            self.viewer.layers[MICE_LABELS_LAYER], 
            self.boxes, 
            self.extract_classes()
        )
        self.viewer.layers[MEDIA_LAYER].opacity = 0.3

        self.toggle_inputs(False)
        self.logger.info("Mice labels built.")
        show_info("Mice labels built!")
        self.pbr.close()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.toggle_inputs(True)
        self.tracking_ready.emit()

    def dump_table(self):
        """
        Dumps the content of the table in a dictionary.
        """
        table = []
        for row_idx in range(self.table.rowCount()):
            row = {}

            color_button_widget = self.table.cellWidget(row_idx, 0)
            if color_button_widget:
                color = color_button_widget.palette().button().color()
                rgb = (color.red(), color.green(), color.blue(), 255)
            else:
                rgb = (0, 0, 0, 255)
            row["color"] = rgb

            class_name_item = self.table.item(row_idx, 1)
            class_name = class_name_item.text() if class_name_item is not None else ""
            row['name'] = class_name

            start_frame = self.start.get(row_idx+1, -1)
            row['start'] = start_frame

            table.append(row)
        
        stringified = json.dumps(table, indent=4)
        path = os.path.join(self.temp_dir, "boxes.json")
        with open(path, "w") as f:
            f.write(stringified)

    def launch_tracking(self):
        if len(self.boxes) == 0:
            show_error("No boxes to track.")
            self.logger.error("No boxes to track.")
            return
        # Building the labels
        self.build_labels_from_polygon()
        apply_lut(
            self.viewer.layers[AREAS_LAYER], 
            self.boxes, 
            self.extract_classes()
        )
        self.launch_mask_processing()



# ---------------------------- TEST PROCEDURE ----------------------------

def launch_test_procedure_1():
    viewer = napari.Viewer()
    miow   = MouseInOutWidget(viewer)
    viewer.window.add_dock_widget(miow)

    print("--- Workflow: WIN_20210830_11_11_50_Pro.mp4 ---")

    media_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/WIN_20210830_11_11_50_Pro.mp4"

    # Set experimental video.
    miow.set_media(media_path)

    # Add boxes...
    b1 = np.array(
        [[158.30374768, 178.23561088],
        [158.30374768, 271.26936533],
        [276.3426219 , 271.26936533],
        [276.3426219 , 178.23561088]])
    
    b2 = np.array(
        [[156.83285828, 277.88836762],
        [156.83285828, 374.96706791],
        [274.8717325 , 374.96706791],
        [274.8717325 , 277.88836762]])
    
    b3 = np.array(
        [[157.93602533, 380.48290316],
        [157.93602533, 477.1938811 ],
        [274.1362878 , 477.1938811 ],
        [274.1362878 , 380.48290316]])
    
    boxes  = [b1, b2, b3]
    starts = [1200, 700, 700]
    colors = ['#ff0000', '#00ff00', '#0000ff']

    for index, (start, box) in enumerate(zip(starts, boxes)):
        miow.add_row()
        viewer.layers.selection.active.data = [box]
        miow.set_frame(start)
        miow.make_start_frame(
            index, 
            miow.table.cellWidget(index, 2)
        )
        miow.select_color(
            index, 
            miow.table.cellWidget(index, 0), 
            QColor(colors[index])
        )


    # Setting min area and threshold value for the mask.
    miow.min_area.setValue(76)
    miow.threshold.setValue(75)

    # Launch the tracking after the background extraction
    miow.background_ready.connect(miow.launch_tracking)
    # Launch measures extraction after the tracking was done.
    miow.tracking_ready.connect(miow.extract_measures)
    # Launch the export of the results after the measures were extracted.
    miow.measures_ready.connect(miow.export_results)
    # Launch the chain: background extraction -> tracking -> measures extraction
    miow.start_extract_background()

    # ---
    napari.run()


def launch_test_procedure_2():
    viewer = napari.Viewer()
    miow   = MouseInOutWidget(viewer)
    viewer.window.add_dock_widget(miow)

    print("--- Workflow: R6S6R7S7_01.09.20.mp4 ---")

    media_path = "/home/benedetti/Documents/projects/25-entry-exit-mouse-monitor/data-samples/R6S6R7S7_01.09.20.mp4"

    # Set experimental video.
    miow.set_media(media_path)

    # Add boxes...
    b1 = np.array([
        [397.31629783,  44.22604657],
        [397.31629783, 173.37378185],
        [233.09827237, 173.37378185],
        [233.09827237,  44.22604657]])
    
    b2 = np.array([
        [395.64628401, 186.73389239],
        [395.64628401, 320.33499785],
        [231.42825855, 320.33499785],
        [231.42825855, 186.73389239]])
    
    b3 = np.array([
        [395.08961273, 331.46842331],
        [395.08961273, 464.51285749],
        [229.75824473, 464.51285749],
        [229.75824473, 331.46842331]])
    
    b4 = np.array([
        [388.96622873, 473.97626913],
        [388.96622873, 600.89731931],
        [228.64490219, 600.89731931],
        [228.64490219, 473.97626913]])
    
    boxes  = [b1, b2, b3, b4]
    starts = [3712, 3712, 162, 162]
    colors = ['#c01c28', '#f5c211', '#2ec27e', '#1c71d8']

    for index, (start, box) in enumerate(zip(starts, boxes)):
        miow.add_row()
        viewer.layers.selection.active.data = [box]
        miow.set_frame(start)
        miow.make_start_frame(
            index, 
            miow.table.cellWidget(index, 2)
        )
        miow.select_color(
            index, 
            miow.table.cellWidget(index, 0), 
            QColor(colors[index])
        )


    # Setting min area and threshold value for the mask.
    miow.min_area.setValue(110)
    miow.threshold.setValue(65)

    # Launch the tracking after the background extraction
    miow.background_ready.connect(miow.launch_tracking)
    # Launch measures extraction after the tracking was done.
    miow.tracking_ready.connect(miow.extract_measures)
    # Launch the export of the results after the measures were extracted.
    miow.measures_ready.connect(miow.export_results)
    # Launch the chain: background extraction -> tracking -> measures extraction
    miow.start_extract_background()

    # ---
    napari.run()


def launch():
    viewer = napari.Viewer()
    miow   = MouseInOutWidget(viewer)
    viewer.window.add_dock_widget(miow)

    # ---
    napari.run()



if __name__ == "__main__":
    # launch()
    # launch_test_procedure_1()
    launch_test_procedure_2()



""" UNIT TESTS

----> MediaManager

    - On ne peut pas faire set_frame si aucun media n'est ouvert.
    - On ne peut pas faire next_frame si aucun media n'est ouvert.
    - On ne peut pas faire previous_frame si aucun media n'est ouvert.
    - On ne peut pas appeler set_frame avec un index invalide.
    - On ne peut pas ouvrir plusieurs fois la même source.
    - Appeler n'importe la quelle des méthodes pour changer de frame actualise les autres.

----> Tracking
    - On ne peut pas lancer de process s'il n'y a aucune box sélectionnée.
    - Un layer de boite ne devrait contenir qu'un objet.
    - On ne peut pas lancer de process si chaque box n'a pas de start frame.

"""