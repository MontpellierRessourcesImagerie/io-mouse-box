from typing import TYPE_CHECKING

import sys, os
import random
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, 
                             QSpinBox, QTableWidget, QTableWidgetItem, QColorDialog, 
                             QGroupBox, QLabel, QTabWidget, QFileDialog, QFrame)

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
import tifffile
import logging
import numpy as np
from skimage.draw import polygon
from skimage.measure import regionprops
from napari.utils.notifications import show_info, show_error, show_warning
from napari.utils import progress
from qtpy.QtCore import QThread
from shapely.geometry import Polygon

from .media_manager import MediaManager
from .results_table import ResultsTable
from .video_mean_processor import QtWorkerVMP
from .mask_from_video import QtWorkerMFV
from .utils import setup_logger

if TYPE_CHECKING:
    import napari
    

"""

- Dedans / dehors (bool)

- Comparaison au centroid précédent pour la vitesse.

- Nombre de pixels à chaque frame (int)

~~~

- Carte de présence -> une image par boite -> Calculable avec 'extractMeanFromVideo' sur le masque.

"""
def process_statistics(path, areas, starts):
    print("====== Processing statistics ======")


def apply_lut(tags, to_pass, classes):
    """
    tags: Layer that will receive the LUT.
    to_pass: List of gray levels to which we will have to bind a color.
    classes: Dictionary of colors to bind to the gray levels.
    """
    lut = {i: (0.0, 0.0, 0.0, 1.0) for i in range(256)}
    
    for index, level in enumerate(to_pass):
        color = classes[level]
        tpl = [float(color[i])/255.0 for i in range(3)] + [1.0]
        lut[index+1] = tuple(tpl)

    tags.color = lut



BG_REF_LAYER      = "bg_ref"
MICE_LABELS_LAYER = "mice_labels"
AREAS_LAYER       = "areas"
MEDIA_LAYER       = "media"
TS_PREVIEW_LAYER  = "threshold_preview" 
FONT              = QFont()
FONT.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")


class MouseInOutWidget(QWidget):

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.boxes  = []
        self.viewer = napari_viewer
        self.mm     = MediaManager(self.viewer)
        self.start  = 1
        self.logger = None
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.add_video_control_box()
        self.add_box_control()
        self.add_tracking()

    def add_video_control_box(self):
        group_box = QGroupBox("Video Control")
        layout = QVBoxLayout(group_box)

        # Bouton pour sélectionner un fichier
        self.file_button = QPushButton("📁 Select File", self)
        self.file_button.setFont(FONT)
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Boutons next et previous
        self.prev_button = QPushButton("◀ Previous", self)
        self.prev_button.setFont(FONT)
        self.prev_button.clicked.connect(self.previous_frame)
        self.next_button = QPushButton("Next ▶", self)
        self.next_button.setFont(FONT)
        self.next_button.clicked.connect(self.next_frame)
        self.backward_button = QPushButton("⏮️ Previous", self)
        self.backward_button.setFont(FONT)
        self.backward_button.clicked.connect(self.jump_backward)
        self.forward_button = QPushButton("Next ⏭️", self)
        self.forward_button.setFont(FONT)
        self.forward_button.clicked.connect(self.jump_forward)
        
        nav_layout_1 = QHBoxLayout()
        nav_layout_1.addWidget(self.prev_button)
        nav_layout_1.addWidget(self.next_button)
        nav_layout_2 = QHBoxLayout()
        nav_layout_2.addWidget(self.backward_button)
        nav_layout_2.addWidget(self.forward_button)
        layout.addLayout(nav_layout_1)
        layout.addLayout(nav_layout_2)

        # Slider
        self.slider = QSlider(self)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setOrientation(Qt.Horizontal)
        layout.addWidget(self.slider)

        # Slot d'entier
        self.frame_input = QSpinBox(self)
        self.frame_input.setValue(1)
        self.frame_input.valueChanged.connect(self.on_spinbox_change)
        self.frame_input.setPrefix("Frame: ")
        layout.addWidget(self.frame_input)

        # Time in seconds
        self.info_label = QLabel("0 sec", self)
        self.info_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)  # Style optionnel
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # Video properties
        self.properties_display = QLabel("0x0 (0 FPS) ↦ 0s", self)
        self.properties_display.setFrameStyle(QFrame.Panel | QFrame.Sunken)  # Style optionnel
        self.properties_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.properties_display)

        self.layout.addWidget(group_box)

    def add_box_control(self):
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

        start_layout = QHBoxLayout()
        self.make_start_frame_button = QPushButton(f"🔘 Set Start", self)
        self.make_start_frame_button.setFont(FONT)
        self.make_start_frame_button.clicked.connect(self.make_start_frame)
        self.start_label = QLabel(f"▶  {self.start}", self)
        # self.start_label = QLabel(f"🏁 {self.start}", self)
        # self.start_label.setFont(FONT)
        # self.start_label.setStyleSheet("font-size: 1.2em; background-color: rgba(54, 214, 0, 1.0); border-radius: 3px; color: black; letter-spacing: -8px;")

        btn_layout.addWidget(self.add_box_button)
        btn_layout.addWidget(self.pop_box_button)
        
        start_layout.addWidget(self.make_start_frame_button)
        start_layout.addWidget(self.start_label)

        layout.addLayout(start_layout)
        layout.addLayout(btn_layout)

        # Tableau
        cols = ["Color", "Name"]
        self.table = QTableWidget(0, len(cols), self)
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)

        self.layout.addWidget(group_box)

    def add_tracking(self):
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

        self.extract_button = QPushButton("♻️ Clean background", self)
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

        self.extract_measures_button = QPushButton("📐 Extract measures", self)
        self.extract_measures_button.setFont(FONT)
        self.extract_measures_button.clicked.connect(self.extract_measures)
        layout.addWidget(self.extract_measures_button)

        self.clear_state_button = QPushButton("✨ Clear state", self)
        self.clear_state_button.setFont(FONT)
        self.clear_state_button.clicked.connect(self.clear_state)
        layout.addWidget(self.clear_state_button)

        self.layout.addWidget(group_box)

    def set_min_area(self):
        if self.mm.get_n_sources() == 0:
            show_error("No media opened.")
            self.logger.error("No media opened.")
            return
        
        if self.viewer.layers.selection.active is None:
            show_error("No layer selected.")
            self.logger.error("No layer selected.")
            return
        
        active_layer = self.viewer.layers.selection.active
        active_layer.scale = self.viewer.layers[MEDIA_LAYER].scale

        if 'add_lines' not in dir(active_layer):
            show_error("A shape layer is expected.")
            self.logger.error("A shape layer is expected.")
            return
        
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

    def on_threshold_update(self, value):
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

    def turn_on_off_inputs(self, t):
        """
        Used to disable the inputs (buttons and text fields) when a long process is running.

        Args:
            t: bool - True to enable the inputs, False to disable them.
        """
        self.clear_state_button.setEnabled(t)
        self.extract_measures_button.setEnabled(t)
        self.track_button.setEnabled(t)
        self.extract_button.setEnabled(t)
        self.threshold.setEnabled(t)
        self.set_min_area_button.setEnabled(t)

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

    def make_start_frame(self):
        """
        Sets the frame at which we start measures from the current frame being displayed.
        The saved index is the one displayed on the screen. (The real one + 1).
        A source is required before we can set the start frame.
        """
        if self.mm.get_n_sources() == 0:
            return
        self.start = self.mm.current_frame+1
        self.start_label.setText(f"▶  {self.start}")

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
    
    def select_color(self, row, button):
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f'background-color: {color.name()};')
            button.setText(color.name())
            nb = len(self.viewer.layers[self.boxes[row]].edge_color)
            self.viewer.layers[self.boxes[row]].edge_color = [color.name() for i in range(nb)]
            self.viewer.layers[self.boxes[row]].current_edge_color = color.name()

    def add_row(self):
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        self.boxes.append(f"Box-{row_count}")
        # Adding color picker to the table
        color_button = QPushButton('Pick a color')
        color_button.clicked.connect(lambda: self.select_color(row_count, color_button))
        self.table.setCellWidget(row_count, 0, color_button)
        # Adding the name slot to the table
        self.table.setItem(row_count, 1, QTableWidgetItem(f"Box-{row_count}"))
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

    def extract_measures(self):
        print("Extracting measures...")

    def clear_state(self):
        """
        Resets the state of the widget to its initial state.
        """
        self.viewer.layers.clear()
        self.boxes = []
        self.slider.setValue(1)
        self.frame_input.setValue(1)
        self.mm.release()
        self.table.setRowCount(0)
        self.start = 1
        self.start_label.setText(f"▶  {self.start}")
        self.update_playback_info()
        self.logger.info("Widget cleared.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a video")
        if not file_path: 
            print("No file selected.")
            return
        
        log_name = ".".join(file_path.split('.')[:-1])+".log"
        print("Logs in: " + log_name)
        self.switch_log_file(log_name)
        self.mm.set_logger(self.logger)
        
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
        
        self.logger.info(f"File opened: '{file_path}'")
        self.logger.info("Properties: " + str(properties))
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
        self.mm.set_frame(n)
        self.slider.setValue(n+1)
        self.frame_input.setValue(n+1)
        self.update_playback_info()

    def previous_frame(self):
        f = self.mm.current_frame-1
        self.set_frame(f)

    def next_frame(self):
        f = self.mm.current_frame+1
        self.set_frame(f)

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

        self.turn_on_off_inputs(False)
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

        file_name = "bg-ref-" + ".".join(os.path.basename(src_path).split(".")[:-1]) + ".tif"
        dir_name  = os.path.dirname(src_path)
        out_path  = os.path.join(dir_name, file_name)
        tifffile.imwrite(out_path, ref)

        self.logger.info("Background extracted.")
        self.logger.info(f"Background saved in '{out_path}'")
        show_info("Background extracted!")
        self.pbr.close()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.turn_on_off_inputs(True)
        
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
            shape = shape_layer.data[0]
            rr, cc = polygon(shape[:, 0], shape[:, 1], canvas_shape)
            label_areas[rr, cc] = lbl+1

        # Build a path to export the labels
        base_path = self.mm.get_source_by_name(MEDIA_LAYER)[0]
        file_name = "areas-" + ".".join(os.path.basename(base_path).split(".")[:-1]) + ".tif"
        dir_name = os.path.dirname(base_path)
        output_path = os.path.join(dir_name, file_name)
        tifffile.imwrite(output_path, label_areas)
        self.logger.info(f"Labels (regions) saved in '{output_path}'")

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

        # Building the output path
        if self.mm.get_n_sources() == 0:
            print("No media opened.")
            return
        base_path = self.mm.get_source_by_name(MEDIA_LAYER)[0]
        file_name = "mask-" + ".".join(os.path.basename(base_path).split(".")[:-1]) + ".avi"
        dir_name = os.path.dirname(base_path)
        output_path = os.path.join(dir_name, file_name)

        # Removing threshold previewer
        if TS_PREVIEW_LAYER in self.viewer.layers:
            del self.viewer.layers[TS_PREVIEW_LAYER]

        # Launching the worker...
        self.turn_on_off_inputs(False)
        show_info("Building mice labels...")
        self.logger.info("Building mice labels with threshold at: " + str(self.threshold.value()))
        self.logger.info("Building mice labels from frame: " + str(self.start))
        self.pbr = progress(total=0)
        self.pbr.set_description("Building mice labels...")
        self.thread = QThread()
        self.mfv = QtWorkerMFV(
            base_path, 
            output_path, 
            bg_ref,
            self.threshold.value(),
            self.start-1
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

        self.turn_on_off_inputs(False)
        self.logger.info("Mice labels built.")
        show_info("Mice labels built!")
        self.pbr.close()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.turn_on_off_inputs(True)

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


""" TODO

    - [X] Ajouter un bouton reset pour repartir dans un environnement propre.
    - [X] Faire en sorte qu'on puisse utiliser le bouton de sélection de fichier pour changer de vidéo.
    - [ ] Calculer un tableau qui donne le centroïde de chaque souris dans sa boite à chaque frame.
    - [ ] Calculer un tableau de visibilité des souris dans chaque boite à chaque frame.
    - [ ] Avoir des barres de progression pour les tâches longues plutôt que le logo de chargement.
    - [ ] S'assurer que le clear state ne laisse pas l'instance dans un état instable.
    - [ ] Ajouter plus de logs pour suivre l'évolution de l'application, et corriger le nom des logs.
    - [ ] Avoir un moyen plus simple de calculer le threshold de création du masque, sans attendre que toute la vidéo soit traitée.
          > Appliquer la soustraction du background une fois qu'il est calculé ?
          > Les souris sont noires sur fond blanc, on ne pourrait garder que les différences d'un certain signe.
          > Quand le champ de threshold est update, on calcule le masque sur un layer "dump" qu'on discard après.
    - [ ] Faire une fonction de test qui permet d'injecter les paramètres sans passer par les menus.

"""


""" UNIT TESTS

----> MediaManager

    - On ne peut pas faire set_frame si aucun media n'est ouvert.
    - On ne peut pas faire next_frame si aucun media n'est ouvert.
    - On ne peut pas faire previous_frame si aucun media n'est ouvert.
    - On ne peut pas appeler set_frame avec un index invalide.
    - On ne peut pas ouvrir plusieurs fois la même source.
    - Appeler n'importe la quelle des méthodes pour changer de frame actualise les autres.

"""