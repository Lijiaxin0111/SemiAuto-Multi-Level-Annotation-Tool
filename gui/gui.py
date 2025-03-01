import functools
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import cv2

from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QSpinBox,
    QPlainTextEdit,
    QVBoxLayout,
    QSizePolicy,
    QButtonGroup,
    QSlider,
    QRadioButton,
    QApplication,
    QFileDialog,
    QColorDialog,
)

from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QTextCursor,
    QImage,
    QPixmap,
    QIcon,
    QColor,
    QMouseEvent,
)
from PySide6.QtCore import Qt, QTimer

from cutie.utils.palette import davis_palette_np
from gui.gui_utils import *
import json
from PySide6.QtGui import QWheelEvent


class GUI(QWidget):
    def __init__(self, controller, cfg: DictConfig) -> None:
        super().__init__()

        # track mouse press
        self.set_mouse_pressed = None

        # callbacks to be set by the controller
        self.on_mouse_motion_xy = None
        self.click_fn = None
        self.paint_erase = None

        self.controller = controller
        self.cfg = cfg
        self.h = controller.h
        self.w = controller.w
        self.T = controller.T

        # set up the window
        self.setWindowTitle(f'Cutie demo: {cfg["workspace"]}')
        self.setGeometry(100, 100, self.w + 200, self.h + 200)
        self.setWindowIcon(QIcon("docs/icon.png"))
        self.size_ratio = 0.5

        # set up some buttons
        self.play_button = QPushButton("Play video")
        self.play_button.clicked.connect(self.on_play_video)
        self.commit_button = QPushButton("Commit to permanent memory")
        self.commit_button.clicked.connect(controller.on_commit)
        self.export_video_button = QPushButton("Export as video")
        self.export_video_button.clicked.connect(controller.on_export_visualization)
        self.export_binary_button = QPushButton("Export binary masks")
        self.export_binary_button.clicked.connect(controller.on_export_binary)

        self.forward_run_button = QPushButton("Propagate forward")
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)
        self.forward_run_button.setMinimumWidth(int(150 * self.size_ratio))

        self.backward_run_button = QPushButton("Propagate backward")
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)
        self.backward_run_button.setMinimumWidth(int(150 * self.size_ratio))

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(int(200 * self.size_ratio))

        self.reset_frame_button = QPushButton("Reset frame")
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_object_button = QPushButton("Reset object")
        self.reset_object_button.clicked.connect(controller.on_reset_object)

        # set up the LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(int(28 * self.size_ratio))
        self.lcd.setMaximumWidth(int(150 * self.size_ratio))
        self.lcd.setText("{: 5d} / {: 5d}".format(0, controller.T - 1))

        # current object id
        self.object_dial = QSpinBox()
        self.object_dial.setReadOnly(False)
        self.object_dial.setMinimumSize(
            int(50 * self.size_ratio), int(30 * self.size_ratio)
        )
        self.object_dial.setMinimumSize(
            int(25 * self.size_ratio), int(15 * self.size_ratio)
        )

        self.object_dial.setMinimum(1)
        self.object_dial.setMaximum(controller.num_objects)
        self.object_dial.editingFinished.connect(controller.on_object_dial_change)

        self.davis_palette_np2viz_palette_np = self.init_palette()

        self.object_color = QLabel()
        self.object_color.setMinimumSize(
            int(100 * self.size_ratio), int(30 * self.size_ratio)
        )
        self.object_color.setMinimumSize(
            int(50 * self.size_ratio), int(15 * self.size_ratio)
        )
        self.object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.viz_object_color = QLabel()
        self.viz_object_color.setMinimumSize(
            int(100 * self.size_ratio), int(30 * self.size_ratio)
        )
        self.viz_object_color.setMinimumSize(
            int(50 * self.size_ratio), int(15 * self.size_ratio)
        )
        self.viz_object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.viz_object_color.mousePressEvent = self.on_show_color_dialog

        # Color Selector
        self.color_dialog = QColorDialog()
        self.color_dialog.setOption(
            QColorDialog.ColorDialogOption.ShowAlphaChannel, True
        )

        self.color_dialog.colorSelected.connect(controller.updateColor)
        self.color_dialog.hide()

        self.frame_name = QLabel()

        self.frame_name.setMinimumSize(
            int(50 * self.size_ratio), int(15 * self.size_ratio)
        )

        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # timeline slider
        self.tl_slider = QSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(controller.on_slider_update)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(controller.T - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("mask")
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("layer")
        self.combo.addItem("rgba")
        self.combo.setCurrentText("davis")
        self.combo.currentTextChanged.connect(controller.set_vis_mode)

        self.save_visualization_combo = QComboBox(self)
        self.save_visualization_combo.addItem("None")
        self.save_visualization_combo.addItem("Always")
        self.save_visualization_combo.addItem("Propagation only (higher quality)")
        self.combo.setCurrentText("None")
        self.save_visualization_combo.currentTextChanged.connect(
            controller.on_set_save_visualization_mode
        )

        self.save_soft_mask_checkbox = QCheckBox(self)
        self.save_soft_mask_checkbox.toggled.connect(
            controller.on_save_soft_mask_toggle
        )
        self.save_soft_mask_checkbox.setChecked(False)

        # controls for output FPS and bitrate
        self.fps_dial = QSpinBox()
        self.fps_dial.setReadOnly(False)
        self.fps_dial.setMinimumSize(
            int(40 * self.size_ratio), int(30 * self.size_ratio)
        )

        self.fps_dial.setMinimum(1)
        self.fps_dial.setMaximum(60)
        self.fps_dial.setValue(cfg["output_fps"])
        self.fps_dial.editingFinished.connect(controller.on_fps_dial_change)

        self.bitrate_dial = QSpinBox()
        self.bitrate_dial.setReadOnly(False)
        self.bitrate_dial.setMinimumSize(
            int(40 * self.size_ratio), int(30 * self.size_ratio)
        )

        self.bitrate_dial.setMinimum(1)
        self.bitrate_dial.setMaximum(100)
        self.bitrate_dial.setValue(cfg["output_bitrate"])
        self.bitrate_dial.editingFinished.connect(controller.on_bitrate_dial_change)

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(
            int(100 * self.size_ratio), int(100 * self.size_ratio)
        )

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        # clearing memory
        self.clear_all_mem_button = QPushButton("Reset all memory")
        self.clear_all_mem_button.clicked.connect(controller.on_clear_memory)
        self.clear_non_perm_mem_button = QPushButton("Reset non-permanent memory")
        self.clear_non_perm_mem_button.clicked.connect(
            controller.on_clear_non_permanent_memory
        )


        #  Test Tree
        self.phase_state = self.cfg["phase"]
        self.before_state_combo_box = TreeComboBox(content_dict=self.phase_state)
        self.before_state_combo_box.setMinimumWidth(
            int(280 * self.size_ratio)
        )  # This value can be changed according to your needs

        self.before_state_combo_box.currentTextChanged.connect(
            controller.set_before_state
        )

        self.before_state_h_layout = QHBoxLayout()
        self.before_state_h_layout.addWidget(QLabel("Initial Phase state"))
        self.before_state_h_layout.addWidget(self.before_state_combo_box)

        self.before_state_layout = QVBoxLayout()
        self.before_state_layout.addLayout(self.before_state_h_layout)

        self.after_state_combo_box = TreeComboBox(content_dict=self.phase_state)
        self.after_state_combo_box.setMinimumWidth(
            int(280 * self.size_ratio)
        )  # This value can be changed according to your needs

        self.after_state_combo_box.currentTextChanged.connect(
            controller.set_after_state
        )

        self.after_state_h_layout = QHBoxLayout()
        self.after_state_h_layout.addWidget(QLabel("Final Phase state"))
        self.after_state_h_layout.addWidget(self.after_state_combo_box)

        self.after_state_layout = QVBoxLayout()
        self.after_state_layout.addLayout(self.after_state_h_layout)

        self.save_state_button = QPushButton("Save state")
        self.save_state_button.clicked.connect(controller.on_save_state)

        self.phase_trans_list = self.get_phase_trans_to_select(
            self.cfg["phase_transition"],
            self.cfg["usr_define_phase_trans_json_path"],
            self.before_state_combo_box.currentText(),
            self.after_state_combo_box.currentText(),
        )
        self.phase_trans_combo_box = TreeComboBox(content_dict=self.phase_trans_list)
        self.phase_trans_combo_box.setMinimumWidth(int(280 * self.size_ratio))

        self.phase_trans_combo_box.currentTextChanged.connect(
            controller.set_phase_trans
        )

        self.phase_trans_h_layout = QHBoxLayout()
        self.phase_trans_h_layout.addWidget(QLabel("Phase state transition"))
        self.phase_trans_h_layout.addWidget(self.phase_trans_combo_box)

        self.phase_trans_layout = QVBoxLayout()
        self.phase_trans_layout.addLayout(self.phase_trans_h_layout)

        self.save_phase_trans_button = QPushButton("Save phase trans")
        self.save_phase_trans_button.clicked.connect(controller.on_save_trans)

        # displaying memory usage
        self.perm_mem_gauge, self.perm_mem_gauge_layout = create_gauge(
            "Permanent memory size"
        )
        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge(
            "Working memory size"
        )
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge(
            "Long-term memory size"
        )
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            "GPU mem. (all proc, w/ caching)"
        )
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            "GPU mem. (torch, w/o caching)"
        )

        # Parameters setting
        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(
            1, 100, "Min. working memory frames", callback=controller.on_work_min_change
        )
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(
            2, 100, "Max. working memory frames", callback=controller.on_work_max_change
        )
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(
            1000,
            100000,
            "Max. long-term memory size",
            step=1000,
            callback=controller.update_config,
        )
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, "Memory frame every (r)", callback=controller.update_config
        )

        # import mask/layer
        self.import_mask_button = QPushButton("Import mask")
        self.import_mask_button.clicked.connect(controller.on_import_mask)
        self.import_layer_button = QPushButton("Import layer")
        self.import_layer_button.clicked.connect(controller.on_import_layer)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(int(100 * self.size_ratio))
        self.console.setMaximumHeight(int(100 * self.size_ratio))

        # Tips for the users
        self.tips = QTextEdit()
        self.tips.setReadOnly(True)
        self.tips.setTextInteractionFlags(Qt.NoTextInteraction)
        self.tips.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        with open(Path(__file__).parent / "TIPS.md", "r") as f:
            self.tips.setMarkdown(f.read())

        # navigator
        navi = QHBoxLayout()

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interact_topbox.addWidget(self.lcd)
        interact_topbox.addWidget(self.play_button)
        interact_topbox.addWidget(self.reset_frame_button)
        interact_topbox.addWidget(self.reset_object_button)
        interact_botbox.addWidget(QLabel("Current object ID:"))
        interact_botbox.addWidget(self.object_dial)
        # interact_botbox.addLayout(self.object_color_layout)
        interact_botbox.addWidget(self.object_color)
        interact_botbox.addWidget(self.viz_object_color)

        interact_botbox.addWidget(self.frame_name)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        interact_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        navi.addLayout(interact_subbox)

        apply_fixed_size_policy = lambda x: x.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        apply_to_all_children_widget(interact_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(interact_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        navi.addStretch(1)
        overlay_subbox = QVBoxLayout()
        overlay_topbox = QHBoxLayout()
        overlay_botbox = QHBoxLayout()
        overlay_topbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_topbox.addWidget(QLabel("Visualization mode"))
        overlay_topbox.addWidget(self.combo)
        overlay_topbox.addWidget(QLabel("Save soft mask during propagation"))
        overlay_topbox.addWidget(self.save_soft_mask_checkbox)
        overlay_topbox.addWidget(self.export_binary_button)
        overlay_botbox.addWidget(QLabel("Save visualization"))
        overlay_botbox.addWidget(self.save_visualization_combo)
        overlay_botbox.addWidget(self.export_video_button)
        overlay_botbox.addWidget(QLabel("Output FPS: "))
        overlay_botbox.addWidget(self.fps_dial)
        overlay_botbox.addWidget(QLabel("Output bitrate (Mbps): "))
        overlay_botbox.addWidget(self.bitrate_dial)
        overlay_subbox.addLayout(overlay_topbox)
        overlay_subbox.addLayout(overlay_botbox)
        navi.addLayout(overlay_subbox)
        apply_to_all_children_widget(overlay_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(overlay_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        control_subbox = QVBoxLayout()
        control_topbox = QHBoxLayout()
        control_botbox = QHBoxLayout()
        control_topbox.addWidget(self.commit_button)
        control_topbox.addWidget(self.forward_run_button)
        control_topbox.addWidget(self.backward_run_button)
        control_botbox.addWidget(self.progressbar)
        control_subbox.addLayout(control_topbox)
        control_subbox.addLayout(control_botbox)
        navi.addLayout(control_subbox)

        # Drawing area main canvas
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # right area
        right_area = QVBoxLayout()
        right_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        right_area.addWidget(self.tips)
        # right_area.addStretch(1)

        # Parameters
        right_area.addLayout(self.perm_mem_gauge_layout)
        right_area.addLayout(self.work_mem_gauge_layout)
        right_area.addLayout(self.long_mem_gauge_layout)
        right_area.addLayout(self.gpu_mem_gauge_layout)
        right_area.addLayout(self.torch_mem_gauge_layout)
        right_area.addWidget(self.clear_all_mem_button)
        right_area.addWidget(self.clear_non_perm_mem_button)
        right_area.addLayout(self.work_mem_min_layout)
        right_area.addLayout(self.work_mem_max_layout)
        right_area.addLayout(self.long_mem_max_layout)
        right_area.addLayout(self.mem_every_box_layout)
        right_area.addLayout(self.before_state_layout)
        right_area.addLayout(self.after_state_layout)
        right_area.addWidget(self.save_state_button)
        right_area.addLayout(self.phase_trans_layout)
        right_area.addWidget(self.save_phase_trans_button)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        right_area.addLayout(import_area)

        # console
        right_area.addWidget(self.console)

        draw_area.addLayout(right_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(controller.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(controller.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # Objects shortcuts
        for i in range(1, controller.num_objects + 1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                functools.partial(controller.hit_number_key, i)
            )
            QShortcut(QKeySequence(f"Ctrl+{i}"), self).activated.connect(
                functools.partial(controller.hit_number_key, i)
            )

        # next/prev frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(
            controller.on_prev_frame
        )
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            controller.on_next_frame
        )

        # +/- 10 frames shortcuts
        QShortcut(
            QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.ShiftModifier), self
        ).activated.connect(functools.partial(controller.on_prev_frame, 10))
        QShortcut(
            QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.ShiftModifier), self
        ).activated.connect(functools.partial(controller.on_next_frame, 10))

        # first/last frame shortcuts
        QShortcut(
            QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.AltModifier), self
        ).activated.connect(functools.partial(controller.on_prev_frame, 999999))
        QShortcut(
            QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.AltModifier), self
        ).activated.connect(functools.partial(controller.on_next_frame, 999999))

        # commit to permanent memory shortcut
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(
            controller.on_commit
        )

        # propagate forward/backward/pause shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(
            controller.on_forward_propagation
        )
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(
            controller.on_forward_propagation
        )
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(
            controller.on_backward_propagation
        )

        # label in Erase / Paint / Auto mode
        QShortcut(QKeySequence(Qt.Key.Key_E), self).activated.connect(
            controller.on_erase_mode
        )
        QShortcut(QKeySequence(Qt.Key.Key_I), self).activated.connect(
            controller.on_paint_mode
        )
        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(
            controller.on_auto_mode
        )

        QShortcut(QKeySequence(Qt.Key.Key_V), self).activated.connect(
            controller.on_paint_void_mode
        )
        QShortcut(QKeySequence(Qt.Key.Key_X), self).activated.connect(
            controller.on_erase_void_mode
        )

        # adjust the size of painter
        QShortcut(QKeySequence(Qt.Key.Key_Minus), self).activated.connect(
            controller.reduce_painter_size
        )
        QShortcut(QKeySequence(Qt.Key.Key_Equal), self).activated.connect(
            controller.add_painter_size
        )

        # add/del the magnifying glass
        QShortcut(QKeySequence(Qt.Key.Key_0), self).activated.connect(
            controller.set_mag_image
        )

        # use the color similar making out
        QShortcut(QKeySequence(Qt.Key.Key_T), self).activated.connect(
            controller.select_color
        )
        QShortcut(QKeySequence(Qt.Key.Key_Y), self).activated.connect(
            controller.paint_in_color_mask
        )


        # adjust the bound ratio of magnifying glass
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self).activated.connect(
            controller.reduce_bound_ratio
        )
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self).activated.connect(
            controller.add_bound_ratio
        )

        # quit shortcut
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)

    def init_palette(self):
        res = {}
        for i in range(len(davis_palette_np)):
            res[str(davis_palette_np[i])] = davis_palette_np[i]
        return res

    def on_show_color_dialog(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.showColorDialog()

    def showColorDialog(self):
        self.color_dialog.show()

    def set_object_color(self, object_id: int):
        r, g, b = davis_palette_np[object_id]
        viz_r, viz_g, viz_b = self.davis_palette_np2viz_palette_np[
            str(davis_palette_np[object_id])
        ]
        rgb = f"rgb({r},{g},{b})"
        self.object_color.setStyleSheet("QLabel {background: " + rgb + ";}")
        self.object_color.setText(f"{object_id}")

        viz_rgb = f"rgb({viz_r},{viz_g},{viz_b})"
        self.viz_object_color.setStyleSheet("QLabel {background: " + viz_rgb + ";}")
        self.viz_object_color.setText(f"{object_id}")
        self.color_dialog.setCurrentColor(QColor(viz_r, viz_g, viz_b))

    def resizeEvent(self, event):
        self.controller.show_current_frame()

    def text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text + "\n")

    def set_canvas(
        self,
        image,
        x=0,
        y=0,
        size=0,
        bound_ratio=1.5,
        zoom_factor=3,
        when_paint_earse=False,
        mag_glass=False,
    ):

        # Draw a white circle on the image
        if when_paint_earse:
            to_circle_image = image.copy()

            cv2.circle(
                to_circle_image, (int(x), int(y)), size, (0, 255, 0), 1
            )  # -1 fills the circle
            image = to_circle_image
            if mag_glass:
                image = self.magnifying_glass(
                    image, x, y, size * bound_ratio, zoom_factor
                )

        height, width, channel = image.shape
        # if the image is RGBA, convert to RGB first by coloring the background green
        if channel == 4:
            image_rgb = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255
            green_bg = np.array([0, 255, 0])
            # soft blending
            image = (
                image_rgb * alpha[:, :, np.newaxis]
                + green_bg[np.newaxis, np.newaxis, :] * (1 - alpha[:, :, np.newaxis])
            ).astype(np.uint8)

        bytesPerLine = 3 * width

        qImg = QImage(
            image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        )
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.main_canvas.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        )

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def magnifying_glass(self, image, x, y, radius, zoom_factor=3):
        x = int(x)
        y = int(y)
        radius = max(1, int(radius))

        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)

    
        # get the region to be zoomed in by the mask
        original_region = cv2.bitwise_and(image, image, mask=mask)

        # zoom the region and add the mask
        zoomed_radius = int(radius * zoom_factor)
        zoomed_region = cv2.resize(
            original_region[y - radius : y + radius, x - radius : x + radius],
            (2 * zoomed_radius, 2 * zoomed_radius),
            interpolation=cv2.INTER_LINEAR,
        )

        zoomed_mask = cv2.resize(
            mask[y - radius : y + radius, x - radius : x + radius],
            (2 * zoomed_radius, 2 * zoomed_radius),
            interpolation=cv2.INTER_LINEAR,
        )

     
        # get the new top-left and bottom-right coordinates of the zoomed region
        new_top_left_x = x - zoomed_radius
        new_top_left_y = y - zoomed_radius
        new_bottom_right_x = x + zoomed_radius
        new_bottom_right_y = y + zoomed_radius

        
        # insert the zoomed region into the original image
        result = image.copy()
        for c in range(3):
            result[
                new_top_left_y:new_bottom_right_y, new_top_left_x:new_bottom_right_x, c
            ] = np.where(
                zoomed_mask > 0,
                zoomed_region[:, :, c],
                result[
                    new_top_left_y:new_bottom_right_y,
                    new_top_left_x:new_bottom_right_x,
                    c,
                ],
            )

        image = result

        
        # Draw a white circle on the image
        color = (255, 255, 255)  # BGR格式
        thickness = 1
        cv2.circle(image, (x, y), zoomed_radius, color, thickness)


        return image

    def update_slider(self, value):
        self.lcd.setText("{: 3d} / {: 3d}".format(value, self.controller.T - 1))
        self.tl_slider.setValue(value)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)
        out_of_bound = (x < 0) or (y < 0) or (x > self.w - 1) or (y > self.h - 1)

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        return x, y

    def forward_propagation_start(self):
        self.backward_run_button.setEnabled(False)
        self.forward_run_button.setText("Pause propagation")

    def backward_propagation_start(self):
        self.forward_run_button.setEnabled(False)
        self.backward_run_button.setText("Pause propagation")

    def pause_propagation(self):
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_all_mem_button.setEnabled(True)
        self.clear_non_perm_mem_button.setEnabled(True)
        self.forward_run_button.setText("Propagate forward")
        self.backward_run_button.setText("propagate backward")
        self.tl_slider.setEnabled(True)

    def process_events(self):
        QApplication.processEvents()

    def on_mouse_press(self, event):
        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            return

        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        if event.button() == Qt.MouseButton.LeftButton:
            action = "left"
        elif event.button() == Qt.MouseButton.RightButton:
            action = "right"
        elif event.button() == Qt.MouseButton.MiddleButton:
            action = "middle"

        self.set_mouse_pressed(True)
        self.click_fn(action, ex, ey)

    def on_mouse_motion(self, event):
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        self.on_mouse_motion_xy(ex, ey)

        self.paint_erase(ex, ey)

    def on_mouse_release(self, event):
        self.set_mouse_pressed(False)
        pass

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.on_scroll_up()
        else:
            self.on_scroll_down()

    def on_scroll_up(self):
        self.controller.add_color_tolerance()
        

    def on_scroll_down(self):
        self.controller.reduce_color_tolerance()
        

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play video")
        else:
            self.timer.start(1000 // 30)
            self.play_button.setText("Stop video")

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, prompt, "", "Image files (*)", options=options
        )
        return file_name

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()

    def get_phase_trans_to_select(
        self, config_dict, usr_define_path, before_state, after_state
    ):
        usr_define_dict = json.load(open(usr_define_path, "r"))
        select_list = []

        if "" == before_state or "" == after_state:
            for key, value in config_dict.items():
                for value_item in value:
                    select_list.append(value_item)

            for key, value in usr_define_dict.items():
                for value_item in value:
                    select_list.append(value_item)

        else:

            for key, value in config_dict.items():
                key_before_state = key.split("-")[0]
                key_after_state = key.split("-")[1]
                if key_before_state in before_state.split(
                    ":"
                ) and key_after_state in after_state.split(":"):
                    for value_item in value:
                        select_list.append(value_item)

            for key, value in usr_define_dict.items():
                key_before_state = key.split("-")[0]
                key_after_state = key.split("-")[1]
                if key_before_state in before_state and key_after_state in after_state:
                    for value_item in value:
                        select_list.append(value_item)

        return select_list
