import os
from os import path
import logging
from typing import Literal

import cv2

# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


import torch

try:
    from torch import mps
except:
    print("torch.MPS not available.")

from torch import autocast

from torchvision.transforms.functional import to_tensor

import numpy as np

from omegaconf import DictConfig, open_dict, OmegaConf

from cutie.model.cutie import CUTIE

from cutie.inference.inference_core import InferenceCore


from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.phase_utils import get_full_transition
from cutie.utils.palette import davis_palette_np


from PySide6.QtGui import QColor
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from scripts.download_models import download_models_if_needed
import time
import json
import time

log = logging.getLogger()


PAINT_MODE = 1
ERASE_MODE = 2
AUTO_MODE = 3
PAINT_VOID_MODE = 4
ERASE_VOID_MODE = 5
MODE2STR = {
    PAINT_MODE: "Paint MODE",
    ERASE_MODE: "Erase MODE",
    AUTO_MODE: "Auto MODE",
    PAINT_VOID_MODE: "Paint Void MODE",
    ERASE_VOID_MODE: "Erase Void MODE",
}


class MainController:
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError(
                    "Either images, video, or workspace has to be specified"
                )

            cfg["workspace"] = path.join(cfg["workspace_root"], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg["num_objects"]
        self.device = cfg["device"]
        self.amp = cfg["amp"]
        print(f"Using device: {self.device}")

        # initializing the network(s)
        self.initialize_networks()
        print("Networks initialized.")

        # main components
        self.res_man = ResourceManager(cfg)
        if "workspace_init_only" in cfg and cfg["workspace_init_only"]:
            return

        if cfg["model_name"] == "cutie":
            self.processor = InferenceCore(self.cutie, self.cfg)
        else:

            raise NotImplementedError
        print("Processor initialized.")
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = "Click"
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal["forward", "backward", "none"] = "none"
        self.last_ex = self.last_ey = 0
        self.curr_video = self.cfg["video_name"]

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros(
            (self.num_objects + 1, self.h, self.w), dtype=torch.float
        ).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = "davis"
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = "None"
        self.save_soft_mask: bool = False

        self.interacted_prob: torch.Tensor = None
        self.interacted_void_mask: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # color mask parameter
        self.selected_color = None
        self.color_tolerance = 50
        self.select_coloring = False

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # for annotation
        self.state_before = ""
        self.state_after = ""
        self.phase_trans = ""
        self.phase_trans_dict = OmegaConf.to_container(cfg["phase_transition"])
        self.usr_def_phase_trans_dict = json.load(
            open(cfg["usr_define_phase_trans_json_path"])
        )

        all_dict = self.phase_trans_dict.copy()
        all_dict.update(self.usr_def_phase_trans_dict)

        self.all_phase_trans_dict = get_full_transition(
            all_dict, OmegaConf.to_container(cfg["phase"])
        )

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()

        if self.cfg["model_name"] == "cutie":
            self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
            self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
            self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
            self.gui.mem_every_box.setValue(self.processor.mem_every)
        else:
            self.gui.work_mem_min.setValue(0)
            self.gui.work_mem_max.setValue(0)
            self.gui.long_mem_max.setValue(0)
            self.gui.mem_every_box.setValue(0)
        self.init_state()
        self.init_trans()

        # for exporting videos
        self.output_fps = cfg["output_fps"]
        self.output_bitrate = cfg["output_bitrate"]

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn
        self.mouse_pressed = False
        self.gui.paint_erase = self.paint_erase
        self.gui.set_mouse_pressed = self.set_mouse_pressed

        # paint_erase parameter
        self.paint_radius = 5
        self.mag_image = False
        self.bound_ratio = 3
        self.zoom_factor = 3

        # add click mod
        self.click_mod = AUTO_MODE
        self.last_click_mod = AUTO_MODE

        self.gui.show()
        self.gui.text("Initialized.")
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer("./docs/uiuc.png")
        self.gui.set_object_color(self.curr_object)
        self.update_config()

    def initialize_networks(self) -> None:
        if self.cfg.model_name == "cutie":
            download_models_if_needed()
            self.cutie = CUTIE(self.cfg).eval().to(self.device)
            model_weights = torch.load(self.cfg.weights, map_location=self.device)
            self.cutie.load_weights(model_weights)
            self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)

        else:
            raise NotImplementedError

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f"Current object changed to {number}.")
        self.gui.set_object_color(number)
        self.init_state()
        self.show_current_frame()

    def updateColor(self, color: QColor):
        rgb_color = [color.red(), color.green(), color.blue()]
        self.gui.davis_palette_np2viz_palette_np[
            str(davis_palette_np[self.curr_object])
        ] = rgb_color

        self.gui.set_object_color(self.curr_object)


        self.compose_current_im()
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def click_fn(self, action: Literal["left", "right", "middle"], x: int, y: int):

        if self.propagating:
            return

        if self.select_coloring and self.click_mod != AUTO_MODE:

            self.selected_color = self.curr_image_np[int(y), int(x), :]
            self.zoom_factor = self.last_zoom
            self.paint_radius = self.last_r
            self.bound_ratio = self.last_b
            self.select_coloring = False
            self.mag_image = False
            self.on_use_similar_color_mask()
            self.gui.text(f"Selected color: {self.selected_color}")
            time.sleep(0.2)
            return

        if self.click_mod != AUTO_MODE:
            self.paint_erase(x, y)

        if self.click_mod == AUTO_MODE:
            self.update_canvas()

            last_interaction = self.interaction
            new_interaction = None

            with autocast(self.device, enabled=(self.amp and self.device == "cuda")):
                if action in ["left", "right"]:
                    # left: positive click
                    # right: negative click
                    self.convert_current_image_mask_torch()
                    image = self.curr_image_torch
                    if (
                        last_interaction is None
                        or last_interaction.tar_obj != self.curr_object
                        or self.last_click_mod != AUTO_MODE
                    ):
                        # create new interaction is needed
                        self.complete_interaction()
                        self.click_ctrl.unanchor()
                        if self.cfg.model_name == "cutie":
                            new_interaction = ClickInteraction(
                                self.cfg,
                                image,
                                self.curr_prob,
                                (self.h, self.w),
                                self.click_ctrl,
                                self.curr_object,
                                self.curr_ti,
                            )
                        else:
                            raise NotImplementedError

                        if new_interaction is not None:
                            self.interaction = new_interaction

                    self.interaction.push_point(x, y, is_neg=(action == "right"))
                    self.interacted_prob = self.interaction.predict().to(
                        self.device, non_blocking=True
                    )
                    self.interacted_void_mask = (
                        self.res_man.get_void_mask(self.curr_ti)
                        if self.res_man.get_void_mask(self.curr_ti) is not None
                        else np.zeros((self.h, self.w))
                    )
                    self.update_interacted_mask()
                    self.update_gpu_gauges()

                elif action == "middle":
                    # middle: select a new visualization object
                    target_object = self.curr_mask[int(y), int(x)]
                    if target_object in self.vis_target_objects:
                        self.vis_target_objects.remove(target_object)
                    else:
                        self.vis_target_objects.append(target_object)
                    self.gui.text(
                        f"Overlay target(s) changed to {self.vis_target_objects}"
                    )
                    self.show_current_frame()
                    return
                else:
                    raise NotImplementedError

    def paint_erase(self, x: int, y: int):
        if self.click_mod != AUTO_MODE:
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )

        if self.mouse_pressed and (
            self.click_mod == PAINT_VOID_MODE or self.click_mod == ERASE_VOID_MODE
        ):

            self.convert_current_image_mask_torch()
            self.complete_interaction()
            self.interaction = PressVOID_Interaction(
                self.curr_prob, self.curr_void_mask, (self.h, self.w), self.paint_radius
            )

            self.interacted_void_mask, self.interacted_prob = self.interaction.predict(
                x, y, is_neg=(self.click_mod == ERASE_VOID_MODE)
            )
            self.update_interacted_mask()

            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )
            self.update_gpu_gauges()

        if self.mouse_pressed and (
            self.click_mod == PAINT_MODE or self.click_mod == ERASE_MODE
        ):
            last_interaction = self.interaction

            self.convert_current_image_mask_torch()
            self.complete_interaction()

            self.interaction = PressInteraction(
                self.curr_prob, (self.h, self.w), self.curr_object, self.paint_radius
            )

            self.interacted_void_mask = (
                self.res_man.get_void_mask(self.curr_ti)
                if self.res_man.get_void_mask(self.curr_ti) is not None
                else np.zeros((self.h, self.w))
            )
            self.interacted_prob = self.interaction.predict(
                x, y, is_neg=(self.click_mod == ERASE_MODE)
            ).to(self.device, non_blocking=True)
            self.update_interacted_mask()
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )

            self.update_gpu_gauges()

    def paint_in_color_mask(self):
        if self.selected_color is not None:
            self.convert_current_image_mask_torch()
            _, color_mask = color_mask_out(
                self.curr_image_np, self.selected_color, self.color_tolerance
            )


            last_obj_mask = self.curr_prob[self.curr_object].unsqueeze(0).unsqueeze(0)

            last_obj_mask[0, 0, color_mask] = 0.9
            obj_mask = last_obj_mask

            out_prob = self.curr_prob.clone()
            out_prob[self.curr_object] = obj_mask
            self.interacted_prob = aggregate_wbg(out_prob[1:], keep_bg=True, hard=True)
            self.update_interacted_mask()
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )
            self.update_gpu_gauges()
        else:
            self.gui.text("Please select a color first")

    def on_auto_mode(self):
        if self.click_mod == AUTO_MODE:
            return
        self.last_click_mod = self.click_mod
        self.click_mod = AUTO_MODE
        self.update_canvas()
        self.gui.text(
            f"From {MODE2STR[self.last_click_mod]} to {MODE2STR[self.click_mod]}"
        )

    def on_paint_mode(self):
        if self.click_mod == PAINT_MODE:
            return
        self.last_click_mod = self.click_mod
        self.click_mod = PAINT_MODE
        self.gui.text(
            f"From {MODE2STR[self.last_click_mod]} to {MODE2STR[self.click_mod]}"
        )

    def on_erase_mode(self):
        if self.click_mod == ERASE_MODE:
            return
        self.last_click_mod = self.click_mod
        self.click_mod = ERASE_MODE
        self.gui.text(
            f"From {MODE2STR[self.last_click_mod]} to {MODE2STR[self.click_mod]}"
        )

    def on_paint_void_mode(self):
        if self.click_mod == PAINT_VOID_MODE:
            return
        self.last_click_mod = self.click_mod

        self.click_mod = PAINT_VOID_MODE
        self.gui.text(
            f"From {MODE2STR[self.last_click_mod]} to {MODE2STR[self.click_mod]}"
        )

    def on_erase_void_mode(self):
        if self.click_mod == ERASE_VOID_MODE:
            return
        self.last_click_mod = self.click_mod
        self.click_mod = ERASE_VOID_MODE
        self.gui.text(
            f"From {MODE2STR[self.last_click_mod]} to {MODE2STR[self.click_mod]}"
        )

    def add_painter_size(self):
        self.paint_radius += 1
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def reduce_painter_size(self):
        self.paint_radius -= 1
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def add_zoom_factor(self):
        self.zoom_factor += 1
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def reduce_zoom_factor(self):
        self.zoom_factor -= 1
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def add_bound_ratio(self):
        self.bound_ratio *= 1.1
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def reduce_bound_ratio(self):
        self.bound_ratio *= 0.9
        self.gui.set_canvas(
            self.vis_image,
            self.last_ex,
            self.last_ey,
            size=self.paint_radius,
            bound_ratio=self.bound_ratio,
            zoom_factor=self.zoom_factor,
            when_paint_earse=True,
            mag_glass=self.mag_image,
        )

    def set_mag_image(self):
        self.mag_image = not self.mag_image
        if self.click_mod != AUTO_MODE:
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )

    def on_use_similar_color_mask(self):
        if self.click_mod != AUTO_MODE:
            self.compose_current_im()
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )

        pass

    def select_color(self):
        if self.selected_color is None and self.click_mod != AUTO_MODE:
            self.last_r = self.paint_radius
            self.last_zoom = self.zoom_factor
            self.last_b = self.bound_ratio
            self.paint_radius = 1
            self.bound_ratio = 10
            self.zoom_factor = 10
            self.mag_image = True
            self.compose_current_im()
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=True,
                mag_glass=self.mag_image,
            )

            self.select_coloring = True
            return
        else:
            self.selected_color = None
            self.select_coloring = False
            self.compose_current_im()
            self.gui.set_canvas(
                self.vis_image,
                self.last_ex,
                self.last_ey,
                size=self.paint_radius,
                bound_ratio=self.bound_ratio,
                zoom_factor=self.zoom_factor,
                when_paint_earse=False,
                mag_glass=self.mag_image,
            )
            return

    def add_color_tolerance(self):
        if self.selected_color is not None:
            self.color_tolerance += 5
            t_bound1 = 255 - np.max(self.selected_color)
            t_bound2 = np.min(self.selected_color)
            t_bound = max(t_bound1, t_bound2)
            self.color_tolerance = min(t_bound, self.color_tolerance)

            self.on_use_similar_color_mask()
 

    def reduce_color_tolerance(self):
        if self.selected_color is not None:
            self.color_tolerance -= 5
            t_bound1 = 255 - np.max(self.selected_color)
            t_bound2 = np.min(self.selected_color)
            t_bound = max(t_bound1, t_bound2)
            self.color_tolerance = min(t_bound, self.color_tolerance)
            self.on_use_similar_color_mask()
    

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)

            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()

            self.curr_prob = None
            self.curr_void_mask = (
                self.res_man.get_void_mask(self.curr_ti)
                if self.res_man.get_void_mask(self.curr_ti) is not None
                else np.zeros((self.h, self.w))
            )

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(
                self.device, non_blocking=True
            )

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(
                self.curr_mask, self.num_objects + 1
            ).to(self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(
            self.vis_mode,
            self.curr_image_np,
            self.curr_mask,
            self.overlay_layer,
            self.vis_target_objects,
            self.curr_void_mask,
            self.selected_color,
            self.color_tolerance,
            self.gui.davis_palette_np2viz_palette_np,
        )

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(
        self, invalid_soft_mask: bool = False, void_mask: torch.Tensor = None
    ):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        self.vis_image = get_visualization_torch(
            self.vis_mode,
            self.curr_image_torch,
            self.curr_prob,
            self.overlay_layer_torch,
            self.vis_target_objects,
            void_mask=void_mask,
            selected_color=self.selected_color,
            color_tolerance=self.color_tolerance,
            viz_color_map=self.gui.davis_palette_np2viz_palette_np,
        )
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            "Propagation only (higher quality)",
            "Always",
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask, self.curr_void_mask)

        else:
            self.compose_current_im()

            if self.save_visualization_mode == "Always":
                self.res_man.save_visualization(
                    self.curr_ti, self.vis_mode, self.vis_image
                )

            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + ".jpg")

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def set_before_state(self):
        self.state_before = self.gui.before_state_combo_box.currentText()
        self.gui.text(f"Before state set to {self.state_before}.")

    def set_after_state(self):
        self.state_after = self.gui.after_state_combo_box.currentText()
        self.gui.text(f"After state set to {self.state_after}.")

    def set_phase_trans(self):
        self.phase_trans = self.gui.phase_trans_combo_box.currentText()
        self.gui.text(f"Phase transition set to {self.phase_trans}.")

    def init_state(self):

        if os.path.exists(self.cfg["state_label_json_pos"]):
            state_label_dict = json.load(open(self.cfg["state_label_json_pos"]))

            if (
                self.curr_video in state_label_dict
                and str(self.curr_object) in state_label_dict[self.curr_video]
                and "before_state"
                in state_label_dict[self.curr_video][str(self.curr_object)]
                and "after_state"
                in state_label_dict[self.curr_video][str(self.curr_object)]
            ):

                self.state_after = state_label_dict[self.curr_video][
                    str(self.curr_object)
                ]["after_state"]
                self.state_before = state_label_dict[self.curr_video][
                    str(self.curr_object)
                ]["before_state"]
                self.gui.before_state_combo_box.setCurrentText(self.state_before)
                self.gui.after_state_combo_box.setCurrentText(self.state_after)
                self.gui.text(
                    f"Video {self.curr_video} of Obj {self.curr_object} Before state init set to {self.state_before}."
                )
                self.gui.text(
                    f"Video {self.curr_video} of Obj {self.curr_object} After state init set to {self.state_after}."
                )
                return

        self.state_before = ""
        self.state_after = ""
        self.gui.before_state_combo_box.setCurrentText(self.state_before)
        self.gui.after_state_combo_box.setCurrentText(self.state_after)
        self.gui.text(
            f"Video {self.curr_video} of Obj {self.curr_object}  not in the phase.json, Don't init the state."
        )

    def init_trans(self):
        if os.path.exists(self.cfg["state_label_json_pos"]):
            state_label_dict = json.load(open(self.cfg["state_label_json_pos"]))
            if (
                self.curr_video in state_label_dict
                and str(self.curr_object) in state_label_dict[self.curr_video]
                and "phase transition"
                in state_label_dict[self.curr_video][str(self.curr_object)]
            ):

                self.phase_trans = state_label_dict[self.curr_video][
                    str(self.curr_object)
                ]["phase transition"]
                self.gui.phase_trans_combo_box.setCurrentText(self.phase_trans)
                self.gui.text(
                    f"Video {self.curr_video} of Obj {self.curr_object} Phase transition init set to {self.phase_trans}."
                )
                self.gui.phase_trans_list = self.gui.get_phase_trans_to_select(
                    self.cfg["phase_transition"],
                    self.cfg["usr_define_phase_trans_json_path"],
                    self.state_before,
                    self.state_after,
                )
                self.gui.phase_trans_combo_box.reset_tree(self.gui.phase_trans_list)
                return

        self.phase_trans = ""
        self.gui.phase_trans_combo_box.setCurrentText(self.phase_trans)
        self.gui.text(
            f"the phase transition of Video {self.curr_video} of Obj {self.curr_object}  not in the phase.json, Don't init the phase transition."
        )

        self.gui.phase_trans_list = self.gui.get_phase_trans_to_select(
            self.cfg["phase_transition"],
            self.cfg["usr_define_phase_trans_json_path"],
            self.state_before,
            self.state_after,
        )
        self.gui.phase_trans_combo_box.reset_tree(self.gui.phase_trans_list)

    def on_save_state(self):

        if os.path.exists(self.cfg["state_label_json_pos"]):
            state_label_dict = json.load(open(self.cfg["state_label_json_pos"]))
        else:
            state_label_dict = {}

        if self.curr_video not in state_label_dict:
            state_label_dict[self.curr_video] = {}

        if str(self.curr_object) not in state_label_dict[self.curr_video]:
            state_label_dict[self.curr_video][str(self.curr_object)] = {}

        self.state_before = self.gui.before_state_combo_box.currentText()
        self.state_after = self.gui.after_state_combo_box.currentText()

        state_label_dict[self.curr_video][str(self.curr_object)][
            "before_state"
        ] = self.gui.before_state_combo_box.currentText()
        state_label_dict[self.curr_video][str(self.curr_object)][
            "after_state"
        ] = self.gui.after_state_combo_box.currentText()

        with open(self.cfg["state_label_json_pos"], "w") as f:
            json.dump(state_label_dict, f, indent=4)
     
        self.gui.text(
            f"[Saved] Video {self.curr_video} of Obj {self.curr_object} from {self.state_before} to {self.state_after} ."
        )

        if (
            state_label_dict[self.curr_video][str(self.curr_object)]["before_state"]
            == ""
            or state_label_dict[self.curr_video][str(self.curr_object)]["after_state"]
            == ""
        ):
            self.gui.text(
                f"[Warning] Video {self.curr_video} of Obj {self.curr_object} has no state label."
            )

        self.gui.phase_trans_list = self.gui.get_phase_trans_to_select(
            self.cfg["phase_transition"],
            self.cfg["usr_define_phase_trans_json_path"],
            self.state_before,
            self.state_after,
        )
        self.gui.phase_trans_combo_box.reset_tree(self.gui.phase_trans_list)

    def on_save_trans(self):
        if os.path.exists(self.cfg["state_label_json_pos"]):
            state_label_dict = json.load(open(self.cfg["state_label_json_pos"]))
        else:
            state_label_dict = {}

        if self.curr_video not in state_label_dict:
            state_label_dict[self.curr_video] = {}

        if str(self.curr_object) not in state_label_dict[self.curr_video]:
            state_label_dict[self.curr_video][str(self.curr_object)] = {}

        self.phase_trans = self.gui.phase_trans_combo_box.currentText()

        state_label_dict[self.curr_video][str(self.curr_object)][
            "phase transition"
        ] = self.gui.phase_trans_combo_box.currentText()

        with open(self.cfg["state_label_json_pos"], "w") as f:
            json.dump(state_label_dict, f, indent=4)
        self.gui.text(
            f"[Saved] Video {self.curr_video} of Obj {self.curr_object} from {self.state_before} to {self.state_after}: {self.phase_trans}  ."
        )

  
        state_before_after = (
            f"{self.state_before.split(':')[-1]}-{self.state_after.split(':')[-1]}"
        )




        if (
            state_before_after not in self.all_phase_trans_dict
            or self.phase_trans not in self.all_phase_trans_dict[state_before_after]
        ):

            self.gui.text(
                f"[ATTENTION] {self.phase_trans} is not in the phase transition list of {self.state_before}-{self.state_after}."
            )
            self.gui.text(
                f'[SAVED]  {self.state_before}-{self.state_after} : {self.phase_trans} will be added into the usr define phase transition list : {self.cfg["usr_define_phase_trans_json_path"]}'
            )
            if state_before_after not in self.usr_def_phase_trans_dict:
                self.usr_def_phase_trans_dict[state_before_after] = []

            self.usr_def_phase_trans_dict[state_before_after].append(self.phase_trans)
            with open(self.cfg["usr_define_phase_trans_json_path"], "w") as f:
                json.dump(self.usr_def_phase_trans_dict, f, indent=4)

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def save_current_void_mask(self):
        to_save_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        to_save_mask[self.curr_void_mask > 0] = 255
        self.res_man.save_void_mask(self.curr_ti, to_save_mask)

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here

        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            # with self.vis_cond:
            #     self.vis_cond.notify()

            if self.curr_frame_dirty:
                self.save_current_mask()
 

            self.curr_frame_dirty = False

            self.reset_this_interaction()

            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()

            self.show_current_frame()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = "none"
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = "forward"
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = "none"
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = "backward"
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f"Propagation stopped at t={self.curr_ti}.")
        self.gui.pause_propagation()

    def on_propagate(self):
        # start to propagate

        with autocast(
            "cuda" if "cuda" in self.device else "cpu",
            enabled=(self.amp and self.device == "cuda"),
        ):
            self.convert_current_image_mask_torch()

            self.gui.text(f"Propagation started at t={self.curr_ti}.")
            self.processor.clear_sensory_memory()



            self.curr_prob = self.processor.step(
                self.curr_image_torch,
                self.curr_prob[1:],
                idx_mask=False,
                curr_ti=self.curr_ti,
            )

            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.curr_void_mask = (
                self.res_man.get_void_mask(self.curr_ti)
                if self.res_man.get_void_mask(self.curr_ti) is not None
                else np.zeros((self.h, self.w))
            )
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(
                self.res_man, self.curr_ti, self.propagate_direction
            )
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end

            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(
                    self.device, non_blocking=True
                )
                self.propagate_fn()

                self.curr_prob = self.processor.step(
                    self.curr_image_torch, curr_ti=self.curr_ti
                )
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
                self.curr_void_mask = (
                    self.res_man.get_void_mask(self.curr_ti)
                    if self.res_man.get_void_mask(self.curr_ti) is not None
                    else np.zeros((self.h, self.w))
                )

                self.save_current_mask()
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == "cuda")):
            self.convert_current_image_mask_torch()
            self.gui.text(f"Permanent memory saved at {self.curr_ti}.")

            self.curr_prob = self.processor.step(
                self.curr_image_torch,
                self.curr_prob[1:],
                idx_mask=False,
                force_permanent=True,
                curr_ti=self.curr_ti,
            )

            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg["workspace"], "visualization", self.vis_mode)
        save_folder = self.cfg["workspace"]
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f"visualization_{self.vis_mode}.mp4")
            self.gui.text(f"Exporting visualization -- please wait")
            self.gui.process_events()
            convert_frames_to_video(
                image_folder,
                output_path,
                fps=self.output_fps,
                bitrate=self.output_bitrate,
                progress_callback=self.gui.progressbar_update,
            )
            self.gui.text(f"Visualization exported to {output_path}")
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f"No visualization images found in {image_folder}")

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg["workspace"], "masks")
        save_folder = path.join(self.cfg["workspace"], "binary_masks")
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f"Exporting binary masks -- please wait")
            self.gui.process_events()
            convert_mask_to_binary(
                mask_folder,
                save_folder,
                self.vis_target_objects,
                progress_callback=self.gui.progressbar_update,
            )
            self.gui.text(f"Binary masks exported to {save_folder}")
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f"No masks found in {mask_folder}")

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.curr_void_mask = (
            self.interacted_void_mask
            if self.interacted_void_mask is not None
            else np.zeros((self.h, self.w))
        )
        self.save_current_mask()
        self.save_current_void_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()

        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()

        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()

        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if "cuda" in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= 2**30
            global_total /= 2**30
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(
                f"{global_used:.1f} GB / {global_total:.1f} GB"
            )
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(
                f"{used_by_torch:.1f} GB / {global_total:.1f} GB"
            )
            self.gui.torch_mem_gauge.setValue(
                round(used_by_torch / global_total * 100 / 1024)
            )
        elif "mps" in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f"{mem_used:.1f} GB")
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat("N/A")
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat("N/A")
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat("N/A")
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(
                f"{curr_perm_tokens} / {curr_perm_tokens}"
            )
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f"{curr_work_tokens} / {max_work_tokens}")
            self.gui.work_mem_gauge.setValue(
                round(curr_work_tokens / max_work_tokens * 100)
            )

            self.gui.long_mem_gauge.setFormat(f"{curr_long_tokens} / {max_long_tokens}")
            self.gui.long_mem_gauge.setValue(
                round(curr_long_tokens / max_long_tokens * 100)
            )

        except AttributeError as e:
            self.gui.work_mem_gauge.setFormat("Unknown")
            self.gui.long_mem_gauge.setFormat("Unknown")
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(), self.gui.work_mem_max.value() - 1)
            )
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(), self.gui.work_mem_min.value() + 1)
            )
            self.update_config()

    def update_config(self):
        if self.initialized and self.cfg["model_name"] == "cutie":
            with open_dict(self.cfg):
                self.cfg.long_term["min_mem_frames"] = self.gui.work_mem_min.value()
                self.cfg.long_term["max_mem_frames"] = self.gui.work_mem_max.value()
                self.cfg.long_term["max_num_tokens"] = self.gui.long_mem_max.value()
                self.cfg["mem_every"] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)



    def on_clear_memory(self):
        self.processor.clear_memory()
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        elif "mps" in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        elif "mps" in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file("Mask")
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = (
            (len(mask.shape) == 2)
            and (mask.shape[-1] == self.w)
            and (mask.shape[-2] == self.h)
        )

        object_condition = mask.max() <= self.num_objects

        if not shape_condition:
            self.gui.text(f"Expected ({self.h}, {self.w}). Got {mask.shape} instead.")
        elif not object_condition:
            self.gui.text(
                f"Expected {self.num_objects} objects. Got {mask.max()} objects instead."
            )
        else:
            self.gui.text(f"Mask file {file_name} loaded.")
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()
            self.save_current_void_mask()

    def on_import_layer(self):
        file_name = self.gui.open_file("Layer")
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f"Layer file {file_name} loaded.")
            self.overlay_layer = layer
            self.overlay_layer_torch = (
                torch.from_numpy(layer).float().to(self.device) / 255
            )
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f"{file_name} not found.")

    def on_set_save_visualization_mode(self):
        self.save_visualization_mode = self.gui.save_visualization_combo.currentText()

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

    def set_mouse_pressed(self, pressed: bool):
        self.mouse_pressed = pressed

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
