# Modified from https://github.com/seoungwugoh/ivs-demo

from typing import Literal, List
import numpy as np

import torch
import torch.nn.functional as F
from cutie.utils.palette import davis_palette
import cv2


def image_to_torch(frame: np.ndarray, device: str = "cuda"):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def torch_prob_to_numpy_mask(prob: torch.Tensor):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


"""
Some constants fro visualization
"""
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
# color_map_np = (color_map_np.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255

grayscale_weights = np.array([[0.3, 0.59, 0.11]]).astype(np.float32)
grayscale_weights_torch = torch.from_numpy(grayscale_weights).to(device).unsqueeze(0)


def get_visualization(
    mode: Literal["image", "mask", "fade", "davis", "light", "popup", "layer", "rgba"],
    image: np.ndarray,
    mask: np.ndarray,
    layer: np.ndarray,
    target_objects: List[int],
    void_mask: np.ndarray = None,
    selected_color=None,
    color_tolerance=50,
    viz_color_map=None,
) -> np.ndarray:

    if selected_color is not None:
        image, color_mask = color_mask_out(image, selected_color, color_tolerance)

    if mode == "image":
        return image
    elif mode == "mask":
        return color_map_np[mask]
    elif mode == "fade":
        return overlay_davis(
            image, mask, fade=True, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "davis":
        return overlay_davis(
            image, mask, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "light":
        return overlay_davis(
            image, mask, 0.9, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "popup":
        return overlay_popup(image, mask, target_objects, void_mask=void_mask)
    elif mode == "layer":
        if layer is None:
            print("Layer file not given. Defaulting to DAVIS.")
            return overlay_davis(
                image, mask, void_mask=void_mask, viz_color_map=viz_color_map
            )
        else:
            return overlay_layer(
                image, mask, layer, target_objects, void_mask=void_mask
            )
    elif mode == "rgba":
        return overlay_rgba(image, mask, target_objects, void_mask=void_mask)
    else:
        raise NotImplementedError


def color_mask_out(image, selected_color, color_tolerance):
    is_image_torch = False
    if isinstance(image, torch.Tensor):
        image = image * 255
        image = image.permute(1, 2, 0).cpu().numpy()
        is_image_torch = True


    # transform the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    selected_color = cv2.cvtColor(
        np.array([[selected_color]], dtype=np.uint8), cv2.COLOR_BGR2RGB
    )[0][0]

    
    # calculate the color range
    lower_bound = np.array(selected_color).astype(int) - color_tolerance
    upper_bound = np.array(selected_color).astype(int) + color_tolerance


    # make sure the color range is within 0-255
    lower_bound = np.maximum(lower_bound, 1)
    upper_bound = np.minimum(upper_bound, 255)

    
    # create a mask to determine which pixels are within the specified color range
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)


    # create a new image that only shows pixels close to the target color, other pixels are set to black
    result_image = np.zeros_like(image_rgb)
    result_image[mask != 0] = image_rgb[mask != 0]


    image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    

    if is_image_torch:
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float().to(device, non_blocking=True) / 255

    return image, (mask != 0)


def get_visualization_torch(
    mode: Literal["image", "mask", "fade", "davis", "light", "popup", "layer", "rgba"],
    image: torch.Tensor,
    prob: torch.Tensor,
    layer: torch.Tensor,
    target_objects: List[int],
    void_mask: torch.Tensor = None,
    selected_color=None,
    color_tolerance=50,
    viz_color_map=None,
) -> np.ndarray:

    if selected_color is not None:
        image, color_mask = color_mask_out(image, selected_color, color_tolerance)

    if mode == "image":
        return image
    elif mode == "mask":
        mask = torch.max(prob, dim=0).indices
        return (color_map_torch[mask] * 255).byte().cpu().numpy()
    elif mode == "fade":
        return overlay_davis_torch(
            image, prob, fade=True, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "davis":
        return overlay_davis_torch(
            image, prob, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "light":
        return overlay_davis_torch(
            image, prob, 0.9, void_mask=void_mask, viz_color_map=viz_color_map
        )
    elif mode == "popup":
        return overlay_popup_torch(image, prob, target_objects, void_mask=void_mask)
    elif mode == "layer":
        if layer is None:
            print("Layer file not given. Defaulting to DAVIS.")
            return overlay_davis_torch(
                image, prob, void_mask=void_mask, viz_color_map=viz_color_map
            )
        else:
            return overlay_layer_torch(
                image, prob, layer, target_objects, void_mask=void_mask
            )
    elif mode == "rgba":
        return overlay_rgba_torch(image, prob, target_objects, void_mask=void_mask)
    else:
        raise NotImplementedError


def overlay_davis(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    fade: bool = False,
    void_mask: np.ndarray = None,
    viz_color_map=None,
):
    """Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    mask = mask.copy()
    if void_mask is not None:
        mask[void_mask > 0] = 255

    if viz_color_map is not None:

        color_map_viz = np.zeros_like(color_map_np)

        for i in range(len(color_map_viz)):
            color_map_viz[i] = viz_color_map[str(color_map_np[i])]

        colored_mask = color_map_viz[mask]

    else:
        colored_mask = color_map_np[mask]

    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = mask > 0



    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]

    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def overlay_popup(
    image: np.ndarray, mask: np.ndarray, target_objects: List[int], void_mask=None
):
    # Keep foreground colored. Convert background to grayscale.
    im_overlay = image.copy()

    binary_mask = ~(np.isin(mask, target_objects))
    colored_region = (im_overlay[binary_mask] * grayscale_weights).sum(-1, keepdims=-1)

    if void_mask is not None:
        binary_mask[void_mask > 0] = 0
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)


def overlay_layer(
    image: np.ndarray,
    mask: np.ndarray,
    layer: np.ndarray,
    target_objects: List[int],
    void_mask=None,
):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[:, :, np.newaxis]
    layer_alpha = layer[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255
    layer_rgb = layer[:, :, :3]
    if void_mask is not None:
        obj_mask[void_mask > 0] = 0
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (
        image * background_alpha
        + layer_rgb * (1 - obj_mask) * layer_alpha
        + image * obj_mask
    ).clip(0, 255)
    return im_overlay.astype(image.dtype)


def overlay_rgba(
    image: np.ndarray, mask: np.ndarray, target_objects: List[int], void_mask=None
):
    # Put the mask is in the alpha channel
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[
        :, :, np.newaxis
    ] * 255
    if void_mask is not None:
        obj_mask[void_mask > 0] = 0
    im_overlay = np.concatenate([image, obj_mask], axis=-1)
    return im_overlay.astype(image.dtype)


def overlay_davis_torch(
    image: torch.Tensor,
    prob: torch.Tensor,
    alpha: float = 0.5,
    fade: bool = False,
    void_mask: torch.Tensor = None,
    viz_color_map=None,
):
    """Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    # NOTE: Make sure you no longer use image after calling this function
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(prob, dim=0).indices

    # 显示void mask
    if void_mask is not None:
        mask[void_mask > 0] = 255

    if viz_color_map is not None:

        color_map_viz = torch.zeros_like(color_map_torch)
        for i in range(len(color_map_viz)):
            color_map_viz[i] = (
                torch.tensor(viz_color_map[str(color_map_np[i])]).to(device) / 255
            )

        colored_mask = color_map_viz[mask]

    else:
        colored_mask = color_map_torch[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = mask > 0
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_popup_torch(
    image: torch.Tensor, prob: torch.Tensor, target_objects: List[int], void_mask=None
):
    # Keep foreground colored. Convert background to grayscale.
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # I should not need to convert this to numpy.
        # Using list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)

    gray_image = (image * grayscale_weights_torch).sum(-1, keepdim=True)
    if void_mask is not None:
        obj_mask[void_mask > 0] = 0
    im_overlay = obj_mask * image + (1 - obj_mask) * gray_image
    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_layer_torch(
    image: torch.Tensor,
    prob: torch.Tensor,
    layer: torch.Tensor,
    target_objects: List[int],
    void_mask=None,
):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    if void_mask is not None:
        obj_mask[void_mask > 0] = 0

    layer_alpha = layer[:, :, 3].unsqueeze(2)
    layer_rgb = layer[:, :, :3]
    # background_alpha = torch.maximum(obj_mask, layer_alpha)
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (
        image * background_alpha
        + layer_rgb * (1 - obj_mask) * layer_alpha
        + image * obj_mask
    ).clip(0, 1)

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_rgba_torch(
    image: torch.Tensor, prob: torch.Tensor, target_objects: List[int], void_mask=None
):
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    if void_mask is not None:
        obj_mask[void_mask > 0] = 0
    im_overlay = torch.cat([image, obj_mask], dim=-1).clip(0, 1)
    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay
