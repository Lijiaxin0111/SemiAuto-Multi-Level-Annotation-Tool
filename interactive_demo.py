import logging
import os
import sys
import json
import multiprocessing

# fix for Windows
if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import time
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument(
        "--images", help="Folders containing input images.", default=None
    )
    parser.add_argument("--model_name", help="Segment Model: cutie", type=str, default="cutie")

    parser.add_argument("--video", help="Video file readable by OpenCV.", default=None)
    # parser.add_argument('--gpu', help='the using gpu id ', default=None)
    parser.add_argument(
        "--workspace",
        help="directory for storing buffered images (if needed) and output masks",
        default=None,
    )
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument(
        "--workspace_init_only",
        action="store_true",
        help="initialize the workspace and exit",
    )

    parser.add_argument(
        "--only_label_challenge",
        action="store_true",
        help="only_label_challenge, the masks must exists",
    )

    parser.add_argument(
        "--gpu", help="using the id of gpu", default=None, required=False
    )

    args = parser.parse_args()
    return args


if __name__ in "__main__":
    # input arguments
    args = get_arguments()

    # perform slow imports after parsing args

    from omegaconf import open_dict
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from PySide6.QtWidgets import QApplication
    import qdarktheme

    # logging
    log = logging.getLogger()

    # getting hydra's config without using its decorator

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch

    # general setup
    torch.set_grad_enabled(False)

    from gui.main_controller import MainController

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    

    if args.model_name == "cutie":
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()
        initialize(version_base="1.3.2", config_path="cutie/config", job_name="gui")
        cfg = compose(config_name="gui_config")
    else:
        raise NotImplementedError(f"Model {args.model_name} is not supported")

    args.device = device
    log.info(f"Using device: {device}")
    if args.gpu is not None:
        log.info(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            assert k not in cfg, f"Argument {k} already exists in config"
            cfg[k] = v

    # start everything

    # ------------------------
    # Add challenge label

    if cfg["images"] is not None:
        basename = os.path.basename(cfg["images"])

    elif cfg["video"] is not None:
        basename = os.path.basename(cfg["video"])[:-4]

    else:
        raise NotImplementedError(
            "Either images, video, or workspace has to be specified"
        )

    if cfg["workspace"] is None:
        state_label_json_pos = os.path.join(cfg["workspace_root"], "phase_label.json")
    else:
        state_label_json_pos = os.path.join(
            os.path.dirname(cfg["workspace"]), "phase_label.json"
        )

    with open_dict(cfg):
        cfg["state_label_json_pos"] = state_label_json_pos
        cfg["video_name"] = basename

    # add_challenge_label_gui( cfg["num_objects"], challenge_label_json_pos,   basename)
    # print(challenge_label_json_pos )
    # print(basename)
    # -------------------


    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    ex = MainController(cfg)

    if "workspace_init_only" in cfg and cfg["workspace_init_only"]:
        sys.exit(0)
    else:
        app.exec()

    video_id = basename


