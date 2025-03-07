# Semi-Auto-Multi-Level-Annotation-Tool

[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux)](https://www.linux.org/)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/torch-2.3.1%2Bcu118-orange?logo=pytorch)](https://pytorch.org/) [![arXiv](https://img.shields.io/badge/arXiv-2412.13803-b31b1b?logo=arxiv)](https://arxiv.org/html/2412.13803v2)  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repo is the office implement of the Semi-Auto Multi-Level Annotation Tool used in [M-cube-VOS](https://zixuan-chen.github.io/M-cube-VOS.github.io/). It allows for user to annotate the mask of target objects efficiently. In [M-cube-VOS](https://zixuan-chen.github.io/M-cube-VOS.github.io/), The pipeline of data collection is as follow:

 ![final_dataset_pipeline](./fig/final_dataset_pipeline.jpg)


## News :fire:

- We release the Annotation Tool.

- [M-cube-VOS](https://zixuan-chen.github.io/M-cube-VOS.github.io/) get accepted in CVPR 2025.

- We release the dataset  M-cube-VOS in [baidu disk](https://pan.baidu.com/s/1_JxrhJRB3UTqwQtUb2kFpg?pwd=mvos).

  
# Installation

Our test environment is :

- `Ubuntu 20.04.6 LTS`

- `Python 3.8.19`

- `torch 2.3.1+cu118` , `torchaudio 2.3.1+cu118`, `torchvision 0.18.1+cu118`

tip: The machine running this tool is expected to need  GeForce GTX and RTX.

**Clone our repository:**

```
git clone https://github.com/Lijiaxin0111/SemiAuto-Multi-Level-Annotation-Tool.git   
```

**Create Environment:**

```
conda create -n SemiAuto_AnnotateTool python=3.8
conda activate SemiAuto_AnnotateTool
```

**Install with pip:**

```
cd SemiAuto-Multi-Level-Annotation-Tool

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

(If you encounter "error: Microsoft Visual C++ 14.0 or greater is required.", get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Usage

### Quick Start

Tips: If you are running this on a remote server, you can use VNC or X11 forwarding.

```
python interactive_demo.py  --video ./demo_data/make_glass.mp4  --workspace ./workspace/make_glass --num_objects 1 --gpu 0
```

### More Usage

- [中文文档](./doc/UI_China.md)
- [English Doc](./doc/UI_ENG.md)


## UI dmeo

![UI](./fig/UI.png)

## Citation

```
@InProceedings{chen2024m3vos_2025_CVPR,
    author    = {Zixuan Chen and Jiaxin Li and Liming Tan and Yejie Guo and Junxuan Liang and Cewu Lu and Yong-Lu Li},
    title     = {M$^3$-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025}
}
```

## References

- This Semi-Auto-Multi-Level-Annotation-Tool is based on [Cutie GUI tool](https://raw.githubusercontent.com/hkchengrex/Cutie), [IVS](https://github.com/seoungwugoh/ivs-demo), [MiVOS](https://github.com/hkchengrex/MiVOS), and [XMem](https://github.com/hkchengrex/XMem).

  - The Cutie GUI tools uses [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) for interactive image segmentation. This repository also contains a redistribution of their code in `gui/ritm`. That part of code follows RITM's license.

  - For automatic video segmentation/integration with external detectors, see [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).

  -  [Cutie GUI tool](https://raw.githubusercontent.com/hkchengrex/Cutie) used [ProPainter](https://github.com/sczhou/ProPainter) in the video inpainting demo.


- Thanks to [Cutie](https://raw.githubusercontent.com/hkchengrex/Cutie), [RTIM](https://github.com/SamsungLabs/ritm_interactive_segmentation), [XMem++](https://github.com/max810/XMem2), [IVS](https://github.com/seoungwugoh/ivs-demo), [MiVOS](https://github.com/hkchengrex/MiVOS), and [XMem](https://github.com/hkchengrex/XMem) for making this possible.


## License

This project is licensed under the [MIT License ](https://chat.sjtu.edu.cn/c/LICENSE). You are free to use, modify, and distribute the code, provided that the original copyright notice and license are included.

