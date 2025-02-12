<p align="center">

<p align="center">
  <h1 align="center">Hierarchical Object-Aware Dual-Level Contrastive Learning for Domain Generalized Stereo Matching</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=qyHfMJAAAAAJ&hl=en&oi=ao"><strong>Yikun Miao</strong></a>
    ·
    <strong>Meiqing Wu</strong></a>
    ·
    <a href="https://siewkeilam.github.io/ei-research-group/index.html"><strong>Siew-Kei Lam</strong></a>
    ·
    <a href="https://cs.bit.edu.cn/szdw/jsml/gjjgccrc/lcs_e253eb02bdf246c4a88e1d2499212546/index.htm"><strong>Changsheng Li</strong></a>
    ·
    <strong>Thambipillai Srikanthan</strong>
    <br>
    <br>
    <b>Beijing Institute of Technology &nbsp; | &nbsp; Nanyang Technological University </b>
  </p>
</p>



## Abstract

Stereo matching algorithms that leverage end-to-end convolutional neural networks have recently demonstrated notable advancements in performance. However, a common issue is their susceptibility to domain shifts, hindering their ability in generalizing to diverse, unseen realistic domains. We argue that existing stereo matching networks overlook the importance of extracting semantically and structurally meaningful features. To address this gap, we propose an effective hierarchical object-aware dual-level contrastive learning (HODC) framework for domain generalized stereo matching. Our framework guides the model in extracting features that support semantically and structurally driven matching by segmenting objects at different scales and enhances correspondence between intra- and inter-scale regions from the left feature map to the right using dual-level contrastive loss. HODC can be integrated with existing stereo matching models in the training stage, requiring no modifications to the architecture. Remarkably, using only synthetic datasets for training, HODC achieves state-of-the-art generalization performance with various existing stereo matching network architectures, across multiple realistic datasets.

## Install

Please install [Git](https://git-scm.com/) and [Anaconda](https://www.anaconda.com/download) first, and then execute the following command:

```shell
git clone https://github.com/joshmiao/HODC.git
cd HODC

conda create -n HODC python=3.9
conda activate HODC
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Preparing Training & Evaluation Data

Prepare the training & evaluation data in the following structure in `work_dir/`:

```
work_dir
└── StereoMatching
    ├── annotations
    ├── ETH3D
    │   ├── two_view_testing
    │   └── two_view_training
    ├── KITTI-2012
    │   ├── testing
    │   │   ├── colored_0
    │   │   ├── colored_1
    │   └── training
    │       ├── colored_0
    │       ├── colored_1
    │       ├── disp_noc
    │       ├── disp_occ
    ├── KITTI-2015
    │   ├── testing
    │   │   ├── image_2
    │   │   └── image_3
    │   └── training
    │       ├── disp_noc_0
    │       ├── disp_noc_1
    │       ├── disp_occ_0
    │       ├── disp_occ_1
    │       ├── flow_noc
    │       ├── flow_occ
    │       ├── image_2
    │       ├── image_3
    ├── Middlebury
    │   └── MiddEval3
    │       ├── testF
    │       ├── testH
    │       ├── testQ
    │       ├── trainingF
    │       ├── trainingH
    │       └── trainingQ
    ├── SceneFlow
    │   ├── driving
    │   │   ├── disparity
    │   │   ├── frames_cleanpass
    │   │   ├── frames_finalpass
    │   │   └── object_index
    │   ├── flyingthings3d
    │   │   ├── disparity
    │   │   ├── frames_cleanpass
    │   │   ├── frames_finalpass
    │   │   └── object_index
    │   └── Monkaa
    │       ├── disparity
    │       ├── frames_cleanpass
    │       ├── frames_finalpass
    │       └── object_index
```

You can download the data on the following websites:

- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [KITTI stereo 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
- [Middlebury v3](https://vision.middlebury.edu/stereo/submit3/)
- [ETH3D](https://www.eth3d.net/datasets#low-res-two-view)

## Model Weights

Pretrained weights are availabe on [Google Drive](https://drive.google.com/file/d/1sdLzxWni7DRA2CgHHdtDhqIjUEEGKyrz/view?usp=sharing).

## Training and Evaluation

1. Install the dependencies and prepare the data.

2. Adjust the `root` in `configs/test_psm_cfg.py`, `configs/train_psm_cfg.py` according to your data ROOT.


Run the following command to train and evaluate HODC. Remember to adjust `CUDA_VISIBLE_DEVICES={0,1,2,...}`, `nproc_per_node={number_of_gpus}` and `gpus={number_of_gpus}` according to your training setup.

```
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 6666 --nproc_per_node=1 tools/train.py configs/train_psm_cfg.py --launcher pytorch --validate --gpus 1
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 6666 --nproc_per_node=1 tools/test.py configs/test_psm_cfg.py --launcher pytorch --validate --gpus 1
```

## Citation

If you find HODC useful, please cite using this BibTeX:

```bibtex
@inproceedings{miaohierarchical,
  title={Hierarchical Object-Aware Dual-Level Contrastive Learning for Domain Generalized Stereo Matching},
  author={Miao, Yikun and Wu, Meiqing and Lam, Siew Kei and Li, Changsheng and Srikanthan, Thambipillai},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Acknowledgement

We thank [FCStereo](https://github.com/jiaw-z/FCStereo) and [PSMNet](https://github.com/JiaRenChang/PSMNet) for their open-source code, which our project depends heavily on.

## License

This Repo is released under MIT License.