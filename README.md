# OPAL: Occlusion Pattern Aware Loss for Unsupervised Light Field Disparity Estimation
### [Project Page](https://penghtyx.github.io/OPAL/) | [Paper](https://arxiv.org/pdf/2203.02231.pdf)

This repository is an implementation of paper "OPAL: Occlusion Pattern Aware Loss for Unsupervised Light Field Disparity Estimation". 

Peng Li, Jiayin Zhao, Jingyao Wu, Chao Deng, Yuqi Han, Haoqian Wang, [Tao Yu](http://ytrock.com/)

Tsinghua University


#### Train:
* Set the hyper-parameters in `option.py` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./checkpoints/`.

#### Test:
* Place the input LFs into `./dataset` (see the attached example).
* Run `test.py` to perform inference on each test scene.
* The result files (i.e., `general_52_eslf_depth.tif`) will be saved to `./results/OPENet/latest/scan_LF/`.
