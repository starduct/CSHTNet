## Introduction

This is an official pytorch implementation of Cross Shaped Heat Tensor.

## Environment

The code is developed using python 3.7 on Centos 7. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA A100 16GB GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start

Our code is based on the [HRNet](https://arxiv.org/abs/1902.09212).  Please refer to [HRNet official code](https://github.com/HRNet/HRNet-Human-Pose-Estimation/tree/master) for installation and dataset preparation.

Specially, please modify the oks_num parameter in the cocoeval.py file, or simply replace it with the cocoeval.py file we have provided.

### Training and Testing

#### Testing on zebrafish dataset

```
python tools/test.py \
    --cfg experiments/zebrafish/hrnet/all_points_w32_w384xh288_adam_lr1e-3_sigma_1_hmCT_msca_rot_cl.yaml \
    TEST.MODEL_FILE path/to/model.pth \
    TEST.USE_GT_BBOX False
```

#### Training on zebrafish dataset

```
python tools/train.py \
    --cfg experiments/zebrafish/hrnet/all_points_w32_w384xh288_adam_lr1e-3_sigma_1_hmCT_msca_rot_cl.yaml \
```

#### Our model on zebrafish dataset

Please download the release file, our model is in

```
CSHTNet_demo/featureNet/result/model_best_CSHT_Net.pth
```
