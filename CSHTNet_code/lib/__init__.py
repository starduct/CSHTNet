# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.pose_resnet
import models.pose_resnet_hmCT
import models.pose_convnext
import models.pose_se_resnet_hmCT
import models.pose_hrnet
import models.pose_hrnet_cross_tensor # 这个不对，没记错的话，是pooling有问题
import models.pose_hrnet_hmCT # 这个才是本尊 
import models.pose_hrnet_hmCT_msca # 当前最优模型，模块最后定下来名字是CSHT
import models.pose_hrnet_CSHT_mlp # 1x1好像没啥用，试着换下mlp
import models.pose_hrnet_hmCT_mscar
import models.pose_hrnet_hmCT_mscar2
import models.pose_hrnet_hmCT_mscar3 # 当前最优结果
import models.pose_hrnet_mscar3 # 当前最优结果，有效性测试
import models.pose_hrnet_hmCT_mscar4
import models.pose_hrnet_hmCT_test # 之前特征图上有错位，测试看看是不是代码问题
import models.pose_hrnet_ca
import models.pose_hrnet_shorter
import models.pose_hrnet_shorter_ca
import models.pose_hrnet_shorter_ca_V2
import models.pose_hrnet_shorter_ca_V3
import models.pose_hrnet_shorter_ca_V4
import models.pose_hrnet_shorter_ca_V42
import models.pose_hrnet_shorter_ca_V5
import models.pose_hrnet_shorter_ca_V6
import models.pose_hrnet_shorter_ca_V7
import models.pose_hrnet_shorter_ca_V8
import models.pose_hrnet_shorter_ca_V9
import models.pose_hrnet_shorter_ca_V72
import models.pose_hrnet_shorter_ca_V82
import models.pose_hrnet_shorter_ca_V83
import models.pose_hrnet_shorter_ca_V84
import models.pose_hrnet_shorter_ca_V85
import models.pose_hrnet_shorter_ca_V86
import models.pose_hrnet_shorter_ca_V87
import models.pose_hrnet_shorter_ca_V88
import models.pose_hrnet_shorter_ca_V89
import models.pose_hrnet_shorter_ca_V89_hmCT
import models.pose_hrnet_shorter_cba_V89_hmCT
import models.pose_hrnet_shorter_V89_hmCT
import models.pose_hrnet_shorter_ca_V89_2
import models.pose_hrnet_shorter_ca_V89_3
import models.pose_hrnet_shorter_ca_V89_35
import models.pose_hrnet_shorter_ca_V89_351
import models.pose_hrnet_shorter_ca_V89_4
import models.pose_hrnet_shorter_SE
import models.pose_hrnet_shorter_CBAM
import models.pose_hrnet_shorter_ca_V810
import models.pose_hrnet_shorter_ca_V811
import models.pose_hrnet_shorter_ca_V812
import models.pose_hrnet_shorter_V86
import models.pose_hrnet_shorter_V88
import models.pose_hrnet_shorter_V89
import models.pose_hrnet_shorter_V89_15
import models.pose_hrnet_shorter_V89_35
import models.pose_hrnet_shorter_V89_35_3d
import models.pose_hrnet_shorter_V89_17
import models.pose_hrnet_shorter_V89_19
import models.pose_hrnet_shorter_V89_111
import models.pose_hrnet_shorter_V89_113
import models.pose_hrnet_ca_V5
import models.pose_hrnet_shorter_ca_V43
import models.pose_hrnet_multi


