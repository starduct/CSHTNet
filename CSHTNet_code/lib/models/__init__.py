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
import models.pose_se_resnet_hmCT
import models.pose_hrnet
import models.pose_hrnet_cross_tensor  # 这个不对，没记错的话，是pooling有问题
import models.pose_hrnet_hmCT  # 这个才是本尊
import models.pose_hrnet_hmCT_msca  # 当前最优模型，模块最后定下来名字是CSHT
import models.pose_hrnet_hmCT_mscar
import models.pose_hrnet_hmCT_mscar2
import models.pose_hrnet_hmCT_mscar3  # 当前最优结果
import models.pose_hrnet_hmCT_mscar4
