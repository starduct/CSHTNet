# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .zebrafish import ZebrafishDataset as zebrafish
from .zebrafishallpoints import ZebrafishAllPointsDataset as zebrafishallpoints
from .zebrafishallpoints_tensor import ZebrafishAllPointsDatasetCrossTensor as zebrafishallpointscrosstensor
