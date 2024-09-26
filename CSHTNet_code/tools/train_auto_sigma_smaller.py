# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from ast import If, arg
import os
import pprint
import shutil
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # philly
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    # 这里如果使用自适应sigma，则在sigma_vector 中生成一个[26,2]的全1向量作为初始值
    if cfg.MODEL.INDEPENDENT_SIGMA:
        print("INDEPENDENT_SIGMA")
        # sigma_vector = torch.ones(cfg.MODEL.NUM_JOINTS, 2) * cfg.MODEL.SIGMA
        # 生成全1的sigma，用于自适应学习

        #         sigma_x_list = [
        #             1,
        #             1.5,
        #             1.5,
        #             1,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1.5,
        #             1,
        #             1.5,
        #             1,
        #             1.5,
        #             1,
        #             1.5,
        #             1,
        #             1.5,
        #             1,
        #             1.5,
        #             1,
        #             1.5,
        #             1.5,
        #             1,
        #         ]
        #         sigma_x_list = np.array(sigma_x_list).reshape(-1,1)
        #         sigma_y_list = [1.5,
        #              1.0,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.5,
        #              1.0,
        #              1.0,
        #              1.5
        #         ]

        #         重新设计了sigma在xy方向的参数
        sigma_x_list = [
            0.75,
            1,
            1,
            0.75,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0.75,
            1,
            0.75,
            1,
            0.75,
            1,
            0.75,
            1,
            0.75,
            1,
            0.75,
            1,
            1,
            0.75,
        ]
        sigma_x_list = np.array(sigma_x_list).reshape(-1, 1)
        sigma_y_list = [
            1,
            1.0,
            1.0,
            1,
            1,
            1,
            1.0,
            1.0,
            1.0,
            1,
            1.0,
            1.0,
            1,
            1.0,
            1,
            1.0,
            1,
            1.0,
            1,
            1.0,
            1,
            1.0,
            1,
            1.0,
            1,
            1,
        ]
        sigma_y_list = np.array(sigma_y_list).reshape(-1, 1)
        sigma_vector = (
            torch.from_numpy(np.hstack((sigma_x_list, sigma_y_list))) * cfg.MODEL.SIGMA
        )
        # sigma_vector = torch.from_numpy(np.hstack((sigma_x_list, sigma_y_list)))
        # 手动改下sigma，看看有没有效果，按照 x, y 的顺序来，这个列表只计x，y与x相反就行
        # 这里在训练的时候把sigma改成2,1.5的就代表改回3，1就代表sigma为2
        # print(sigma_vector)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # 调试位置，确认参数是否输入成功
    # return 0

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, "../lib/models", cfg.MODEL.NAME + ".py"),
        final_output_dir,
    )
    # logger.info(pprint.pformat(model))

    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=tb_log_dir),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transforms.Compose([transforms.ToTensor(), normalize,]),
        sigma_vector,
    )

    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([transforms.ToTensor(), normalize,]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, "checkpoint.pth")

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint["epoch"]
        best_perf = checkpoint["perf"]
        last_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint["epoch"]
            )
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        new_sigma_vector = train(
            cfg,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            final_output_dir,
            tb_log_dir,
            sigma_vector,
        )

        # evaluate on validation set
        perf_indicator = validate(
            cfg,
            valid_loader,
            valid_dataset,
            model,
            criterion,
            final_output_dir,
            tb_log_dir,
            sigma_vector,
        )

        # train 完，在这里修改cfg的值，把新的sigma值弄出来
        sigma_vector = new_sigma_vector

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info("=> saving checkpoint to {}".format(final_output_dir))
        # logger.info('=> sigma_vector is {}'.format(sigma_vector))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": cfg.MODEL.NAME,
                "state_dict": model.state_dict(),
                "best_state_dict": model.module.state_dict(),
                "perf": perf_indicator,
                "optimizer": optimizer.state_dict(),
            },
            best_model,
            final_output_dir,
        )

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth")
    logger.info("=> saving final model state to {}".format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    # writer_dict['writer'].close()


if __name__ == "__main__":
    main()
