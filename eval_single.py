from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from PIL import ExifTags
import time
import os
import cv2
import numpy as np

from featureNet.ImageProcess import *

from featureNet.models import *
from featureNet.config import cfg
from featureNet.config import update_config
from featureNet.core import inference
from featureNet.utils import save_labelme_format

base_path = os.getcwd()
# base_path = 'F:\CX\python-tkiner-image-editor-main'


def parse_args(is_standardize=False):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    if not is_standardize:
        parser.add_argument('--cfg',
                            help='experiment configure files name',
                            default=os.path.join(base_path, 'featureNet/all_points_w32_w384xh288_adam_lr1e-3_sigma_1_hmCT_msca_rot_cl.yaml'),
                            # required=True,
                            type=str)
    else:
        parser.add_argument('--cfg',
                            help='experiment configure files name',
                            default=os.path.join(base_path, 'featureNet/three_points_w32_w256xh192_adam_lr1e-3.yaml'),
                            # required=True,
                            type=str)


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

def PIL_rato(img):
    # 放置PIL读取图像时发生自动旋转
    # 根据exif中的orientation信息将图片转正
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        print(exif)
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
            # expand=True 将图片尺寸也进行相应的变换，不写则size不变
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except:
        pass

    return img

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False


# 解决cv读取路径中有中文的图片问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img


def tensor2map(input_tensor, config):
    # 这个专门用来处理预测结果，即output
    # print("--------------tensor2map---------------")
    # print(input_tensor.shape) # 96, 26, 168, 1
    h,w = config.MODEL.HEATMAP_SIZE # 96 72
    # print("heat map size h is {}, w is {}".format(h, w))
    x_tensor, y_tensor = torch.split(input_tensor, [w, h], dim=2)
    # print("x_tensor_shape is {}".format(x_tensor.shape)) # 96 26 72 1
    y_t = y_tensor.permute(0, 1, 3, 2) # 96 26 1 96
    # 测试用
    # z = x_tensor * y_t
    # print("heatmap size is {}".format(z.shape))
    # 测试用
    return x_tensor * y_t


def run(img_path, is_standardize=False):
    # 修改一下这里，使得可以兼容运行三点校正网络
    args = parse_args(is_standardize)
    update_config(cfg, args)
    model_name = 'featureNet\\result\\model_best_CSHT_Net.pth' if not is_standardize else 'featureNet\\result' \
                                                                                 '\\model_best_new3_points.pth '
    model_dir = os.path.join(base_path, model_name)
    # img_dir = img_path
    # image = Image.open(img_dir).convert('RGB')
    # image= image.resize((1088, 740), Image.ANTIALIAS)

    model = eval('featureNet.models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    # model = pose_hrnet.get_pose_net(cfg, is_train=False)
    print('开始加载模型')
    # model.load_state_dict(torch.load(model_dir), strict=False)
    model.load_state_dict(torch.load(model_dir, map_location='cpu'), strict=False)
    model.eval()
    print('模型加载完成')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_data = transforms.Compose([
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        normalize,
    ])
    img = img_path

    if len(img.split('.')) == 1:
        return 0 # 这里应该有一个没图像的时候的退出机制
    
    elif img.split('.')[1] in ['jpg', 'png', 'tif']:
        imgPath = os.path.join(img_path, img)
        if is_chinese(imgPath):
            image = cv_imread(imgPath)
        else:
            image = cv2.imread(imgPath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # image.save(imgPath)
        width, height = image.size[0], image.size[1]

        fin = open(imgPath, 'rb')
        sql_image = fin.read()
        fin.close()

        with torch.no_grad():
            trans_image = transform_data(image)
            trans_image = torch.unsqueeze(trans_image, 0)
            print('输入图像尺寸：', trans_image.shape)
            input_var = torch.autograd.Variable(trans_image)
            start = time.time()
            print('------开始计算------')
            output = model(input_var)
            end = time.time()
            print('------结束计算------')
            print('detection time {}.'.format(end - start))
            if output.shape[-1] == 1:
                # 这里是使用CSHT-Net里的输出，需要后处理一下
                print('original output shape is {}.'.format(output.numpy().shape))
                output = tensor2map(output, cfg)
                print('after tensor2map output shape is {}.'.format(output.numpy().shape))
            preds, _ = inference.get_max_preds(output.numpy())
            preds[:, :, 0] *= (image.size[0] / output.numpy().shape[3])
            preds[:, :, 1] *= (image.size[1] / output.numpy().shape[2])
            print(preds[0], preds[0].shape)
            print('--------------------------')

            if is_standardize:
                tmp = np.zeros((1,26,2))
                tmp[0][0] = preds[0][0]
                tmp[0][3] = preds[0][1]
                tmp[0][4] = preds[0][2]
                preds = tmp
                # print(tmp,tmp.shape)
                # 将3点结果补全到26点
            return preds[0]
            points = preds.reshape(-1)
            print('output shape is {}.'.format(points.shape))
            return points
            # label_path = os.path.join(os.path.split(img_path)[0], 'labels_3points')
            # # print(label_path)
            # label_name = img.split('.')[0]
            # # print(points, label_name)
            # # 这里需要存储特征点
            # result = save_labelme_format.save_feature_point_coordinates(numpy_list=points,
            #                                                     img_name=label_name,
            #                                                     label_path=label_path
            #                                                     )
            # print(result)





if __name__ == '__main__':
    # run(img_path=r'F:\CX\autoZebrafishLabel\images', is_standardize=True)


    # run(img_path=r'F:\CX\HRNet\HRNet-Facial-Landmark-Detection-master\data\zebrafish\22thirteenth_batch_pic\原始文件\segmentation\seg\images', is_standardize=True)

    result = run(img_path=r'E:\zebrafish_extend\train_images\1.png', is_standardize=False)
    print(result[0])
    #     [   0. 1050.   60.   15. 1600.  225. 1900.  240. 1900.  300.  360.  990.
    #   480.  360. 1820.    0.  360.  345.  260. 1035.  140. 1035.    0. 1050.
    #   280.  540.    0.   45.    0.  120.    0. 1005.   60.   75.  180.   75.
    #   180.  180.   60.  930. 1600.  210. 1700.  285. 1760.  330.  700.  495.
    #   200.  870.  300.  465.]