import numpy as np
import copy
import os
import json

# 这两个字典是输出模板
dict_template = {
    "version": "4.5.6",
    "flags": {},
    "shapes": [],
    "imagePath": "",
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None
}

shape_template = {
    "label": "0",
    "points": [],
    "group_id": None,
    "shape_type": "point",
    "flags": {}
}


# 获取桌面路径
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), 'Desktop')


# def save_feature_point_coordinates(numpy_list, img_name):
#     '''
#     把numpy格式的一维列表存为指定格式，保存路径是桌面的labels文件夹下
#     :param numpy_list: 一维numpy格式列表，正常的长度应该是26*2 52个数值
#     :param img_name: 图像名称
#     :return: 返回是否存储成功
#     '''
#     save_path = os.path.join(get_desktop_path(), "labels")
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     json_path = os.path.join(save_path, "{}.json".format(img_name))
#     output_dict = copy.deepcopy(dict_template)
#     output_dict["imagePath"] = "..\\images\\{}.jpg".format(img_name)
#     np_list = numpy_list.reshape(-1, 2)
#     if not np.size(np_list, 0) == 26:
#         return "特征点数量错误"
#     for index, coordinate in enumerate(np_list):
#         shape = copy.deepcopy(shape_template)
#         shape["label"] = str(index + 1)
#         shape["points"].append(coordinate.tolist())
#         output_dict["shapes"].append(shape)
#     try:
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(output_dict, f)
#         return "存储成功"
#     except:
#         return "存储失败"


def save_feature_point_coordinates(numpy_list, img_name, label_path=None):
    '''
    把numpy格式的一维列表存为指定格式，保存路径是桌面的labels文件夹下
    :param label_path: label文件存储根目录
    :param numpy_list: 一维numpy格式列表，正常的长度应该是26*2 52个数值
    :param img_name: 图像名称
    :return: 返回是否存储成功
    '''
    save_path = os.path.join(get_desktop_path(), "labels") if not label_path else label_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path)
    json_path = os.path.join(save_path, "{}.json".format(img_name))
    output_dict = copy.deepcopy(dict_template)
    output_dict["imagePath"] = "..\\images\\{}.jpg".format(img_name)
    np_list = numpy_list.reshape(-1, 2)
    if not np.size(np_list, 0) == 26:
        return "特征点数量错误"
    for index, coordinate in enumerate(np_list):
        shape = copy.deepcopy(shape_template)
        shape["label"] = str(index + 1)
        shape["points"].append(coordinate.tolist())
        output_dict["shapes"].append(shape)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f)
        return "存储成功"
    except:
        return "存储失败"


if __name__ == "__main__":
    # 数据准备
    test_point_coordinate = [float(i) for i in
                             "117.75	508.25	222.75	470.75	505.25	535.75	1495.25	414.5	1435.25	488.25	984	" \
                             "703.25	684	735.75	444	770.75	347.75	737	312.75	732	252.75	694.5	137.75	" \
                             "649.5	127.75	548.25	219	510.75	249	598.25	175.25	622	294	604.5	356.5	568.25	" \
                             "391.5	627	327.75	650.75	481.5	662	535.25	648.25	582.75	684.5	517.75	690.75	" \
                             "359	675.75	429	717".split("\t")]
    test_pic_name = "00100002"
    # 数据准备结束

    print("示例数组格式")
    print(type(test_point_coordinate))
    print(test_point_coordinate)
    print(("------------------"))

    numpy_list = np.array(test_point_coordinate)
    print("函数的输入数组格式")
    print(type(numpy_list))
    print(numpy_list)
    print("----------------")
    result = save_feature_point_coordinates(numpy_list, test_pic_name)
    print(result)







