import tkinter as tk
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageDraw
from tkinter import ttk
import eval_single
import os
import json

root = tk.Tk()
root.geometry("1000x500")
root.title("CSHT-Net demo")
root.config(bg="white")

# pen_color = "black"
# pen_size = 5


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 400



def get_gt_points(file_path):
    """
    这个函数需要根据图片路径，寻找同路径下关键点标注信息
    """
    
    if file_path.endswith(".jpg") or file_path.endswith(".png"):
        file_path = os.path.dirname(file_path) + os.path.splitext(os.path.basename(file_path))[0] + ".json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        data = json.load(f)

        return data


def generate_points_image(img, points, width, height, point_width=10):
    """
    这个函数需要根据关键点信息，生成关键点标注图，并返回标注图的image对象，用于显示在画布上。
    """
    print("----------------")
    point_width = max(point_width, img.size[0] // 100, img.size[1] // 100)
    print("Generating image with points")
    draw = ImageDraw.Draw(img)
    for point in points:
        # print(point)
        draw.ellipse((point[0] - point_width / 2, point[1] - point_width / 2, point[0] + point_width / 2, point[1] + point_width / 2), fill="red")
    output = img.resize((width, height))
    img.save("output.jpg")
    return output


# 定义一个函数add_image，用于添加图片
def add_image():
    """
    这里的图像已经被resize过了，直接用于预测可能会导致问题，最好是模型预测的时候重新读图，直接生成出预测图和GT图
    """
    # 定义一个全局变量file_path
    global FILE_PATH
    global OR_IMAGE 
    # global GT_IMAGE
    global PR_IMAGE

    # 使用filedialog模块的askopenfilename函数，获取文件路径，初始路径为当前工作目录下的images文件夹
    FILE_PATH = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "images"))
    # 使用Image模块的open函数，打开文件路径
    image = Image.open(FILE_PATH)
    # 定义图片的宽度和高度
    width, height = CANVAS_WIDTH, CANVAS_HEIGHT
    
    # 使用Image模块的resize函数，将图片的宽度和高度设置为CANVAS_WIDTH和CANVAS_HEIGHT，并使用Image模块的ANTIALIAS参数
    original_image = image.resize((width, height), Image.ANTIALIAS)
    OR_IMAGE = original_image

    # gt_points = get_gt_points(FILE_PATH)
    # GT_IMAGE = generate_points_image(image, gt_points, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

    pr_points = eval_single.run(FILE_PATH, is_standardize=False)
    PR_IMAGE = generate_points_image(image, pr_points, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)


    # 设置画布的宽度和高度
    canvas.config(width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    # 使用ImageTk模块的PhotoImage函数，将图片转换为PhotoImage格式
    original_image = ImageTk.PhotoImage(original_image)
    # 将图片赋值给canvas的image属性
    canvas.image = original_image
    # 使用canvas的create_image函数，在画布上添加图片，并设置图片的位置
    canvas.create_image(0, 0, image=original_image, anchor="nw")

    filter_combobox.set("Original Image")


def apply_filter(filter):
    if filter == "Original Image":
        image = OR_IMAGE
    elif filter == "Predicted result":
        image = PR_IMAGE
    # elif filter == "Ground Truth":
    #     image = GT_IMAGE
        # 这里得判断下
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")



# 从这里开始应该是界面部分
# 创建一个名为left_frame的tk.Frame，宽度为200，高度为600，背景色为白色
left_frame = tk.Frame(root, width=200, height=600, bg="white")
# 将left_frame放置在root的左侧，填充方式为y
left_frame.pack(side="left", fill="y")

# 创建一个名为canvas的tk.Canvas，宽度为800，高度为600
canvas = tk.Canvas(root, width=800, height=600)
# 将canvas放置在root中
canvas.pack()

# 创建一个名为image_button的tk.Button，文本为"Add Image"，命令为add_image，背景色为白色
image_button = tk.Button(left_frame, text="Add Image",
                         command=add_image, bg="white")
# 将image_button放置在left_frame中，填充方式为pady=15
image_button.pack(pady=15)


# 创建标签，显示过滤器选项
filter_label = tk.Label(left_frame, text="Select Filter", bg="white")
filter_label.pack()
# 创建下拉框，显示过滤器选项
filter_combobox = ttk.Combobox(left_frame, values=["Original Image",  "Predicted result"])
filter_combobox.pack()
filter_combobox.set("Original Image")

# 绑定下拉框，当选择过滤器选项时，调用apply_filter函数
filter_combobox.bind("<<ComboboxSelected>>",
                     lambda event: apply_filter(filter_combobox.get()))

filter_combobox.bind("<<ComboboxSelected>>",
                     lambda event: apply_filter(filter_combobox.get()))


# canvas.bind("<B1-Motion>", draw)

root.mainloop()