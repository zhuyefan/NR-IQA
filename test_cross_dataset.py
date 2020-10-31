"""
Test Cross Dataset
    For help
    ```bash
    python test_cross_dataset.py --help
    ```
 Date: 2018/5/26
"""
import pandas as pd
from argparse import ArgumentParser
import torch
from torch import nn
import cv2
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches
import numpy as np
import h5py, os


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h  = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        return q


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test on the whole cross dataset')
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="dataset dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")
    parser.add_argument("--save_path", type=str, default='data/scores_pepper.xlsx',
                        help="save path (default: score)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file))

    # Info = h5py.File(args.names_info)
    # im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        # [::2].decode() for i in range(len(Info['im_names'][0, :]))]
    File_name = pd.read_excel("data/NR-IQA.xlsx")


# 对普通的540p进行得分
    im_names = File_name["images540p"]
    model.eval()
    scores = []
    for i in range(len(im_names)):  # pandas中第一行为clounms列名
        im = Image.open((im_names[i])).convert('L')
        patches = NonOverlappingCropPatches(im, 32, 32)
        patch_scores = model(torch.stack(patches).to(device))
        score = patch_scores.mean().item()
        print(score)
        scores.append(score)
    save540 = pd.Series(scores)
    File_name["scores540p"] = save540 # 将其进行新的一列添加
    # File_name.to_excel(args.save_path)  # 将其进行保存

#  进行1080p的图片进行得分
    im_names = File_name["images1080p"]
    # model.eval()
    scores = []
    for i in range(len(im_names)):  # pandas中第一行为clounms列名
        im = Image.open((im_names[i])).convert('L') # 将图片转换为PIL.Image.Image格式加载
        patches = NonOverlappingCropPatches(im, 32, 32)
        patch_scores = model(torch.stack(patches).to(device))
        score = patch_scores.mean().item()
        print(score)
        scores.append(score)
    save1080 = pd.Series(scores)
    File_name["scores1080p"] = save1080  # 将其进行新的一列添加

#  进行540p加盐噪声后的得分 比例为0.001
    im_names = File_name["images540p"]
    model.eval()
    scores = []
    for i in range(len(im_names)):  # pandas中第一行为clounms列名
        # 进行噪声处理
        image = Image.open((im_names[i])).convert('L') # （960，540）
        image = np.array(image)
        # peppers = np.array(im) # 将image对象转化为array，unit8格式
        # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
        row, column = image.shape
        noise_salt = np.random.randint(0, 256, (row,column))
        # noise_pepper = np.random.randint(0, 256, (row, column))
        rand = 0.001
        noise_salt = np.where(noise_salt < rand * 256, 0, 255)
        # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
        image = image.astype("float")
        noise_salt = noise_salt.astype("float")
        # noise_pepper.astype("float")
        image = image + noise_salt
        # pepper = peppers + noise_pepper
        image = np.where(image > 255, 0, image)
        image = image.astype("uint8")
        # pepper = np.where(pepper < 0, 0, pepper)

        # 将nump.ndarray格式转化为PIL.Image.Image格式
        image = Image.fromarray(image)
        # im = img_pil.convert('L')
        patches = NonOverlappingCropPatches(image, 32, 32)
        patch_scores = model(torch.stack(patches).to(device))
        score = patch_scores.mean().item()
        print(score)
        scores.append(score)
    save540 = pd.Series(scores)

    File_name["salt540p(0.001)"] = save540  # 将其进行新的一列添加

#  进行540p加盐噪声后的得分 比例为0.1
    im_names = File_name["images540p"]
    model.eval()
    scores = []
    for i in range(len(im_names)):  # pandas中第一行为clounms列名
        # 进行噪声处理
        image = Image.open((im_names[i])).convert('L')  # （960，540）
        image = np.array(image)
        # peppers = np.array(im) # 将image对象转化为array，unit8格式
        # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
        row, column = image.shape
        noise_salt = np.random.randint(0, 256, (row, column))
        # noise_pepper = np.random.randint(0, 256, (row, column))
        rand = 0.1
        noise_salt = np.where(noise_salt < rand * 256, 0, 255)
        # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
        image = image.astype("float")
        noise_salt = noise_salt.astype("float")
        # noise_pepper.astype("float")
        image = image + noise_salt
        # pepper = peppers + noise_pepper
        image = np.where(image > 255, 0, image)
        image = image.astype("uint8")
        # pepper = np.where(pepper < 0, 0, pepper)

        # 将nump.ndarray格式转化为PIL.Image.Image格式
        image = Image.fromarray(image)
        # im = img_pil.convert('L')
        patches = NonOverlappingCropPatches(image, 32, 32)
        patch_scores = model(torch.stack(patches).to(device))
        score = patch_scores.mean().item()
        print(score)
        scores.append(score)
    save540 = pd.Series(scores)

    File_name["salt540p(0.1)"] = save540  # 将其进行新的一列添加



#  进行540p加盐噪声后的得分 比例为0.5

    im_names = File_name["images540p"]
    model.eval()
    scores = []
    for i in range(len(im_names)):  # pandas中第一行为clounms列名
        # 进行噪声处理
        image = Image.open((im_names[i])).convert('L')  # （960，540）
        image = np.array(image)
        # peppers = np.array(im) # 将image对象转化为array，unit8格式
        # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
        row, column = image.shape
        noise_salt = np.random.randint(0, 256, (row, column))
        # noise_pepper = np.random.randint(0, 256, (row, column))
        rand = 0.5
        noise_salt = np.where(noise_salt < rand * 256,0,255)
        # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
        image = image.astype("float")
        noise_salt = noise_salt.astype("float")
        # noise_pepper.astype("float")
        image = image + noise_salt
        # pepper = peppers + noise_pepper
        image = np.where(image > 255,0, image)
        image = image.astype("uint8")
        # pepper = np.where(pepper < 0, 0, pepper)

        # 将nump.ndarray格式转化为PIL.Image.Image格式
        image = Image.fromarray(image)
        # im = img_pil.convert('L')
        patches = NonOverlappingCropPatches(image, 32, 32)
        patch_scores = model(torch.stack(patches).to(device))
        score = patch_scores.mean().item()
        print(score)
        scores.append(score)
    save540 = pd.Series(scores)

    File_name["salt540p(0.5)"] = save540  # 将其进行新的一列添加
    #
    # #  进行540p加盐噪声后的得分 比例为0.001
    # im_names = File_name["images1080p"]
    # model.eval()
    # scores = []
    # for i in range(len(im_names)):  # pandas中第一行为clounms列名
    #     # 进行噪声处理
    #     image = Image.open((im_names[i])).convert('L')  # （960，540）
    #     image = np.array(image)
    #     # peppers = np.array(im) # 将image对象转化为array，unit8格式
    #     # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
    #     row, column = image.shape
    #     noise_salt = np.random.randint(0, 256, (row, column))
    #     # noise_pepper = np.random.randint(0, 256, (row, column))
    #     rand = 0.001
    #     noise_salt = np.where(noise_salt < rand * 256, 0, 255)
    #     # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
    #     image = image.astype("float")
    #     noise_salt = noise_salt.astype("float")
    #     # noise_pepper.astype("float")
    #     image = image + noise_salt
    #     # pepper = peppers + noise_pepper
    #     image = np.where(image > 255, 0, image)
    #     image = image.astype("uint8")
    #     # pepper = np.where(pepper < 0, 0, pepper)
    #
    #     # 将nump.ndarray格式转化为PIL.Image.Image格式
    #     image = Image.fromarray(image)
    #     # im = img_pil.convert('L')
    #     patches = NonOverlappingCropPatches(image, 32, 32)
    #     patch_scores = model(torch.stack(patches).to(device))
    #     score = patch_scores.mean().item()
    #     print(score)
    #     scores.append(score)
    # save540 = pd.Series(scores)
    #
    # File_name["salt1080p(0.001)"] = save540  # 将其进行新的一列添加
    #
    # #  进行540p加盐噪声后的得分 比例为0.1
    # im_names = File_name["images1080p"]
    # model.eval()
    # scores = []
    # for i in range(len(im_names)):  # pandas中第一行为clounms列名
    #     # 进行噪声处理
    #     image = Image.open((im_names[i])).convert('L')  # （960，540）
    #     image = np.array(image)
    #     # peppers = np.array(im) # 将image对象转化为array，unit8格式
    #     # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
    #     row, column = image.shape
    #     noise_salt = np.random.randint(0, 256, (row, column))
    #     # noise_pepper = np.random.randint(0, 256, (row, column))
    #     rand = 0.1
    #     noise_salt = np.where(noise_salt < rand * 256, 0, 255)
    #     # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
    #     image = image.astype("float")
    #     noise_salt = noise_salt.astype("float")
    #     # noise_pepper.astype("float")
    #     image = image + noise_salt
    #     # pepper = peppers + noise_pepper
    #     image = np.where(image > 255, 0, image)
    #     image = image.astype("uint8")
    #     # pepper = np.where(pepper < 0, 0, pepper)
    #
    #     # 将nump.ndarray格式转化为PIL.Image.Image格式
    #     image = Image.fromarray(image)
    #     # im = img_pil.convert('L')
    #     patches = NonOverlappingCropPatches(image, 32, 32)
    #     patch_scores = model(torch.stack(patches).to(device))
    #     score = patch_scores.mean().item()
    #     print(score)
    #     scores.append(score)
    # save540 = pd.Series(scores)
    #
    # File_name["salt1080p(0.1)"] = save540  # 将其进行新的一列添加
    #
    # #  进行540p加盐噪声后的得分 比例为0.5
    #
    # im_names = File_name["images1080p"]
    # model.eval()
    # scores = []
    # for i in range(len(im_names)):  # pandas中第一行为clounms列名
    #     # 进行噪声处理
    #     image = Image.open((im_names[i])).convert('L')  # （960，540）
    #     image = np.array(image)
    #     # peppers = np.array(im) # 将image对象转化为array，unit8格式
    #     # image = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
    #     row, column = image.shape
    #     noise_salt = np.random.randint(0, 256, (row, column))
    #     # noise_pepper = np.random.randint(0, 256, (row, column))
    #     rand = 0.5
    #     noise_salt = np.where(noise_salt < rand * 256, 255, 0)
    #     # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
    #     image = image.astype("float")
    #     noise_salt = noise_salt.astype("float")
    #     # noise_pepper.astype("float")
    #     image = image + noise_salt
    #     # pepper = peppers + noise_pepper
    #     image = np.where(image > 255, 255, image)
    #     image = image.astype("uint8")
    #     # pepper = np.where(pepper < 0, 0, pepper)
    #
    #     # 将nump.ndarray格式转化为PIL.Image.Image格式
    #     image = Image.fromarray(image)
    #     # im = img_pil.convert('L')
    #     patches = NonOverlappingCropPatches(image, 32, 32)
    #     patch_scores = model(torch.stack(patches).to(device))
    #     score = patch_scores.mean().item()
    #     print(score)
    #     scores.append(score)
    # save540 = pd.Series(scores)
    #
    # File_name["salt1080p(0.5)"] = save540  # 将其进行新的一列添加

    # #  进行540p加盐噪声后的得分 比例为0.1
#     im_names = File_name["images540p"]
#     model.eval()
#     scores = []
#     for i in range(len(im_names)):  # pandas中第一行为clounms列名
#         # 进行噪声处理
#         # peppers = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载  cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1
#         # # 。cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。cv2.IMREAD_UNCHANGED：包括alpha，可以直接写-1
#         im = Image.open((im_names[i])).convert('L')
#         peppers = np.array(im)  # 将image对象转化为array，unit8格式
#         row, column = peppers.shape
#         noise_salt = np.random.randint(0, 256, (row, column))
#         print(noise_salt.shape) # 960
#         # noise_pepper = np.random.randint(0, 256, (row, column))
#         rand = 0.1
#         noise_salt = np.where(noise_salt < rand * 256, 255, 0)
#         # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
#         peppers.astype("float")
#         noise_salt.astype("float")
#         # noise_pepper.astype("float")
#         salt = peppers + noise_salt
#         # pepper = peppers + noise_pepper
#         salt = np.where(salt > 255, 255, salt)
#         # pepper = np.where(pepper < 0, 0, pepper)
#
#         # 将nump.ndarray格式转化为PIL.Image.Image格式
#         img_pil = Image.fromarray(salt)
#         im = img_pil.convert('L')
#         patches = NonOverlappingCropPatches(img_pil, 32, 32)
#         patch_scores = model(torch.stack(patches).to(device))
#         score = patch_scores.mean().item()
#         print(score)
#         scores.append(score)
#     save540 = pd.Series(scores)
#     File_name["salt540p(0.1)"] = save540  # 将其进行新的一列添加

# #  进行540p加盐噪声后的得分 比例为0.2
#     im_names = File_name["images540p"]
#     model.eval()
#     scores = []
#     for i in range(len(im_names)):  # pandas中第一行为clounms列名
#         # 进行噪声处理
#         peppers = cv2.imread(im_names[i], 0) # 将图片转化为numpy.ndarray格式加载
#         row, column = peppers.shape
#         noise_salt = np.random.randint(0, 256, (row, column))
#         # noise_pepper = np.random.randint(0, 256, (row, column))
#         rand = 0.2
#         noise_salt = np.where(noise_salt < rand * 256, 255, 0)
#         # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
#         peppers.astype("float")
#         noise_salt.astype("float")
#         # noise_pepper.astype("float")
#         salt = peppers + noise_salt
#         # pepper = peppers + noise_pepper
#         salt = np.where(salt > 255, 255, salt)
#         # pepper = np.where(pepper < 0, 0, pepper)
#
#
#         # 将nump.ndarray格式转化为PIL.Image.Image格式
#         img_pil = Image.fromarray(peppers)
#         # im = img_pil.convert('L')
#         patches = NonOverlappingCropPatches(img_pil, 32, 32)
#         patch_scores = model(torch.stack(patches).to(device))
#         score = patch_scores.mean().item()
#         print(score)
#         scores.append(score)
#     save540 = pd.Series(scores)
#
#
#     File_name["salt540p(0.2)"] = save540  # 将其进行新的一列添加
#
#
# #  进行540p加盐噪声后的得分 比例为0.5
#     im_names = File_name["images540p"]
#     model.eval()
#     scores = []
#     for i in range(len(im_names)):  # pandas中第一行为clounms列名
#         # 进行噪声处理
#         peppers = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
#         row, column = peppers.shape
#         noise_salt = np.random.randint(0, 256, (row, column))
#         # noise_pepper = np.random.randint(0, 256, (row, column))
#         rand = 0.5
#         noise_salt = np.where(noise_salt < rand * 256, 255, 0)
#         # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
#         peppers.astype("float")
#         noise_salt.astype("float")
#         # noise_pepper.astype("float")
#         salt = peppers + noise_salt
#         # pepper = peppers + noise_pepper
#         salt = np.where(salt > 255, 255, salt)
#         # pepper = np.where(pepper < 0, 0, pepper)
#
#         # 将nump.ndarray格式转化为PIL.Image.Image格式
#         img_pil = Image.fromarray(peppers)
#         # im = img_pil.convert('L')
#         patches = NonOverlappingCropPatches(img_pil, 32, 32)
#         patch_scores = model(torch.stack(patches).to(device))
#         score = patch_scores.mean().item()
#         print(score)
#         scores.append(score)
#     save540 = pd.Series(scores)
#
#     File_name["salt540p(0.5)"] = save540  # 将其进行新的一列添加
#
# #  进行540p加盐噪声后的得分 比例为0.7
#     im_names = File_name["images540p"]
#     model.eval()
#     scores = []
#     for i in range(len(im_names)):  # pandas中第一行为clounms列名
#         # 进行噪声处理
#         peppers = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
#         row, column = peppers.shape
#         noise_salt = np.random.randint(0, 256, (row, column))
#         # noise_pepper = np.random.randint(0, 256, (row, column))
#         rand = 0.7
#         noise_salt = np.where(noise_salt < rand * 256, 255, 0)
#         # noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
#         peppers.astype("float")
#         noise_salt.astype("float")
#         # noise_pepper.astype("float")
#         salt = peppers + noise_salt
#         # pepper = peppers + noise_pepper
#         salt = np.where(salt > 255, 255, salt)
#         # pepper = np.where(pepper < 0, 0, pepper)
#
#         # 将nump.ndarray格式转化为PIL.Image.Image格式
#         img_pil = Image.fromarray(peppers)
#         # im = img_pil.convert('L')
#         patches = NonOverlappingCropPatches(img_pil, 32, 32)
#         patch_scores = model(torch.stack(patches).to(device))
#         score = patch_scores.mean().item()
#         print(score)
#         scores.append(score)
#     save540 = pd.Series(scores)
#
#     File_name["salt540p(0.7)"] = save540  # 将其进行新的一列添加
#
# # 对540p的椒噪声进行得分 0.2比例
#
#     im_names = File_name["images540p"]
#     model.eval()
#     scores = []
#     for i in range(len(im_names)):  # pandas中第一行为clounms列名
#         # 进行噪声处理
#         peppers = cv2.imread(im_names[i], 0)  # 将图片转化为numpy.ndarray格式加载
#         row, column = peppers.shape
#         # noise_salt = np.random.randint(0, 256, (row, column))
#         noise_pepper = np.random.randint(0, 256, (row, column))
#         rand = 0.1
#         # noise_salt = np.where(noise_salt < rand * 256, -255, 0)
#         noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
#         peppers.astype("float")
#         # noise_salt.astype("float")
#         noise_pepper.astype("float")
#         # salt = peppers + noise_salt
#         pepper = peppers + noise_pepper
#         # salt = np.where(salt > 255, 255, salt)
#         pepper = np.where(pepper < 0, 0, pepper)
#
#         # 将nump.ndarray格式转化为PIL.Image.Image格式
#         img_pil = Image.fromarray(peppers)
#         # im = img_pil.convert('L')
#         patches = NonOverlappingCropPatches(img_pil, 32, 32)
#         patch_scores = model(torch.stack(patches).to(device))
#         score = patch_scores.mean().item()
#         print(score)
#         scores.append(score)
#     save540 = pd.Series(scores)
#
#     File_name["pepper540p(0.2)"] = save540  # 将其进行新的一列添加
#


    File_name.to_excel(args.save_path)  # 将其进行保存

