# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py


def default_loader(path):
    return Image.open(path).convert('L') #


def LocalNormalization(patch, P=3, Q=3, C=1): # 对像素块进行转换操作   ####在论文中体现出来得是特征优化  
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same') # 2维的卷积,返回2维数据，卷积核为3x3 图像的均值滤波
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')# 对平方求卷积 对图像像素平方求均值滤波
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C # 进行求差，再与0比较
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0) # 转换为tensor格式并增加一个维度（通道）
    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):    ####在论文中体现出来得是特征学习
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size))) # 裁剪得到图片的方块
            # 变为（c,w,h），image object对象（w,h）转化为（1，w,h）
            """1. pytorch中tensor和numpy的通道位置不同，numpy的通道在H，W之后，即（H，W，C），用np.shape(object)可以查看；
            而tensor的通道在H和W之前，
            即（C，H，W），用np.shape(object)或者object.shape可以查看； 若numpy没有通道,则tensor通道看为为1
            所以，读取图像后得到numpy数组，要变成tensor进行数据处理，需要transpose操作，改变通道的位置；
            2. 处理针对的是tensor格式，需要通过from_numpy等方法，将输入图像时的numpy格式转换为tensor格式
            """
            patch = LocalNormalization(patch[0].numpy()) # 大小仍然是（1，h，w）的块
            patches = patches + (patch,)  # # 因为patches是tuple，所以要把patch变成tuple尾部追加进去
    return patches


class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader  #  输入图像转化为灰度图
        im_dir = conf['im_dir'] # 它只是单层字典，需要双层字典（但是在main()中config.update已经添加过了
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')  # # 加载了存放主观得分 的文件
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        """
        将数据按照比例划分为0.6的训练，0.2的评估，0.2的测试数据
        """
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1-test_ratio) * len(index)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in self.index]  # 找到单张图片的路径
        
        self.patches = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))

            patches = NonOverlappingCropPatches(im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches  ## 直接将patches这个元组加进去，里面的内容是每个处理过后的像素块
                for i in range(len(patches)):  # 对该图片每个像素块添加该图片的主观得分
                    self.label.append(self.mos[idx])  # mod[]中存储的是'subjective_scores'主观分数
                    self.label_std.append(self.mos_std[idx])  # mos_std[]中存储的是subjective_scoresSTD
                    # 将处理过后的像素块打包，外加一个元组，里面的内容是每个图片的全部像素块
                    # 在进行model时，将整个图像的所有像素块放进去，用train训练过的网络结构的参数进行得分
                    """
                    一定要注意 这种评价得分模式和train中不一样,train是像素块都有主观得分进行比较
                    这个是直接一张图片放进去，得到每个像素块预测分进行求平均值，与主观得分比较    
                    """
            else:
                self.patches = self.patches + (torch.stack(patches), ) # 添加方式不同  直接对一个图片进行一个得分
                # 在tensor中外多加一层tensor,调用时直接调用 得到的y_pred是一个[a,b...]，里面的a,b是每一像素块的得分
                # 该图片的所有像素块进行evaluator或test
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):  # 得到（类型为下面的 __getitem__的return类型）返回数据的大小
        return len(self.patches)

    def __getitem__(self, idx): # 得到该函数的返回的数据
        """"
               在  x,y = _prepare_batch(batch, device=device)
               就是分别的 x = self.patches[idx],存储的是tensor
                         Y = (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]])
               """
        return self.patches[idx], (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]]))
