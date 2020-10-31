"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
 Date: 2018/5/26
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches


"""
该类和main.py相同，构造相同的网络架构，以便使用存储好的训练参数
"""
class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size) #输入为1通道，输出为50通道 7x7卷积核
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #数据相同，形状不同

        h  = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1))) # return 50
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling  100
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h)) # 进行一次线性变换得到   800
        h  = self.dropout(h) # 进行 0.5的抽取
        h  = F.relu(self.fc2(h)) # 再进行一次线性变换，  800
        q  = self.fc3(h) # 再进行一次线性变换得到1个得分值
        return q


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test demo')
    parser.add_argument("--im_path", type=str, default='data/000000.png',
                        help="image path")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file)) # 网络结构加载训练好的参数

    im = Image.open(args.im_path).convert('L') #  由3通道的rgb变为一通道的灰度图，形状矩阵均为（h,w） 像素点由三维数据变为1维
    patches = NonOverlappingCropPatches(im, 32, 32) # 对补块进行初始特征学习和优化

    model.eval() # Sets the m?odule in evaluat?ion mode 将训练模式改为评估模式
    # with torch.no_grad(): # 不训练梯度
    patch_scores = model(torch.stack(patches).to(device)) # 将paches依次传输
    # patches中都i是tensor，不过由tuple包着
    # 将patches中的patch在新的维度上链接起来,
    # 在这里只是一个tuple，转化为tensor，里面仍然是tensor
    # 依次进行model
    print(patch_scores.mean().item())
