"""
PyTorch 1.3 implementation of the following paper:
Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=logger --port=6006
    ```
    Run the main.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0
    ```
 Implemented by Dingquan Li
 Email: dingquanli@pku.edu.cn
 Date: 2019/11/8
"""

from argparse import ArgumentParser
import os
import numpy as np
import random
from scipy import stats
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from IQADataset import IQADataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from tensorboardX import SummaryWriter
import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""""
这里的y 是上面patchs中返回的y = (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]])
"""
def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y[0])  # Function that takes the mean element-wise absolute value difference\
                                    # 取平均元素绝对值差的 函数


"""
这是定义进行损失函数的系数
默认将“y”pred“、”y“作为“update”函数的输入。当有多个变量时再进行参数output_transform设置
"""
class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self): # 在__init__函数中调用  将度量重置为其初始状态。默认情况下，在每个历元开始时调用此函数。
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output): # 在__init__函数中调用 更新然后添加到列表中，默认情况下，每个批次调用一次

        y_pred, y = output  # 默认将“y”pred“、”y“作为“update”函数的输入 这里的y代表的是主观评价分
        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        self._y_pred.append(torch.mean(y_pred).item())

    def compute(self):  # 自动调用该函数
        sq = np.reshape(np.asarray(self._y), (-1,))  # 将h x W转换为一列 （-1，）等价于 -1,一维数组
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0] # 计算斯皮尔曼等级相关系数 在axis=0上
        krocc = stats.stats.kendalltau(sq, q)[0] # 计算肯德尔等级相关系数 预测单调性
        plcc = stats.pearsonr(sq, q)[0]  # 计算皮尔逊线性相关系数 预测准确性
        rmse = np.sqrt(((sq - q) ** 2).mean()) # 计算预测的一致性 由均方根误差
        mae = np.abs((sq - q)).mean() # 计算均方误差的绝对值
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc, krocc, plcc, rmse, mae, outlier_ratio
"""
这是cnn网络架构
"""
class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size) # 由1通道变为50通道
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes) #
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x): # 调用__call__时，自动调用forward函数
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  ##数据相同，形状不同

        h  = self.conv1(x)   # 深度 50

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))# return 50
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1))) # 由于relu函数 不能负值，所以这样--为正
        h  = torch.cat((h1, h2), 1)  # max-min pooling，maxpooling 和 minpooling 在指定维度连接起来 返回100
        h  = h.squeeze(3).squeeze(2) # 降维度，减少无用信息

        h = F.relu(self.fc1(h))  # 进行一次线性变换得到800
        h = self.dropout(h)  # 进行 0.5的抽取
        h = F.relu(self.fc2(h))  # 再进行一次线性变换，800
        q = self.fc3(h)  # 再进行一次线性变换得到1个得分值

        return q


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4)  #数据加载器，结合了数据集和取样器，并且
    # 可以提供多个线程处理数据集。在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直至
    # 把所有的数据都抛出。就是做一个数据的初始化

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir, trained_model_file, save_result_file, disable_gpu=False):
    if config['test_ratio']: # 为0.2 非0则真
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=config['kernel_size'],
                      n_kers=config['n_kernels'],
                      n1_nodes=config['n1_nodes'],
                      n2_nodes=config['n2_nodes'])
    writer = SummaryWriter(log_dir=log_dir) # 日志会被存入log_dir文件
    model = model.to(device)
    print(model)
    # if multi_gpu and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # 优化方法  权重衰减为0
    # (defaul) beats用于计算梯度以及梯度平方的运行平均值的系数
    # betas = （beta1，beta2）
    # beta1：一阶矩估计的指数衰减率（如 0.9）。
    # beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
    global best_criterion # 系数阈值进行比较
    best_criterion = -1  # SROCC>=-1
    # 对loss进行反向传播
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)  # Factory function for creating a trainer for supervised models.
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()}, # metrics是一个字典来存储需要度量的指标。
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED) # 当一个 iteration 结束时, 会触发此事件
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration) # 日志中增加了该参数，并可被可视化

    @trainer.on(Events.EPOCH_COMPLETED) # 当一个 epoch 结束时, 会触发此事件，该程序有500次
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion: # 将值不断增加的参数，添加到日志中
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            torch.save(model.state_dict(), trained_model_file) # 将整个模型的字典中参数存入到这个trained_model_file文件名
            """
            # 保存训练的模型：
             torch.save(model.state_dict(),"Linear.pth")
           # 加载预训练模型的参数
           model.load_state_dict(torch.load("Linear.pth"))
            """


    @trainer.on(Events.EPOCH_COMPLETED) #当一个 epoch 结束时, 会触发此事件
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED) # 当全部运行完后触发该事件 run is completed
    def final_testing_results(engine):
        if config["test_ratio"]:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            global best_epoch
            print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            np.save(save_result_file, (SROCC, KROCC, PLCC, RMSE, MAE, OR))  # 将测试结果中的系数保存在save_result_file文件里

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs) # 直接运行程序

    writer.close()
"""
pytorch中高级API  ignite 训练模型 的步骤
创建模型, 创建 Dataloader
创建 trainer
创建 evaluator
为一些事件注册函数, @trainer.on()
trainer.run()
"""

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA') # 这就是一个简单的一个示例说明，告诉读者各个参数的意义，方便理解
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')
    parser.add_argument('--model', default='CNNIQA', type=str,
                        help='model name (default: CNNIQA)')
    # parser.add_argument('--resume', default=None, type=str,
    #                     help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    # parser.add_argument('--multi_gpu', action='store_true',
    #                     help='flag whether to use multiple GPUs')

    args = parser.parse_args() # 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。

    torch.manual_seed(args.seed)  #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.backends.cudnn.deterministic = True # 由于含随机种子，提升计算速度，避免波动
    torch.backends.cudnn.benchmark = False # 由于含随机种子，提升计算速度，避免波动
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True #为了帮助识别你的代码中通过广播介绍向后不兼容性可能存在的情况下，
    # 可以设定torch.utils.backcompat.broadcast_warning.enabled到True，这将产生在这种情况下，一个python警告。

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader) # 打开文件 将解析流中的第一个YAML文档转换为python对象,字典嵌套
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    """
    以下两行代码就是将config参数进行更新，
    """
    config.update(config[args.database]) # 将config[args.database]中字典内容进行添加到config中，增加了（减少一层嵌套的字典）
    config.update(config[args.model]) # 将config[args.model]中字典内容进行添加到config中，增加了（减少一层嵌套的字典）
    # 创造了一个日志路径
    log_dir = '{}/EXP{}-{}-{}-lr={}-{}'.format(args.log_dir, args.exp_id, args.database, args.model, args.lr, 
                                                  datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    # 创造了一个目录
    ensure_dir('checkpoints')
    # 在上个目录中 创造一个保存训练模型参数的文件名
    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)
    ensure_dir('results')
    # 创造一个保存测试结果中系数的文件名
    save_result_file = 'results/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        log_dir, trained_model_file, save_result_file, args.disable_gpu)
