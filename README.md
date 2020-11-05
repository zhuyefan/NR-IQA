# NR-IQA
  This project use  CNN network, supervised learning, to construct a NR-IQA
 train data is LIVE
 test data is TID2013
1. 首先将LIVE数据集中图片分成32x32像素块，对每个像素块进行特征学习和优化，特征化可
    提高模型效率。(在LIVE数据集上，对于特定于失真的实验，我们对JP2K、JPEG、WN、BLUR和FF这五种失真进行了训练和测试。 对于非失真   		     特定的实验，所有五种失真的图像都被训练和测试在一起，而不提供失真类型。)
2. 将每个像素块进行CNN网络模型训练，网络结构为7x7卷积、max-min池化、2层relu、 
    dropout、线性得分（32×32− 26 × 26 × 50 − 2 × 50 − 800 − 800 − 1 structure）。
3. 将CNN得分与事先准备好的主观得分进行SVR,并将其作为损失函数。
4. 利用损失函数对模型作反向传播，并通过对LIVE数据集进行60%train，20%evaluate，20%test          
    得到训练好的网络结构参数。
5. 将TID2008数据和LIVE数据集进行交叉验证，比较其他IQA模型，得出该模型优于其他模型
   结论。
