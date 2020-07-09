import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__() 
        self.kernel_num = kernel_num # 核数量
        self.kernel_mul = kernel_mul # 相邻两个核的倍数关系
        self.fix_sigma = None # 高斯核的sigma值
        self.kernel_type = kernel_type # 核类型

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None): # 高斯核，source.shape:(n, len(x)); target.shape:(m, len(y))
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))) # 将整个total复制(n+m)份
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))) # 将total每一行都复制(n+m)行，即每条数据都扩展(n+m)份
        L2_distance = ((total0-total1)**2).sum(2) # 沿着dim=2 相加
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples) # 方差（计算结果与标准方差公式相同）
        bandwidth /= kernel_mul ** (kernel_num // 2) # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) # 高斯核数学表达式
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) # 最终核矩阵

    def linear_mmd2(self, f_of_X, f_of_Y): # 线性mmd损失
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0) # source和target数据特征的均值差
        loss = delta.dot(delta.T) # 特征均值差的平方和
        return loss 

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0]) # 一般默认源域和目标域的batchsize相同
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma) # 根据公式将核矩阵分为4个部分
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX) # 因为一般源域与目标域输入数据数量相同(即batchsize相同)，所以L矩阵不加入计算
            torch.cuda.empty_cache()
            return loss
