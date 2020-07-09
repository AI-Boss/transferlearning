import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = [] # # 存放每个epoch的损失值

def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset) # 目标域测试集样本个数
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1] # 计算s_output每行最大值并取其列索引，即每条数据的概率(max括号里的1为dim=1,max()返回值有两个，分别为值和索引的tensor，[1]为取索引tnesor)
            correct += torch.sum(pred == target)

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader) # 源域数据的批次个数？
    len_target_loader = len(target_train_loader)
    for e in range(CFG['epoch']):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train() # model.train()用于训练，model.eval()用于测试，区别在于对BN和Dropout的处理方式不同
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader) # source_loader和target_train_loader均为iterable，使用iter函数将其变为iterator
        n_batch = min(len_source_loader, len_target_loader) # 批次个数
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next() # 对应iter操作，取一批数据
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                    DEVICE), label_source.to(DEVICE) # 把数据copy一份在DEVICE指定的GPU或CPU上运行
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad() # 把loss关于weight的梯度置零，简单的说就是进来一个batch，更新并计算一次梯度，以防梯度累加
            label_source_pred, transfer_loss = model(data_source, data_target) # 返回源域预测标签和迁移损失(源码在model.py?)
            clf_loss = criterion(label_source_pred, label_source) # 计算源域的分类损失
            loss = clf_loss + CFG['lambda'] * transfer_loss # 计算总损失Loss(total)=Loss(clf)+λLoss(transfer_loss)
            loss.backward() # 反向传播求解梯度
            optimizer.step() # 更新权重参数
            train_loss_clf.update(clf_loss.item()) # 更新损失，item()为取出元素张量里面的元素值
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0: # 每10个epoch打印一次......
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg]) # 存放每个epoch的损失值
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f') # 保存为txt文件，delimiter为分隔符，fmt为指定保存的文件格式(这里为保存小数点后6位)
        # Test
        test(model, target_test_loader)
    

def load_data(src, tar, root_dir):
    folder_src = root_dir + src + '/images/'
    folder_tar = root_dir + tar + '/images/'
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)

    source_name = "amazon"
    target_name = "webcam"

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, CFG['data_path'])

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)
