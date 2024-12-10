from __future__ import print_function, absolute_import
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
import os.path as osp
import pandas as pd
from trainers import Trainer
from datasets import ActionData
from utils.logging import Logger
from torch.backends import cudnn
from utils.meters import AverageMeter
from torch.utils.data import DataLoader
from utils.lr_scheduler import WarmupMultiStepLR
from models import CognitiveProcessor, PercepProcessor, MHP, LipschitzGraph
from models.cognitive_s import CognitiveProcessor_s

def set_seed(seed):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def train(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))    # 将标准输出重定向到日志文件
    print("==========\nArgs:{}\n==========".format(args))    # 打印传入的参数
    set_seed(args.seed)  # 设置随机种子

    num_frames = args.num_frames    # 帧数设置为50
    stride = args.stride            # 步幅设置为25
    edge_dim = args.edge_dim        # 边的维度设置为8
    num_neighbor = args.neighbors   # 邻居数量设置为6

    Cog = CognitiveProcessor(input_dim=64, convert_type=args.convert_type, num_features=num_frames,  # 初始化CognitiveProcessor对象
                             n_channels=edge_dim, k=num_neighbor)
    Per = PercepProcessor(only_fuse=True)  # 初始化PercepProcessor对象
    Mot = LipschitzGraph(edge_channel=edge_dim, n_layers=args.layers, act_type=args.act,  # 初始化LipschitzGraph对象
                         num_features=num_frames, norm=args.norm, get_logdets=args.get_logdets)
    
    Cog_s = CognitiveProcessor_s(input_dim=68, convert_type=args.convert_type, num_features=num_frames,  # 初始化CognitiveProcessor对象
                             n_channels=edge_dim, k=num_neighbor)
    

    # model = MHP(p=Per, c=Cog, m=Mot, no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern)  # 初始化MHP模型
    model = MHP(p=Per, c=Cog, m=Mot, c_s=Cog_s, no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern)  # 初始化MHP模型
    model = model.cuda()  # 将模型移动到GPU上

    # train_path = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, delimiter=',')
    train_path = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, delimiter=',')  # 读取训练数据路径
    train_path = train_path.drop(0)  # 删除第一行（通常是标题行）
    speaker_path = [path for path in list(train_path.values[:, 1])] + [path for path in list(train_path.values[:, 2])]  # 提取说话者路径
    listener_path = [path for path in list(train_path.values[:, 2])] + [path for path in list(train_path.values[:, 1])]  # 提取听者路径

    train_neighbour_path = os.path.join(args.data_dir, 'neighbour_emotion_train.npy')  # 读取邻居数据路径
    train_neighbour = np.load(train_neighbour_path)  # 加载邻居数据

    neighbors = {  # 创建邻居字典
        'speaker_path': speaker_path,
        'listener_path': listener_path,
        'neighbors': train_neighbour
    }



    # # 定义保存文本文件的路径
    # output_file_path = 'neighbors_content.txt'

    # # 保存 `neighbors` 的内容到文本文件
    # with open(output_file_path, 'w') as file:
    #     file.write("speaker_path:\n")
    #     for item in neighbors['speaker_path']:
    #         file.write(f"{item}\n")
        
    #     file.write("\nlistener_path:\n")
    #     for item in neighbors['listener_path']:
    #         file.write(f"{item}\n")
        
    #     file.write("\nneighbors:\n")
    #     # 使用 `numpy.savetxt` 确保矩阵内容不被省略
    #     np.savetxt(file, neighbors['neighbors'], fmt='%d', delimiter=', ')


    # dataset = ActionData(root=args.data_dir, data_type='train',  neighbors=None,
    #                      neighbor_pattern=args.neighbor_pattern, num_frames=num_frames, stride=stride)
    
    dataset = ActionData(root=args.data_dir, data_type='train',  neighbors=None,  # 定义数据集
                         neighbor_pattern=args.neighbor_pattern, num_frames=num_frames, stride=stride)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)  # 创建数据加载器

    trainer = Trainer(model=model, neighbors=neighbors, loss_name=args.loss_name,  # 初始化训练器
                      no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern,
                      num_frames=num_frames, stride=stride, loss_mid=args.loss_mid,
                      cal_logdets=args.get_logdets)

    params = []  # 设置优化器的参数
    for key, value in model.named_parameters():  # 遍历模型参数
        if not value.requires_grad:  # 检查参数是否需要梯度更新
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]  # 添加参数到优化器列表

    optimizer = torch.optim.Adam(params)  # 初始化优化器
    lr_scheduler = WarmupMultiStepLR(optimizer, gamma=args.gamma, warmup_factor=args.warmup_factor,  # 初始化学习率调度器
                                     milestones=args.milestones, warmup_iters=args.warmup_step)

    # for epoch in range(100):
    for epoch in range(101):  # 训练模型
        lr_scheduler.step(epoch)  # 更新学习率
        print('Epoch [{}] LR [{:.6f}]'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))  # 打印当前学习率
        trainer.train(epoch=epoch, dataloader=dataloader, optimizer=optimizer, train_iters=args.train_iters)  # 训练模型
        if epoch > 0 and epoch % 5 == 0:  # 每5个epoch保存一次模型
            torch.save(model, osp.join(args.logs_dir, "mhp-epoch{0}-seed{1}.pth").format(epoch, args.seed))  # 保存模型

def test():
    sys.stdout = Logger(osp.join(args.logs_dir, 'test.txt'))  # 将标准输出重定向到测试日志文件
    print("==========\nArgs:{}\n==========".format(args))  # 打印参数信息

    set_seed(args.seed)  # 设置随机种子

    num_frames = args.num_frames  # 获取帧数参数
    stride = args.stride  # 获取步幅参数

    test_batch = len([i for i in range(0, 750 - num_frames + 1, stride)])  # 计算测试批次的大小

    model_pth = args.model_pth  # 获取模型路径参数

    save_base = os.path.join(args.data_dir, 'outputs_avs2', model_pth.split('/')[-2])  # 设置输出保存路径，假设模型路径为 'path/to/your/model/file.pth' ，获取路径中的倒数第二个目录名，结果是 'model'
    if not os.path.isdir(save_base):  # 如果保存路径不存在
        os.makedirs(save_base)  # 创建保存路径

    testset = ActionData(root=args.data_dir, data_type='test', neighbors=None,  # 定义测试数据集
                         neighbor_pattern=args.neighbor_pattern, num_frames=num_frames, stride=stride)
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=False)  # 创建数据加载器，不打乱数据

    model = torch.load(model_pth)  # 加载模型
    model = model.cuda()  # 将模型移动到GPU上

    val_path = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), header=None, delimiter=',')  # 读取验证数据路径
    val_path = val_path.drop(0)  # 删除第一行（通常是标题行）
    speaker_path = [path for path in list(val_path.values[:, 1])] + [path for path in list(val_path.values[:, 2])]  # 提取说话者路径
    listener_path = [path for path in list(val_path.values[:, 2])] + [path for path in list(val_path.values[:, 1])]  # 提取听者路径

    val_neighbour_path = os.path.join(args.data_dir, 'neighbour_emotion_test.npy')  # 读取邻居数据路径
    val_neighbour = np.load(val_neighbour_path)  # 加载邻居数据

    neighbors = {  # 创建邻居字典
        'speaker_path': speaker_path,
        'listener_path': listener_path,
        'neighbors': val_neighbour
    }

    trainer = Trainer(model=model, neighbors=neighbors, neighbor_pattern=args.neighbor_pattern,  # 初始化训练器
                      no_inverse=args.no_inverse)

    trainer.threshold = 0.06  # 设置训练器的阈值
    trainer.test(testloader, modify=args.modify, save_base=save_base)  # 使用训练器进行测试并保存结果



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Actiton Generation")
    # pattern
    parser.add_argument('--test', action='store_true')
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    # model
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--no-inverse', action='store_true')
    parser.add_argument('--convert-type', type=str, default='indirect')
    parser.add_argument('--edge-dim', type=int, default=8)
    parser.add_argument('--neighbors', type=int, default=6)
    # optimizer
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 15])
    parser.add_argument('--warmup-factor', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--loss-name', type=str, default='MSE')
    parser.add_argument('--train-iters', type=int, default=100)
    parser.add_argument('--get-logdets', action='store_true')
    parser.add_argument('--loss-mid', action='store_true')
    parser.add_argument('--neighbor-pattern', type=str, default='nearest', choices=['nearest', 'pair', 'all'])
    parser.add_argument('--num-frames', type=int, default=50)
    parser.add_argument('--stride', type=int, default=25)
    # testing configs
    parser.add_argument('--modify', action='store_true')  # 添加--modify参数，类型为布尔值，指定是否修改数据
    parser.add_argument('--model-pth', type=str, metavar='PATH', default=' ')  # 添加--model-pth参数，指定模型文件的路径
    # path
    working_dir = osp.dirname(osp.abspath(__file__))  # 获取当前脚本所在的目录路径
    parser.add_argument('--data-dir', type=str, metavar='PATH',  # 添加--data-dir参数，指定数据目录路径
                        default=osp.join(working_dir, '/home/lqc/test/data'))  # 默认值为当前脚本目录的上一级目录下的data/react_clean
    parser.add_argument('--logs-dir', type=str, metavar='PATH',  # 添加--logs-dir参数，指定日志目录路径
                        default=osp.join(working_dir, 'logs'))  # 默认值为当前脚本目录下的logs目录

    args = parser.parse_args()
    if args.test:
        test()
    else:
        train(args)