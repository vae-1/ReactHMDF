import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torchvision  # 导入torchvision库，用于计算机视觉任务
from models import SwinTransformer, VGGish  # 从自定义模型文件中导入SwinTransformer和VGGish类
import pandas as pd  # 导入pandas库，用于数据处理
import os  # 导入os模块，用于操作文件系统
from PIL import Image  # 导入PIL库，用于图像处理

from torch.utils import data  # 导入PyTorch的数据工具模块
from torchvision import transforms  # 从torchvision导入transforms模块，用于数据预处理

import numpy as np  # 导入numpy库，用于数值计算
import random  # 导入random模块，用于生成随机数
from decord import VideoReader  # 从decord库导入VideoReader类，用于视频读取
from decord import cpu  # 从decord库导入cpu模块

import argparse  # 导入argparse模块，用于解析命令行参数

from tqdm import tqdm  # 导入tqdm库

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')  # 创建一个ArgumentParser对象
    # 参数定义
    parser.add_argument('--data-dir', default="../data/react/cropped_face", type=str, help="dataset path")  # 添加数据集路径参数
    parser.add_argument('--save-dir', default="../data/react_clean", type=str, help="the dir to save features")  # 添加保存特征路径参数
    parser.add_argument('--split', type=str, help="split of dataset", choices=["train", "train1", "val", "test"], required=True)  # 添加数据集划分参数
    parser.add_argument('--type', type=str, help="type of features to extract", choices=["audio", "video"], required=True)  # 添加特征类型参数
    
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回解析后的参数

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size  # 初始化图像大小
        self.crop_size = crop_size  # 初始化裁剪大小
        
    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 定义归一化变换
        transform = transforms.Compose([
            transforms.Resize(self.img_size),  # 调整图像大小
            transforms.CenterCrop(self.crop_size),  # 中心裁剪图像
            transforms.ToTensor(),  # 将图像转换为张量
            normalize  # 归一化
        ])
        img = transform(img)  # 应用变换
        return img  # 返回变换后的图像

def extract_audio_features(args):
    model = VGGish(preprocess=True)  # 初始化VGGish模型，启用预处理
    model = model.cuda()  # 将模型移动到GPU上
    model.eval()  # 设置模型为评估模式

    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')  # 读取数据列表
    _list_path = _list_path.drop(0)  # 删除第一行

    all_path = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]  # 获取所有路径

    # for path in all_path:
    for path in tqdm(all_path, desc="Processing audio files"):
        # ab_audio_path = os.path.join(args.data_dir, args.split, 'Audio_files', path+'.wav')  # 获取绝对音频路径
        ab_audio_path = os.path.join(args.data_dir, 'Audio_files', path+'.wav')  # 获取绝对音频路径

        with torch.no_grad():  # 禁用梯度计算
            audio_features = model.forward(ab_audio_path, fs=25).cpu()  # 提取音频特征
            
        site, group, pid, clip = path.split('/')  # 分割路径
        if not os.path.exists(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid)):  # 检查目录是否存在
            os.makedirs(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid))  # 创建目录

        torch.save(audio_features, os.path.join(args.save_dir, args.split, 'Audio_features', path+'.pth'))  # 保存音频特征

def extract_video_features(args):
    _transform = Transform(img_size=256, crop_size=224)  # 初始化图像变换

    model = SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.2, num_classes=7)  # 初始化SwinTransformer模型
    model.load_state_dict(torch.load(r"models/torchvggish/swin_fer.pth", map_location='cpu'))  # 加载预训练权重
    model = model.cuda()  # 将模型移动到GPU上
    model.eval()  # 设置模型为评估模式

    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')  # 读取数据列表
    _list_path = _list_path.drop(0)  # 删除第一行

    all_path = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]  # 获取所有路径

    total_length = 751  # 视频总长度

    # for path in all_path:
    for path in tqdm(all_path, desc="Processing video files"):
        clip = []
        ab_video_path = os.path.join(args.data_dir,'Video_files', path+'.mp4')  # 获取绝对视频路径
        with open(ab_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))  # 读取视频
        for i in range(total_length):
            frame = vr[i]  # 获取视频帧
            img = Image.fromarray(frame.asnumpy())  # 将帧转换为图像
            img = _transform(img)  # 应用图像变换
            clip.append(img.unsqueeze(0))  # 添加到剪辑列表中

        video_clip = torch.cat(clip, dim=0).cuda()  # 将剪辑列表拼接成张量并移动到GPU
        with torch.no_grad():  # 禁用梯度计算
            video_features = model.forward_features(video_clip).cpu()  # 提取视频特征
        
        site, group, pid, clip = path.split('/')  # 分割路径
        if not os.path.exists(os.path.join(args.save_dir, args.split, 'Video_features', site, group, pid)):  # 检查目录是否存在
            os.makedirs(os.path.join(args.save_dir, args.split, 'Video_features', site, group, pid))  # 创建目录

        torch.save(video_features, os.path.join(args.save_dir, args.split, 'Video_features', path+'.pth'))  # 保存视频特征

def main(args):
    if args.type == 'video':
        extract_video_features(args)  # 提取视频特征
    elif args.type == 'audio':
        extract_audio_features(args)  # 提取音频特征

# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arg()  # 解析命令行参数
    os.environ["NUMEXPR_MAX_THREADS"] = '32'  # 设置环境变量
    main(args)  # 调用main函数
