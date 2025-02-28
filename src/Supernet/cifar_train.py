import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
import logging
import argparse
from network import ShuffleNetV2_OneShot
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from flops import get_cand_flops

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def _data_transforms_cifar10():
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor(),
            ])

        valid_transform = transforms.Compose([
            OpencvResize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
            ])
        return train_transform, valid_transform

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--val-batch-size', type=int, default=200, help='val batch size')
    parser.add_argument('--total-iters', type=int, default=150000, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='report frequency')
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=10000, help='report frequency')
    parser.add_argument('--save-interval', type=int, default=10000, help='report frequency')

    parser.add_argument('--root', type=str, default='data/cifar10', help='path to dataset')
    #parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):                                                                                         # log存储文件夹
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True 
    
    train_transform, valid_transform = _data_transforms_cifar10()
    
    train_dataset = datasets.CIFAR10(root=args.root, train=True, download=False, transform=train_transform)
    assert os.path.exists(args.root)
    valid_dataset = datasets.CIFAR10(root=args.root, train=False, download=False, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
      valid_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=use_gpu, num_workers=4)
    
    train_dataprovider = DataIterator(train_loader)
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    model = ShuffleNetV2_OneShot(n_class=10)                                                                            # 对于cifar10,输出是10

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):                                                                                      # 将lr_scheduler自动走到中断的地方
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint, strict=True)
            validate(model, device, args, all_iters=all_iters)
        exit(0)

    while all_iters < args.total_iters:
        all_iters = train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters)
    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, *, val_interval, bn_process=False, all_iters=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st
        
        get_random_cand = lambda:tuple(np.random.randint(4) for i in range(20))             # 匿名采样函数：生成20个4以下的整数，20是层，4是算子的类型
        flops_l, flops_r, flops_step = 290, 360, 10
        bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]             # [[290, 300], [300, 310], [310, 320], [320, 330], [330, 340], [340, 350], [350, 360]]

        def get_uniform_sample_cand(*,timeout=500):
            idx = np.random.randint(len(bins))                                              # 随机生成bins的下标
            l, r = bins[idx]                                                                # 提取某个bin的左右 flops界限
            for i in range(timeout):
                cand = get_random_cand()
                if l*1e6 <= get_cand_flops(cand) <= r*1e6:                                  # 满足flops限制条件的，返回;    get_cand_flops计算仅骨干网的flops，不包括最后的GAP和全连接分类层
                    return cand
            return get_random_cand()                                                        # 若无满足的，直接返回新采样的随机数

        output = model(data, get_uniform_sample_cand())                                     # 根据采样的结果激活单通路模型计算
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100                      # 错误率的小数，是display_interval个iters的average
        Top5_err += 1 - prec5.item() / 100

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_last_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, all_iters)
        
        scheduler.step()

    return all_iters

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)


if __name__ == "__main__":
    main()

