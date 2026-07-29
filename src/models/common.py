# YOLOv5 common modules

import math
from copy import copy, deepcopy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import logger, non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

import math

from typing import List, Tuple, Optional
# torch.autograd.set_detect_anomaly(True)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion*c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion*c2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=self.expansion*c2, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*c2),
                                          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(c2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion*c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class DMAF(nn.Module):
    def __init__(self, c2):
        super(DMAF, self).__init__()


    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with torch.amp.autocast(device_type='cuda' if p.device.type != 'cpu' else 'cpu',
                                    enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with torch.amp.autocast(device_type='cuda' if p.device.type != 'cpu' else 'cpu',
                                enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


from torch.nn.init import xavier_uniform_, constant_

# modified on ConvolutionalGLU from TransNeXt(arxiv: 2311.17132; https://github.com/DaiShiResearch/TransNeXt)
class ConvGLUDownsample(nn.Module):

    def __init__(self, n_features=256, kernel_size=3):
        super(ConvGLUDownsample, self).__init__()

        self.n_features = n_features
        self.kernel_size = kernel_size
        _N_INPUTS = n_features * 2
        self.n_inputs = _N_INPUTS

        self.fc1 = nn.Conv2d(_N_INPUTS, _N_INPUTS, kernel_size=1, stride=1)
        self.dw_conv = nn.Conv2d(_N_INPUTS, n_features, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_features)
        self.sigmoid = nn.Sigmoid()
        # self.fc2 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1)

        # self._reset_parameters()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == self.n_inputs
        x1, x2 = torch.chunk(x, chunks=2, dim=1) # [B, c, H, W]

        x = self.fc1(x) # [B, C, H, W]
        x = x.view(B, 2, self.n_features, H, W).transpose(1, 2).reshape(B, C, H, W)
        x = self.sigmoid(self.dw_conv(x)) # [B, c, H, W]
        y = x1 * x + x2 * (1 - x)
        # y = self.fc2(y) # [B, c, H, W]

        return y


def _safe_deform_sampling_grid(grid: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Keep deformable sampling locations finite and away from exact borders."""
    grid = torch.nan_to_num(grid, nan=0.0, posinf=1.0 - eps, neginf=-1.0 + eps)
    return grid.clamp(min=-1.0 + eps, max=1.0 - eps)


def _safe_attention_logits(logits: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    """Bound attention logits before softmax to avoid rare cross-domain overflow."""
    logits = torch.nan_to_num(logits, nan=0.0, posinf=limit, neginf=-limit)
    return logits.clamp(min=-limit, max=limit)


class DeformScaledDotAttnLayerLocal(nn.Module):

    def __init__(self, n_features=256, n_heads=8, n_points=8, sampling_field=7, attn_pdrop=.1,
                 generate_offsets_with_positional_embedding=False):
        super(DeformScaledDotAttnLayerLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_points = n_points
        _N_GROUPS = n_points*2
        assert n_features % n_heads == 0
        assert n_features % _N_GROUPS == 0
        self.n_features_per_head = n_features // n_heads

        assert sampling_field % 2 == 1
        self._SAMPLING_FIELD = sampling_field
        self._N_SAMPLERS = (self._SAMPLING_FIELD - 1) // 2
        def _gen_sampling_blocks(n_features, n_points, n_samplers):
            sampling_block_top = ConvGLUDownsample(n_features, kernel_size=3)
            sampling_block = nn.Sequential(nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, groups=n_features),
                                           nn.GroupNorm(_N_GROUPS, n_features),
                                           nn.SiLU())
            sampling_block_bottom = nn.Conv2d(n_features, n_points*2, kernel_size=1)
            return nn.Sequential(*([sampling_block_top] + [sampling_block for _ in range(n_samplers-1)] + [sampling_block_bottom]))
        self.sampling_offsets = _gen_sampling_blocks(n_features, n_points, self._N_SAMPLERS)
        self.query_proj = nn.Conv2d(n_features, n_features, 1, groups=n_heads)
        self.key_proj = nn.Conv2d(n_features*n_points, n_features*n_points, 1, groups=n_heads*n_points)
        self.value_proj = nn.Conv2d(n_features*n_points, n_features*n_points, 1, groups=n_heads*n_points)
        self.output_proj = nn.Conv2d(n_features, n_features, 1)

        self._offsets_with_pos = generate_offsets_with_positional_embedding
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
                    (1, self.n_heads, self._SAMPLING_FIELD, self._SAMPLING_FIELD)), requires_grad=True)
        self._bias_forward = self._table_bias_forward

        self.dropout = nn.Dropout(p=attn_pdrop)

        self._scale = math.sqrt(n_features // n_heads)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _with_pos_emb(self, x: torch.Tensor, pos_emb: torch.Tensor):
        if not pos_emb is None:
            return x + pos_emb
        else:
            return x

    def _table_bias_forward(self, actual_offsets, bs, H, W, n_query): # generate offset bias with interpolation
        positions = actual_offsets.view(bs, self.n_points, n_query, 2).transpose(1, 2).reshape(bs*n_query, 1, self.n_points, 2) # [bsxp, H, W, 2] -> [bsxHxW, 1, p, 2]
        sampled_bias = F.grid_sample(input=self.relative_position_bias_table.expand(bs*n_query, -1, -1, -1),
                                     grid=positions, mode='bilinear', align_corners=False) # [bsxHxW, h, 1, p]
        attn_bias = sampled_bias.view(bs, n_query, self.n_heads, self.n_points).transpose(1, 2).reshape(
            bs*self.n_heads*n_query, 1, self.n_points) # [bsxhxHxW, 1, p]
        return attn_bias

    def forward(self, query: torch.Tensor, x: torch.Tensor, reference_points: torch.Tensor, positional_embeddings: Optional[torch.Tensor] = None):
        bs_q, d_q, H_q, W_q = query.shape
        bs, d_x, H, W = x.shape
        n_query = H_q*W_q
        assert bs == bs_q, "inconsistent batch size, query: %d, x: %d" %(bs_q, bs)
        assert d_q == d_x, "inconsistent channels, query: %d, x: %d" %(d_q, d_x)
        assert H_q == H and W_q == W, "inconsistent feature map size, query: (%d, %d), x: (%d, %d)" %(H_q, W_q, H, W)

        # generating offsets
        query_embed = self._with_pos_emb(query, positional_embeddings)
        if self._offsets_with_pos:
            x_embed = self._with_pos_emb(x, positional_embeddings)
            reception = torch.cat([query_embed, x_embed], dim=1) # [bs, 2xC, H, W]
        else:
            reception = torch.cat([query, x], dim=1)
        offsets = self.sampling_offsets(reception).tanh() # [bs, 2xp, H, W]; scaled to [-1, 1]

        # normalizing and truncating sampling positions
        offsets = offsets.reshape(bs*self.n_points, 2, H, W).permute(0, 2, 3, 1) # [bs, 2xp, H, W] -> [bsxp, 2, H, W] -> [bsxp, H, W, 2]
        scaler = torch.as_tensor([self._SAMPLING_FIELD/2/W, self._SAMPLING_FIELD/2/H],
                                  dtype=offsets.dtype, device=offsets.device)[None, None, None, :] # [1, 1, 1, 2]
        offsets = offsets * scaler # scale offsets
        sampling_loc = _safe_deform_sampling_grid(reference_points + offsets).view(bs, self.n_points, H, W, 2) # [bsxp, H, W, 2] -> [bs, p, H, W, 2]

        # sampling deformed features
        sampled = []
        for i in range(self.n_points):
            sampled.append(F.grid_sample(input=x, grid=sampling_loc[:, i], mode='bilinear', align_corners=False))
        sampled = torch.stack(sampled, dim=1).view(bs, self.n_points*self.n_features, H, W) # [bs, p, C, H, W] -> [bs, p*C, H, W]

        # obtain query, key, and value
        query = self.query_proj(query_embed).reshape(bs*self.n_heads, self.n_features_per_head, n_query) # [bsxh, c, HxW]
        query = query.transpose(1, 2).reshape(bs*self.n_heads*n_query, 1, self.n_features_per_head) # [bsxhxHxW, 1, c]
        key = self.key_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query) # [bs, p, h, c, HxW]
        key = key.permute(0, 2, 4, 1, 3).reshape(bs*self.n_heads*n_query, self.n_points, self.n_features_per_head) # [bsxhxHxW, p, c]
        value = self.value_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query) # [bs, p, h, c, HxW]
        value = value.permute(0, 2, 4, 1, 3).reshape(bs*self.n_heads*n_query, self.n_points, self.n_features_per_head) # [bsxhxHxW, p, c]

        # calculating scaled-dot attention
        actual_offsets = sampling_loc.view(-1, H, W, 2) - reference_points # [bsxp, H, W, 2]
        attn_bias = self._bias_forward(actual_offsets, bs, H, W, n_query) # [bsxhxHxW, 1, p]
        attn = torch.matmul(query, key.transpose(1, 2)) / self._scale + attn_bias # [bsxhxHxW, 1, p]
        attn = _safe_attention_logits(attn)

        attn = self.dropout(F.softmax(attn, dim=2, dtype=attn.dtype))
        out = torch.matmul(attn, value).view(bs*self.n_heads, n_query, self.n_features_per_head) # [bsxhxHxW, 1, c] -> [bsxh, H*W, c]
        out = out.transpose(1, 2).reshape(bs, self.n_features, H, W) # [bs, C, H, W]
        output = self.output_proj(out) # [bs, C, H, W]

        return output


class DeformScaledDotTransformerFusionLayerLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7,
                 attn_pdrop: float = .1, ffn_pdrop: float = .1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True):
        super(DeformScaledDotTransformerFusionLayerLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_ffn_features = n_features * ffn_exp
        self.n_points = n_points

        # deformAttn\
        self.rgb_cross_attn = DeformScaledDotAttnLayerLocal(n_features, n_heads, n_points, s_sampling, attn_pdrop,
                                                            generate_offsets_with_positional_embedding)
        self.LN_rgb_attn = nn.LayerNorm(n_features)

        self.ir_cross_attn = DeformScaledDotAttnLayerLocal(n_features, n_heads, n_points, s_sampling, attn_pdrop,
                                                           generate_offsets_with_positional_embedding)
        self.LN_ir_attn = nn.LayerNorm(n_features)

        # ffn
        self.rgb_ffn = nn.Sequential(nn.Linear(n_features, n_features*ffn_exp),
                                     nn.GELU(),
                                     nn.Linear(n_features*ffn_exp, n_features),
                                     nn.Dropout(p=ffn_pdrop))
        self.LN_rgb_ffn = nn.LayerNorm(n_features)

        self.ir_ffn = nn.Sequential(nn.Linear(n_features, n_features*ffn_exp),
                                    nn.GELU(),
                                    nn.Linear(n_features*ffn_exp, n_features),
                                    nn.Dropout(p=ffn_pdrop))
        self.LN_ir_ffn = nn.LayerNorm(n_features)


    def forward(self, rgb_feat_map, ir_feat_map, pos_emb, reference_points):

        B, C, H, W = rgb_feat_map.shape

        rgb_x_ir = rgb_feat_map + self.rgb_cross_attn(rgb_feat_map, ir_feat_map, reference_points, pos_emb)
        rgb_feat_flat = rgb_x_ir.flatten(2).transpose(1, 2) # [B, C, H, W] -> [B, HxW, C]
        rgb_feat_flat = self.LN_rgb_attn(rgb_feat_flat)
        rgb_feat_flat = self.LN_rgb_ffn(rgb_feat_flat + self.rgb_ffn(rgb_feat_flat))
        rgb_feat_out = rgb_feat_flat.transpose(1, 2).view(B, C, H, W) # [B, HxW, C] -> [B, C, H, W]

        ir_x_rgb = ir_feat_map + self.ir_cross_attn(ir_feat_map, rgb_feat_map, reference_points, pos_emb)
        ir_feat_flat = ir_x_rgb.flatten(2).transpose(1, 2) # [B, C, H, W] -> [B, HxW, C]
        ir_feat_flat = self.LN_ir_attn(ir_feat_flat)
        ir_feat_flat = self.LN_ir_ffn(ir_feat_flat + self.ir_ffn(ir_feat_flat))
        ir_feat_out = ir_feat_flat.transpose(1, 2).view(B, C, H, W) # [B, HxW, C] -> [B, C, H, W]

        return [rgb_feat_out, ir_feat_out]


class DeformScaledDotTransformerFusionBlockLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7, n_layers: int = 1,
                 attn_pdrop: float = 0.1, ffn_pdrop: float = 0.1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True):
        super(DeformScaledDotTransformerFusionBlockLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_points = n_points

        transformer_layer = DeformScaledDotTransformerFusionLayerLocal(n_features, n_heads, n_points, s_sampling, attn_pdrop=attn_pdrop, ffn_pdrop=ffn_pdrop, ffn_exp=ffn_exp,
                                                                       generate_offsets_with_positional_embedding=generate_offsets_with_positional_embedding)
        self.transformer_layers = nn.ModuleList([deepcopy(transformer_layer) for _ in range(self.n_layers)])
        self.output_layer = ConvGLUDownsample(n_features, 3)

        self._temperature = 10000
        self._scale = 2*math.pi
        self._normalize = True
        self._using_cuda = torch.cuda.is_available()

    @torch.no_grad()
    def _get_pos_emb(self, x: torch.Tensor):
        H_, W_ = x.shape[2:]
        y_embed = torch.arange(0, H_, 1, dtype=x.dtype, device=x.device)[None, :, None].repeat(1, 1, W_) # [1, H_, W_]
        x_embed = torch.arange(0, W_, 1, dtype=x.dtype, device=x.device)[None, None, :].repeat(1, H_, 1) # [1, H_, W_]
        # print(x_embed.shape, y_embed.shape)

        if self._normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self._scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self._scale

        dim_t = torch.arange(self.n_features//2, dtype=x.dtype, device=x.device)
        dim_t = (self._temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / (self.n_features//2)))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

    @torch.no_grad()
    def _get_reference_points(self, x: torch.Tensor):
        B, _, H, W, = x.shape
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5/H, (H-0.5)/H, H, dtype=x.dtype, device=x.device),
                                      torch.linspace(0.5/W, (W-0.5)/W, W, dtype=x.dtype, device=x.device), indexing='ij')
        ref = torch.stack((ref_x[None, :], ref_y[None, :]), -1).expand(B*self.n_points, -1, -1, -1) # [Bxp, H, W, 2]
        ref = ref * 2 - 1
        return ref

    def forward(self, feat_maps):

        assert len(feat_maps) == 2
        rgb_feat_map, ir_feat_map = feat_maps

        pos_emb = self._get_pos_emb(rgb_feat_map)
        reference_points = self._get_reference_points(rgb_feat_map)

        rgb_feat_out, ir_feat_out = rgb_feat_map, ir_feat_map
        for i in range(self.n_layers):
            rgb_feat_out, ir_feat_out = self.transformer_layers[i](rgb_feat_out, ir_feat_out, pos_emb, reference_points)

        out = torch.cat([rgb_feat_out, ir_feat_out], dim=1) # [B, 2xC, H, W]
        output = self.output_layer(out) # [B, C, H, W]

        return output


class ConsensusDetailSceneRefiner(nn.Module):

    def __init__(self, n_features: int = 256, refine_hidden_ratio: float = 0.5):
        super(ConsensusDetailSceneRefiner, self).__init__()

        hidden = int(n_features * refine_hidden_ratio) if refine_hidden_ratio < 1 else int(refine_hidden_ratio)
        hidden = max(32, make_divisible(hidden, 8))
        hidden = min(hidden, n_features)

        self.rgb_proj = Conv(n_features, hidden, 1, 1)
        self.ir_proj = Conv(n_features, hidden, 1, 1)
        self.fused_proj = Conv(n_features, hidden, 1, 1)

        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.rgb_detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1)
        )
        self.ir_detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1)
        )

        self.consensus_encoder = nn.Sequential(
            Conv(hidden * 2, hidden, 3, 1),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        scene_hidden = max(16, hidden // 2)
        self.scene_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 4, scene_hidden, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(scene_hidden, 2, kernel_size=1, stride=1, bias=True)
        )

        self.output_proj = nn.Sequential(
            Conv(hidden * 3, n_features, 1, 1),
            Conv(n_features, n_features, 3, 1)
        )
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, feat_maps):

        assert len(feat_maps) == 3
        rgb_feat_map, ir_feat_map, fused_feat_map = feat_maps

        rgb_proj = self.rgb_proj(rgb_feat_map)
        ir_proj = self.ir_proj(ir_feat_map)
        fused_proj = self.fused_proj(fused_feat_map)

        difference = torch.abs(rgb_proj - ir_proj)
        agreement = rgb_proj * ir_proj
        consensus = self.consensus_encoder(torch.cat([agreement, difference], dim=1))

        rgb_detail = self.rgb_detail(rgb_proj - self.local_pool(rgb_proj))
        ir_detail = self.ir_detail(ir_proj - self.local_pool(ir_proj))

        scene_logits = self.scene_gate(torch.cat([rgb_proj, ir_proj, fused_proj, difference], dim=1))
        scene_weights = torch.softmax(scene_logits, dim=1)
        detail_mix = scene_weights[:, :1] * rgb_detail + scene_weights[:, 1:2] * ir_detail
        detail_mix = detail_mix * (1.0 + consensus)

        residual = self.output_proj(torch.cat([fused_proj, fused_proj * consensus, detail_mix], dim=1))
        return fused_feat_map + torch.sigmoid(self.res_scale) * residual


class DeformConsensusDetailFusionBlockLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7, n_layers: int = 1,
                 attn_pdrop: float = 0.1, ffn_pdrop: float = 0.1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True, refine_hidden_ratio: float = 0.5):
        super(DeformConsensusDetailFusionBlockLocal, self).__init__()

        self.deform_fusion = DeformScaledDotTransformerFusionBlockLocal(
            n_features=n_features,
            n_heads=n_heads,
            n_points=n_points,
            s_sampling=s_sampling,
            n_layers=n_layers,
            attn_pdrop=attn_pdrop,
            ffn_pdrop=ffn_pdrop,
            ffn_exp=ffn_exp,
            generate_offsets_with_positional_embedding=generate_offsets_with_positional_embedding,
        )
        self.refiner = ConsensusDetailSceneRefiner(n_features=n_features, refine_hidden_ratio=refine_hidden_ratio)

    def forward(self, feat_maps):

        assert len(feat_maps) == 2
        fused_feat_map = self.deform_fusion(feat_maps)
        return self.refiner([feat_maps[0], feat_maps[1], fused_feat_map])


class DaytimeReliabilityRefiner(nn.Module):

    def __init__(self, n_features: int = 256, refine_hidden_ratio: float = 0.375):
        super(DaytimeReliabilityRefiner, self).__init__()

        hidden = int(n_features * refine_hidden_ratio) if refine_hidden_ratio < 1 else int(refine_hidden_ratio)
        hidden = max(32, make_divisible(hidden, 8))
        hidden = min(hidden, n_features)

        self.rgb_proj = Conv(n_features, hidden, 1, 1)
        self.ir_proj = Conv(n_features, hidden, 1, 1)
        self.fused_proj = Conv(n_features, hidden, 1, 1)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.foreground_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.background_conflict_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.ambiguity_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.groupness_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.uncertainty_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        scene_hidden = max(16, hidden // 2)
        self.day_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 5, scene_hidden, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(scene_hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.modality_router = nn.Sequential(
            Conv(hidden * 5, hidden, 3, 1),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, bias=True),
        )
        self.detail_mixer = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            Conv(hidden, hidden, 1, 1),
        )
        self.output_proj = nn.Sequential(
            Conv(hidden * 3, n_features, 1, 1),
            Conv(n_features, n_features, 3, 1),
        )

        self.res_scale = nn.Parameter(torch.tensor(-4.0))
        self.suppress_scale = nn.Parameter(torch.tensor(-2.2))
        self.rgb_noise_scale = nn.Parameter(torch.tensor(-2.4))
        self.ir_noise_scale = nn.Parameter(torch.tensor(-2.8))
        self.countability_scale = nn.Parameter(torch.tensor(-2.4))
        self.semantic_probe_detach = False
        self.typed_semantic_residual_scale = 0.0
        self.typed_semantic_residual_temperature = 1.0
        self.typed_semantic_residual_group_gain = 1.0
        self.typed_semantic_residual_uncertainty_gain = 1.0
        self.typed_semantic_residual_center = 0.50
        self.typed_countability_group_gain = 0.0
        self.typed_countability_uncertainty_gain = 0.0
        self.typed_countability_activation_floor = 0.35
        self.typed_countability_min_gate = 0.70
        self.last_aux = None
        self.export_aux = False
        self._reset_parameters()

    def _ensure_runtime_defaults(self):
        if not hasattr(self, 'semantic_probe_detach'):
            self.semantic_probe_detach = False
        if not hasattr(self, 'typed_semantic_residual_scale'):
            self.typed_semantic_residual_scale = 0.0
        if not hasattr(self, 'typed_semantic_residual_temperature'):
            self.typed_semantic_residual_temperature = 1.0
        if not hasattr(self, 'typed_semantic_residual_group_gain'):
            self.typed_semantic_residual_group_gain = 1.0
        if not hasattr(self, 'typed_semantic_residual_uncertainty_gain'):
            self.typed_semantic_residual_uncertainty_gain = 1.0
        if not hasattr(self, 'typed_semantic_residual_center'):
            self.typed_semantic_residual_center = 0.50
        if not hasattr(self, 'typed_countability_group_gain'):
            self.typed_countability_group_gain = 0.0
        if not hasattr(self, 'typed_countability_uncertainty_gain'):
            self.typed_countability_uncertainty_gain = 0.0
        if not hasattr(self, 'typed_countability_activation_floor'):
            self.typed_countability_activation_floor = 0.35
        if not hasattr(self, 'typed_countability_min_gate'):
            self.typed_countability_min_gate = 0.70
        if not hasattr(self, 'last_aux'):
            self.last_aux = None
        if not hasattr(self, 'export_aux'):
            self.export_aux = False

    def __getstate__(self):
        # `last_aux` stores forward-time tensors for auxiliary supervision.
        # They are runtime caches, not model state, and may be non-leaf tensors.
        # Clearing them here keeps checkpoint deepcopy/save stable without
        # changing the training graph or serialized parameters.
        state = super().__getstate__()
        state['last_aux'] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._ensure_runtime_defaults()

    def set_typed_semantic_runtime_config(self, detach_semantic_probe=None, residual_scale=None, residual_temperature=None,
                                          residual_group_gain=None, residual_uncertainty_gain=None,
                                          residual_center=None,
                                          countability_group_gain=None, countability_uncertainty_gain=None,
                                          countability_activation_floor=None, countability_min_gate=None):
        self._ensure_runtime_defaults()
        if detach_semantic_probe is not None:
            self.semantic_probe_detach = bool(detach_semantic_probe)
        if residual_scale is not None:
            self.typed_semantic_residual_scale = max(float(residual_scale), 0.0)
        if residual_temperature is not None:
            self.typed_semantic_residual_temperature = max(float(residual_temperature), 1e-3)
        if residual_group_gain is not None:
            self.typed_semantic_residual_group_gain = max(float(residual_group_gain), 0.0)
        if residual_uncertainty_gain is not None:
            self.typed_semantic_residual_uncertainty_gain = max(float(residual_uncertainty_gain), 0.0)
        if residual_center is not None:
            self.typed_semantic_residual_center = min(max(float(residual_center), 0.0), 1.0)
        if countability_group_gain is not None:
            self.typed_countability_group_gain = max(float(countability_group_gain), 0.0)
        if countability_uncertainty_gain is not None:
            self.typed_countability_uncertainty_gain = max(float(countability_uncertainty_gain), 0.0)
        if countability_activation_floor is not None:
            self.typed_countability_activation_floor = min(max(float(countability_activation_floor), 0.0), 0.95)
        if countability_min_gate is not None:
            self.typed_countability_min_gate = min(max(float(countability_min_gate), 0.0), 1.0)

    def _activate_typed_countability_signal(self, semantic_map):
        self._ensure_runtime_defaults()
        floor = min(max(float(self.typed_countability_activation_floor), 0.0), 0.95)
        scale = max(1.0 - floor, 1e-3)
        return ((semantic_map - floor).clamp(min=0.0) / scale).clamp(max=1.0)

    def _center_typed_residual_signal(self, semantic_map):
        self._ensure_runtime_defaults()
        center = min(max(float(self.typed_semantic_residual_center), 0.0), 1.0)
        return semantic_map - center

    def _reset_parameters(self):
        constant_(self.foreground_gate[-2].bias.data, -0.6)
        constant_(self.background_conflict_gate[-2].bias.data, -1.2)
        constant_(self.ambiguity_gate[-2].bias.data, -1.8)
        constant_(self.groupness_gate[-2].bias.data, -2.0)
        constant_(self.uncertainty_gate[-2].bias.data, -2.2)
        constant_(self.day_router[-2].weight.data, 0.0)
        constant_(self.day_router[-2].bias.data, -1.2)
        constant_(self.modality_router[-1].weight.data, 0.0)
        constant_(self.modality_router[-1].bias.data, 0.0)

    def forward(self, feat_maps):
        self._ensure_runtime_defaults()

        assert len(feat_maps) == 3
        rgb_feat_map, ir_feat_map, fused_feat_map = feat_maps

        rgb_proj = self.rgb_proj(rgb_feat_map)
        ir_proj = self.ir_proj(ir_feat_map)
        fused_proj = self.fused_proj(fused_feat_map)

        difference = torch.abs(rgb_proj - ir_proj)
        agreement = rgb_proj * ir_proj
        fused_detail = fused_proj - self.local_pool(fused_proj)
        rgb_detail = rgb_proj - self.local_pool(rgb_proj)
        ir_detail = ir_proj - self.local_pool(ir_proj)

        foreground = self.foreground_gate(torch.cat([fused_proj, agreement, fused_detail], dim=1))
        conflict = self.background_conflict_gate(torch.cat([difference, fused_detail, fused_proj], dim=1))
        background_conflict = conflict * (1.0 - foreground)
        ambiguity = None
        groupness = None
        uncertainty = None
        countable_gate = 1.0
        typed_countable_gate = None
        typed_countability_signal = None
        typed_residual_signal = None
        if hasattr(self, 'ambiguity_gate'):
            ambiguity = self.ambiguity_gate(torch.cat([difference, fused_detail, fused_proj], dim=1))
            ambiguous_background = ambiguity * (1.0 - foreground)
        if hasattr(self, 'groupness_gate'):
            group_source = torch.cat([difference, fused_detail, agreement], dim=1)
            if getattr(self, 'semantic_probe_detach', False):
                group_source = group_source.detach()
            groupness = self.groupness_gate(group_source)
        if hasattr(self, 'uncertainty_gate'):
            uncertainty_source = torch.cat([difference, fused_detail, fused_proj], dim=1)
            if getattr(self, 'semantic_probe_detach', False):
                uncertainty_source = uncertainty_source.detach()
            uncertainty = self.uncertainty_gate(uncertainty_source)

        rgb_energy = rgb_detail.abs()
        ir_energy = ir_detail.abs()
        day_context = torch.cat([rgb_proj, ir_proj, fused_proj, difference, rgb_energy - ir_energy], dim=1)
        day_gate = self.day_router(day_context)

        modality_logits = self.modality_router(torch.cat([rgb_proj, ir_proj, difference, agreement, fused_detail], dim=1))
        modality_weights = torch.softmax(modality_logits, dim=1)
        rgb_weight_raw = modality_weights[:, :1]
        ir_weight_raw = modality_weights[:, 1:2]
        rgb_weight = rgb_weight_raw * (
            1.0 - day_gate * torch.sigmoid(self.rgb_noise_scale) * background_conflict
        ).clamp(min=0.20)
        ir_weight = ir_weight_raw * (
            1.0 - (0.35 + 0.35 * day_gate) * torch.sigmoid(self.ir_noise_scale) * background_conflict
        ).clamp(min=0.25)

        stable_fused = fused_proj * (
            1.0 - day_gate * torch.sigmoid(self.suppress_scale) * background_conflict
        ).clamp(min=0.45)
        if ambiguity is not None and hasattr(self, 'countability_scale'):
            countable_gate = (
                1.0 - day_gate * torch.sigmoid(self.countability_scale) * ambiguous_background
            ).clamp(min=0.55)
            typed_countable_gate = torch.ones_like(countable_gate)
            if self.typed_countability_group_gain > 0.0 or self.typed_countability_uncertainty_gain > 0.0:
                typed_terms = []
                if groupness is not None and self.typed_countability_group_gain > 0.0:
                    typed_terms.append(
                        self.typed_countability_group_gain * self._activate_typed_countability_signal(groupness)
                    )
                if uncertainty is not None and self.typed_countability_uncertainty_gain > 0.0:
                    typed_terms.append(
                        self.typed_countability_uncertainty_gain * self._activate_typed_countability_signal(uncertainty)
                    )
                if typed_terms:
                    typed_countability_signal = torch.stack(typed_terms, dim=0).sum(dim=0).clamp(max=1.0)
                    if getattr(self, 'semantic_probe_detach', False):
                        typed_countability_signal = typed_countability_signal.detach()
                    typed_countable_gate = (
                        1.0 - day_gate * ambiguous_background * typed_countability_signal
                    ).clamp(min=self.typed_countability_min_gate, max=1.0)
                    countable_gate = countable_gate * typed_countable_gate
            stable_fused = stable_fused * countable_gate

        rgb_contribution = (rgb_weight * rgb_detail.abs()).mean(dim=1, keepdim=True)
        ir_contribution = (ir_weight * ir_detail.abs()).mean(dim=1, keepdim=True)
        reliability_detail = rgb_weight * rgb_detail + ir_weight * ir_detail
        detail_mix = self.detail_mixer(torch.cat([reliability_detail, difference, agreement], dim=1))
        detail_mix = detail_mix * foreground * (0.5 + day_gate)
        if self.typed_semantic_residual_scale > 0.0 and (groupness is not None or uncertainty is not None):
            semantic_terms = []
            semantic_weights = []
            if groupness is not None and self.typed_semantic_residual_group_gain > 0.0:
                semantic_terms.append(
                    self.typed_semantic_residual_group_gain * self._center_typed_residual_signal(groupness)
                )
                semantic_weights.append(self.typed_semantic_residual_group_gain)
            if uncertainty is not None and self.typed_semantic_residual_uncertainty_gain > 0.0:
                semantic_terms.append(
                    self.typed_semantic_residual_uncertainty_gain * self._center_typed_residual_signal(uncertainty)
                )
                semantic_weights.append(self.typed_semantic_residual_uncertainty_gain)
            if semantic_terms:
                denom = max(sum(semantic_weights), 1e-3)
                typed_residual_signal = torch.stack(semantic_terms, dim=0).sum(dim=0) / denom
                typed_residual_signal = torch.tanh(
                    typed_residual_signal / max(self.typed_semantic_residual_temperature, 1e-3)
                )
                if getattr(self, 'semantic_probe_detach', False):
                    typed_residual_signal = typed_residual_signal.detach()
                detail_mix = detail_mix * (1.0 + self.typed_semantic_residual_scale * typed_residual_signal)

        residual = self.output_proj(torch.cat([stable_fused, stable_fused * foreground, detail_mix], dim=1))
        should_export = (self.training and torch.is_grad_enabled()) or getattr(self, 'export_aux', False)
        if should_export:
            self.last_aux = {
                'foreground': foreground if self.training else foreground.detach(),
                'conflict': conflict if self.training else conflict.detach(),
                'background_conflict': background_conflict if self.training else background_conflict.detach(),
                'day_gate': day_gate if self.training else day_gate.detach(),
                'rgb_weight_raw': rgb_weight_raw if self.training else rgb_weight_raw.detach(),
                'ir_weight_raw': ir_weight_raw if self.training else ir_weight_raw.detach(),
                'rgb_weight': rgb_weight if self.training else rgb_weight.detach(),
                'ir_weight': ir_weight if self.training else ir_weight.detach(),
                'rgb_contribution': rgb_contribution if self.training else rgb_contribution.detach(),
                'ir_contribution': ir_contribution if self.training else ir_contribution.detach(),
            }
            if ambiguity is not None:
                self.last_aux['ambiguity'] = ambiguity if self.training else ambiguity.detach()
                self.last_aux['countable_gate'] = countable_gate if self.training else countable_gate.detach()
                if typed_countable_gate is not None:
                    self.last_aux['typed_countable_gate'] = typed_countable_gate if self.training else typed_countable_gate.detach()
                if typed_countability_signal is not None:
                    self.last_aux['typed_countability_signal'] = typed_countability_signal if self.training else typed_countability_signal.detach()
            if groupness is not None:
                self.last_aux['groupness'] = groupness if self.training else groupness.detach()
            if uncertainty is not None:
                self.last_aux['uncertainty'] = uncertainty if self.training else uncertainty.detach()
            if typed_residual_signal is not None:
                self.last_aux['typed_residual_signal'] = typed_residual_signal if self.training else typed_residual_signal.detach()
        else:
            self.last_aux = None
        return fused_feat_map + torch.sigmoid(self.res_scale) * residual


class DeformDASRFusionBlockLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7, n_layers: int = 1,
                 attn_pdrop: float = 0.1, ffn_pdrop: float = 0.1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True, refine_hidden_ratio: float = 0.5,
                 reliability_hidden_ratio: float = 0.375):
        super(DeformDASRFusionBlockLocal, self).__init__()

        self.deform_fusion = DeformScaledDotTransformerFusionBlockLocal(
            n_features=n_features,
            n_heads=n_heads,
            n_points=n_points,
            s_sampling=s_sampling,
            n_layers=n_layers,
            attn_pdrop=attn_pdrop,
            ffn_pdrop=ffn_pdrop,
            ffn_exp=ffn_exp,
            generate_offsets_with_positional_embedding=generate_offsets_with_positional_embedding,
        )
        self.refiner = ConsensusDetailSceneRefiner(n_features=n_features, refine_hidden_ratio=refine_hidden_ratio)
        self.reliability_refiner = DaytimeReliabilityRefiner(
            n_features=n_features,
            refine_hidden_ratio=reliability_hidden_ratio,
        )

    def forward(self, feat_maps):

        assert len(feat_maps) == 2
        fused_feat_map = self.deform_fusion(feat_maps)
        refined_feat_map = self.refiner([feat_maps[0], feat_maps[1], fused_feat_map])
        return self.reliability_refiner([feat_maps[0], feat_maps[1], refined_feat_map])


class IlluminationAwareScaleRefiner(nn.Module):

    def __init__(self, n_features: int = 256, refine_hidden_ratio: float = 0.5, detail_boost: float = 1.0):
        super(IlluminationAwareScaleRefiner, self).__init__()

        hidden = int(n_features * refine_hidden_ratio) if refine_hidden_ratio < 1 else int(refine_hidden_ratio)
        hidden = max(32, make_divisible(hidden, 8))
        hidden = min(hidden, n_features)

        self.detail_boost = detail_boost

        self.rgb_proj = Conv(n_features, hidden, 1, 1)
        self.ir_proj = Conv(n_features, hidden, 1, 1)
        self.fused_proj = Conv(n_features, hidden, 1, 1)

        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.rgb_detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1)
        )
        self.ir_detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1)
        )

        self.consensus_encoder = nn.Sequential(
            Conv(hidden * 2, hidden, 3, 1),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        scene_hidden = max(16, hidden // 2)
        self.scene_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 4, scene_hidden, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(scene_hidden, 3, kernel_size=1, stride=1, bias=True)
        )

        self.suppress_gate = nn.Sequential(
            Conv(hidden * 2, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        self.thermal_gate = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        self.output_proj = nn.Sequential(
            Conv(hidden * 3, n_features, 1, 1),
            Conv(n_features, n_features, 3, 1)
        )

        self.res_scale = nn.Parameter(torch.tensor(-2.0))
        self.suppress_scale = nn.Parameter(torch.tensor(-2.2))
        self.thermal_scale = nn.Parameter(torch.tensor(-1.8))

    def forward(self, feat_maps):

        assert len(feat_maps) == 3
        rgb_feat_map, ir_feat_map, fused_feat_map = feat_maps

        rgb_proj = self.rgb_proj(rgb_feat_map)
        ir_proj = self.ir_proj(ir_feat_map)
        fused_proj = self.fused_proj(fused_feat_map)

        difference = torch.abs(rgb_proj - ir_proj)
        agreement = rgb_proj * ir_proj
        consensus = self.consensus_encoder(torch.cat([agreement, difference], dim=1))

        rgb_detail = self.rgb_detail(rgb_proj - self.local_pool(rgb_proj))
        ir_detail = self.ir_detail(ir_proj - self.local_pool(ir_proj))

        scene_logits = self.scene_gate(torch.cat([rgb_proj, ir_proj, fused_proj, difference], dim=1))
        scene_weights = torch.softmax(scene_logits, dim=1)

        detail_mix = scene_weights[:, :1] * rgb_detail + scene_weights[:, 1:2] * ir_detail
        detail_mix = detail_mix * (1.0 + self.detail_boost * consensus)

        thermal_rescue = self.thermal_gate(torch.cat([ir_proj, difference, ir_detail], dim=1)) * ir_detail
        thermal_rescue = scene_weights[:, 2:3] * thermal_rescue

        suppress_mask = self.suppress_gate(torch.cat([difference, fused_proj], dim=1))
        stable_fused = fused_proj * (1.0 - torch.sigmoid(self.suppress_scale) * suppress_mask)

        adaptive_detail = detail_mix + torch.sigmoid(self.thermal_scale) * thermal_rescue
        residual = self.output_proj(torch.cat([stable_fused, stable_fused * consensus, adaptive_detail], dim=1))

        return fused_feat_map + torch.sigmoid(self.res_scale) * residual


class DeformAIMSFusionBlockLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7, n_layers: int = 1,
                 attn_pdrop: float = 0.1, ffn_pdrop: float = 0.1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True, refine_hidden_ratio: float = 0.5, detail_boost: float = 1.0):
        super(DeformAIMSFusionBlockLocal, self).__init__()

        self.deform_fusion = DeformScaledDotTransformerFusionBlockLocal(
            n_features=n_features,
            n_heads=n_heads,
            n_points=n_points,
            s_sampling=s_sampling,
            n_layers=n_layers,
            attn_pdrop=attn_pdrop,
            ffn_pdrop=ffn_pdrop,
            ffn_exp=ffn_exp,
            generate_offsets_with_positional_embedding=generate_offsets_with_positional_embedding,
        )
        self.refiner = IlluminationAwareScaleRefiner(
            n_features=n_features,
            refine_hidden_ratio=refine_hidden_ratio,
            detail_boost=detail_boost,
        )

    def forward(self, feat_maps):

        assert len(feat_maps) == 2
        fused_feat_map = self.deform_fusion(feat_maps)
        return self.refiner([feat_maps[0], feat_maps[1], fused_feat_map])


class TargetAwareConflictContextEncoder(nn.Module):

    def __init__(self, n_features: int = 256, hidden_ratio: float = 0.5):
        super(TargetAwareConflictContextEncoder, self).__init__()

        hidden = int(n_features * hidden_ratio) if hidden_ratio < 1 else int(hidden_ratio)
        hidden = max(32, make_divisible(hidden, 8))
        hidden = min(hidden, n_features)

        self.rgb_proj = Conv(n_features, hidden, 1, 1)
        self.ir_proj = Conv(n_features, hidden, 1, 1)

        self.prior_head = nn.Sequential(
            Conv(hidden * 4, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
        )
        self.conflict_head = nn.Sequential(
            Conv(hidden * 3, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, bias=True),
        )

        mode_hidden = max(16, hidden // 2)
        self.mode_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 4, mode_hidden, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mode_hidden, 3, kernel_size=1, stride=1, bias=True),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Start close to the DeformCAT fusion behaviour. The target/conflict maps
        # should become influential after learning, not through random early bias.
        constant_(self.prior_head[-1].bias.data, -2.0)
        constant_(self.conflict_head[-1].bias.data, -2.2)
        constant_(self.mode_router[-1].weight.data, 0.0)
        constant_(self.mode_router[-1].bias.data, 0.0)

    def forward(self, rgb_feat_map: torch.Tensor, ir_feat_map: torch.Tensor):

        rgb_proj = self.rgb_proj(rgb_feat_map)
        ir_proj = self.ir_proj(ir_feat_map)
        difference = torch.abs(rgb_proj - ir_proj)
        agreement = rgb_proj * ir_proj

        context = torch.cat([rgb_proj, ir_proj, difference, agreement], dim=1)
        target_prior = torch.sigmoid(self.prior_head(context))
        conflict_mask = torch.sigmoid(self.conflict_head(torch.cat([difference, rgb_proj + ir_proj, agreement], dim=1)))
        mode_weights = torch.softmax(self.mode_router(context), dim=1)

        return target_prior, conflict_mask, mode_weights


class TargetAwareConflictDeformScaledDotAttnLayerLocal(nn.Module):

    def __init__(self, n_features=256, n_heads=8, n_points=8, sampling_field=7, attn_pdrop=.1,
                 generate_offsets_with_positional_embedding=False):
        super(TargetAwareConflictDeformScaledDotAttnLayerLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_points = n_points
        _N_GROUPS = n_points * 2
        assert n_features % n_heads == 0
        assert n_features % _N_GROUPS == 0
        self.n_features_per_head = n_features // n_heads

        assert sampling_field % 2 == 1
        self._SAMPLING_FIELD = sampling_field
        self._N_SAMPLERS = (self._SAMPLING_FIELD - 1) // 2

        def _gen_sampling_blocks(num_features, num_points, num_samplers):
            sampling_block_top = ConvGLUDownsample(num_features, kernel_size=3)
            sampling_block = nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, groups=num_features),
                nn.GroupNorm(_N_GROUPS, num_features),
                nn.SiLU(),
            )
            sampling_block_bottom = nn.Conv2d(num_features, num_points * 2, kernel_size=1)
            return nn.Sequential(*([sampling_block_top] + [sampling_block for _ in range(num_samplers - 1)] + [sampling_block_bottom]))

        self.sampling_offsets = _gen_sampling_blocks(n_features, n_points, self._N_SAMPLERS)
        self.query_proj = nn.Conv2d(n_features, n_features, 1, groups=n_heads)
        self.key_proj = nn.Conv2d(n_features * n_points, n_features * n_points, 1, groups=n_heads * n_points)
        self.value_proj = nn.Conv2d(n_features * n_points, n_features * n_points, 1, groups=n_heads * n_points)
        self.output_proj = nn.Conv2d(n_features, n_features, 1)

        self._offsets_with_pos = generate_offsets_with_positional_embedding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((1, self.n_heads, self._SAMPLING_FIELD, self._SAMPLING_FIELD)), requires_grad=True
        )
        self._bias_forward = self._table_bias_forward

        self.dropout = nn.Dropout(p=attn_pdrop)
        self._scale = math.sqrt(n_features // n_heads)

        self.query_prior_scale = nn.Parameter(torch.tensor(-2.8))
        self.source_prior_scale = nn.Parameter(torch.tensor(-2.8))
        self.source_conflict_scale = nn.Parameter(torch.tensor(-2.8))
        self.source_target_conflict_scale = nn.Parameter(torch.tensor(-3.0))
        self.offset_conflict_scale = nn.Parameter(torch.tensor(-3.0))
        self.offset_target_conflict_scale = nn.Parameter(torch.tensor(-3.2))
        self.sample_prior_scale = nn.Parameter(torch.tensor(-3.0))
        self.sample_conflict_scale = nn.Parameter(torch.tensor(-3.0))
        self.sample_target_conflict_scale = nn.Parameter(torch.tensor(-3.2))
        self.prior_attn_scale = nn.Parameter(torch.tensor(-2.8))
        self.conflict_attn_scale = nn.Parameter(torch.tensor(-2.8))
        self.target_conflict_attn_scale = nn.Parameter(torch.tensor(-3.2))

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _with_pos_emb(self, x: torch.Tensor, pos_emb: torch.Tensor):
        if pos_emb is not None:
            return x + pos_emb
        return x

    def _table_bias_forward(self, actual_offsets, bs, H, W, n_query):
        positions = actual_offsets.view(bs, self.n_points, n_query, 2).transpose(1, 2).reshape(bs * n_query, 1, self.n_points, 2)
        sampled_bias = F.grid_sample(
            input=self.relative_position_bias_table.expand(bs * n_query, -1, -1, -1),
            grid=positions,
            mode='bilinear',
            align_corners=False,
        )
        attn_bias = sampled_bias.view(bs, n_query, self.n_heads, self.n_points).transpose(1, 2).reshape(
            bs * self.n_heads * n_query, 1, self.n_points
        )
        return attn_bias

    def _sample_guidance(self, guidance: torch.Tensor, sampling_loc: torch.Tensor):
        sampled = []
        for i in range(self.n_points):
            sampled.append(F.grid_sample(input=guidance, grid=sampling_loc[:, i], mode='bilinear', align_corners=False))
        return torch.stack(sampled, dim=1)

    def forward(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        reference_points: torch.Tensor,
        positional_embeddings: Optional[torch.Tensor] = None,
        target_prior: Optional[torch.Tensor] = None,
        conflict_mask: Optional[torch.Tensor] = None,
        query_mode_weight: Optional[torch.Tensor] = None,
        source_mode_weight: Optional[torch.Tensor] = None,
        conflict_mode_weight: Optional[torch.Tensor] = None,
    ):
        bs_q, d_q, H_q, W_q = query.shape
        bs, d_x, H, W = x.shape
        n_query = H_q * W_q
        assert bs == bs_q, "inconsistent batch size, query: %d, x: %d" % (bs_q, bs)
        assert d_q == d_x, "inconsistent channels, query: %d, x: %d" % (d_q, d_x)
        assert H_q == H and W_q == W, "inconsistent feature map size, query: (%d, %d), x: (%d, %d)" % (H_q, W_q, H, W)

        if target_prior is None:
            target_prior = query.new_zeros((bs, 1, H, W))
        if conflict_mask is None:
            conflict_mask = query.new_zeros((bs, 1, H, W))
        if query_mode_weight is None:
            query_mode_weight = query.new_ones((bs, 1, 1, 1))
        if source_mode_weight is None:
            source_mode_weight = query.new_ones((bs, 1, 1, 1))
        if conflict_mode_weight is None:
            conflict_mode_weight = query.new_zeros((bs, 1, 1, 1))

        target_conflict = target_prior * conflict_mask
        background_conflict = (1.0 - target_prior) * conflict_mask
        query_embed = self._with_pos_emb(query, positional_embeddings)
        source_embed = self._with_pos_emb(x, positional_embeddings) if self._offsets_with_pos else x

        query_gain = 1.0 + torch.sigmoid(self.query_prior_scale) * target_prior * query_mode_weight
        source_gain = 1.0 + torch.sigmoid(self.source_prior_scale) * target_prior * source_mode_weight
        source_gain = source_gain * (
            1.0 - torch.sigmoid(self.source_conflict_scale) * background_conflict * (0.5 + conflict_mode_weight)
        ).clamp(min=0.30)
        source_gain = source_gain * (
            1.0 + torch.sigmoid(self.source_target_conflict_scale) * target_conflict * source_mode_weight
        ).clamp(max=1.35)

        reception_query = query_embed if self._offsets_with_pos else query
        reception_source = source_embed
        reception = torch.cat([reception_query * query_gain, reception_source * source_gain], dim=1)
        offsets = self.sampling_offsets(reception).tanh()
        background_conflict_offsets = background_conflict.repeat(1, self.n_points * 2, 1, 1)
        target_conflict_offsets = target_conflict.repeat(1, self.n_points * 2, 1, 1)
        offset_gate = (
            1.0
            - torch.sigmoid(self.offset_conflict_scale) * background_conflict_offsets
            + torch.sigmoid(self.offset_target_conflict_scale) * target_conflict_offsets
        ).clamp(min=0.35, max=1.35)
        offsets = offsets * offset_gate

        offsets = offsets.reshape(bs * self.n_points, 2, H, W).permute(0, 2, 3, 1)
        scaler = torch.as_tensor(
            [self._SAMPLING_FIELD / 2 / W, self._SAMPLING_FIELD / 2 / H],
            dtype=offsets.dtype,
            device=offsets.device,
        )[None, None, None, :]
        offsets = offsets * scaler
        sampling_loc = _safe_deform_sampling_grid(reference_points + offsets).view(bs, self.n_points, H, W, 2)

        sampled = []
        for i in range(self.n_points):
            sampled.append(F.grid_sample(input=x, grid=sampling_loc[:, i], mode='bilinear', align_corners=False))
        sampled = torch.stack(sampled, dim=1)

        sampled_prior = self._sample_guidance(target_prior, sampling_loc)
        sampled_conflict = self._sample_guidance(conflict_mask, sampling_loc)
        sampled_target_conflict = sampled_prior * sampled_conflict
        sampled_background_conflict = (1.0 - sampled_prior) * sampled_conflict
        sampled_gain = (
            1.0
            + torch.sigmoid(self.sample_prior_scale) * sampled_prior * source_mode_weight.unsqueeze(1)
            - torch.sigmoid(self.sample_conflict_scale) * sampled_background_conflict * (0.5 + conflict_mode_weight.unsqueeze(1))
            + torch.sigmoid(self.sample_target_conflict_scale) * sampled_target_conflict * source_mode_weight.unsqueeze(1)
        ).clamp(min=0.25, max=1.50)
        sampled = (sampled * sampled_gain).view(bs, self.n_points * self.n_features, H, W)

        query_tokens = self.query_proj(query_embed).reshape(bs * self.n_heads, self.n_features_per_head, n_query)
        query_tokens = query_tokens.transpose(1, 2).reshape(bs * self.n_heads * n_query, 1, self.n_features_per_head)
        key = self.key_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query)
        key = key.permute(0, 2, 4, 1, 3).reshape(bs * self.n_heads * n_query, self.n_points, self.n_features_per_head)
        value = self.value_proj(sampled).view(bs, self.n_points, self.n_heads, self.n_features_per_head, n_query)
        value = value.permute(0, 2, 4, 1, 3).reshape(bs * self.n_heads * n_query, self.n_points, self.n_features_per_head)

        actual_offsets = sampling_loc.view(-1, H, W, 2) - reference_points
        attn_bias = self._bias_forward(actual_offsets, bs, H, W, n_query)

        prior_bias = sampled_prior.squeeze(2).view(bs, self.n_points, n_query).transpose(1, 2)
        prior_bias = (
            torch.sigmoid(self.prior_attn_scale) * source_mode_weight.view(bs, 1, 1) * prior_bias
        ).unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(bs * self.n_heads * n_query, 1, self.n_points)
        background_conflict_bias = sampled_background_conflict.squeeze(2).view(bs, self.n_points, n_query).transpose(1, 2)
        background_conflict_bias = (
            torch.sigmoid(self.conflict_attn_scale) * (0.5 + conflict_mode_weight.view(bs, 1, 1)) * background_conflict_bias
        ).unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(bs * self.n_heads * n_query, 1, self.n_points)
        target_conflict_bias = sampled_target_conflict.squeeze(2).view(bs, self.n_points, n_query).transpose(1, 2)
        target_conflict_bias = (
            torch.sigmoid(self.target_conflict_attn_scale) * source_mode_weight.view(bs, 1, 1) * target_conflict_bias
        ).unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(bs * self.n_heads * n_query, 1, self.n_points)

        attn = (
            torch.matmul(query_tokens, key.transpose(1, 2)) / self._scale
            + attn_bias
            + prior_bias
            + target_conflict_bias
            - background_conflict_bias
        )
        attn = _safe_attention_logits(attn)
        attn = self.dropout(F.softmax(attn, dim=2, dtype=attn.dtype))
        out = torch.matmul(attn, value).view(bs * self.n_heads, n_query, self.n_features_per_head)
        out = out.transpose(1, 2).reshape(bs, self.n_features, H, W)
        output = self.output_proj(out)

        return output


class TargetAwareConflictTransformerFusionLayerLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7,
                 attn_pdrop: float = .1, ffn_pdrop: float = .1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True):
        super(TargetAwareConflictTransformerFusionLayerLocal, self).__init__()

        self.n_features = n_features

        self.rgb_cross_attn = TargetAwareConflictDeformScaledDotAttnLayerLocal(
            n_features, n_heads, n_points, s_sampling, attn_pdrop, generate_offsets_with_positional_embedding
        )
        self.LN_rgb_attn = nn.LayerNorm(n_features)
        self.ir_cross_attn = TargetAwareConflictDeformScaledDotAttnLayerLocal(
            n_features, n_heads, n_points, s_sampling, attn_pdrop, generate_offsets_with_positional_embedding
        )
        self.LN_ir_attn = nn.LayerNorm(n_features)

        self.rgb_ffn = nn.Sequential(
            nn.Linear(n_features, n_features * ffn_exp),
            nn.GELU(),
            nn.Linear(n_features * ffn_exp, n_features),
            nn.Dropout(p=ffn_pdrop),
        )
        self.LN_rgb_ffn = nn.LayerNorm(n_features)

        self.ir_ffn = nn.Sequential(
            nn.Linear(n_features, n_features * ffn_exp),
            nn.GELU(),
            nn.Linear(n_features * ffn_exp, n_features),
            nn.Dropout(p=ffn_pdrop),
        )
        self.LN_ir_ffn = nn.LayerNorm(n_features)

    def forward(
        self,
        rgb_feat_map: torch.Tensor,
        ir_feat_map: torch.Tensor,
        pos_emb: torch.Tensor,
        reference_points: torch.Tensor,
        target_prior: torch.Tensor,
        conflict_mask: torch.Tensor,
        mode_weights: torch.Tensor,
    ):
        B, C, H, W = rgb_feat_map.shape

        rgb_mode = mode_weights[:, :1]
        ir_mode = mode_weights[:, 1:2]
        conflict_mode = mode_weights[:, 2:3]

        rgb_x_ir = rgb_feat_map + self.rgb_cross_attn(
            rgb_feat_map,
            ir_feat_map,
            reference_points,
            pos_emb,
            target_prior=target_prior,
            conflict_mask=conflict_mask,
            query_mode_weight=rgb_mode,
            source_mode_weight=ir_mode,
            conflict_mode_weight=conflict_mode,
        )
        rgb_feat_flat = rgb_x_ir.flatten(2).transpose(1, 2)
        rgb_feat_flat = self.LN_rgb_attn(rgb_feat_flat)
        rgb_feat_flat = self.LN_rgb_ffn(rgb_feat_flat + self.rgb_ffn(rgb_feat_flat))
        rgb_feat_out = rgb_feat_flat.transpose(1, 2).view(B, C, H, W)

        ir_x_rgb = ir_feat_map + self.ir_cross_attn(
            ir_feat_map,
            rgb_feat_map,
            reference_points,
            pos_emb,
            target_prior=target_prior,
            conflict_mask=conflict_mask,
            query_mode_weight=ir_mode,
            source_mode_weight=rgb_mode,
            conflict_mode_weight=conflict_mode,
        )
        ir_feat_flat = ir_x_rgb.flatten(2).transpose(1, 2)
        ir_feat_flat = self.LN_ir_attn(ir_feat_flat)
        ir_feat_flat = self.LN_ir_ffn(ir_feat_flat + self.ir_ffn(ir_feat_flat))
        ir_feat_out = ir_feat_flat.transpose(1, 2).view(B, C, H, W)

        return [rgb_feat_out, ir_feat_out]


class TargetAwareConflictFusionProjector(nn.Module):

    def __init__(self, n_features: int = 256, hidden_ratio: float = 0.5):
        super(TargetAwareConflictFusionProjector, self).__init__()

        hidden = int(n_features * hidden_ratio) if hidden_ratio < 1 else int(hidden_ratio)
        hidden = max(32, make_divisible(hidden, 8))
        hidden = min(hidden, n_features)

        self.base_fusion = ConvGLUDownsample(n_features, 3)
        self.rgb_proj = Conv(n_features, hidden, 1, 1)
        self.ir_proj = Conv(n_features, hidden, 1, 1)
        self.base_proj = Conv(n_features, hidden, 1, 1)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.target_detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
        )

        self.output_proj = nn.Sequential(
            Conv(hidden * 5, n_features, 1, 1),
            Conv(n_features, n_features, 3, 1),
        )

        self.res_scale = nn.Parameter(torch.tensor(-2.8))
        self.conflict_scale = nn.Parameter(torch.tensor(-2.8))
        self.target_detail_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        rgb_feat_map: torch.Tensor,
        ir_feat_map: torch.Tensor,
        target_prior: torch.Tensor,
        conflict_mask: torch.Tensor,
        mode_weights: torch.Tensor,
    ):
        fused_feat_map = self.base_fusion(torch.cat([rgb_feat_map, ir_feat_map], dim=1))

        rgb_proj = self.rgb_proj(rgb_feat_map)
        ir_proj = self.ir_proj(ir_feat_map)
        fused_proj = self.base_proj(fused_feat_map)

        target_conflict = target_prior * conflict_mask
        background_conflict = (1.0 - target_prior) * conflict_mask
        stable_fused = fused_proj * (
            1.0 - torch.sigmoid(self.conflict_scale) * background_conflict * (1.0 + mode_weights[:, 2:3])
        ).clamp(min=0.30)
        target_context = fused_proj * (target_prior + target_conflict).clamp(max=1.0)
        target_detail = self.target_detail(fused_proj - self.local_pool(fused_proj))
        target_detail = (
            torch.sigmoid(self.target_detail_scale)
            * target_detail
            * (target_prior + target_conflict).clamp(max=1.0)
        )
        rgb_target = rgb_proj * target_prior * mode_weights[:, :1]
        ir_target = ir_proj * target_prior * mode_weights[:, 1:2]

        residual = self.output_proj(torch.cat([stable_fused, target_context, rgb_target, ir_target, target_detail], dim=1))
        return fused_feat_map + torch.sigmoid(self.res_scale) * residual


class DeformTACMFusionBlockLocal(nn.Module):

    def __init__(self, n_features: int = 256, n_heads: int = 8, n_points: int = 8, s_sampling: int = 7, n_layers: int = 1,
                 attn_pdrop: float = 0.1, ffn_pdrop: float = 0.1, ffn_exp: int = 4,
                 generate_offsets_with_positional_embedding: bool = True, context_hidden_ratio: float = 0.5,
                 projector_hidden_ratio: float = 0.5):
        super(DeformTACMFusionBlockLocal, self).__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_points = n_points

        transformer_layer = TargetAwareConflictTransformerFusionLayerLocal(
            n_features,
            n_heads,
            n_points,
            s_sampling,
            attn_pdrop=attn_pdrop,
            ffn_pdrop=ffn_pdrop,
            ffn_exp=ffn_exp,
            generate_offsets_with_positional_embedding=generate_offsets_with_positional_embedding,
        )
        self.transformer_layers = nn.ModuleList([deepcopy(transformer_layer) for _ in range(self.n_layers)])
        self.context_encoder = TargetAwareConflictContextEncoder(n_features=n_features, hidden_ratio=context_hidden_ratio)
        self.output_layer = TargetAwareConflictFusionProjector(n_features=n_features, hidden_ratio=projector_hidden_ratio)

        self._temperature = 10000
        self._scale = 2 * math.pi
        self._normalize = True
        self._using_cuda = torch.cuda.is_available()

    @torch.no_grad()
    def _get_pos_emb(self, x: torch.Tensor):
        H_, W_ = x.shape[2:]
        y_embed = torch.arange(0, H_, 1, dtype=x.dtype, device=x.device)[None, :, None].repeat(1, 1, W_)
        x_embed = torch.arange(0, W_, 1, dtype=x.dtype, device=x.device)[None, None, :].repeat(1, H_, 1)

        if self._normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self._scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self._scale

        dim_t = torch.arange(self.n_features // 2, dtype=x.dtype, device=x.device)
        dim_t = self._temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / (self.n_features // 2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

    @torch.no_grad()
    def _get_reference_points(self, x: torch.Tensor):
        B, _, H, W = x.shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5 / H, (H - 0.5) / H, H, dtype=x.dtype, device=x.device),
            torch.linspace(0.5 / W, (W - 0.5) / W, W, dtype=x.dtype, device=x.device),
            indexing='ij',
        )
        ref = torch.stack((ref_x[None, :], ref_y[None, :]), -1).expand(B * self.n_points, -1, -1, -1)
        ref = ref * 2 - 1
        return ref

    def forward(self, feat_maps):

        assert len(feat_maps) == 2
        rgb_feat_map, ir_feat_map = feat_maps

        pos_emb = self._get_pos_emb(rgb_feat_map)
        reference_points = self._get_reference_points(rgb_feat_map)

        rgb_feat_out, ir_feat_out = rgb_feat_map, ir_feat_map
        for i in range(self.n_layers):
            target_prior, conflict_mask, mode_weights = self.context_encoder(rgb_feat_out, ir_feat_out)
            rgb_feat_out, ir_feat_out = self.transformer_layers[i](
                rgb_feat_out, ir_feat_out, pos_emb, reference_points, target_prior, conflict_mask, mode_weights
            )

        target_prior, conflict_mask, mode_weights = self.context_encoder(rgb_feat_out, ir_feat_out)
        return self.output_layer(rgb_feat_out, ir_feat_out, target_prior, conflict_mask, mode_weights)
