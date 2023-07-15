import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.layer import MergedConv2d
from engine.layer import MaxPooling2d
from engine.layer import FC
from engine.layer import ReLULayer
from engine.imageStar import ImageStar
import imageio
from torch.quantization import QuantStub, DeQuantStub


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.re3_2 = nn.ReLU()
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.re4_2 = nn.ReLU()
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.re5_2 = nn.ReLU()
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.re1_1(self.conv1_1(x))
        x = self.pool1(self.re1_2(self.conv1_2(x)))
        x = self.re2_1(self.conv2_1(x))
        x = self.pool2(self.re2_2(self.conv2_2(x)))
        x = self.re3_1(self.conv3_1(x))
        x = self.re3_2(self.conv3_2(x))
        x = self.pool3(self.re3_3(self.conv3_3(x)))
        x = self.re4_1(self.conv4_1(x))
        x = self.re4_2(self.conv4_2(x))
        x = self.pool4(self.re4_3(self.conv4_3(x)))
        x = self.re5_1(self.conv5_1(x))
        x = self.re5_2(self.conv5_2(x))
        x = self.pool5(self.re5_3(self.conv5_3(x)))
        x = torch.flatten(x, 1)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.re3_2 = nn.ReLU()
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.re4_2 = nn.ReLU()
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.re5_2 = nn.ReLU()
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # x = x.to('cpu')
        x = self.quant(x)
        x = self.re1_1(self.conv1_1(x))
        x = self.pool1(self.re1_2(self.conv1_2(x)))
        x = self.re2_1(self.conv2_1(x))
        x = self.pool2(self.re2_2(self.conv2_2(x)))
        x = self.re3_1(self.conv3_1(x))
        x = self.re3_2(self.conv3_2(x))
        x = self.pool3(self.re3_3(self.conv3_3(x)))
        x = self.re4_1(self.conv4_1(x))
        x = self.re4_2(self.conv4_2(x))
        x = self.pool4(self.re4_3(self.conv4_3(x)))
        x = self.re5_1(self.conv5_1(x))
        x = self.re5_2(self.conv5_2(x))
        x = self.pool5(self.re5_3(self.conv5_3(x)))
        x = torch.flatten(x, 1)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


def quant_star(input, scale, offset):
    if isinstance(input, list):
        w = input[0].V.shape[1]
        ch = input[0].V.shape[2] // 2
        if w != 1:
            for i in range(len(input)):
                input[i].V[w:, :, :, 0] = np.round(input[i].V[w:, :, :, 0] / scale) + offset
                input[i].V[w:, :, :, 0] = (input[i].V[w:, :, :, 0] - offset) * scale
        elif (w == 1) & (input[0].V.shape[0] == 2):
            for i in range(len(input)):
                input[i].V[1, 0, :, 0] = np.round(input[i].V[1, 0, :, 0] / scale) + offset
                input[i].V[1, 0, :, 0] = (input[i].V[1, 0, :, 0] - offset) * scale
        else:
            for i in range(len(input)):
                input[i].V[0, 0, ch:, 0] = np.round(input[i].V[0, 0, ch:, 0] / scale) + offset
                input[i].V[0, 0, ch:, 0] = (input[i].V[0, 0, ch:, 0] - offset) * scale
    elif isinstance(input, ImageStar):
        w = input.V.shape[1]
        input.V[w:, :, :, 0] = np.round(input.V[w:, :, :, 0] / scale) + offset
        input.V[w:, :, :, 0] = (input.V[w:, :, :, 0] - offset) * scale
    return input


def err_range(Istar):
    diff_list = []
    n = len(Istar)
    for i in range(n):
        Star1 = Istar[i].toStar()
        vertices = Star1.toPolyhedron()
        if i == 0:
            for ind in range(10):
                diff_list.append([min(vertices[ind, :]), max(vertices[ind, :])])
        else:
            for ind in range(10):
                diff_list[ind] = [min([diff_list[ind][0], min(vertices[ind, :])]),
                                  max([diff_list[ind][1], max(vertices[ind, :])])]
    return diff_list


def compute_net_diff(result_path1, result_path2, IM, lb, ub):
    net1 = torch.load(result_path1, map_location=torch.device('cpu'))  # only load for parameter reading
    net2 = torch.load(result_path2, map_location=torch.device('cpu'))

    conv1_1_weight1 = net1['conv1_1.weight'].to(torch.float32)
    conv1_1_bias1 = net1['conv1_1.bias'].to(torch.float32)
    conv1_2_weight1 = net1['conv1_2.weight'].to(torch.float32)
    conv1_2_bias1 = net1['conv1_2.bias'].to(torch.float32)
    conv2_1_weight1 = net1['conv2_1.weight'].to(torch.float32)
    conv2_1_bias1 = net1['conv2_1.bias'].to(torch.float32)
    conv2_2_weight1 = net1['conv2_2.weight'].to(torch.float32)
    conv2_2_bias1 = net1['conv2_2.bias'].to(torch.float32)
    conv3_1_weight1 = net1['conv3_1.weight'].to(torch.float32)
    conv3_1_bias1 = net1['conv3_1.bias'].to(torch.float32)
    conv3_2_weight1 = net1['conv3_2.weight'].to(torch.float32)
    conv3_2_bias1 = net1['conv3_2.bias'].to(torch.float32)
    conv3_3_weight1 = net1['conv3_3.weight'].to(torch.float32)
    conv3_3_bias1 = net1['conv3_3.bias'].to(torch.float32)
    conv4_1_weight1 = net1['conv4_1.weight'].to(torch.float32)
    conv4_1_bias1 = net1['conv4_1.bias'].to(torch.float32)
    conv4_2_weight1 = net1['conv4_2.weight'].to(torch.float32)
    conv4_2_bias1 = net1['conv4_2.bias'].to(torch.float32)
    conv4_3_weight1 = net1['conv4_3.weight'].to(torch.float32)
    conv4_3_bias1 = net1['conv4_3.bias'].to(torch.float32)
    conv5_1_weight1 = net1['conv5_1.weight'].to(torch.float32)
    conv5_1_bias1 = net1['conv5_1.bias'].to(torch.float32)
    conv5_2_weight1 = net1['conv5_2.weight'].to(torch.float32)
    conv5_2_bias1 = net1['conv5_2.bias'].to(torch.float32)
    conv5_3_weight1 = net1['conv5_3.weight'].to(torch.float32)
    conv5_3_bias1 = net1['conv5_3.bias'].to(torch.float32)
    fc1_weight1 = net1['fc1.weight'].to(torch.float32)
    fc1_bias1 = net1['fc1.bias'].to(torch.float32)
    fc2_weight1 = net1['fc2.weight'].to(torch.float32)
    fc2_bias1 = net1['fc2.bias'].to(torch.float32)
    fc3_weight1 = net1['fc3.weight'].to(torch.float32)
    fc3_bias1 = net1['fc3.bias'].to(torch.float32)

    conv1_1_weight2 = net2['conv1_1.weight'].dequantize().to(torch.float32)
    conv1_1_bias2 = net2['conv1_1.bias'].dequantize().to(torch.float32)
    conv1_2_weight2 = net2['conv1_2.weight'].dequantize().to(torch.float32)
    conv1_2_bias2 = net2['conv1_2.bias'].dequantize().to(torch.float32)
    conv2_1_weight2 = net2['conv2_1.weight'].dequantize().to(torch.float32)
    conv2_1_bias2 = net2['conv2_1.bias'].dequantize().to(torch.float32)
    conv2_2_weight2 = net2['conv2_2.weight'].dequantize().to(torch.float32)
    conv2_2_bias2 = net2['conv2_2.bias'].dequantize().to(torch.float32)
    conv3_1_weight2 = net2['conv3_1.weight'].dequantize().to(torch.float32)
    conv3_1_bias2 = net2['conv3_1.bias'].dequantize().to(torch.float32)
    conv3_2_weight2 = net2['conv3_2.weight'].dequantize().to(torch.float32)
    conv3_2_bias2 = net2['conv3_2.bias'].dequantize().to(torch.float32)
    conv3_3_weight2 = net2['conv3_3.weight'].dequantize().to(torch.float32)
    conv3_3_bias2 = net2['conv3_3.bias'].dequantize().to(torch.float32)
    conv4_1_weight2 = net2['conv4_1.weight'].dequantize().to(torch.float32)
    conv4_1_bias2 = net2['conv4_1.bias'].dequantize().to(torch.float32)
    conv4_2_weight2 = net2['conv4_2.weight'].dequantize().to(torch.float32)
    conv4_2_bias2 = net2['conv4_2.bias'].dequantize().to(torch.float32)
    conv4_3_weight2 = net2['conv4_3.weight'].dequantize().to(torch.float32)
    conv4_3_bias2 = net2['conv4_3.bias'].dequantize().to(torch.float32)
    conv5_1_weight2 = net2['conv5_1.weight'].dequantize().to(torch.float32)
    conv5_1_bias2 = net2['conv5_1.bias'].dequantize().to(torch.float32)
    conv5_2_weight2 = net2['conv5_2.weight'].dequantize().to(torch.float32)
    conv5_2_bias2 = net2['conv5_2.bias'].dequantize().to(torch.float32)
    conv5_3_weight2 = net2['conv5_3.weight'].dequantize().to(torch.float32)
    conv5_3_bias2 = net2['conv5_3.bias'].dequantize().to(torch.float32)
    fc1_weight2 = net2['fc1._packed_params._packed_params'][0].dequantize().to(torch.float32)
    fc1_bias2 = net2['fc1._packed_params._packed_params'][1].dequantize().to(torch.float32)
    fc2_weight2 = net2['fc2._packed_params._packed_params'][0].dequantize().to(torch.float32)
    fc2_bias2 = net2['fc2._packed_params._packed_params'][1].dequantize().to(torch.float32)
    fc3_weight2 = net2['fc3._packed_params._packed_params'][0].dequantize().to(torch.float32)
    fc3_bias2 = net2['fc3._packed_params._packed_params'][1].dequantize().to(torch.float32)

    method = 'exact-star'
    poolsize = np.array([2, 2])
    stride = np.array([2, 2])
    p_padding = np.array([0, 0])

    LB = np.zeros((32, 32, 3), dtype=np.single)
    UB = np.zeros((32, 32, 3), dtype=np.single)
    LB[0, 0, 0] = lb
    UB[0, 0, 0] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    l_conv1_1 = MergedConv2d(conv1_1_weight1, conv1_1_bias1, conv1_1_weight2, conv1_1_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv1_2 = MergedConv2d(conv1_2_weight1, conv1_2_bias1, conv1_2_weight2, conv1_2_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv2_1 = MergedConv2d(conv2_1_weight1, conv2_1_bias1, conv2_1_weight2, conv2_1_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv2_2 = MergedConv2d(conv2_2_weight1, conv2_2_bias1, conv2_2_weight2, conv2_2_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv3_1 = MergedConv2d(conv3_1_weight1, conv3_1_bias1, conv3_1_weight2, conv3_1_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv3_2 = MergedConv2d(conv3_2_weight1, conv3_2_bias1, conv3_2_weight2, conv3_2_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv3_3 = MergedConv2d(conv3_3_weight1, conv3_3_bias1, conv3_3_weight2, conv3_3_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv4_1 = MergedConv2d(conv4_1_weight1, conv4_1_bias1, conv4_1_weight2, conv4_1_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv4_2 = MergedConv2d(conv4_2_weight1, conv4_2_bias1, conv4_2_weight2, conv4_2_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv4_3 = MergedConv2d(conv4_3_weight1, conv4_3_bias1, conv4_3_weight2, conv4_3_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv5_1 = MergedConv2d(conv5_1_weight1, conv5_1_bias1, conv5_1_weight2, conv5_1_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv5_2 = MergedConv2d(conv5_2_weight1, conv5_2_bias1, conv5_2_weight2, conv5_2_bias2, pad1=(1, 1), pad2=(1, 1))
    l_conv5_3 = MergedConv2d(conv5_3_weight1, conv5_3_bias1, conv5_3_weight2, conv5_3_bias2, pad1=(1, 1), pad2=(1, 1))
    l_pool1 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool1')
    l_pool2 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool2')
    l_pool3 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool3')
    l_pool4 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool4')
    l_pool5 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool5')
    l_fc1 = FC(fc1_weight1, fc1_bias1, fc1_weight2, fc1_bias2)
    l_fc2 = FC(fc2_weight1, fc2_bias1, fc2_weight2, fc2_bias2)
    l_fc3 = FC(fc3_weight1, fc3_bias1, fc3_weight2, fc3_bias2)
    fc4_weight1 = np.eye(10, dtype=np.single)
    fc4_weight2 = -np.eye(10, dtype=np.single)
    fc4_b1 = np.zeros((1, 2))
    fc4_b2 = np.zeros((1, 2))
    l_fc4 = FC(fc4_weight1, fc4_b1, fc4_weight2, fc4_b2, 'last')
    l_relu = ReLULayer()

    I1_m = quant_star(I1_m, net2['quant.scale'].numpy(), net2['quant.zero_point'].numpy())
    Istar1 = l_conv1_1.reach(I1_m)
    Istar2 = l_relu.reach(Istar1, method)

    Istar2 = quant_star(Istar2, net2['conv1_1.scale'].numpy(), net2['conv1_1.zero_point'].numpy())
    Istar3 = l_conv1_2.reach(Istar2)
    Istar4 = l_relu.reach(Istar3, method)
    Istar5 = l_pool1.reach(Istar4, method)

    Istar5 = quant_star(Istar5, net2['conv1_2.scale'].numpy(), net2['conv1_2.zero_point'].numpy())
    Istar6 = l_conv2_1.reach(Istar5)
    Istar7 = l_relu.reach(Istar6, method)

    Istar7 = quant_star(Istar7, net2['conv2_1.scale'].numpy(), net2['conv2_1.zero_point'].numpy())
    Istar8 = l_conv2_2.reach(Istar7)
    Istar9 = l_relu.reach(Istar8, method)
    Istar10 = l_pool2.reach(Istar9, method)

    Istar10 = quant_star(Istar10, net2['conv2_2.scale'].numpy(), net2['conv2_2.zero_point'].numpy())
    Istar11 = l_conv3_1.reach(Istar10)
    Istar12 = l_relu.reach(Istar11, method)

    Istar12 = quant_star(Istar12, net2['conv3_1.scale'].numpy(), net2['conv3_1.zero_point'].numpy())
    Istar13 = l_conv3_2.reach(Istar12)
    Istar14 = l_relu.reach(Istar13, method)

    Istar14 = quant_star(Istar14, net2['conv3_2.scale'].numpy(), net2['conv3_2.zero_point'].numpy())
    Istar15 = l_conv3_3.reach(Istar14)
    Istar16 = l_relu.reach(Istar15, method)
    Istar17 = l_pool3.reach(Istar16, method)

    Istar17 = quant_star(Istar17, net2['conv3_3.scale'].numpy(), net2['conv3_3.zero_point'].numpy())
    Istar18 = l_conv4_1.reach(Istar17)
    Istar19 = l_relu.reach(Istar18, method)

    Istar19 = quant_star(Istar19, net2['conv4_1.scale'].numpy(), net2['conv4_1.zero_point'].numpy())
    Istar20 = l_conv4_2.reach(Istar19)
    Istar21 = l_relu.reach(Istar20, method)

    Istar21 = quant_star(Istar21, net2['conv4_2.scale'].numpy(), net2['conv4_2.zero_point'].numpy())
    Istar22 = l_conv4_3.reach(Istar21)
    Istar23 = l_relu.reach(Istar22, method)
    Istar24 = l_pool4.reach(Istar23, method)

    Istar24 = quant_star(Istar24, net2['conv4_3.scale'].numpy(), net2['conv4_3.zero_point'].numpy())
    Istar25 = l_conv5_1.reach(Istar24)
    Istar26 = l_relu.reach(Istar25, method)

    Istar26 = quant_star(Istar26, net2['conv5_1.scale'].numpy(), net2['conv5_1.zero_point'].numpy())
    Istar27 = l_conv5_2.reach(Istar26)
    Istar28 = l_relu.reach(Istar27, method)

    Istar28 = quant_star(Istar28, net2['conv5_2.scale'].numpy(), net2['conv5_2.zero_point'].numpy())
    Istar29 = l_conv5_3.reach(Istar28)
    Istar30 = l_relu.reach(Istar29, method)
    Istar31 = l_pool5.reach(Istar30, method)

    Istar31 = quant_star(Istar31, net2['conv5_3.scale'].numpy(), net2['conv5_3.zero_point'].numpy())
    Istar32 = l_fc1.reach(Istar31)
    Istar33 = l_relu.reach(Istar32, method)

    Istar33 = quant_star(Istar33, net2['fc1.scale'].numpy(), net2['fc1.zero_point'].numpy())
    Istar34 = l_fc2.reach(Istar33)
    Istar35 = l_relu.reach(Istar34, method)

    Istar35 = quant_star(Istar35, net2['fc2.scale'].numpy(), net2['fc2.zero_point'].numpy())
    Istar36 = l_fc3.reach(Istar35)

    Istar36 = quant_star(Istar36, net2['fc3.scale'].numpy(), net2['fc3.zero_point'].numpy())
    Istar37 = l_fc4.reach(Istar36)

    diff_list = err_range(Istar37)
    return diff_list


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    small_path = './cifar_vgg_s.pth'
    large_path = './cifar_vgg_l.pth'
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for iter in range(1):
        index = np.random.randint(0, 1000)
        # index_s = str(index).zfill(4)
        index_s = '0004'
        image_path = 'D:/Cody/phd/research1/CIFAR-10-images/test/airplane/' + index_s + '.jpg'
        # test on one image set
        IM = imageio.v2.imread(image_path)
        IM = IM / 255
        IM = (IM - 0.5) / 0.5
        lb = -0.05
        ub = 0.05
        max_range = compute_net_diff(large_path, small_path, IM, lb, ub)
        fig = plt.Figure()
        x = np.linspace(0, 9, 10, dtype=int)
        max_diff_total = 0

        net1 = Net1()
        net1.load_state_dict(torch.load(large_path, map_location=torch.device('cpu')))  # load whole network
        fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                      ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                      ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                      ['conv5_3', 're5_3']]
        net2 = Net2()
        net2.eval()
        net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
        net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
        net2 = torch.quantization.convert(net2.eval(), inplace=True)
        net2.load_state_dict(torch.load(small_path, map_location=torch.device('cpu')))

        with torch.no_grad():
            outputs1 = net1(torch.from_numpy(IM.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32))
            upp_list = np.zeros(10)
            low_list = np.zeros(10)
            for l in range(10):
                upp_list[l] = (outputs1[0, l] - max_range[l][0]).squeeze()
                low_list[l] = (outputs1[0, l] - max_range[l][1]).squeeze()
                plt.plot([l, l], [low_list[l], upp_list[l]], linewidth=1, color='r')
                plt.plot(l, upp_list[l], marker='_', markersize=15, color='r')
                plt.plot(l, low_list[l], marker='_', markersize=15, color='r')
                plt.plot(x[l], outputs1[0, l], color='b', marker='.')

            for i in range(500):
                noise = lb + (ub - lb) * np.random.random()
                IM_sample = IM.copy()
                IM_sample[0, 0, 0] += noise

                outputs2 = net2(torch.from_numpy(IM_sample.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32))

        plt.xticks(x, classes)
        plt.title('Cifar10 Experiment')
        plt.xlabel('Label')
        plt.ylabel('Ranges')
        plt.show()
