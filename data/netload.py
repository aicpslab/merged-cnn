import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary

from layer import MergedConv2d
from layer import MaxPooling2d
from layer import FC
from layer import ReLULayer
from imageStar import ImageStar


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def err_range(Istar):
    diff_list = []
    n = len(Istar)
    for i in range(n):
        Star1 = Istar[i].toStar()
        vertices = Star1.toPolyhedron()
        if i == 0:
            for ind in range(1):
                diff_list.append([min(vertices[ind, :]), max(vertices[ind, :])])
        else:
            for ind in range(1):
                diff_list[ind] = [min([diff_list[ind][0], min(vertices[ind, :])]),
                                  max([diff_list[ind][1], max(vertices[ind, :])])]
    return diff_list


if __name__ == '__main__':
    pth1 = './numer1.pth'
    pth2 = './numer2.pth'
    net1_model = Net1()
    net1 = torch.load(pth1)
    net2_model = Net2()
    net2 = torch.load(pth2)
    # print(net1_model)
    summary(net1_model, (1, 6, 6))

    IM = np.random.rand(6, 6, 1).astype(np.single)
    lb = -0.05
    ub = 0.05

    LB = np.zeros((6, 6, 1), dtype=np.single)
    UB = np.zeros((6, 6, 1), dtype=np.single)
    LB[0:2, 0:2, :] = lb
    UB[0:2, 0:2, :] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    name_list = []
    Istar_pre = I1_m
    p_i = 1
    method = 'exact-star'
    relu_layer = ReLULayer()
    for name, m in net1_model.named_children():
        print(name, ">>>", m)
        if isinstance(m, nn.Conv2d):
            c_wei1 = net1[name+'.weight'].to(torch.float32)
            c_bias1 = net1[name+'.bias'].to(torch.float32)
            c_wei2 = net2[name + '.weight'].to(torch.float32)
            c_bias2 = net2[name + '.bias'].to(torch.float32)
            layer = MergedConv2d(c_wei1, c_bias1, c_wei2, c_bias2)
            Istar = layer.reach(Istar_pre)
        elif isinstance(m, nn.MaxPool2d):
            pad = m.padding
            pad = np.array([pad, pad])
            stride = m.stride
            stride = np.array([stride, stride])
            pool = m.kernel_size
            pool = np.array([pool, pool])
            layer = MaxPooling2d(pool, stride, pad, 'maxpool'+str(p_i))
            Istar = layer.reach(Istar_pre, method)
            p_i += 1
        elif isinstance(m, nn.Linear):
            fc_wei1 = net1[name+'.weight'].to(torch.float32)
            fc_bias1 = net1[name+'.bias'].to(torch.float32)
            fc_wei2 = net2[name + '.weight'].to(torch.float32)
            fc_bias2 = net2[name + '.bias'].to(torch.float32)
            layer = FC(fc_wei1, fc_bias1, fc_wei2, fc_bias2)
            Istar = layer.reach(Istar_pre)
            output_size = m.out_features
        elif isinstance(m, nn.ReLU):
            Istar = relu_layer.reach(Istar, method)
        Istar_pre = Istar

    fc_last_w1 = np.eye(output_size, dtype=np.single)
    fc_last_w2 = -np.eye(output_size, dtype=np.single)
    fc_last_b1 = np.zeros(output_size)
    fc_last_b2 = np.zeros(output_size)
    layer = FC(fc_last_w1, fc_last_b1, fc_last_w2, fc_last_b2, 'last')
    Istar = layer.reach(Istar_pre)
    max_range = err_range(Istar)

