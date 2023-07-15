import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys

sys.path.append("..")

from engine.layer import MergedConv2d
from engine.layer import MaxPooling2d
from engine.layer import FC
from engine.layer import ReLULayer
from engine.imageStar import ImageStar


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
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


def plot_result(net1, net2):
    x = np.linspace(0, 20, 20, dtype=int)
    y = np.zeros(20, )
    for idx, val in enumerate(x):
        input_x = np.random.rand(6, 6).astype(np.single)
        input_x = torch.from_numpy(input_x).unsqueeze(0).unsqueeze(0)
        output = net1(input_x)
        y[idx] = output
    plt.figure()
    plt.plot(x, y)
    for idx, val in enumerate(x):
        input_x = np.random.rand(6, 6).astype(np.single)
        input_x = torch.from_numpy(input_x).unsqueeze(0).unsqueeze(0)
        output = net2(input_x)
        y[idx] = output
    plt.plot(x, y)
    plt.show()


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


def compute_net_diff(result_path1, result_path2, IM, lb, ub):
    net1 = torch.load(result_path1, map_location=torch.device('cpu'))  # only load for parameter reading
    net2 = torch.load(result_path2, map_location=torch.device('cpu'))

    conv1_weight1 = net1['conv1.weight'].to(torch.float32)
    conv1_bias1 = net1['conv1.bias'].to(torch.float32)
    conv2_weight1 = net1['conv2.weight'].to(torch.float32)
    conv2_bias1 = net1['conv2.bias'].to(torch.float32)
    fc1_weight1 = net1['fc1.weight'].to(torch.float32)
    fc1_bias1 = net1['fc1.bias'].to(torch.float32)

    conv1_weight2 = net2['conv1.weight'].to(torch.float32)
    conv1_bias2 = net2['conv1.bias'].to(torch.float32)
    conv2_weight2 = net2['conv2.weight'].to(torch.float32)
    conv2_bias2 = net2['conv2.bias'].to(torch.float32)
    fc1_weight2 = net2['fc1.weight'].to(torch.float32)
    fc1_bias2 = net2['fc1.bias'].to(torch.float32)

    method = 'exact-star'
    poolsize = np.array([2, 2])
    stride = np.array([2, 2])
    p_padding = np.array([0, 0])

    LB = np.zeros((6, 6, 1), dtype=np.single)
    UB = np.zeros((6, 6, 1), dtype=np.single)
    LB[0:2, 0:2, :] = lb
    UB[0:2, 0:2, :] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    l_conv1 = MergedConv2d(conv1_weight1, conv1_bias1, conv1_weight2, conv1_bias2)
    l_conv2 = MergedConv2d(conv2_weight1, conv2_bias1, conv2_weight2, conv2_bias2)
    l_pool1 = MaxPooling2d(poolsize, stride, p_padding, 'maxpool1')
    l_fc1 = FC(fc1_weight1, fc1_bias1, fc1_weight2, fc1_bias2)
    fc2_weight1 = np.eye(1, dtype=np.single)
    fc2_weight2 = -np.eye(1, dtype=np.single)
    fc2_b1 = np.zeros((1, 2))
    fc2_b2 = np.zeros((1, 2))
    l_fc2 = FC(fc2_weight1, fc2_b1, fc2_weight2, fc2_b2, 'last')
    l_relu = ReLULayer()

    Istar1 = l_conv1.reach(I1_m)
    Istar2 = l_relu.reach(Istar1, method)
    Istar3 = l_conv2.reach(Istar2)
    Istar4 = l_relu.reach(Istar3, method)
    Istar5 = l_pool1.reach(Istar4, method)
    Istar6 = l_fc1.reach(Istar5)
    Istar7 = l_fc2.reach(Istar6)

    return err_range(Istar7)


if __name__ == '__main__':
    pth1 = '../data/numer1.pth'
    pth2 = '../data/numer2.pth'
    net1 = Net1()
    net2 = Net2()
    net1.load_state_dict(torch.load(pth1))
    net2.load_state_dict(torch.load(pth2))
    fig = plt.figure()
    sample_n = 10
    x = np.linspace(1, sample_n, sample_n, dtype=int)
    upp_list = np.zeros(10)
    low_list = np.zeros(10)
    for iter in range(sample_n):
        IM = np.random.rand(6, 6, 1).astype(np.single)
        lb = -0.05
        ub = 0.05
        max_range = compute_net_diff(pth1, pth2, IM, lb, ub)
        with torch.no_grad():
            IM_input = torch.from_numpy(IM.transpose(2, 0, 1)).unsqueeze(0)
            output1 = net1(IM_input)
            upp_list[iter] = output1.squeeze().numpy() - max_range[0][0]
            low_list[iter] = output1.squeeze().numpy() - max_range[0][1]
            for i in range(20):
                noise = 0.1 * np.random.rand(2, 2) - 0.05
                IM_n = IM.copy()
                IM_n[0:2, 0:2, 0] += noise
                output2 = net2(torch.from_numpy(IM_n.transpose(2, 0, 1)).unsqueeze(0))
                plt.plot(x[iter], output2, color='k', marker='.', markersize=4)
    plt.plot(x, upp_list, color='b', linestyle='solid')
    plt.plot(x, low_list, color='g', linestyle='solid')
    plt.fill_between(x, low_list, upp_list, color='yellow', alpha=0.2)
    plt.title('Numerical Example')
    plt.xlabel('Index of Sample')
    plt.ylabel('Ranges')
    plt.show()
