import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
import collections
import imageio
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)

import sys
sys.path.append("..//")

from engine.netload import compute_quant_net_diff, compute_net_diff


class VGG1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.refc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.refc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG2(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.refc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.refc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


def quant_repair(model, model_s, pth_l, pth_s, IM, lb, ub, new_pth, alpha, fx):
    # max_range = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    max_range = compute_quant_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    optimizer = optim.SGD(model_s.parameters(), lr=0.0001, momentum=0.9)
    loss_func = torch.nn.MSELoss()
    dis_list = []

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            outputs2 = model_s(inputs)
            labels = outputs2.squeeze() + (torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(torch.float32)) / alpha

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            outputs = model_s(inputs.to(torch.float32))
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # torch.save(model_s.state_dict(), new_pth)
        if fx:
            net_temp = copy.deepcopy(model_s)
            net_temp = quantize_fx.convert_fx(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            temp1 = pth_s_new['conv1_1_input_scale_0']
            temp2 = pth_s_new['conv1_1_input_zero_point_0']
            pth_s_new = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                            else (k, v) for k, v in pth_s_new.items()])
            pth_s_new = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                            else (k, v) for k, v in pth_s_new.items()])
        else:
            net_temp = copy.deepcopy(model_s)
            net_temp = torch.quantization.convert(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
        new_range = compute_quant_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
        new_range = np.array(new_range)
        dis = np.linalg.norm(np.mean(abs(new_range), 1))
        dis_list.append(dis)
        temp_range = new_range
        print(dis)

    return dis_list


def pruning_repair(model, model_s, pth_l, pth_s, IM, lb, ub, new_pth, alpha, p_method):
    # max_range = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    max_range = compute_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    # model_s.load_state_dict(torch.load(model_s_pth, map_location=torch.device('cpu')))
    optimizer = optim.SGD(model_s.parameters(), lr=0.0001, momentum=0.9)
    loss_func = torch.nn.MSELoss()
    dis_list = []

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            outputs2 = model_s(inputs)
            labels = outputs2.squeeze() + (
                torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(torch.float32)) / alpha

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            outputs = model_s(inputs.to(torch.float32))
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if p_method == 'local_unstru':
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'global_unstru':
            parameters_to_prune = ((model_s.conv1_1, 'weight'), (model_s.conv1_2, 'weight'), (model_s.conv2_1, 'weight'),
                                   (model_s.conv2_2, 'weight'), (model_s.conv3_1, 'weight'), (model_s.conv3_2, 'weight'),
                                   (model_s.conv3_3, 'weight'), (model_s.conv4_1, 'weight'), (model_s.conv4_2, 'weight'),
                                   (model_s.conv4_3, 'weight'), (model_s.conv5_1, 'weight'), (model_s.conv5_2, 'weight'),
                                   (model_s.conv5_3, 'weight'), (model_s.fc1, 'weight'), (model_s.fc2, 'weight'),
                                   (model_s.fc3, 'weight'))
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'l1_stru':
            for name, module in net2.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'rand_stru':
            for name, module in net2.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range

    return dis_list



if __name__ == '__main__':
    image_path = '../data/0022.jpg'
    IM = imageio.v2.imread(image_path)
    IM = IM / 255
    IM = (IM - 0.5) / 0.5
    lb = -0.05
    ub = 0.05

    print("FX Mode QAT")
    net1 = torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu'))
    net2 = torch.load('../data/cifar_vgg_fx_s.pth', map_location=torch.device('cpu'))
    temp1 = net2['conv1_1_input_scale_0']
    temp2 = net2['conv1_1_input_zero_point_0']
    net3 = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                    else (k, v) for k, v in net2.items()])
    net3 = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                    else (k, v) for k, v in net3.items()])
    net2 = VGG2()
    example_input = (torch.randn(1, 3, 32, 32))
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    net2 = quantize_fx.prepare_qat_fx(net2.train(), qconfig_mapping, example_input)
    net2.load_state_dict(torch.load('../data/cifar_vgg_fx_s_be.pth', map_location=torch.device('cpu')))

    dis_list = quant_repair(VGG2(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', 2, True)
    print(dis_list)

    print('Eager Mode QAT')
    net3 = torch.load('../data/cifar_vgg_s.pth', map_location=torch.device('cpu'))
    net2 = VGG2()
    net2.eval()
    net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                  ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                  ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                  ['conv5_3', 're5_3']]
    net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
    net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
    net2.load_state_dict(torch.load('../data/cifar_vgg_s_be.pth', map_location=torch.device('cpu')))
    dis_list = quant_repair(VGG2(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_s_temp.pth', 2, False)
    print(dis_list)

    print('Local Unstructured Pruning')
    net3 = torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu'))
    net2 = VGG1()
    net2.load_state_dict(torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu')))
    dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth', 2, 'local_unstru')
    print(dis_list)

    print('Global Unstructured Pruning')
    net3 = torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu'))
    net2 = VGG1()
    net2.load_state_dict(torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu')))
    dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth', 2, 'global_unstru')
    print(dis_list)

    print('Local Structured Pruning')
    net3 = torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu'))
    net2 = VGG1()
    net2.load_state_dict(torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu')))
    dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_strupru_s_temp.pth', 2, 'l1_stru')
    print(dis_list)

    print('Random Structured Pruning')
    net3 = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
    net2 = VGG1()
    net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
    dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_randstrupru_s_temp.pth', 2, 'rand_stru')
    print(dis_list)

    repair_list = [[16.3547, 28.9640, 23.2230, 14.8200, 8.4599, 9.0541], [17.0159, 18.1480, 11.3299, 6.9041, 4.2543, 2.2669],
                   [1.9602, 2.1744, 2.9405, 2.5069, 1.1136, 0.5915], [0.9487, 2.1489, 0.376, 0.2090, 0.1326, 0.1633],
                   [32.5631, 12.7388, 15.6981, 8.4703, 2.7437, 0.9242], [34.4265, 37.0091, 36.8323, 36.8606, 36.5052, 37.2667]]
    model_list = ['Eager QAT', 'FX QAT', 'L-Unstru', 'G-Unstru', 'L-Stru', 'R-Stru']
    x = [0, 3, 6, 9, 12, 15]
    plt.figure()
    plt.plot(x, repair_list[0], color='blue', label=model_list[0])
    plt.plot(x, repair_list[1], color='yellow', label=model_list[1])
    plt.plot(x, repair_list[2], color='red', label=model_list[2])
    plt.plot(x, repair_list[3], color='green', label=model_list[3])
    plt.plot(x, repair_list[4], color='purple', label=model_list[4])
    plt.plot(x, repair_list[5], color='orange', label=model_list[5])
    plt.legend()
    plt.xticks(np.array(x), x)
    plt.xlabel('Epoch')
    plt.ylabel('Discrepancy')
    plt.show()

