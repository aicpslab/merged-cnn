import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
import onnx
import onnxruntime
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    pth1 = './numer1.pth'
    net1 = Net1()
    net1.load_state_dict(torch.load(pth1))
    net1.eval()
    IM = np.random.rand(6, 6, 1).astype(np.single)
    input_x = torch.from_numpy(IM.transpose(2, 0, 1)).unsqueeze(0)
    output_y = net1(input_x)
    # torch.onnx.export(net1, input_x, 'numer1_onnx.onnx', export_params=True, opset_version=10,
    #                   do_constant_folding=True, input_names=['input_x'], output_names=['output_y'])

    onnx_model = onnx.load("./numer1_onnx.onnx")
    onnx.checker.check_model(onnx_model)
    # print(onnx_model)

    for i in range(len(onnx_model.graph.node)):
        Node = onnx_model.graph.node[i]
        input_name = onnx_model.graph.node[i].input
        print(Node)

    x = torch.randn(1, 1, 6, 6, requires_grad=True)
    # ort_session = onnxruntime.InferenceSession('./numer1_onnx.onnx')
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # np.testing.assert_allclose(to_numpy(output_y), ort_outs[0], rtol=1e-03, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # # Create one input (ValueInfoProto)
    # X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    # pads = helper.make_tensor_value_info("pads", TensorProto.FLOAT, [1, 4])
    #
    # value = helper.make_tensor_value_info("value", AttributeProto.FLOAT, [1])
    #
    # # Create one output (ValueInfoProto)
    # Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
    #
    # # Create a node (NodeProto) - This is based on Pad-11
    # node_def = helper.make_node(
    #     "Pad",  # name
    #     ["X", "pads", "value"],  # inputs
    #     ["Y"],  # outputs
    #     mode="constant",  # attributes
    # )
    #
    # # Create the graph (GraphProto)
    # graph_def = helper.make_graph(
    #     [node_def],  # nodes
    #     "test-model",  # name
    #     [X, pads, value],  # inputs
    #     [Y],  # outputs
    # )
    #
    # model_def = helper.make_model(graph_def, producer_name="onnx-example")
    #
    # print(f"The model is:\n{model_def}")
