# MergedCNN
 This Python tool (in development) provides reachability-analysis-based discrepancy computation between two similar convolution neural networks. The operation supported by this tool currently: convolution, max pooling, fully connected, and ReLU.
 # Related tools and software
 The network structure and computation are based on PyTorch. The reachability representation of images is based on ImageStar in NNV tool ([NNV](https://github.com/verivital/nnv))
# Installation:
1) Clone from [Merged-CNN](https://github.com/aicpslab/merged-cnn/)
2) Install all required dependencies.
3) Run experiment1.py in the example folder.
# Features
1) This tool can compute the discrepancy between two convolution neural networks. In the example, the VGG16 structure is used to build the original network, and the network is trained with CIFAR10 dataset. The compression network is compressed with the quantization method (QAT in PyTorch). The computed error range between the original network and the compression network can be used to verify the robustness of the compression network.
<figure>
  <img src="/results/fig exp2.png" width="600"> <figcaption>Original network output and guarantee output range of quantization network.</figcaption>
</figure>
