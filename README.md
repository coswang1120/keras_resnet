# A Residual net in Keras

Here lies an implementation of the residual model from
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

I use Keras 1.0.1 and a recent nightly build of TensorFlow.

The CIFAR10 model compiles and runs but at the moment I'm not doing anything with it.
Where the paper is unclear I follow the open source [Torch resnet](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua).

The full ImageNet model is there too but if you want to train it that's your problem.

TODOs:

1. Bottleneck architecture

2. Compare CIFAR10 results with the results in the paper

3. Fix ImageNet model
