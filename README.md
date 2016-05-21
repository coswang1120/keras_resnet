# A residual network in Keras

Here's an implementation of the residual model from
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) and the updated [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027)
for [Keras](keras.io) 1.0.1 and a recent build of TensorFlow.

The CIFAR10 model compiles and runs but at the moment I'm not doing anything with it.
Where the paper is unclear I follow the open source [Torch ResNet](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua).
