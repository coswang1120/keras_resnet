# A Residual net in Keras

Here lies an implementation of the residual model from
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

This uses Keras 1.0.1, TensorFlow 0.8.0 or Theano 0.9.0

The CIFAR10 model compiles and trains with `train_cifar10.py`, it uses the basic block.

The latest Theano allows strides greater than the pool size but TensorFlow does not.

TODOs:

1. Bottleneck residual architecture

3. Implement zero padding identity connections across layers with dimension changes (when stride=2)

4. Compare CIFAR results with the results in the paper
