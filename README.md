# A Residual net in Keras

Here lies an implementation of the 18/34-layer model from
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

This uses Keras 1.0.1 and TensorFlow 0.8.0

TODOs:

1. Test with CIFAR10 dataset

2. Bottleneck residual architecture

3. Implement zero padding identity connections across layers with dimension changes (when stride=2)

4. Compare CIFAR results with the results in the paper
