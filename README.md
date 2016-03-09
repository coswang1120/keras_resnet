# keras_resnet

Here lies an implementation of the 18-layer model from
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385). I used *Keras* with a
graph model which facilitates the identity connections at the heart of ResNets.

This is the skeleton of another project I'm working on...so there's
nothing to run just yet.  

My TODOs:

1. Build a graph factory to construct a graph with an arbitrary number
of layers.  

2. Add image classification datasets to work with

  - https://www.cs.toronto.edu/~kriz/cifar.html

  - http://www.vision.caltech.edu/visipedia/CUB-200.html

3. Implement zero padding identity residual connections across layers with dimension changes (when stride=2)
