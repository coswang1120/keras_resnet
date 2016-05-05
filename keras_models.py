from __future__ import division
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape, Lambda
from keras.layers import merge, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import backend as K


WEIGHT_DECAY = 0.0001


def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=1)


def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 2D tensors
    shape[1] *= 2
    return tuple(shape)


def base_convolution(input, nb_filters, conv_shape=(3, 3), stride=(1, 1),
                     relu_activation=True, **kwargs):
    """Convolution2D -> BatchNormalization -> ReLU"""

    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=conv_shape[0], nb_col=conv_shape[1],
                      W_regularizer=l2(WEIGHT_DECAY),
                      subsample=stride,
                      border_mode='same',
                      init='he_normal',
                      **kwargs)(input)

    x = BatchNormalization()(x)
    if relu_activation:
        x = Activation('relu')(x)

    return x


def shortcut(input_layer, nb_filters, output_shape, zeros_upsample=True):
    """Used to increase dimensions, ie 16 filters to 32 filters.

    Parameters
    ----------
    zeros_upsample : Bool  (for now)

        A: identity shortcut with zero-padding for increasing dimensions. This is used for all CIFAR-10 experiments.
        B: identity shortcut with 1x1 convolutions for increasing dimesions. This is used for most ImageNet experiments.
        C: 1x1 convolutions for all shortcut connections.
    """
    if zeros_upsample:
        # TODO: Figure out why zeros_upsample doesn't work in Theano
        # TODO: Change to A/B options
        # Option A: pad with zeros
        x = MaxPooling2D(pool_size=(1,1),
                         strides=(2,2),
                         border_mode='same')(input_layer)
        x = Lambda(zeropad, output_shape=zeropad_output_shape)(x)
    else:
        # My style: Take a 1x1 convolution over entire image with 1/4 the
        # filters and reshapes to the desired output shape with 2x the
        # number of filters. Works with old versions of TensorFlow where
        # the stride cannot be larger than the filter size.
        x = Convolution2D(nb_filter=nb_filters//4,
                          nb_row=1, nb_col=1,
                          W_regularizer=l2(WEIGHT_DECAY),
                          border_mode='same')(input_layer)
        x = BatchNormalization()(x)
        x = Reshape(output_shape[1:])(x)
    return x


def basic_block(input_layer, nb_filters, first_stride=(1, 1)):
    """Add a residual building block

    A residual block consists of 2 base convolutions with a short/identity
    connection between the input and output activation

    Parameters
    ----------
    input_layer : name of input node
    nb_filters : int

    Returns
    -------
    output_name : name of output node, string
    """

    # First convolution
    x = base_convolution(input=input_layer, nb_filters=nb_filters,
                         stride=first_stride)
    output_shape = x._keras_shape

    # Second Convolution, with Batch Normalization, without ReLU activation
    x = base_convolution(input=x, nb_filters=nb_filters, stride=(1, 1),
                         relu_activation=False)

    # Add the short convolution, with Batch Normalization
    if first_stride == (2, 2):
        input_layer = shortcut(input_layer, nb_filters, output_shape)

    x = merge(inputs=[x, input_layer], mode='sum')
    x = Activation('relu')(x)

    return x


def stack_units(block_unit, input_layer, nb_blocks, nb_filters, stride=(1, 1)):
    x = block_unit(input_layer=input_layer, nb_filters=nb_filters,
                   first_stride=stride)

    for _ in range(nb_blocks-1):
        x = block_unit(input_layer=x, nb_filters=nb_filters)
    return x


def build_residual_imagenet(nb_blocks=[1, 3, 4, 6, 3],
                            input_shape=(3, 224, 224),
                            initial_nb_filters=64,
                            first_conv_shape=(7, 7)):
    """Construct a residual network with ImageNet architecture.

    Parameters
    ----------
    nb_blocks : list
       The number of residual blocks for each layer group. For the 18-layer
       model nb_blocks=[1,2,2,2,2] and 34-layer nb_blocks=[1,3,4,6,3].
    initial_nb_filters : int, optional
       The initial number of filters to use. The number of filters is doubled
       for each layer.
    first_conv_shape : tuple of 2 ints
       The shape of the first convolution, also known as the kernel size.

    Returns
    -------
    input_image, output : input tensor, output class probabilities


    From the paper: input image 224x224 RGB image

    layer name      output size     18-layer        34-layer
    conv1           112x112      7x7, 64, stride 2 -> 3x3 max pool, stride 2
    conv2_x         56x56           [3x3, 64]x2     [3x3, 64]x3
                                    [3x3, 64]       [3x3, 64]
    conv3_x         28x28           [3x3, 128]x2    [3x3, 128]x4
                                    [3x3, 128]      [3x3, 128]
    conv4_x         14x14           [3x3, 256]x2    [3x3, 256]x6
                                    [3x3, 256]      [3x3, 256]
    conv5_x         7x7             [3x3, 512]x2    [3x3, 512]x3
                                    [3x3, 512]      [3x3, 512]
                    1x1          average pool, 1000-d fc, softmax

    Reference: http://arxiv.org/abs/1512.03385
    """

    # -------------------------- Layer Group 1 ----------------------------
    input_image = Input(shape=input_shape)
    x = base_convolution(input=input_image, nb_filters=initial_nb_filters,
                         conv_shape=first_conv_shape,
                         stride=(2, 2))
    # Output shape = (None,64,112,112)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)
    # Output shape = (None,64,56,56)
    # Output size = 56x56
    # -------------------------- Layer Group 2 ----------------------------
    for i in range(1, nb_blocks[1] + 1):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters)
    # Output size = 56x56
    # -------------------------- Layer Group 3 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 2,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[2]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 2)
    # Output size = 28x28
    # -------------------------- Layer Group 4 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 4,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[3]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 4)
    # Output size = 14x14
    # -------------------------- Layer Group 5 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 8,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[4]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 8)
    # Output size = 7x7

    pool_size = x.get_shape().as_list()[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size), border_mode='same')(x)
    # Output size = 1x1
    x = Flatten()(x)
    output = Dense(10, activation='sigmoid')(x)

    return input_image, output


if __name__ == '__main__':
    nb_classes = 10
    input_tensor, output_tensor = build_residual_imagenet(initial_nb_filters=16,
                                                          nb_blocks=[1, 9, 9, 9, 9],
                                                          first_conv_shape=(3,3),
                                                          input_shape=(3,32,32))

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    model.summary()