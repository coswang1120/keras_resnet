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
SHORTCUT_OPTION = 'B'

def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=1)


def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 4D tensors
    shape[1] *= 2
    return tuple(shape)


def featuremap_reduction_shortcut(input_layer, nb_filters,
                                  option='B'):
    """Used to increase dimensions and reduce feature map size

    For example: shortcut(?, 16, 32, 32) -> (?, 32, 16, 16)

    Parameters
    ----------
    option : 'A'/'B'

        A: identity shortcut with zero-padding for increasing dimensions.
           This is used for all CIFAR-10 experiments.
        B: identity shortcut with 1x1 convolutions for increasing dimensions
           This is used for most ImageNet experiments.
    """
    if option == 'A':
        # TODO: Figure out why zeros_upsample doesn't work in Theano
        x = MaxPooling2D(pool_size=(1, 1),
                         strides=(2, 2),
                         border_mode='same')(input_layer)
        x = Lambda(zeropad, output_shape=zeropad_output_shape)(x)
    elif option == 'B':
        x = BatchNormalization()(input_layer)
        x = Convolution2D(nb_filter=nb_filters, nb_col=1, nb_row=1,
                          subsample=(2, 2),
                          border_mode='same',
                          bias=False)(x)
    else:
        # My style: Take a 1x1 convolution over entire image with 1/4 the
        # filters and reshapes to the desired output shape with 2x the
        # number of filters. Works with old versions of TensorFlow where
        # the stride cannot be larger than the filter size.
        x = Convolution2D(nb_filter=nb_filters,
                          nb_row=1, nb_col=1,
                          W_regularizer=l2(WEIGHT_DECAY),
                          border_mode='same')(input_layer)
        x = BatchNormalization()(x)
    return x


def basic_unit(input, nb_filters, first_stride=(1, 1)):
    """Add a residual building block consisting of 2 convolutions

    A residual block consists of 2 base convolutions with a short/identity
    connection between the input and output activation
    """
    x = Convolution2D(nb_filters, 3, 3,
                      W_regularizer=l2(WEIGHT_DECAY),
                      subsample=first_stride,
                      border_mode='same',
                      init='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Convolution, with Batch Normalization, without ReLU activation
    x = Convolution2D(nb_filters, 3, 3,
                      W_regularizer=l2(WEIGHT_DECAY),
                      border_mode='same',
                      init='he_normal')(x)
    x = BatchNormalization()(x)

    if first_stride == (2, 2):
        input = featuremap_reduction_shortcut(input, nb_filters,
                                              option=SHORTCUT_OPTION)

    x = merge(inputs=[x, input], mode='sum')
    x = Activation('relu')(x)

    return x


def bottleneck_unit(input, nb_filters, first_stride=(1, 1)):
    """The bottleneck unit comprising 1x1 -> 3x3 -> 1x1 convolutions

    This is the new pre-activation residual unit, Batch Norm -> ReLU -> Conv
    http://arxiv.org/abs/1603.05027
    """
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters, 1, 1,
                      W_regularizer=l2(WEIGHT_DECAY),
                      subsample=first_stride,
                      border_mode='same',
                      init='he_normal')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters, 3, 3,
                      W_regularizer=l2(WEIGHT_DECAY),
                      border_mode='same',
                      init='he_normal')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(4*nb_filters, 1, 1,
                      W_regularizer=l2(WEIGHT_DECAY),
                      border_mode='same',
                      init='he_normal')(x)

    if first_stride == (2, 2):
        input = featuremap_reduction_shortcut(input_layer=input,
                                              nb_filters=4*nb_filters,
                                              option='B')

    x = merge(inputs=[x, input], mode='sum')
    x = Activation('relu')(x)

    return x


def stack_units(input, block_unit, nb_blocks, nb_filters, stride=(1, 1)):
    x = block_unit(input=input, nb_filters=nb_filters,
                   first_stride=stride)

    for _ in range(nb_blocks - 1):
        x = block_unit(input=x, nb_filters=nb_filters)

    return x


def build_residual_imagenet(nb_blocks=[1, 3, 4, 6, 3],
                            input_shape=(3, 224, 224),
                            initial_nb_filters=64):
    """Construct a residual network with ImageNet architecture.

    Parameters
    ----------
    nb_blocks : list
       The number of residual blocks for each layer group. For the 18-layer
       model nb_blocks=[1,2,2,2,2] and 34-layer nb_blocks=[1,3,4,6,3].
    initial_nb_filters : int, optional
       The initial number of filters to use. The number of filters is doubled
       for each layer.

    Returns
    -------
    input_image, output : input tensor, output class probabilities


    input: 224x224 RGB image

    layer name      output size     50-layer
    conv1           112x112      7x7, 64, stride 2
                    56x56        3x3 max pool, stride 2
    conv2_x         56x56           [1x1, 64]
                                    [3x3, 64]  x3
                                    [1x1, 128]
    conv3_x         28x28           [1x1, 128]
                                    [3x3, 128] x4
                                    [1x1, 256]
    conv4_x         14x14           [1x1, 256] x6
                                    [3x3, 256]
                                    [1x1, 1024]
    conv5_x         7x7             [1x1, 1024]
                                    [3x3, 1024] x3
                                    [1x1, 2048]
                    1x1       average pool, 1000-d fc, softmax

    Reference: http://arxiv.org/abs/1512.03385
    """

    # ------------------------------ Unit Group 1 -----------------------------
    input_image = Input(shape=input_shape)
    x = Convolution2D(initial_nb_filters, 7, 7, subsample=(2, 2),
                      border_mode='same')(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Output shape = (None,64,112,112)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)
    x = Convolution2D(4*initial_nb_filters, 1, 1, bias=False)(x)
    # Output shape = (None,64,56,56)
    # Output size = 56x56

    # ------------------------------ Unit Group 2 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters,
                    stride=(1, 1))
    # Output size = 56x56

    # ------------------------------ Unit Group 3 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters * 2,
                    stride=(2, 2))
    # Output size = 28x28

    # ------------------------------ Unit Group 4 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters * 4,
                    stride=(2, 2))
    # Output size = 14x14
    # ------------------------------ Unit Group 5 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters * 8,
                    stride=(2, 2))
    # Output size = 7x7

    pool_size = x._keras_shape[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size), border_mode='same')(x)
    # Output size = 1x1
    x = Flatten()(x)
    output = Dense(1000, activation='sigmoid')(x)

    return input_image, output


if __name__ == '__main__':
    input_tensor, output_tensor = build_residual_imagenet(initial_nb_filters=64,
                                                          nb_blocks=[1, 3, 4, 6, 3],
                                                          input_shape=(3, 224, 224))

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # model.summary()
    model.to_json()