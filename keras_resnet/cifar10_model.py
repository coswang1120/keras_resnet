from __future__ import division
from keras.layers import Input
from keras.layers.convolutional import AveragePooling2D, Convolution2D
from keras.layers.core import Dense, Flatten
from keras.regularizers import l2

from keras_resnet.resnet_utils import basic_unit, bottleneck_unit, stack_units
from keras_resnet.resnet_utils import WEIGHT_DECAY


def build_cifar_model(nb_blocks=[1, 3, 3, 3], input_shape=(3, 32, 32),
                      initial_nb_filters=16,
                      nb_classes=10,
                      residual_unit=basic_unit):
    """Construct a residual network model for CIFAR10.

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


    From the paper: input image 32x32 RGB image
    layer name      output size
    conv1           32x32           3x3, 16, stride 1
    conv2_x         32x32           3x3, 32, stride 2
    conv3_x         16x16           3x3, 32, stride 2
    conv4_x         8x8             8x8, average pool
    pool            1x1             10-d fc, softmax

    Reference: http://arxiv.org/abs/1512.03385
    """

    # ------------------------------ Unit Group 1 -----------------------------
    input_image = Input(shape=input_shape)
    x = Convolution2D(initial_nb_filters, 3, 3,
                      W_regularizer=l2(WEIGHT_DECAY),
                      border_mode='same',
                      init='he_normal')(input_image)
    # Output size = 32x32

    # ------------------------------ Unit Group 2 -----------------------------
    # x = Convolution2D(4*initial_nb_filters, 1, 1,
    #                   W_regularizer=l2(WEIGHT_DECAY),
    #                   init='he_normal',
    #                   bias=False)(x)  # For bottleneck
    x = stack_units(input=x, block_unit=residual_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters)
    # Output size = 32x32

    # ------------------------------ Unit Group 3 -----------------------------
    x = stack_units(input=x, block_unit=residual_unit, nb_blocks=nb_blocks[2],
                    nb_filters=2*initial_nb_filters,
                    stride=(2, 2))
    # Output size = 16x16

    # ------------------------------ Unit Group 4 -----------------------------
    x = stack_units(input=x, block_unit=residual_unit, nb_blocks=nb_blocks[3],
                    nb_filters=4*initial_nb_filters,
                    stride=(2, 2))
    # Output size = 8x8

    pool_size = x._keras_shape[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size))(x)
    # Output size = 1x1
    x = Flatten()(x)
    output = Dense(nb_classes, W_regularizer=l2(WEIGHT_DECAY),
                   activation='softmax')(x)

    return input_image, output


if __name__ == '__main__':
    from keras.models import Model
    from keras.optimizers import SGD

    NB_CLASSES = 10
    input_tensor, output_tensor = build_cifar_model(initial_nb_filters=16,
                                                    nb_blocks=[1, 5, 5, 5],
                                                    input_shape=(3, 32, 32),
                                                    nb_classes=NB_CLASSES)

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print(model.to_json())
    model.summary()
