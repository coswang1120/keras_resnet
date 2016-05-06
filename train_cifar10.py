from __future__ import division
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras_models import base_convolution, basic_block, stack_units


def build_residual_network(nb_blocks=[1, 3, 3, 3], input_shape=(3, 32, 32),
                           initial_nb_filters=16,
                           first_conv_shape=(3, 3)):
    """Construct a residual network model for CIFAR10.

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


    From the paper: input image 32x32 RGB image
    layer name      output size
    conv1           32x32           3x3, 16, stride 1
    conv2_x         32x32           3x3, 32, stride 2
    conv3_x         16x16           3x3, 32, stride 2
    conv4_x         8x8             8x8, average pool
    pool            1x1             10-d fc, softmax

    Reference: http://arxiv.org/abs/1512.03385
    """

    # -------------------------- Layer Group 1 ----------------------------
    input_image = Input(shape=input_shape)
    x = base_convolution(input=input_image, nb_filters=initial_nb_filters,
                         conv_shape=first_conv_shape)
    # Output size = 32x32
    # -------------------------- Layer Group 2 ----------------------------
    x = stack_units(input=x, block_unit=basic_block, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters)
    # Output size = 32x32
    # -------------------------- Layer Group 3 ----------------------------
    x = stack_units(input=x, block_unit=basic_block, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters * 2,
                    stride=(2, 2))
    # Output size = 16x16
    # -------------------------- Layer Group 4 ----------------------------
    x = stack_units(input=x, block_unit=basic_block, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters * 4,
                    stride=(2, 2))
    # Output size = 8x8

    pool_size = x._keras_shape[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size))(x)
    # Output size = 1x1
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    return input_image, output


if __name__ == '__main__':
    nb_classes = 10
    input_tensor, output_tensor = build_residual_network(initial_nb_filters=16,
                                                         nb_blocks=[1, 5, 5, 5],
                                                         first_conv_shape=(3,3),
                                                         input_shape=(3,32,32))

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # history = model.fit(X_train, y_train,
    #                     validation_data=(X_test, y_test),
    #                     batch_size=128)

    model.summary()
