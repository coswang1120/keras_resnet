from __future__ import division

from keras.layers import Input
from keras.layers.convolutional import AveragePooling2D, Convolution2D
from keras.layers.core import Dense, Flatten


from keras_resnet.resnet_utils import basic_unit, bottleneck_unit, stack_units

def build_cifar_model(nb_blocks=[1, 3, 3, 3], input_shape=(3, 32, 32),
                      initial_nb_filters=16,
                      nb_classes=10):
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
    x = Convolution2D(initial_nb_filters, 3, 3, border_mode='same')(input_image)
    # Output size = 32x32

    # ------------------------------ Unit Group 2 -----------------------------
    x = Convolution2D(4*initial_nb_filters, 1, 1, bias=False)(x)  # For bottleneck
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[1],
                    nb_filters=initial_nb_filters)
    # Output size = 32x32

    # ------------------------------ Unit Group 3 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[2],
                    nb_filters=2*initial_nb_filters,
                    stride=(2, 2))
    # Output size = 16x16

    # ------------------------------ Unit Group 4 -----------------------------
    x = stack_units(input=x, block_unit=bottleneck_unit, nb_blocks=nb_blocks[3],
                    nb_filters=4*initial_nb_filters,
                    stride=(2, 2))
    # Output size = 8x8

    pool_size = x._keras_shape[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size))(x)
    # Output size = 1x1
    x = Flatten()(x)
    output = Dense(nb_classes, activation='softmax')(x)

    return input_image, output


if __name__ == '__main__':
    from keras.datasets import cifar10
    from keras.utils import np_utils
    from keras.models import Model
    from keras.optimizers import SGD

    NB_CLASSES = 10
    input_tensor, output_tensor = build_cifar_model(initial_nb_filters=16,
                                                    nb_blocks=[1, 2, 2, 2],
                                                    input_shape=(3, 32, 32),
                                                    nb_classes=NB_CLASSES)

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.to_json())
    # model.summary()
    #
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    #
    # # convert class vectors to binary class matrices
    # y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    # y_test = np_utils.to_categorical(y_test, NB_CLASSES)
    #
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    #
    #
    # history = model.fit(X_train, y_train,
    #                     validation_data=(X_test, y_test),
    #                     batch_size=128)
