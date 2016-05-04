from keras_models import base_convolution, basic_block


def build_residual_network(nb_blocks=[1, 3, 4, 6, 3],
                           input_shape=(3, 224, 224),
                           initial_nb_filters=64,
                           first_conv_shape=(7, 7),
                           first_stride=(1, 1)):
    """Construct a residual convolutional network graph from scratch.

    Parameters
    ----------
    nb_blocks : list
       The number of residual blocks for each layer group. For the 18-layer
       model nb_blocks=[1,2,2,2,2] and 34-layer nb_blocks=[1,3,4,6,3].
    initial_nb_filters : int, optional
       The initial number of filters to use. The number of filters is doubled
       for each layer.
    first_conv_shape : tuple of ints
       The shape of the first convolution, also known as the kernel size.

    Returns
    -------
    self.graph : A new Keras graph

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
                         stride=first_stride)
    # Output shape = (None,16,112,112)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)
    # Output shape = (None,initial_nb_filters,56,56)
    # -------------------------- Layer Group 2 ----------------------------
    for i in range(1, nb_blocks[1] + 1):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters)
    # self.graph.nodes[output_name] = (None,initial_nb_filters,56,56)
    # output size = 14x14
    # -------------------------- Layer Group 3 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 2,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[2]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 2)
    # -------------------------- Layer Group 4 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 4,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[3]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 4)
    # output size = 14x14
    # -------------------------- Layer Group 5 ----------------------------
    x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 8,
                    first_stride=(2, 2))
    for _ in range(1, nb_blocks[4]):
        x = basic_block(input_layer=x, nb_filters=initial_nb_filters * 8)
    # output size = 7x7

    pool_size = x.get_shape().as_list()[-2:]
    x = AveragePooling2D(pool_size=tuple(pool_size), border_mode='same')(x)
    x = Flatten()(x)
    output_tensor = Dense(1000, activation='softmax')(x)

    return input_image, output_tensor


if __name__ == '__main__':
    nb_classes = 1000
    input_tensor, output_tensor = build_residual_network(initial_nb_filters=64,
                                                         nb_blocks=[1, 9, 9, 9, 9],
                                                         first_conv_shape=(7, 7),
                                                         first_stride=(2, 2),
                                                         input_shape=(3, 224, 224))

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

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

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=256)
