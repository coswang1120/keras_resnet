from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers import merge, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.regularizers import l2


class ResidualModel(object):
    """ Keras model and a method to create an arbitrary residual network."""

    def __init__(self, weight_decay=0.0001):
        self.weight_decay = weight_decay
        self.model = None

    def base_convolution(self, input_layer, nb_filters, conv_shape=(3, 3),
                         stride=(1, 1),
                         relu_activation=True,
                         **kwargs):
        """Convolution2D -> BatchNormalization -> ReLU

        :param nb_filters: number of filters
        :param input_layer: name of input
        """

        x = Convolution2D(nb_filter=nb_filters,
                          nb_row=conv_shape[0], nb_col=conv_shape[1],
                          W_regularizer=l2(self.weight_decay),
                          subsample=stride,
                          border_mode='same',
                          **kwargs)(input_layer)

        x = BatchNormalization()(x)
        if relu_activation:
            x = Activation('relu')(x)

        return x

    def residual_block(self, input_layer, nb_filters, first_stride=(1, 1)):
        """Add a residual building block

        A residual block consists of 2 base convolutions with a short/identity
        connection between the input and output activation

        Input:
        input_name: name of input node, string
        :type nb_filters: int
        :type input_layer: str

        Output:
        output_name: name of output node, string
        """

        # First convolution
        x = self.base_convolution(input_layer=input_layer, nb_filters=nb_filters,
                                  stride=first_stride)
        output_shape = x._shape_as_list()

        # Second Convolution, with Batch Normalization, without ReLU activation
        x = self.base_convolution(input_layer=x, nb_filters=nb_filters, stride=(1, 1),
                                  relu_activation=False)

        # Add the short convolution, with Batch Normalization
        if first_stride == (2, 2):
            input_layer = Convolution2D(nb_filter=nb_filters//4,
                                              nb_row=1,
                                              nb_col=1,
                                              W_regularizer=l2(self.weight_decay),
                                              border_mode='same')(input_layer)

            input_layer = BatchNormalization()(x)
            input_layer = Reshape(output_shape[1:])(x)

        x = merge(inputs=[x, input_layer], mode='sum')
        x = Activation('relu')(x)

        return x

    def build_residual_network(self, nb_blocks=[1,3,4,6,3],
                               initial_nb_filters=64,
                               first_conv_shape=(7, 7)):
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
        imsize = 224
        self.model = Sequential()
        # -------------------------- Layer Group 1 ----------------------------
        input_image = Input(shape=(3, imsize, imsize))
        x = self.base_convolution(input_layer=input_image,
                                  nb_filters=initial_nb_filters,
                                  conv_shape=first_conv_shape,
                                  stride=(2, 2))
        # Output shape = (None,16,112,112)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)
        # Output shape = (None,initial_nb_filters,56,56)
        # -------------------------- Layer Group 2 ----------------------------
        for i in range(1, nb_blocks[1]+1):
            x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters)
        # self.graph.nodes[output_name] = (None,initial_nb_filters,56,56)
        # output size = 14x14
        # -------------------------- Layer Group 3 ----------------------------
        x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters*2, first_stride=(2, 2))
        for i in range(1, nb_blocks[2]):
            x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters * 2)
        # -------------------------- Layer Group 4 ----------------------------
        x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters * 4,
                                          first_stride=(2, 2))
        for i in range(1, nb_blocks[3]):
            x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters * 4)
        # output size = 14x14
        # -------------------------- Layer Group 5 ----------------------------
        x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters * 8,
                                          first_stride=(2, 2))
        for i in range(1, nb_blocks[4]):
            x = self.residual_block(input_layer=x, nb_filters=initial_nb_filters * 8)
        # output size = 7x7
        x = AveragePooling2D(pool_size=(7, 7), border_mode='same')(x)
        x = Flatten()(x)
        output_tensor = Dense(9, activation='sigmoid')(x)

        return input_image, output_tensor


if __name__ == '__main__':
    model_factory = ResidualModel()
    input_tensor, output_tensor = model_factory.build_residual_network()

    model = Model(input=input_tensor, output=output_tensor)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    model.summary()
