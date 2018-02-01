from keras import layers
from keras import models
# from keras.utils import plot_model
# from keras.optimizers import SGD
#



def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    return y


def residual_block(y, nb_channels):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = layers.BatchNormalization()(y)

    y = layers.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = layers.LeakyReLU()(y)

    return y



def streamProcess(x):
    # x_orig = x
    # x = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    # x = layers.Cropping2D(((0, 0), (2, 2)))(x)
    # x = layers.LeakyReLU()(x)

    x_orig = x
    x = layers.ZeroPadding2D(padding=(2, 0), data_format=None)(x)
    x = layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='valid')(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    x = residual_block(x,32)
    # x = residual_block(x,64)

    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([shortcut, x])
    x = layers.LeakyReLU()(x)

    x = layers.ZeroPadding2D(padding=(1, 0), data_format=None)(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = add_common_layers(x)



    y = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(x_orig)

    y = layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 1), padding='valid')(y)
    y = add_common_layers(y)
    # y = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(y)

    y = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 1), padding='valid')(y)
    y = add_common_layers(y)
    # y = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(y)

    y = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(y)
    y = add_common_layers(y)
    y = layers.pooling.MaxPooling2D(pool_size=(2, 2), padding='valid')(y)
    #
    # y = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid')(y)
    # y = add_common_layers(y)
    # y = layers.pooling.MaxPooling2D(pool_size=(2, 2), padding='valid')(y)

    y = layers.Flatten()(y)
    y = layers.Dense(1*255*5)(y)
    y = add_common_layers(y)
    # y = layers.LeakyReLU()(y)
    y = layers.Reshape([1,255,5])(y)
    # y = layers.ZeroPadding2D(padding=(0, 3), data_format=None)(y)

    x = layers.concatenate([x,y], axis=1)

    x = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    # x = add_common_layers(x)

    # network_output = layers.Conv2D(2, kernel_size=(7, 7), strides=(1, 1), padding='same')(x)
    # # we remove the beginning and ending several frames, since some of the temporal information is not included
    # network_output = layers.Cropping2D(((0, 0), (5, 5)))(network_output)


    # In the last layer, to save the computational complexity, we first zero padding on the frequency axis,
    # then apply convolution without zero padding, with filter kernel spanning the same temporal length as the previous layer, i.e 7 frames
    x = layers.ZeroPadding2D(padding=(2, 0), data_format=None)(x)
    network_output = layers.Conv2D(2, kernel_size=(5, 5), strides=(1, 1), padding='valid')(x)


    return network_output



def streamProcessOhio(x):
    y = layers.Dense(3000)(x)
    y = add_common_layers(y)
    y = layers.Dense(3000)(y)
    y = add_common_layers(y)
    y = layers.Dense(3000)(y)
    y = add_common_layers(y)
    network_output = layers.Dense(255, activation='sigmoid')(y)


    return network_output



def streamProcessMLP(x):
    y = layers.Dense(3000)(x)
    y = add_common_layers(y)
    y = layers.Dense(3000)(y)
    y = add_common_layers(y)
    y = layers.Dense(3000)(y)
    y = add_common_layers(y)
    network_output = layers.Dense(510)(y)

    return network_output









def generateModel():
    channels = 3
    spectrum_height = 255
    spectrum_width = 11

    spectrum_tensor = layers.Input(shape=(channels, spectrum_height, spectrum_width))
    spectrum_output = streamProcess(spectrum_tensor)

    model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])
    return model



def generateModelRaw():
    channels = 2 # The only difference is the input features, the same DNN structure is used
    spectrum_height = 255
    spectrum_width = 11

    spectrum_tensor = layers.Input(shape=(channels, spectrum_height, spectrum_width))
    spectrum_output = streamProcess(spectrum_tensor)

    model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])
    return model


# model = generateModel()
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


# model = generateModelRaw()
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, to_file='modelRaw.png')


def generateModelOhio():
    Dim_in = 251*11
    # Dim_out = 255
    Dim_out = 255 * 2

    spectrum_tensor = layers.Input(shape=(Dim_in,))
    # spectrum_output = streamProcessOhio(spectrum_tensor)
    spectrum_output = streamProcessMLP(spectrum_tensor)

    model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])
    return model

def generateModelMLP():
    Dim_in = 765 * 11
    Dim_out = 255 * 2

    spectrum_tensor = layers.Input(shape=(Dim_in,))
    spectrum_output = streamProcessMLP(spectrum_tensor)

    model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])
    return model

def generateModelRawMLP():
    Dim_in = 510 * 11
    Dim_out = 255 * 2

    spectrum_tensor = layers.Input(shape=(Dim_in,))
    spectrum_output = streamProcessMLP(spectrum_tensor)

    model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])
    return model



# model = generateModel()
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


# model = generateModelRaw()
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, to_file='modelRaw.png')