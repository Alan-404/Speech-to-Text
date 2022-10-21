import tensorflow as tf
from keras.layers import Layer, BatchNormalization, Conv2D
from convolutional_attention import ConvolutionalAttention

class PreNet(Layer):
    def __init__(self, num_M = 2, n=64, c=64):
        super(PreNet, self).__init__()

        self.num_M = num_M
        self.n = n
        self.c = c

        self.conv1 = Conv2D(n, 3, 1, padding='same', activation='tanh', kernel_initializer='glorot_normal')
        self.conv2 = Conv2D(n, 3, 1, padding='same', activation='tanh', kernel_initializer='glorot_normal')

        self.batch1 = BatchNormalization()
        self.batch2 = BatchNormalization()

        self.conv_attention = [ConvolutionalAttention(n, c) for _ in range(num_M)]

    def call(self, inputs, training=True):
        inputs = tf.cast(inputs, dtype=tf.float32)

        output = self.batch1(self.conv1(inputs), training=training)
        output = self.batch2(self.conv2(output), training=training)

        for i in range(self.num_M):
            output = self.conv_attention[i](output, training)

        return output
