import tensorflow as tf
from keras.layers import Layer, Conv2D, BatchNormalization, LayerNormalization, Activation
from transformer.behaviors.self_attention import scaled_dot_product_attention

class ConvolutionalAttention(Layer):
    def __init__(self, n = 64, c = 64):
        super(ConvolutionalAttention, self).__init__()
        self.n = n
        self.c = c

        self.conv_q = Conv2D(c, 3, 1, padding='same', kernel_initializer='glorot_normal')
        self.conv_k = Conv2D(c, 3, 1, padding='same', kernel_initializer='glorot_normal')
        self.conv_v = Conv2D(c, 3, 1, padding='same', kernel_initializer='glorot_normal')

        self.conv = Conv2D(n, 3, 1, padding='same', kernel_initializer='glorot_normal')

        self.batch_q = BatchNormalization()
        self.batch_k = BatchNormalization()
        self.batch_v = BatchNormalization()

        self.normal_layer = LayerNormalization()

        self.final_conv1 = Conv2D(n, 3, 1, padding='same', kernel_initializer='glorot_normal', activation='relu')
        self.final_conv2 = Conv2D(n, 3, 1, padding='same', kernel_initializer='glorot_normal', activation='relu')

        self.final_batch1 = BatchNormalization()
        self.final_batch2 = BatchNormalization()

        self.outpu = Activation('relu')

    def call(self, inputs, training=True):
        residual = inputs

        batch_size = tf.shape(inputs)[0]

        q = self.batch_q(self.conv_q(inputs), training=training)
        k = self.batch_k(self.conv_k(inputs), training=training)
        v = self.batch_v(self.conv_v(inputs), training=training)

        # time dimention
        q_time = tf.transpose(q, [0,3,1,2])
        k_time = tf.transpose(k, [0,3,1,2])
        v_time = tf.transpose(v, [0,3,1,2])

        # frequency dimention
        q_frequency = tf.transpose(q, [0,3,2,1])
        k_frequency = tf.transpose(k, [0,3,2,1])
        v_frequency = tf.transpose(v, [0,3,2,1])

        scaled_attention_time, attention_weights_time = scaled_dot_product_attention(q_time, k_time, v_time, None)

        scaled_attetion_frequency, attention_weights_frequency = scaled_dot_product_attention(q_frequency, k_frequency, v_frequency, None)

        output = tf.concat([scaled_attention_time, scaled_attetion_frequency], -1)

        output = self.normal_layer(self.conv(output))

        final_output = self.final_batch1(self.final_conv1(output), training=training)
        final_output = self.final_batch2(self.final_conv2(final_output), training=training)

        return final_output
