import tensorflow as tf
from behaviors.self_attention import scaled_dot_product_attention
from keras.layers import Layer, Dense

class MultiHeadAttention(Layer):
    def __init__(self, d_model=512, h=8):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model

        self.dense_k = Dense(d_model)
        self.dense_q = Dense(d_model)
        self.dense_v = Dense(d_model)

        self.dense_output = Dense(d_model)

    def splitting_head(self, tensor):
        batch_size = tf.shape(tensor)[0]
        length = tf.shape(tensor)[1]
        d_model = tf.shape(tensor)[2]

        heading_value = d_model//self.h

        tensor = tf.reshape(tensor, (batch_size, length, self.h, heading_value))

        tensor_head = tf.transpose(tensor, [0,2,1,3])

        return tensor_head

    def call(self, q, k, v, mask=None):

        batch_size = tf.shape(q)[0]

        qw = self.dense_q(q)
        kw = self.dense_k(k)
        vw = self.dense_v(q)

        heading_q = self.splitting_head(qw)
        heading_k = self.splitting_head(kw)
        heading_v = self.splitting_head(vw)

        output, attention_weights = scaled_dot_product_attention(heading_q, heading_k, heading_v, mask)

        output = tf.transpose(output, [0,2,1,3])

        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense_output(output)

        return output, attention_weights


