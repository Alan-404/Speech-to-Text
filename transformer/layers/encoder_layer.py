from keras.layers import Layer, LayerNormalization, Dropout
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.behaviors.position_wise_feed_forward_networks import ffn

class EncoderLayer(Layer):
    def __init__(self, h, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = ffn(d_ff, d_model, activation)

        self.norm_layer_1 = LayerNormalization(epsilon=eps)
        self.norm_layer_2 = LayerNormalization(epsilon=eps)

        self.dropout_layer_1 = Dropout(rate=dropout_rate)
        self.dropout_layer_2 = Dropout(rate=dropout_rate)



    def call(self, inputs, is_train, mask):
        multi_head_attention_output, _ = self.multi_head_attention(inputs, inputs, inputs, mask)

        inputs = self.norm_layer_1(inputs + self.dropout_layer_1(multi_head_attention_output, training=is_train))

        ffn_out = self.feed_forward(inputs)

        encoder_layer_out = self.norm_layer_2(inputs + self.dropout_layer_2(ffn_out, training=is_train))

        return encoder_layer_out


