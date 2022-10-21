import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask=None):
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)

    attention_scores = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(dk)

    if mask is not None:
        attention_scores += (mask*(-1e30))

    attention_weights = tf.nn.softmax(attention_scores, axis=-1)

    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights