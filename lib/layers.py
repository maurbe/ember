import tensorflow as tf
import tensorflow.keras.backend as K


class MinibatchStatConcatLayer(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(MinibatchStatConcatLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # standard deviation over minibatch n for each feature and location
        x = inputs
        y = K.sqrt(K.mean(K.square(x - K.mean(x, axis=0, keepdims=True)), axis=0, keepdims=True) + 1e-8)

        # compute the mean standard deviation
        y = K.mean(y, keepdims=True)

        # repeat to match n, h and w and have 1 feature map
        x_shape = tf.shape(x)
        y = K.tile(y, [x_shape[0], x_shape[1], x_shape[2], 1])

        return K.concatenate([x, y], axis=-1)

    def compute_output_shape(self, input_shape):
        # output_shape is input_shape plus one feature map
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class noise_injection(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(noise_injection, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.dtypes.float32, trainable=True, name='w')

    def call(self, x, training=None, mask=None):
        x_shape = tf.shape(x)
        noise = tf.random.normal(shape=(x_shape[0], x_shape[1], x_shape[2], 1), dtype=tf.dtypes.float32)

        x += noise * self.noise_strength
        return x