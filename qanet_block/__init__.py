import tensorflow as tf
from tricks import layer_norm, layer_dropout, separable_convolution, multihead_attention, count_trainable_params
'''
    the implement of encoder block in QANet,
    you can create a encoder instance from class EncoderBlock_QANet,
    a encoder instance is called for encoding a hidden vector or a embedding vector.
'''

'''
======================================================================
    define regularizer and initializer
======================================================================
'''

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=1.0,
    mode='FAN_AVG',
    uniform=True,
    dtype=tf.float32
)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    dtype=tf.float32
)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)

initializer_zero = tf.zeros_initializer()

'''
======================================================================
    define positional encoding
======================================================================
'''


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        tf.math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


'''
======================================================================
    the encoder block in the QANet
======================================================================
'''


class EncoderBlock_QANet():
    def __init__(self, num_blocks, num_conv_layers, conv_kernel_size, num_hidden, input_projection=False, num_heads=8, bias_multi_attn=True, dropout=0.0, scope='encoder_block', is_training=True):
        '''
            Args:
                num_blocks: int
                    the number of encoder blocks

                num_conv_layers: int
                    the number of convolution layers in the encoder block

                conv_kernel_size: int
                    the width of kernel in the depth-wise filter of 1D separable convolution

                num_hidden: int
                    the number of hidden units in the linear projection layer,
                    and the number of kernels in the point-wise convolution of separable convolution,
                    and the number of hidden units in the multi-head attention,
                    and the number of hidden units in the feed forward network.

                input_projection: bool
                    whether apply linear projection layer to inputs firstly.

                num_heads: int
                    the number of heads in the multi-head attention

                bias_multi_attn: bool
                    whether use bias in the multi-head attention

                dropout: float
                    the rate of dropout in the encoder block.

                is_training: bool
                    whether used for training, main affect tf.layers.dropout

        '''
        self.initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
            factor=1.0,
            mode='FAN_AVG',
            uniform=True,
            dtype=tf.float32
        )
        self.initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_IN',
            uniform=False,
            dtype=tf.float32
        )
        self.regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

        self.initializer_zero = tf.zeros_initializer()

        self.num_blocks = num_blocks
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.num_hidden = num_hidden
        self.input_projection = input_projection
        self.num_heads = num_heads
        self.bias_multi_attn = bias_multi_attn
        self.dropout = dropout
        self.scope = scope
        self.is_training = is_training

    def __call__(self, inputs, inputs_pad_mask=None):
        '''
            Args:
                inputs:
                    a 3D float tensor.
                    for example in NLP: inputs.shape = (N, T, E)
                    N: batch size
                    T: the max length of text in all data or batch data
                    E: the size of the word embedding or the other hidden vector
                inputs_pad_mask:
                    a 2-D bool tensor
                    the mask for the <pad> in the inputs
                    inputs_pad_mask.shape = (N, T)

            Return:
                outputs:
                    a 3D float tensor. outputs.shape = (N, T, num_hidden)
        '''
        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):
            if self.bias_multi_attn:
                inputs = tf.layers.dense(inputs, self.num_hidden, use_bias=False, kernel_initializer=self.initializer(), kernel_regularizer=self.regularizer)

            outputs = inputs
            sublayer_index = 1
            total_sublayers = (self.num_conv_layers + 2) * self.num_blocks

            for i in range(self.num_blocks):
                with tf.variable_scope(name_or_scope=f'encoder_block_{i}', reuse=tf.AUTO_REUSE):
                    # add positional encoding
                    outputs = add_timing_signal_1d(outputs)
                    outputs, sublayer_index = self.conv_block(
                        outputs,
                        num_conv_layers=self.num_conv_layers,
                        conv_kernel_size=self.conv_kernel_size,
                        num_hidden=self.num_hidden,
                        dropout=self.dropout,
                        layer_info=(sublayer_index, total_sublayers),
                        scope=f'conv_block_{i}',
                    )
                    outputs, sublayer_index = self.self_attention_block(
                        outputs,
                        num_hidden=self.num_hidden,
                        inputs_pad_mask=inputs_pad_mask,
                        num_heads=self.num_heads,
                        bias=self.bias_multi_attn,
                        dropout=self.dropout,
                        layer_info=(sublayer_index, total_sublayers),
                        is_training=self.is_training,
                        scope=f'self_attention_block{i}'
                    )

        return outputs

    def conv_block(self, inputs, num_conv_layers, conv_kernel_size, num_hidden, dropout=0.0, layer_info=(1, 1), scope = 'conv_block'):
        '''
            Args:
                inputs:
                    a 3-D float tensor, inputs.shape = (N, T, E)
                num_conv_layers: int
                    the number of convolution layers
                conv_kernel_size: int
                    the width of kernel in the depth-wise filter of 1D separable convolution
                num_hidden: int
                    the number of kernels in the point-wise convolution of separable convolution
                dropout: float
                    the rate of dropout
                layer_info: a tuple
                    layer_info[0]: the index of current sublayer
                    layer_info[1]: the total number of sublayers

        '''
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            outputs = tf.expand_dims(inputs, dim=2)
            l, L = layer_info
            for i in range(num_conv_layers):
                residual = outputs
                # layer normalization
                outputs = layer_norm(outputs, scope=f'layer_norm_{i}')

                if (i % 2) == 0:
                    outputs = tf.layers.dropout(outputs, dropout, training=self.is_training)

                outputs = separable_convolution(
                    outputs,
                    kernel_size=(conv_kernel_size, 1),
                    num_depth_wise_kernels=1,
                    num_point_wise_kernels=num_hidden,
                    regularizer=self.regularizer,
                    initializer=self.initializer_relu(),
                    scope=f'separable_conv_{i}',
                )
                outputs = layer_dropout(outputs, residual, dropout * float(l) / L)

        return tf.squeeze(outputs, axis=2), l

    def self_attention_block(self, inputs, num_hidden, num_hidden_ffn=None, inputs_pad_mask=None, num_heads=8, bias=True, dropout=0.0, layer_info=(1, 1), is_training=True, scope='self_attention_block'):
        '''
            self_attention_block includes self-attention and FFN
            Args:
                inputs:
                    a 3-D float tensor, inputs.shape = (N, T, E)
                num_hidden: int
                    the number of hidden units in the multi-head attention
                num_hidden_ffn: int
                    the number of hidden units in the feed forward network,
                    if num_hidden_inner is None:
                        num_hidden_inner = num_hidden * 4
                inputs_pad_mask:
                    a 2-D bool tensor
                    the mask for the <pad> in the inputs
                    inputs_pad_mask.shape = (N, T)
                num_heads: int
                    the number of heads in the multi-head attention
                bias: bool
                    whether use bias
                dropout: float
                    the rate of dropout in the encoder block.
                layer_info: a tuple
                    layer_info[0]: the index of current sublayer
                    layer_info[1]: the total number of sublayer
                is_training: bool
                    whether used for training, main affect tf.layers.dropout
        '''
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            l, L = layer_info

            # layer normalize
            outputs = layer_norm(inputs, regularizer=self.regularizer, scope='layer_norm')
            outputs = tf.layers.dropout(outputs, dropout, training=is_training)

            # multi-head attention
            outputs = multihead_attention(
                queries=outputs,
                num_hidden=num_hidden,
                num_heads=num_heads,
                mask=inputs_pad_mask,
                bias=bias,
                dropout=dropout,
                is_training=is_training,
            )

            # layer dropout
            residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
            l += 1

            # feed forward network
            outputs = layer_norm(residual, regularizer=self.regularizer, scope='layer_norm_2')
            outputs = tf.layers.dropout(outputs, dropout, training=is_training)

            if num_hidden_ffn is None:
                num_hidden_ffn = num_hidden * 4
            outputs = tf.layers.dense(
                outputs,
                num_hidden_ffn,
                use_bias=True,
                activation=tf.nn.relu,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer_relu(),
                bias_regularizer=self.regularizer,
                bias_initializer=self.initializer_zero,
                name='FNN_1'
            )
            outputs = tf.layers.dense(
                outputs,
                num_hidden,
                use_bias=True,
                activation=None,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer_relu(),
                bias_regularizer=self.regularizer,
                bias_initializer=self.initializer_zero,
                name='FNN_2'
            )

            # layer dropout
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)

            l += 1

        return outputs, l


if __name__ == '__main__':
    import numpy as np
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(
        log_device_placement=True
    )
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            inputs_fake = tf.Variable(np.random.randn(10, 20, 16), dtype=tf.float32, trainable=False)

            a = np.ones((10, 20))
            for i in range(10):
                np.random.seed(i)
                pad_length = np.random.randint(0, 10)
                a[i, -pad_length:] = np.zeros((pad_length))
            inputs_pad_masks_fake = tf.Variable(a, dtype=tf.bool, trainable=False)

            print('===' * 20)
            count_trainable_params()

            print('===' * 20)
            encoder = EncoderBlock_QANet(
                num_blocks=1,
                num_conv_layers =2,
                conv_kernel_size=7,
                num_hidden=16,
                input_projection=False,
                num_heads=1,
                bias_multi_attn=False,
                dropout=0.0,
            )
            enc = encoder(inputs_fake, inputs_pad_mask=inputs_pad_masks_fake)
            count_trainable_params()

            print('===' * 20)
            enc2 = encoder(inputs_fake, inputs_pad_mask=inputs_pad_masks_fake)
            count_trainable_params()

            print('===' * 20)
            encoder2 = EncoderBlock_QANet(
                num_blocks=1,
                num_conv_layers=2,
                conv_kernel_size=7,
                num_hidden=16,
                input_projection=False,
                num_heads=1,
                bias_multi_attn=False,
                dropout=0.0,
            )
            enc3 = encoder2(inputs_fake, inputs_pad_mask=inputs_pad_masks_fake)
            count_trainable_params()

            print('===' * 20)
            encoder3 = EncoderBlock_QANet(
                num_blocks=1,
                num_conv_layers=2,
                conv_kernel_size=7,
                num_hidden=16,
                input_projection=False,
                num_heads=1,
                bias_multi_attn=False,
                dropout=0.0,
                scope='encoder2'
            )
            enc4 = encoder3(inputs_fake, inputs_pad_mask=inputs_pad_masks_fake)
            count_trainable_params()

            sess.run(tf.global_variables_initializer())

            print(f'enc.shape = {sess.run(enc).shape}')
            print(f'enc2.shape = {sess.run(enc2).shape}')
            print(f'enc3.shape = {sess.run(enc3).shape}')
            print(f'enc4.shape = {sess.run(enc4).shape}')


            print('end')

