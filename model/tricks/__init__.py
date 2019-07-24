import tensorflow as tf

'''
    implement of some DL tricks and some useful tensorflow op
    
    1. layer_norm():
        define layer normalize
    2. layer_dropout():
        define layer dropout
    3. separable_convolution():
        define separable convolution
    4. multihead_attention():
        define multi-head attention and dot-product scaled attention
    5. count_trainable_params():
        count the number of trainable parameters 
    7. lr_warm_up():
        a learning rate warm-up scheme in QANet, a kind of dynamic learning rate
    8. highway():
        the highway network, SGS201512
    9. char_cnn_embed():
        use CNN to build char embedding, which was used in BiDAF and QANet
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
    define layer normalize
======================================================================
'''


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, regularizer=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=tf.AUTO_REUSE):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


'''
======================================================================
    define separable convolution
======================================================================
'''


def separable_convolution(inputs, kernel_size, num_depth_wise_kernels, num_point_wise_kernels, bias=True, strides=None, regularizer=None, initializer=None, padding='SAME', activation=tf.nn.relu, scope='separable_convolution'):
    '''
        Args:
            inputs: a tensor
                inputs.shape = (N, in_height, in_width, in_channels)
            kernel_size: a tuple
                kernel_size = (kernel_height, kernel_width)
            num_depth_wise_kernels: int
                the number of depth-wise convolution kernels
            num_point_wise_kernels: int
                the number of point_wise convolution kernels
            bias: bool
                whether use bias
            strides: a list
                control the strides in the depth-wise convolution, default [1, 1, 1, 1]
            padding: a string
                'SAME' or 'VALID'
            activation:
                the function of activation

    '''
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable(
            'depthwise_filter',
            (kernel_size[0], kernel_size[1], shapes[-1], num_depth_wise_kernels),
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer,
        )
        pointwise_filter = tf.get_variable(
            'pointwise_filter',
            (1, 1, shapes[-1]*num_depth_wise_kernels, num_point_wise_kernels),
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer,
        )
        outputs = tf.nn.separable_conv2d(
            inputs,
            depthwise_filter=depthwise_filter,
            pointwise_filter=pointwise_filter,
            strides=strides,
            padding=padding,
        )

        if bias:
            bias_ = tf.get_variable(
                'bias',
                outputs.shape[-1],
                regularizer=regularizer,
                initializer=initializer,
            )
            outputs += bias_
        outputs = activation(outputs)
        return outputs


'''
======================================================================
    define layer dropout
======================================================================
'''


def layer_dropout(inputs, residual, dropout):
    '''
        Args:
            inputs: a tensor
                a tensor A after doing some processing
            residual: a tensor
                a tensor A before doing some processing
            dropout:
                the rate of dropout
    '''
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


'''
======================================================================
    multi-head attention and dot-product scaled attention
======================================================================
'''


def multihead_attention(queries, num_hidden, num_heads, memory=None, mask=None, bias=True, dropout=0.0, is_training=True, scope='multihead_attention'):
    '''
        Args:
            queries:
                a tensor, queries.shape = (N, T, E).
            num_hidden: int
                the number of hidden units in multihead-attention
                the value must be 8x.
            num_heads: int
                the number of heads in multihead-attention
            memory:
                a tensor , memory.shape = (N, T2, E)
                if the attention is vanilla attention , not self-attention, it need memory as reference vector.
            mask:
                a bool tensor, mask.shape = (N, T)
                mask the <pad> char in  queries.
            bias: bool
                whether use bias
            dropout: float
                the rate of dropout.
            is_training: bool
                whether used for training, main affect tf.layers.dropout

        if self-attention:
            Q = K = V = queries
        if vanilla attention:
            Q = queries
            K = V = memory
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        # if self-attention
        if memory is None:
            memory = queries

        memory = tf.layers.dense(
            memory,
            2*num_hidden, # K and V
            use_bias=False,
            kernel_initializer=initializer(),
            kernel_regularizer=regularizer,
            name='memory_projection'
        )

        queries = tf.layers.dense(
            queries,
            num_hidden,
            use_bias=False,
            kernel_initializer=initializer(),
            kernel_regularizer=regularizer,
            name='queries_projection'
        )

        Q = tf.concat(tf.split(tf.expand_dims(queries, axis=1), num_heads, axis=-1), axis=1)
        K, V = [tf.concat(tf.split(tf.expand_dims(tensor, axis=1), num_heads, axis=-1), axis=1) for tensor in tf.split(memory, 2, axis=-1)]

        # scaled
        scaled = num_hidden // num_heads
        Q *= scaled ** (-0.5)
        # dot-product attention
        outputs = dot_production_attention(
            Q,
            K,
            V,
            bias=bias,
            mask=mask,
            dropout=dropout,
            is_training=is_training,
        )
        return tf.concat(tf.unstack(outputs, axis=1), axis=-1)


def dot_production_attention(Q, K, V, bias=False, mask=None, dropout=0.0, is_training=True, scope='dot_production_attention'):
    '''
        Args:
            Q:
                a tensor, (N, num_heads, T, num_hidden/num_heads)
            K:
                a tensor, (N, num_heads, T2, num_hidden/num_heads)
            V:
                a tensor, (N, num_heads, T2, num_hidden/num_heads)
            bias: bool
                whether use bias
            mask:
                a bool tensor, mask the <pad> char in Q, mask.shape = (N, T)
            dropout: float
                the rate of dropout
            is_training: bool
                whether used for training, main affect tf.layers.dropout
        if self-attention:
            T = T2
        if vanilla attention:
            T != T2
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        logits = tf.matmul(Q, K, transpose_b=True)
        if bias:
            bias = tf.get_variable(
                'bias',
                logits.shape[-1],
                regularizer=regularizer,
                initializer=initializer_zero,
            )
            logits += bias

        if mask is not None:
            shapes = [x if x is not None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)

        weights = tf.nn.softmax(logits)
        weights = tf.layers.dropout(weights, dropout, training=is_training)

        return tf.matmul(weights, V)


def mask_logits(inputs, mask, mask_value = -1e30):
    '''
        the op of mask out in the Transformer

        Args:
            inputs:
                a float tensor, tensor.shape = (N, num_heads, T, T2)
                it produce from tf.matmul(Q, K)
                if self-attention
                    T = T2
                if vanilla attention
                    T != T2
            mask:
                a bool tensor, tensor.shape = (N, 1, 1, T2)
            mask_value:
                the value is used for mask the <pad>
    '''
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


'''
======================================================================
    count the number of trainable parameters
======================================================================
'''


def count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))


'''
======================================================================
    a learning rate warm-up scheme in QANet, a kind of dynamic learning rate
======================================================================
'''


def lr_warm_up(final_lr, global_step, warm_up_step=999):
    '''
        the implement of a learning rate warm-up scheme in QANet
        Args:
            final_lr: float
                the final value of learning rate
            global_step:
                a tensor, as global step in the tf.Graph, global_step.shape = (1, )
            warm_up_step: int
                the step of warm up in the starting


    '''
    return tf.minimum(
        final_lr,
        final_lr / tf.log(tf.cast(warm_up_step, tf.float32)) * tf.log(tf.cast(global_step, tf.float32) + 1)
    )


'''
======================================================================
    the highway network which develop from SGS201512 were used in BiDAF and QANet
======================================================================
'''


def highway(inputs, num_hidden=None, activation=None, num_layers=2, dropout=0.0, regularizer=regularizer, initializer=initializer_relu(), bias_initializer=initializer_zero, is_training=True, scope='highway_networks'):
    '''
        the implement of the highway network which develop from SGS201512 were used in BiDAF and QANet
        it likes RNNs, also have a gate T and new states H, but it bases on FNNs.
        Args:
            inputs:
                a float32 tensor, inputs.shape = (N, T, E)
            num_hidden: int
                the number of hidden units in the highway networks, default num_hidden = inputs.shape[-1] = E
            activation: tf activation function
                the activation of the highway networks
            num_layers: int
                the number of layers in the highway networks
            dropout: float
                the rate of dropout
            is_training: bool
                    whether used for training, main affect tf.layers.dropout
        Return：
            a tensor, shape = (N, T, num_hidden)
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        if num_hidden is None:
            num_hidden = inputs.get_shape().as_list()[-1]
        else:
            inputs = tf.layers.dense(
                inputs,
                num_hidden,
                use_bias=False,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer,
                name='input_projection',
            )

        for i in range(num_layers):
            T = tf.layers.dense(
                inputs,
                num_hidden,
                use_bias=True,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer,
                bias_regularizer=regularizer,
                bias_initializer=bias_initializer,
                activation=tf.sigmoid,
                name=f'gate_{i}'
            )
            H = tf.layers.dense(
                inputs,
                num_hidden,
                use_bias=True,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer,
                bias_regularizer=regularizer,
                bias_initializer=bias_initializer,
                activation=activation,
                name=f'activation_{i}'
            )
            H = tf.layers.dropout(H, dropout, training=is_training)
            inputs = H * T + inputs * (1.0 - T)

    return inputs


'''
======================================================================
    use CNN to build char embedding, which was used in BiDAF and QANet
======================================================================
'''


def char_cnn_embed(char_inputs, char_mat, maxlen, num_hidden=None, kernel_size=5, dropout=0.0, is_training=True, regularizer=regularizer, initializer=initializer_relu(), bias_initializer=initializer_zero, dtype=tf.float32, scope='char_conv'):
    '''
        use 1D CNN to build char embeddings
        Args:
            char_inputs:
                a int tensor, char_inputs.shape = (N, T, char_limit),
                the value is the index of char in the char vocabulary.
            char_mat:
                a float tensor, index-embeddings of char vocabulary
                char_mat.shape = (total_char, char_dim)
                total_char is the total number of chars
                char_dim is the dimension of char embedding
            maxlen: int
                it is equal to T
            num_hidden: int
                the number of kernel in the 1D CNN
                the default value is equal to char_dim
            kernel_size: int
                the kernel width in the 1D CNN
            dropout: float
                the rate of dropout
            is_training: bool
                    whether used for training, main affect tf.layers.dropout
        Return:
            a float tensor, tensor.shape = (N, T, num_hidden)
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        ch_emb = tf.nn.embedding_lookup(char_mat, char_inputs)  # shape = (N, T, char_limit, char_dim)
        ch_emb = tf.concat(tf.unstack(ch_emb, axis=1), axis=0)  # shape = (N * T, char_limit, char_dim)

        char_dim = ch_emb.get_shape().as_list()[-1]
        if num_hidden is None:
            num_hidden = char_dim
        kernel_shape = [kernel_size, char_dim, num_hidden]
        bias_shape = [1, 1, num_hidden]
        kernel_ = tf.get_variable(
            'char_conv_kernel',
            kernel_shape,
            dtype=dtype,
            regularizer=regularizer,
            initializer=initializer,
        )
        bias_ = tf.get_variable(
            'bias_',
            bias_shape,
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer,
        )

        # char 1D convolution
        char_conv_outputs = tf.nn.conv1d(
            ch_emb,
            kernel_,
            stride=1,
            padding='VALID'
        )
        char_conv_outputs += bias_
        char_conv_outputs = tf.nn.relu(char_conv_outputs)  # shape = (N * T, char_limit, num_hidden)

        # max pooling
        char_pooling_outputs = tf.reduce_max(char_conv_outputs, keepdims=True, axis=1)  # shape = (N * T, 1, num_hidden)

        outputs = tf.concat(tf.split(char_pooling_outputs, maxlen, axis=0), axis=1)  # shape = (N, T, num_hidden)
        outputs = tf.layers.dropout(outputs, dropout, training=is_training)

        return outputs

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
            total_char = 200
            char_dim = 300
            maxlen = 50

            inputs = np.random.randint(0, total_char, (10, maxlen, 16))
            inputs = tf.Variable(inputs, dtype=tf.int32, trainable=False)

            char_mat = np.random.randn(total_char, char_dim)
            char_mat = tf.Variable(char_mat, dtype=tf.float32, trainable=True)

            char_emb = char_cnn_embed(
                inputs,
                char_mat,
                maxlen=maxlen,
                kernel_size=5,
                dropout=0.0,
                is_training=True,
            )

            sess.run(tf.global_variables_initializer())
            re = sess.run(char_emb)

            print('end')