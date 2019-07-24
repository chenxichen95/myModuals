import tensorflow as tf

'''
    implement some common attention mechanism
    
    1. context_query_attention
        implement the c2q and q2c attention in QANet
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
    define context_query_attention
======================================================================
'''


def context_query_attention(Q, C, q_maxlen, c_maxlen, q_mask=None, c_mask=None, scope='context_query_attention', dtype=tf.float32, kernel_regularizer=regularizer, kernel_initializer=initializer, bias_initializer=initializer_zero):
    '''
        the implement of the C2Q attention and Q2C attention which were used in the QANet
        Args:
            Q:
                a float tensor, Q.shape = (N, T1, E)
            C:
                a float tensor, C.shape = (N, T2, E)
            q_maxlen: int
                it is equal to T1 , it may represents the max len of sentence in total data or current batch data
            c_maxlen: int
                it is equal to T2, it may represents the max len of sentence in total data or current batch data
            q_mask:
                a bool tensor, q_mask.shape = (N, T1), mark the <pad> char in the Q
            c_mask:
                a bool tensor, c_mask.shape = (N, T2), mark the <pad> char in the C

        Return:
            c2q:
                the attention vector from context attend query
                c2q.shape = (N, T2, E)
            q2c:
                the attention vector from query attend context
                q2c.shape = (N, T2, E)
    '''
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        d = C.get_shape().as_list()[-1]

        # calculate the similar matrix S between C and Q
        subres0 = tf.layers.dense(
            C,
            1,
            use_bias=False,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer(),
        )
        subres0 = tf.tile(subres0, [1, 1, q_maxlen])
        subres1 = tf.layers.dense(
            Q,
            1,
            use_bias=False,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer(),
        )
        subres1 = tf.transpose(tf.tile(subres1, [1, 1, c_maxlen]), [0, 2, 1])
        W_mlu = tf.get_variable(
            'W_mlu',
            [1, 1, d],
            dtype=dtype,
            regularizer=kernel_regularizer,
            initializer=kernel_initializer(),
        )
        bias = tf.get_variable(
            'bias',
            [1],
            regularizer=regularizer,
            initializer=bias_initializer,
        )
        subres2 = tf.matmul(C * W_mlu, tf.transpose(Q, [0, 2, 1]))
        S = subres0 + subres1 + subres2
        S += bias

        # the attention wight S_ from C attend Q
        if q_mask is not None:
            q_mask = tf.expand_dims(q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, q_mask), axis=-1)
        else:
            S_ = tf.nn.softmax(S, axis=-1)  # the attention wight from C attend Q

        if c_mask is not None:
            c_mask = tf.expand_dims(c_mask, 2)
            S_T = tf.nn.softmax(tf.transpose(mask_logits(S, c_mask), [0, 2, 1]), axis=-1)
        else:
            S_T = tf.nn.softmax(tf.transpose(S, [0, 2, 1]), axis=-1)  # the attention wight from Q attend C

        c2q = tf.matmul(S_, Q)
        q2c = tf.matmul(tf.matmul(S_, S_T), C)

        return c2q, q2c


def mask_logits(inputs, mask, mask_value = -1e30):
    '''
        the op of mask out in the Transformer

        Args:
            inputs:
                a float tensor, tensor.shape = (N, T1, T2)
            mask:
                a bool tensor, tensor.shape = (N, 1, T2) or tensor.shape = (N, T1, 1)
            mask_value:
                the value is used for mask the <pad>
    '''
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


if __name__ == '__main__':
    import numpy as np
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.ConfigProto(
        device_count={"CPU": 3},  # CPU 数量
        log_device_placement=True
    )


    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            Q = tf.Variable(np.random.randn(10, 20, 100), dtype=tf.float32, trainable=False)
            C = tf.Variable(np.random.randn(10, 50, 100), dtype=tf.float32, trainable=False)

            a = np.ones((10, 20))
            for i in range(10):
                np.random.seed(i)
                pad_length = np.random.randint(0, 10)
                a[i, -pad_length:] = np.zeros((pad_length))
            q_mask = tf.Variable(a, dtype=tf.bool, trainable=False)

            a = np.ones((10, 50))
            for i in range(10):
                np.random.seed(i*10)
                pad_length = np.random.randint(0, 30)
                a[i, -pad_length:] = np.zeros((pad_length))
            c_mask = tf.Variable(a, dtype=tf.bool, trainable=False)

            c2q, q2c = context_query_attention(
                Q,
                C,
                q_maxlen=20,
                c_maxlen=50,
                q_mask=q_mask,
                c_mask=c_mask,
            )
            sess.run(tf.global_variables_initializer())
            re = sess.run([c2q, q2c, q_mask, c_mask])
            print('end')


