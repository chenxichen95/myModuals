import tensorflow as tf
import os
import numpy as np


class TransformerVanillaBlock:
    '''
        the core/vanilla block in the transformer
    '''

    def __init__(
            self,
            x_maxLen,
            x_ref_maxLen=None,
            d_model=512,
            d_ff=512*4,
            num_heads=8,
            dropout_rate=0,
            encode_type=0,
            training=True,
            first_block=True,
    ):
        '''
            args:
                x_maxLen:
                    the maximum value of the second dimension(x) in the total data, not in the current batch data
                x_ref_maxLen:
                    the maximum value of the second dimension(x_ref) in the total data, not in the current batch data
                d_model:
                    the hidden dimension
                d_ff:
                    the number of hidden layer in the position-wise feedforward network
                num_heads:
                    the number of heads in multi-head attention
                encode_type:
                    encode_type = 0: mean that the block is based on the self-attention,
                        at this time, Q=x, K=x, V=x
                    encode_type = 1: mean that the block is based on the vanilla attention,
                        at this time, Q=x, K=x_ref, V=x_ref
                first_block:
                    the first block in the stacked transformer block, need positional encoding.
        '''
        self.x_maxLen = x_maxLen
        self.x_ref_maxLen = x_ref_maxLen
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.encode_type = encode_type
        self.training = training
        self.f_dtype = tf.float32
        self.first_block = first_block

        self.check_args()

    def __call__(self, x, x_ref=None):
        '''
            args:
                x:
                    the input of sequence, it should be a tensor, tensor.shape = (N, T1, d_embed)
                x_ref:
                    if encode_type == 1:
                        x_ref is a reference sequence for x, it also should be a tensor, tensor.shape = (N, T2, d_embed)
            return:
                x_encoding:
                    the encoding result from the transformer block for x.
                    x_encoding.shape = (N, T1, d_model)
        '''

        if self.first_block:
            # positional encoding
            x += self.positional_encoding(x, self.x_maxLen, masking=True)
            if self.encode_type in [1, '1']:
                x_ref += self.positional_encoding(x_ref, self.x_ref_maxLen, masking=True)

        # multi head attention encoding
        if self.encode_type in [0, '0']:
            x_encoding = self.multi_head_attention(Q=x, K=x, V=x, scope='multi_head_attention')
        elif self.encode_type in [1, '1']:
            x_encoding = self.multi_head_attention(Q=x, K=x_ref, V=x_ref, scope='multi_head_attention')

        x_encoding = self.ff(x_encoding, [self.d_ff, self.d_model])

        return x_encoding

    def check_args(self):
        if self.encode_type in [0, '0']:
            tf.logging.info('the block is based on self-attention')

        elif self.encode_type in [1, '1']:
            tf.logging.info('the block is based on vanilla attention')
            if self.x_ref is None:
                raise Exception('x_ref does not assgin')

            elif self.x_ref_maxLen is None:
                raise Exception('x_ref_maxLen does not assgin')

        else:
            raise Exception('encode_type should be 0 or 1')

    def positional_encoding(self, inputs, maxLen, masking=True, scope='positional_encoding'):
        '''
            inputs
                the sequences are used for calculating the positional encoding
                inputs.shape = (N, T, d_embed)
            maxLen
                the maximum value of the second dimension in the total data, not in the current batch data
            mask:
                whether mask the positional encoding of <pad>

            return:
                outputs:
                    positional encoding, outputs.shape = (N, T, d_embed)
        '''
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            N, T, E = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
            pos_index = tf.tile(tf.expand_dims(tf.range(T), axis=0), [N, 1])  # pos_index.shape = (N, T)

            A1 = tf.cast(tf.tile(tf.expand_dims(tf.range(maxLen), axis=-1), [1, E]), dtype=self.f_dtype)
            A2 = tf.tile(
                tf.pow(
                    tf.cast(tf.ones((1, E)) * 10000, dtype=self.f_dtype),
                    tf.cast(tf.expand_dims(tf.range(E), axis=0) * 2 / self.d_model, dtype=self.f_dtype),
                ),
                [maxLen, 1]
            )
            pos_encoding = tf.divide(A1, A2)

            outputs = tf.nn.embedding_lookup(pos_encoding, pos_index)

            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return outputs

    def multi_head_attention(self, Q, K, V, scope='multi_head_attention'):
        '''
            args:
                Q.shape = (N, Tq, dk)
                K.shape = (N, Tk, dk)
                V.shape = (N, Tv, dv)
            if block was based on self-attention:
                Q = K = V = x
                Tq = Tk = Tv = T1
                dk = dv = d_embed
            if block was based on vanilla attention:
                Q = x, K = V = x_ref,
                Tq = T1, Tk = Tv = T2
                dk = dv = d_embed

            because need to residual connection, d_embed = d_model

            return:
                outputs:
                    encoding by multi head attention
                    outputs.shape = (N, T1, d_model)
        '''
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # linear project
            # Q => (linear project) => Q_; Q_.shape = (N, Tq, d_model)
            # K => (linear project) => K_; K_.shape = (N, Tk, d_model)
            # V => (linear project) => V_; V_.shape = (N, Tv, d_model)
            Q_, K_, V_ = self.linear_project_layer(Q, K, V)

            # divided into the number of heads
            # Q_ => Q_mh ; Q_mh.shape = (h*N, Tq, d_model/h)
            # K_ => K_mh ; K_mh.shape = (h*N, Tk, d_model/h)
            # V_ => V_mh ; V_mh.shape = (h*N, Tv, d_model/h)
            Q_mh = tf.concat(tf.split(Q_, num_or_size_splits=self.num_heads, axis=-1), axis=0)
            K_mh = tf.concat(tf.split(K_, num_or_size_splits=self.num_heads, axis=-1), axis=0)
            V_mh = tf.concat(tf.split(V_, num_or_size_splits=self.num_heads, axis=-1), axis=0)

            outputs = self.scaled_dot_product_attention(Q_mh, K_mh, V_mh)

            # restore shape (N, Tq, d_model)
            outputs = tf.concat(tf.split(outputs, num_or_size_splits=self.num_heads, axis=0), axis=-1)

            outputs = tf.layers.dense(outputs, self.d_model, use_bias=False)

            # residual connection [because this op, d_embed should equal to d_model!!]
            outputs += Q

            # layer normalizing
            outputs = self.ln(outputs)

            return outputs

    def linear_project_layer(self, Q, K, V, scope='linear_project_layer'):
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            Q_ = tf.layers.dense(Q, self.d_model, use_bias=False)
            K_ = tf.layers.dense(K, self.d_model, use_bias=False)
            V_ = tf.layers.dense(V, self.d_model, use_bias=False)
        return Q_, K_, V_

    def scaled_dot_product_attention(self, Q_mh, K_mh, V_mh):
        '''
            Q_mh.shape = (h*N, Tq, d_model/h)
            K_mh.shape = (h*N, Tk, d_model/h)
            V_mh.shape = (h*N, Tv, d_model/h)
        '''

        # dot product
        outputs = tf.matmul(Q_mh, tf.transpose(K_mh, [0, 2, 1]))  # shape = (h*N, Tq, Tk)

        # scale
        outputs /= self.d_model ** 0.5

        # masking base on K_mh
        outputs = self.masking(outputs, Q_mh, K_mh, mask_type="key")

        outputs = tf.nn.softmax(outputs)

        # masking base on Q_mh
        outputs = self.masking(outputs, Q_mh, K_mh, mask_type="query")

        outputs = tf.layers.dropout(outputs, rate=self.dropout_rate, training=self.training)

        outputs = tf.matmul(outputs, V_mh)  # shape = (h*N, Tq, d_model/h)


        return outputs

    def masking(self, inputs, q, k, mask_type='key'):
        '''
            inputs:
                product from tf.matmul(Q_mh, tf.transpose(K_mh, [0, 2, 1]))
                inputs.shape = (N, Tq, Tk)

            q:
                q.shape = (h*N, Tq, d_model/h)
            k:
                k.shape = (h*N, Tk, d_model/h)
            type:
                k, key, keys
                    the masking basing on k
                q, query, queries
                    the masking basing on q
                f, feature
                    the masking basing on the lower triangular identity matrix

            Because need to prevent leftward information flow in the decoder to preserve the auto-regressive property,
            they implement this inside of scaled dot-product attention by masking out all values in the input (minus infinity)
             of the softmax which correspond to illegal connections.
        '''

        padding_num = -2 ** 32 + 1
        if mask_type in ("k", "key", "keys"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(k), axis=-1))  # shape = (h*N, Tk)
            masks = tf.expand_dims(masks, 1)  # shape = (h*N, 1, Tk)
            masks = tf.tile(masks, [1, tf.shape(q)[1], 1])  # shape = (h*N, Tq, Tk)

            # Apply masks to inputs
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # shape = (h*N, Tq, Tk)

        elif mask_type in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(q), axis=-1))  # shape = (h*N, Tq)
            masks = tf.expand_dims(masks, -1)  # shape = (h*N, Tq, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(k)[1]])  # shape = (h*N, Tq, Tk)

            # Apply masks to inputs
            outputs = inputs * masks

        elif mask_type in ("f", "future"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # shape = (Tq, Tk)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # shape = (Tq, Tk)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # shape = (h*N, Tq, Tk)

            paddings = tf.ones_like(masks) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

        else:
            print("Check if you entered type correctly!")

        return outputs

    def ln(self, inputs, epsilon=1e-8, scope="layer_normalizing"):
        '''
            layer normalization
                J. L. Ba, J. R. Kiros, and G. E. Hinton, ``Layer normalization," https://arxiv.org/abs/1607.06450, 2016.
            args:
                inputs:
                    inputs.shape = (N, T, d_model)
        '''
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", shape=self.d_model, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", shape=self.d_model, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta

        return outputs

    def ff(self, inputs, num_units, scope="positionwise_feedforward"):
        '''
            position-wise feedforward network
            args:
                inputs:
                    inputs.shape = (N, T, d_model)
                num_units:
                    a list, [d_ff, d_model]
                        d_ff:
                            the number of hidden units in the positionwise_feedforward network.
                        d_model:
                            the number of output units in the positionwise_feedforward network.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)  # shape = (N, T, d_ff)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])  # shape = (N, T, d_model)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.ln(outputs)

        return outputs

if __name__ == '__main__':
    '''
        test code
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.ConfigProto(
        device_count={"CPU": 3},  # CPU 数量
        log_device_placement=True
    )
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            x = tf.Variable(np.random.rand(10, 20, 512), dtype=tf.float32, name='x')
            #x = tf.placeholder(dtype=tf.float32, shape=(10, 20, 512), name='x')

            # use 6 transformer blocks to encode x
            x_enc = x
            for i in range(6):
                with tf.variable_scope(name_or_scope=f'transformer_block_{i}', reuse=tf.AUTO_REUSE):
                    if i == 0:
                        block = TransformerVanillaBlock(x_maxLen=50, first_block=True)
                    else:
                        block = TransformerVanillaBlock(x_maxLen=50, first_block=False)
                    x_enc = block(x_enc)

            sess.run(tf.global_variables_initializer())
            re = sess.run(x_enc)
