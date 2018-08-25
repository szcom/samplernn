import keras
import keras.backend as K
from keras.layers import GRUCell, Dense, RNN, TimeDistributed
from keras.layers import Lambda, InputSpec
from keras.utils import to_categorical
import numpy as np
from wnorm import GRUCellWithWeightNorm


class GravesAttentionCell(GRUCellWithWeightNorm):
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self,
                 units,
                 nof_mixtures,
                 state_selector,
                 learned_h,
                 abet_size,
                 **kwargs):
        super(GravesAttentionCell, self).__init__(units, **kwargs)
        self.nof_mixtures = nof_mixtures
        self.abet_size = abet_size
        self.state_selector = state_selector
        self.learned_h = learned_h
        self.state_size = (units + abet_size, # gru state
                           self.nof_mixtures # K
                           )
        self.abet_size_scalar = K.constant(self.abet_size, dtype='int32')

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise TypeError('expects constants shape')
        [input_shape, txt_shape, txt_mask_shape] = input_shape

        super(GravesAttentionCell, self).build(input_shape)
        # changed kernel dim form abet x untis to abet x input size
        # as attention mask is overlayed onto input to the gru
        graves_initializer = keras.initializers.RandomNormal(stddev=0.075)
        window_b_initializer = keras.initializers.RandomNormal(mean=-3.0, stddev=.25)
        regularizer = keras.regularizers.l2(0.01)
        self.attn_tm1_kernel = self.add_weight(shape=(self.abet_size, input_shape[-1]),
                                               initializer=graves_initializer,
                                               regularizer=regularizer,
                                               name="attention_tm1_kernel")
        self.alpha_kernel = self.add_weight(shape=(self.units, self.nof_mixtures),
                                            name="alpha_kernel",
                                            regularizer=regularizer,
                                            initializer=graves_initializer)
        self.beta_kernel = self.add_weight(shape=(self.units, self.nof_mixtures),
                                           name="beta_kernel",
                                           regularizer=regularizer,
                                           initializer = graves_initializer)
        self.kappa_kernel = self.add_weight(shape=(self.units, self.nof_mixtures),
                                            name="kappa_kernel",
                                            regularizer=regularizer,
                                            initializer=graves_initializer)
        self.bias_a = self.add_weight(shape=(self.nof_mixtures,),
                                            name="window_bias_a",
                                            initializer=window_b_initializer)
        self.bias_k = self.add_weight(shape=(self.nof_mixtures,),
                                            name="window_bias_k",
                                            initializer=window_b_initializer)

        self.index_range = K.constant(np.tile(np.arange(500).reshape(1,500,1), (200, 1, self.nof_mixtures)))
        # mu        bs x K
        # index     bs x L x K
        self._trainable_weights.append(self.learned_h)

    def repeater(self, t, count):
        """
        :param t: 2d tensor bs x K
        :param count: number of steps to repeat 2nd dimension
        :return: 3d tensor  bs x count x K
        """
        return K.tile(K.expand_dims(t, axis=1), (1, count, 1))

    def call(self, inputs, states, constants):
#        import tensorflow as tf

        [h, k_tm1] = states
        [txt, txt_mask] = constants
        batch_size = K.shape(txt)[0]
        char_seq_len = K.shape(txt)[1]
        abet_size = self.abet_size_scalar #K.shape(txt)[2]
        phi_tm1 = h[:, -abet_size:]
        h = h[:,:-abet_size]
        h = K.switch(self.state_selector, K.tile(self.learned_h, (batch_size, 1)), h)
        #h = tf.Print(h, [h, states[0], K.tile(self.learned_h, (batch_size, 1))], summarize=80, message='hhhhh')

        # phi_tm1: bs x ABET_SIZE
        # self.attn_tm1_to_input: ABET_SIZE x units
        input_with_attention = inputs + K.dot(phi_tm1, self.attn_tm1_kernel)
        print('gru h state', K.int_shape(h), K.int_shape(input_with_attention))
        output, [gru_state] = super(GravesAttentionCell, self).call(input_with_attention, [h])
        a = K.dot(output, self.alpha_kernel)
        b = K.dot(output, self.beta_kernel)
        k = K.dot(output, self.kappa_kernel)
        a = K.bias_add(a, self.bias_a)
        k = K.bias_add(k, self.bias_k)

        a_t = K.softmax(a) + K.epsilon()
        b_t = K.exp(b) + K.epsilon()
        index_unrolled = self.index_range[:K.shape(inputs)[0], :char_seq_len, :]
        k_t = k_tm1 + K.exp(k)
        b_t_unrolled = self.repeater(b_t, char_seq_len)
        k_t_unrolled = self.repeater(k_t, char_seq_len)
        a_t_unrolled = self.repeater(a_t, char_seq_len)
        #b_t_unrolled = tf.Print(b_t_unrolled, [b_t_unrolled, (k_t_unrolled - index_unrolled) ** 2], summarize=80, message='b_t')
        print("unrolled", K.int_shape(b_t_unrolled), K.int_shape(k_t_unrolled), K.int_shape(index_unrolled))
        #(bs, L, K) (bs, L, K) (bs, L, K)
        phi_t = a_t_unrolled * K.exp(-0.5 * b_t_unrolled * (k_t_unrolled - index_unrolled) ** 2)
        print("phi1", K.int_shape(phi_t), K.int_shape(txt_mask))
        # phi_t bs x L x K
        phi_t = K.sum(phi_t, axis=-1)
        # phi_t bs x L
        print("phi11", K.int_shape(phi_t), K.int_shape(txt_mask))
        phi_t = phi_t * txt_mask
        phi_t_output = phi_t
        phi_t = K.tile(K.expand_dims(phi_t, axis=-1), (1, 1, self.abet_size))
        # phi_t bs x L x ABET_SIZE
        # char_seq  bs x L x ABET_SIZE
        print("before mult", K.int_shape(phi_t), K.int_shape(txt))
        #phi_t: (bs, L, ABET_SIZE)(bs, L, ABET_SIZE)
        phi_t = phi_t * txt[:,:char_seq_len, :]
        print("phi2", K.int_shape(phi_t))
        # phi_t : bs x L x abet_size
        # output
        phi_t = self.COEF * K.sum(phi_t, axis=1)
        print("phi2", K.int_shape(phi_t))
        # phi_t : bs x abet_size

        print(K.int_shape(phi_tm1), K.int_shape(k), K.int_shape(txt), K.int_shape(output))

        return K.concatenate([output, phi_t]), [K.concatenate([gru_state, phi_t]), k_t]

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] += self.abet_size
        return tuple(shape)

    def get_char_pval(self):
        return Lambda(
            lambda x: get_char_window(x, units=self.units),
            output_shape=(self.abet_size,))

    def get_gru_output(self):
        return Lambda(
            lambda x: get_gru_output(x, units=self.units),
            output_shape=(self.units,))

class GravesRnn(RNN):
    def __init__(self, *args, **kwargs):
        super(GravesRnn, self).__init__(*args, **kwargs)
        self.keep_input_spec = self.input_spec
        self.batch_size = None

    def compute_output_shape(self, input_shape):
        shapes = super(GravesRnn, self).compute_output_shape(input_shape)
        output_shape = list(shapes[0])
        assert len(output_shape) == 3  # only valid for 3D tensors
        output_shape[-1] += self.cell.abet_size
        rv = [tuple(output_shape)] + shapes[1:]
        self.batch_size = output_shape[0]

        return rv

    def build(self, input_shape):
        super(GravesRnn, self).build(input_shape)
        self.keep_input_spec = [InputSpec(shape=input_shape[0],ndim=3)]

    def reset_states(self, states=None):
        if self.built:
            self.input_spec = self.keep_input_spec
        super(GravesRnn, self).reset_states(
            [np.zeros((self.input_spec[0].shape[0], self.cell.units + self.cell.abet_size)),
             np.zeros((self.input_spec[0].shape[0], self.cell.nof_mixtures))])



def get_gru_output(o, units):
    return o[:, :units]

def get_char_window(o, units):
    return o[:, units:]

def main():
    np.random.seed(222)
    ABET_SIZE = 5
    BATCH_SIZE = 10
    UNITS = 32
    state_selector = K.zeros((), dtype=K.floatx(), name='slow_state_mask')
    rnn_h = K.variable(np.zeros((1, UNITS)), dtype=K.floatx(), name='h0')
    cell = GravesAttentionCell(UNITS,
                               nof_mixtures=10,
                               state_selector=state_selector,
                               learned_h=rnn_h,
                               abet_size=ABET_SIZE)
    data = np.ones((BATCH_SIZE, 6, UNITS+1))
    target = np.ones((BATCH_SIZE, 6, UNITS)) + 2
    txt = np.zeros((BATCH_SIZE, 3))
    txt[:, 0] = 0
    txt[:, 1] = 1
    txt[:, 2] = 3
    txt_1hot = to_categorical(txt, num_classes=ABET_SIZE)
    txt_mask = np.ones((BATCH_SIZE, 3))
    x = keras.Input(batch_shape=(BATCH_SIZE, None, data.shape[-1]))
    c = keras.Input((None, ABET_SIZE))
    c_mask = keras.Input((None,))
    layer = GravesRnn(cell,
                return_sequences=True,
                return_state=True,
                stateful=True)

    y, h0, k_offs = layer(x, constants=[c, c_mask])
    c_w = TimeDistributed(cell.get_char_pval())(y)

    y = TimeDistributed(cell.get_gru_output())(y)

    print("char win shape", K.int_shape(c_w))
    c_w_proj = TimeDistributed(Dense(units=UNITS))(c_w)
    c_w_proj2 = TimeDistributed(Dense(units=UNITS))(c_w)
    y = keras.layers.add([y, c_w_proj, c_w_proj2])
    model = keras.models.Model(inputs=[x, c, c_mask], outputs=y)
    model.compile(loss='mse', optimizer='sgd')
    print(model.summary())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from keras.utils import plot_model
        plot_model(model, to_file='graves_attention_model.png')
    except Exception as e:
        print('failed to plot models to png', e)
        pass
    model.reset_states()
    print('before fit', model.predict([data, txt_1hot, txt_mask])[0,0])
    model.fit(
        [data, txt_1hot, txt_mask],
        target,
        batch_size=BATCH_SIZE)
    model.fit(
        [data, txt_1hot, txt_mask],
        target,
        batch_size=BATCH_SIZE)
    model.reset_states()

    model.fit(
        [data[:,:-2,:], txt_1hot[:,:-2, :], txt_mask[:,:-2]],
        target[:,:-2,:],
        batch_size=BATCH_SIZE)

    print('after fit', model.predict([data, txt_1hot, txt_mask])[0,0])
    print('learned h0', K.eval(rnn_h))

# Here's how to use the cell to build a stacked RNN:
if __name__ == "__main__":
    main()


