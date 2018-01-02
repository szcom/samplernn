import numpy as np
import keras.optimizers
import keras.regularizers
from keras import layers
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import Lambda, GRU, Reshape
from keras.models import Model
from keras import backend as K

from keras.engine import InputSpec

from sframe import SFrame

SLOW_DIM   = 10 # 1st slow tier dim (top level tier with the biggest frame size)
DIM    = 10 # 2&3 tier dim with smaller frame size
Q_LEVELS   = 2  # quantization number of steps
SLOW_FS    = 8
MID_FS     = 2
SEQ_LEN    = 32
OVERLAP    = SLOW_FS
SUB_SEQ_LEN = 16
N_TRAIN = 20
BATCH_SIZE = N_TRAIN

def l2norm(x, axis=0):
    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.constant(1e-7)))
    return norm
    
def weight_norm_regularizer(layer, weight):
    """Splits weight direction and norm to optimize them separately
    # Arguments
        l: Layer to apply w norm
        w: Float; Initial weights
    """
    w_norm = K.cast_to_floatx(np.linalg.norm(K.get_value(weight), axis=0))
    g = layer.add_weight("{}_{}_g".format(layer.name, weight.name.split(':')[-1]),
                         w_norm.shape,
                         initializer=keras.initializers.Constant(w_norm))
    normed_weight = weight * (g/l2norm(weight))
    return normed_weight


class GruLearnedH0(GRU):
    def __init__(self, *args, **kwargs):
        self.state_selector = kwargs.pop('state_selector')
        if 'weight_norm' in kwargs:
            self.weight_norm = kwargs.pop('weight_norm')
        else:
            self.weight_norm = True

        super(GruLearnedH0, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.weight_norm:
            self.kernel = weight_norm_regularizer(self, self.kernel)
            self.recurrent_kernel = weight_norm_regularizer(self, self.recurrent_kernel)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if not self.stateful:
            raise ValueError("Must be stateful")
        if initial_state is None:
            raise ValueError("Must have initial state to learn")

        initial_state = [K.switch(self.state_selector, init, state) for state, init in zip(self.states, initial_state)]

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output


class DenseWithWeightNorm(Dense):
    def __init__(self, dim, **kw):
        kernel_initializer='he_uniform'
        if 'kernel_initializer' in kw:
            kernel_initializer = kw.get('kernel_initializer')
            kw.pop('kernel_initializer')
        if 'weight_norm' in kw:
            self.weight_norm = kw.pop('weight_norm')
        else:
            self.weight_norm = True

        super(DenseWithWeightNorm, self).__init__(dim,
                                      kernel_initializer=kernel_initializer,
                                      **kw)
    def build(self, input_shape):
        super(DenseWithWeightNorm, self).build(input_shape)
        if self.weight_norm:
            self.kernel = weight_norm_regularizer(self, self.kernel)

def scale_samples_for_rnn(frames, q_levels):
    frames = (K.cast(frames, dtype='float32') / K.cast_to_floatx(q_levels/2)) - K.cast_to_floatx(1)
    frames *= K.cast_to_floatx(2)
    return frames


#
# input audio seq batch x samples x 1
# input slow tier samples
#   - start at 0
#   - stop at slow_fs samples before end
# input mid tier
#   - start at slow_fs - mid_fs, 8-2 = at sample nr 6
#   - stop at mid_fs samples before end
# input top tier(samples level)
#   - same as mid tier
# target
#   - start at slow_fs
#

class SRNN(object):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 seq_len=SUB_SEQ_LEN,
                 slow_fs=SLOW_FS,
                 slow_dim=SLOW_DIM,
                 dim=DIM,
                 mid_fs=MID_FS,
                 q_levels=Q_LEVELS,
                 mlp_activation='relu'):
        self.weight_norm = True
        self.stateful = True
        self.slow_fs = slow_fs
        self.mid_fs = mid_fs
        self.q_levels = q_levels
        self.dim = dim
        self.slow_dim = slow_dim
        self.batch_size = batch_size
        slow_seq_len = max(1, seq_len // slow_fs)
        mid_seq_len = max(1, seq_len  // mid_fs)
        prev_sample_seq_len = seq_len + 1
        
        ################################################################################
        ################## Model to train
        ################################################################################
        
        self.slow_tier_model_input = Input(batch_shape=(batch_size, slow_seq_len * slow_fs, 1))
        self.slow_tier_model = Lambda(lambda x: scale_samples_for_rnn(x, q_levels=q_levels), name='slow_scale')(self.slow_tier_model_input)
        self.slow_tier_model = Reshape((slow_seq_len, self.slow_fs), name='slow_reshape4rnn')(self.slow_tier_model)

        self.slow_rnn_h =  K.variable(np.zeros((1, self.slow_dim)), dtype=K.floatx(), name='show_h0')
        self.slow_rnn_h0 = K.tile(self.slow_rnn_h, (batch_size,1))
        self.mid_rnn_h = K.variable(np.zeros((1, self.dim)), dtype=K.floatx(), name='mid_h0')
        self.mid_rnn_h0 = K.tile(self.mid_rnn_h, (batch_size, 1))

        self.state_selector = K.zeros((), dtype=K.floatx(), name='slow_state_mask')
        self.slow_rnn = GruLearnedH0(slow_dim,
                                      use_bias=True,
                                      name='slow_rnn',
                                      recurrent_activation='sigmoid',
                                      return_sequences=True,
                                      stateful=self.stateful,
                                      state_selector=self.state_selector,
                                      weight_norm=self.weight_norm)
        self.slow_rnn._trainable_weights.append(self.slow_rnn_h)
        self.slow_tier_model = self.slow_rnn(self.slow_tier_model, initial_state=self.slow_rnn_h0)

        # upscale slow rnn output to mid tier ticking freq
        self.slow_tier_model = TimeDistributed(
            DenseWithWeightNorm(dim * slow_fs / mid_fs,
                    weight_norm=self.weight_norm,
                    ), name='slow_project2mid')\
            (self.slow_tier_model)
        self.slow_tier_model = Reshape((mid_seq_len, dim), name='slow_reshape4mid')(self.slow_tier_model)
                
        self.mid_tier_model_input = Input(batch_shape=(batch_size, mid_seq_len * mid_fs, 1))
        self.mid_tier_model = Lambda(lambda x: scale_samples_for_rnn(x, q_levels=q_levels), name='mid_scale')(self.mid_tier_model_input)
        self.mid_tier_model = Reshape((mid_seq_len, self.mid_fs), name='mid_reshape2rnn')(self.mid_tier_model)
        mid_proj = DenseWithWeightNorm(dim, name='mid_project2rnn', weight_norm=self.weight_norm)
        self.mid_tier_model = TimeDistributed(mid_proj, name='mid_project2rnn')(self.mid_tier_model)
        self.mid_tier_model = layers.add([self.mid_tier_model, self.slow_tier_model])
        self.mid_rnn = GruLearnedH0(dim,
                                 name='mid_rnn',
                                 return_sequences=True,
                                 recurrent_activation='sigmoid',
                                 stateful=self.stateful,
                                 state_selector=self.state_selector)

        self.mid_rnn._trainable_weights.append(self.mid_rnn_h)
        self.mid_tier_model = self.mid_rnn(self.mid_tier_model, initial_state=self.mid_rnn_h0)
        self.mid_adapter = DenseWithWeightNorm(dim * mid_fs, name='mid_project2top',
                                   weight_norm=self.weight_norm)
        self.mid_tier_model = TimeDistributed(self.mid_adapter, name='mid_project2top')(self.mid_tier_model)
        self.mid_tier_model = Reshape((mid_seq_len * mid_fs, dim), name='mid_reshape4top')(self.mid_tier_model)
        self.embed_size=256
        self.sframe = SFrame()
        self.top_tier_model_input = self.sframe.build_sframe_model((batch_size, prev_sample_seq_len, 1),
                                                              frame_size=self.mid_fs,
                                                              q_levels=self.q_levels,
                                                              embed_size=self.embed_size)
        self.top_adapter = DenseWithWeightNorm(dim,
                                   use_bias=False,
                                   name='top_project2mlp',
                                   kernel_initializer='lecun_uniform',
                                   weight_norm=self.weight_norm)
        self.top_tier_model = TimeDistributed(self.top_adapter, name='top_project2mpl')(self.top_tier_model_input.output)

        self.top_tier_model_input_from_mid_tier = Input(batch_shape=(batch_size, 1, dim))
        self.top_tier_model_input_predictor = Input(batch_shape=(batch_size, mid_fs, 1))
        self.top_tier_model = layers.add([self.mid_tier_model, self.top_tier_model])

        self.top_tier_mlp_l1 = DenseWithWeightNorm(dim, activation=mlp_activation, name='mlp_1', weight_norm=self.weight_norm)
        self.top_tier_mlp_l2 = DenseWithWeightNorm(dim, activation=mlp_activation, name='mlp_2', weight_norm=self.weight_norm)
        self.top_tier_mlp_l3 = DenseWithWeightNorm(q_levels,
                                       kernel_initializer='lecun_uniform',
                                       name='mlp_3',
                                       weight_norm=self.weight_norm)

        self.top_tier_model = TimeDistributed(self.top_tier_mlp_l1, name='mlp_1')(self.top_tier_model)
        self.top_tier_model = TimeDistributed(self.top_tier_mlp_l2, name='mlp_2')(self.top_tier_model)
        self.top_tier_model = TimeDistributed(self.top_tier_mlp_l3, name='mlp_3')(self.top_tier_model)


        self.mid_tier_model_input_from_slow_tier = Input(batch_shape=(batch_size, 1, dim))
        self.mid_tier_model_input_predictor = Input(batch_shape=(batch_size, mid_fs, 1))
        
        self.srnn = Model([self.slow_tier_model_input, self.mid_tier_model_input,
                           self.top_tier_model_input.input],
                          self.top_tier_model)
        
        ################################################################################
        ################## Model to sample from (predictor)
        ################################################################################

        ################################################################################
        ################## Slow tier predictor
        ################################################################################
        self.slow_tier_model_predictor = Model(inputs=self.slow_tier_model_input,
                                               outputs=self.slow_tier_model)


        ################################################################################
        ################## Mid tier predictor
        ################################################################################
        
        self.mid_tier_model_predictor = Lambda(lambda x: scale_samples_for_rnn(x, q_levels=q_levels))(self.mid_tier_model_input_predictor)
        self.mid_tier_model_predictor = Reshape((1, self.mid_fs))(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = TimeDistributed(mid_proj)(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = layers.add([self.mid_tier_model_predictor, self.mid_tier_model_input_from_slow_tier])
        """ Creating new layer instead of sharing it with the model to train
        due to https://github.com/keras-team/keras/issues/6939
        Sharing statefull layers gives a crosstalk
        """
        self.predictor_mid_rnn = GruLearnedH0(self.dim,
                               name='mid_rnn',
                               return_sequences=True,
                               recurrent_activation='sigmoid',
                               stateful=self.stateful,
                               state_selector=self.state_selector)
        self.mid_tier_model_predictor = self.predictor_mid_rnn(self.mid_tier_model_predictor, initial_state=self.mid_rnn_h0)
        self.predictor_mid_rnn.set_weights(self.mid_rnn.get_weights()[1:])
        self.mid_tier_model_predictor = TimeDistributed(self.mid_adapter)(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = Reshape((mid_fs, dim))(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = Model([self.mid_tier_model_input_predictor,
                                               self.mid_tier_model_input_from_slow_tier],
                                              self.mid_tier_model_predictor)
        
        ################################################################################
        ################## Top tier predictor
        ################################################################################

        self.top_predictor_embedding = self.sframe.get_embedding()
        self.top_tier_model_predictor = self.top_predictor_embedding(self.top_tier_model_input_predictor)
        self.top_tier_model_predictor = Reshape((1, mid_fs*self.embed_size))(self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_adapter)(self.top_tier_model_predictor)
        self.top_tier_model_predictor = layers.add([self.top_tier_model_predictor, self.top_tier_model_input_from_mid_tier])

        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l1)(self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l2)(self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l3)(self.top_tier_model_predictor)

        self.top_tier_model_predictor = Model([self.top_tier_model_input_predictor,
                                               self.top_tier_model_input_from_mid_tier],
                                              self.top_tier_model_predictor)

        def categorical_crossentropy(target, output):
            new_target_shape = [
                K.shape(output)[i]
                for i in xrange(K.ndim(output) - 1)
            ]
            output = K.reshape(output, (-1, self.q_levels))
            xdev = output - K.max(output, axis=1, keepdims=True)
            lsm = xdev - K.log(K.sum(K.exp(xdev), axis=1, keepdims=True))
            cost = - K.sum(lsm * K.reshape(target, (-1, self.q_levels)), axis=1)
            log2e = K.variable(np.float32(np.log2(np.e)))
            print ('a', new_target_shape)
            return K.reshape(cost, new_target_shape) * log2e
        
        self.srnn.compile(loss=categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(clipvalue=1.),
                          sample_weight_mode='temporal')
        
    def set_h0_selector(self, use_learned_h0):
        if use_learned_h0:
            self.srnn.reset_states()
            self.slow_rnn.reset_states()
            self.mid_rnn.reset_states()
            self.slow_tier_model_predictor.reset_states()
            self.mid_tier_model_predictor.reset_states()
            K.set_value(self.state_selector, np.ones(()))
        else:
            K.set_value(self.state_selector, np.zeros(()))
        
    def save_weights(self, file_name):
        self.mid_rnn._trainable_weights.remove(self.mid_rnn_h)
        self.slow_rnn._trainable_weights.remove(self.slow_rnn_h)

        self.srnn.save_weights(file_name)

        self.mid_rnn._trainable_weights.append(self.mid_rnn_h)
        self.slow_rnn._trainable_weights.append(self.slow_rnn_h)

    def load_weights(self, file_name):
        self.mid_rnn._trainable_weights.remove(self.mid_rnn_h)
        self.slow_rnn._trainable_weights.remove(self.slow_rnn_h)
        self.srnn.load_weights(file_name)
        self.predictor_mid_rnn.set_weights(self.mid_rnn.get_weights())
        self.mid_rnn._trainable_weights.append(self.mid_rnn_h)
        self.slow_rnn._trainable_weights.append(self.slow_rnn_h)

    def numpy_one_hot(self, labels_dense, n_classes):
        """Convert class labels from scalars to one-hot vectors."""
        labels_shape = labels_dense.shape[:-1]
        labels_dtype = labels_dense.dtype
        labels_dense = labels_dense.ravel().astype("int32")
        n_labels = labels_dense.shape[0]
        index_offset = np.arange(n_labels) * n_classes
        labels_one_hot = np.zeros((n_labels, n_classes))
        labels_one_hot[np.arange(n_labels).astype("int32"),
                       labels_dense.ravel()] = 1
        labels_one_hot = labels_one_hot.reshape(labels_shape+(n_classes,))
        return labels_one_hot.astype(labels_dtype)

    def _prep_batch(self, x, mask):
        x_slow = x[:, :-self.slow_fs]
        x_mid = x[:, self.slow_fs-self.mid_fs:-self.mid_fs]
        x_prev = x[:, self.slow_fs-self.mid_fs:-1]
        target = x[:, self.slow_fs:]
        target = self.numpy_one_hot(target, self.q_levels)
        if mask is None:
            mask = np.ones((x.shape[0], x.shape[1]))
        target_mask = mask[:, self.slow_fs:]
        return x_slow, x_mid, x_prev, target, target_mask
    
    def train_on_batch(self, x, mask=None):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)

        return self.model().train_on_batch([x_slow, x_mid, x_prev], target,
                                            sample_weight=target_mask)


    def predict_on_batch(self, x, mask=None):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)
        return self.model().predict_on_batch([x_slow, x_mid, x_prev])
    
    def test_on_batch(self, x, mask=None):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)
        return self.model().test_on_batch([x_slow, x_mid, x_prev], target,
                                           sample_weight=target_mask)

    def model(self):
        return self.srnn
    
    def numpy_sample_softmax2d(self, coeff, random_state, debug=False):
        if coeff.ndim > 2:
            raise ValueError("Unsupported dim")
        if debug:
            idx = coeff.argmax(axis=1)
        else:
            # renormalize to avoid numpy errors about summation...
            coeff = coeff / (coeff.sum(axis=1, keepdims=True) + 1E-6)
            idxs = [np.argmax(random_state.multinomial(1, pvals=coeff[i]))
                    for i in range(len(coeff))]
            idx = np.array(idxs)
        return idx.astype(K.floatx())

    def numpy_sample_softmax(self, logits, random_state, debug=False):
        old_shape = logits.shape
        flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
        new_shape = list(old_shape)
        new_shape[-1] = 1
        samples = self.numpy_sample_softmax2d(flattened_logits, random_state, debug).reshape(new_shape)
        return samples

    def numpy_softmax(self, X, temperature=1.):
        # should work for both 2D and 3D
        dim = X.ndim
        X = X / temperature
        e_X = np.exp((X - X.max(axis=dim - 1, keepdims=True)))
        out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
        return out

    def sample(self, ts, random_state, debug):
        samples = np.zeros((1, ts, 1), dtype='int32')
        Q_ZERO=self.q_levels // 2
        samples[:, :self.slow_fs] = Q_ZERO
        big_frame_level_outputs = None
        frame_level_outputs = None
        self.set_h0_selector(False)

        for t in xrange(self.slow_fs, ts):
            if t % self.slow_fs == 0:
                big_frame_level_outputs = self.slow_tier_model_predictor. \
                                          predict_on_batch([samples[:, t-self.slow_fs:t,:]])

            if t % self.mid_fs == 0:
                frame_level_outputs = self.mid_tier_model_predictor. \
                                      predict_on_batch([samples[:, t-self.mid_fs:t],
                                                        big_frame_level_outputs[:, (t / self.mid_fs) % (self.slow_fs / self.mid_fs)][:,np.newaxis,:]])

            sample_prob = self.top_tier_model_predictor. \
                            predict_on_batch([samples[:, t-self.mid_fs:t],
                                              frame_level_outputs[:, t % self.mid_fs][:,np.newaxis,:]])
            sample_prob = self.numpy_softmax(sample_prob)
            samples[:, t] = self.numpy_sample_softmax(sample_prob, random_state, debug=debug>0)
        return samples[0].astype('float32')
    


