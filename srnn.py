import numpy as np
import keras.optimizers
import keras.regularizers
import warnings
from keras import layers
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import Lambda, GRU, Reshape, GRUCell
from keras.models import Model
from keras import backend as K
from kdllib import get_vocabulary, get_code2char_char2code_maps
from kdllib import filter_tokenize_ind
from wnorm import DenseWithWeightNorm, GRUCellWithWeightNorm, GruWithWeightNorm
import gravesatt

from sframe import SFrame

SLOW_DIM = 10  # 1st slow tier dim (top level tier with the biggest frame size)
DIM = 10  # 2&3 tier dim with smaller frame size
Q_LEVELS = 2  # quantization number of steps
SLOW_FS = 8
MID_FS = 2
SEQ_LEN = 32
OVERLAP = SLOW_FS
SUB_SEQ_LEN = 16
N_TRAIN = 20
BATCH_SIZE = N_TRAIN
ABET_SIZE = 31




def scale_samples_for_rnn(frames, q_levels):
    frames = (K.cast(frames, dtype='float32') / K.cast_to_floatx(
        q_levels / 2)) - K.cast_to_floatx(1)
    frames *= K.cast_to_floatx(2)
    return frames



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
        mid_seq_len = max(1, seq_len // mid_fs)
        prev_sample_seq_len = seq_len + 1

        ################################################################################
        ################## Model to train
        ################################################################################

        self.slow_tier_model_input = Input(
            batch_shape=(batch_size, slow_seq_len * slow_fs, 1))
        self.slow_tier_model_input_txt = keras.Input(batch_shape=(batch_size, None, ABET_SIZE))
        self.slow_tier_model_input_txt_mask = keras.Input(batch_shape=(batch_size, None))

        self.slow_tier_model = Lambda(
            lambda x: scale_samples_for_rnn(x, q_levels=q_levels),
            name='slow_scale')(self.slow_tier_model_input)
        self.slow_tier_model = Reshape(
            (slow_seq_len, self.slow_fs),
            name='slow_reshape4rnn')(self.slow_tier_model)

        self.slow_rnn_h = K.variable(
            np.zeros((1, self.slow_dim)), dtype=K.floatx(), name='show_h0')
        self.mid_rnn_h = K.variable(
            np.zeros((1, self.dim)), dtype=K.floatx(), name='mid_h0')
        self.mid_rnn_h0 = K.tile(self.mid_rnn_h, (batch_size, 1))

        self.state_selector = K.zeros(
            (), dtype=K.floatx(), name='slow_state_mask')
        self.slow_rnn_cell = gravesatt.GravesAttentionCell(slow_dim,
                                                 nof_mixtures=10,
                                                 state_selector=self.state_selector,
                                                 learned_h=self.slow_rnn_h,
                                                 abet_size=ABET_SIZE
                                                 )
        self.slow_rnn = gravesatt.GravesRnn(self.slow_rnn_cell,
                                  return_sequences=True,
                                  return_state=True,
                                  stateful=True)



        slow_output, slow_rnn_state, k_offset_slow = self.slow_rnn(
            self.slow_tier_model,
            constants=[self.slow_tier_model_input_txt,
                       self.slow_tier_model_input_txt_mask]
        )
        print('shape of offset', K.int_shape(k_offset_slow))
        c_w = TimeDistributed(self.slow_rnn_cell.get_char_pval())(slow_output)

        self.slow_tier_model = TimeDistributed(self.slow_rnn_cell.get_gru_output())(slow_output)
        print("char win shape", K.int_shape(c_w))
        phi_slow = TimeDistributed(self.slow_rnn_cell.get_char_pval())(slow_output)
        c_w_proj_slow = TimeDistributed(
            DenseWithWeightNorm(slow_dim,
                                weight_norm=self.weight_norm),
            name='char_win2slow')(c_w)
        self.slow_tier_model = layers.add(
            [c_w_proj_slow, self.slow_tier_model])
        # upscale slow rnn output to mid tier ticking freq
        self.slow_tier_model = TimeDistributed(
            DenseWithWeightNorm(dim * slow_fs / mid_fs,
                                weight_norm=self.weight_norm,
                                ), name='slow_project2mid') \
            (self.slow_tier_model)
        self.slow_tier_model = Reshape(
            (mid_seq_len, dim), name='slow_reshape4mid')(self.slow_tier_model)

        self.mid_tier_model_input = Input(
            batch_shape=(batch_size, mid_seq_len * mid_fs, 1))
        self.mid_tier_model = Lambda(
            lambda x: scale_samples_for_rnn(x, q_levels=q_levels),
            name='mid_scale')(self.mid_tier_model_input)
        self.mid_tier_model = Reshape(
            (mid_seq_len, self.mid_fs),
            name='mid_reshape2rnn')(self.mid_tier_model)
        mid_proj = DenseWithWeightNorm(
            dim, name='mid_project2rnn', weight_norm=self.weight_norm)
        self.mid_tier_model = TimeDistributed(
            mid_proj, name='mid_project2rnn')(self.mid_tier_model)
        self.mid_tier_model = layers.add(
            [self.mid_tier_model, self.slow_tier_model])
        self.mid_rnn = GruWithWeightNorm(
            dim,
            name='mid_rnn',
            return_sequences=True,
            recurrent_activation='sigmoid',
            stateful=self.stateful,
            state_selector=self.state_selector)

        self.mid_rnn.cell._trainable_weights.append(self.mid_rnn_h)
        self.mid_tier_model = self.mid_rnn(
            self.mid_tier_model, initial_state=self.mid_rnn_h0)
        self.mid_adapter = DenseWithWeightNorm(
            dim * mid_fs, name='mid_project2top', weight_norm=self.weight_norm)
        self.mid_tier_model = TimeDistributed(
            self.mid_adapter, name='mid_project2top')(self.mid_tier_model)
        self.mid_tier_model = Reshape(
            (mid_seq_len * mid_fs, dim),
            name='mid_reshape4top')(self.mid_tier_model)
        self.embed_size = 256
        self.sframe = SFrame()
        self.top_tier_model_input = self.sframe.build_sframe_model(
            (batch_size, prev_sample_seq_len, 1),
            frame_size=self.mid_fs,
            q_levels=self.q_levels,
            embed_size=self.embed_size)
        self.top_adapter = DenseWithWeightNorm(
            dim,
            use_bias=False,
            name='top_project2mlp',
            kernel_initializer='lecun_uniform',
            weight_norm=self.weight_norm)
        self.top_tier_model = TimeDistributed(
            self.top_adapter,
            name='top_project2mpl')(self.top_tier_model_input.output)

        self.top_tier_model_input_from_mid_tier = Input(
            batch_shape=(batch_size, 1, dim))
        self.top_tier_model_input_predictor = Input(
            batch_shape=(batch_size, mid_fs, 1))
        self.top_tier_model = layers.add(
            [self.mid_tier_model, self.top_tier_model])

        self.top_tier_mlp_l1 = DenseWithWeightNorm(
            dim,
            activation=mlp_activation,
            name='mlp_1',
            weight_norm=self.weight_norm)
        self.top_tier_mlp_l2 = DenseWithWeightNorm(
            dim,
            activation=mlp_activation,
            name='mlp_2',
            weight_norm=self.weight_norm)
        self.top_tier_mlp_l3 = DenseWithWeightNorm(
            q_levels,
            kernel_initializer='lecun_uniform',
            name='mlp_3',
            weight_norm=self.weight_norm)

        self.top_tier_model = TimeDistributed(
            self.top_tier_mlp_l1, name='mlp_1')(self.top_tier_model)
        self.top_tier_model = TimeDistributed(
            self.top_tier_mlp_l2, name='mlp_2')(self.top_tier_model)
        self.top_tier_model = TimeDistributed(
            self.top_tier_mlp_l3, name='mlp_3')(self.top_tier_model)

        self.mid_tier_model_input_from_slow_tier = Input(
            batch_shape=(batch_size, 1, dim))
        self.mid_tier_model_input_predictor = Input(
            batch_shape=(batch_size, mid_fs, 1))

        self.srnn = Model([
            self.slow_tier_model_input,
            self.mid_tier_model_input,
            self.top_tier_model_input.input,
            self.slow_tier_model_input_txt,
            self.slow_tier_model_input_txt_mask
        ], self.top_tier_model)

        ################################################################################
        ################## Model to sample from (predictor)
        ################################################################################

        ################################################################################
        ################## Slow tier predictor
        ################################################################################
        self.slow_tier_model_predictor = Model(
            inputs=[self.slow_tier_model_input,
                    self.slow_tier_model_input_txt,
                    self.slow_tier_model_input_txt_mask],
            outputs=[self.slow_tier_model, slow_rnn_state, phi_slow])

        ################################################################################
        ################## Mid tier predictor
        ################################################################################

        self.mid_tier_model_predictor = Lambda(
            lambda x: scale_samples_for_rnn(x, q_levels=q_levels))(
                self.mid_tier_model_input_predictor)
        self.mid_tier_model_predictor = Reshape(
            (1, self.mid_fs))(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = TimeDistributed(mid_proj)(
            self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = layers.add([
            self.mid_tier_model_predictor,
            self.mid_tier_model_input_from_slow_tier
        ])
        """ Creating new layer instead of sharing it with the model to train
        due to https://github.com/keras-team/keras/issues/6939
        Sharing statefull layers gives a crosstalk
        """
        self.predictor_mid_rnn = GruWithWeightNorm(
            self.dim,
            name='mid_rnn_pred',
            return_sequences=True,
            recurrent_activation='sigmoid',
            stateful=self.stateful,
            state_selector=self.state_selector)
        self.predictor_mid_rnn.cell._trainable_weights.append(self.mid_rnn_h)
        self.mid_tier_model_predictor = self.predictor_mid_rnn(
            self.mid_tier_model_predictor, initial_state=self.mid_rnn_h0)
        self.predictor_mid_rnn.set_weights(self.mid_rnn.get_weights())
        self.mid_tier_model_predictor = TimeDistributed(self.mid_adapter)(
            self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = Reshape(
            (mid_fs, dim))(self.mid_tier_model_predictor)
        self.mid_tier_model_predictor = Model([
            self.mid_tier_model_input_predictor,
            self.mid_tier_model_input_from_slow_tier
        ], self.mid_tier_model_predictor)

        ################################################################################
        ################## Top tier predictor
        ################################################################################

        self.top_predictor_embedding = self.sframe.get_embedding()
        self.top_tier_model_predictor = self.top_predictor_embedding(
            self.top_tier_model_input_predictor)
        self.top_tier_model_predictor = Reshape(
            (1, mid_fs * self.embed_size))(self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_adapter)(
            self.top_tier_model_predictor)
        self.top_tier_model_predictor = layers.add([
            self.top_tier_model_predictor,
            self.top_tier_model_input_from_mid_tier
        ])

        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l1)(
            self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l2)(
            self.top_tier_model_predictor)
        self.top_tier_model_predictor = TimeDistributed(self.top_tier_mlp_l3)(
            self.top_tier_model_predictor)

        self.top_tier_model_predictor = Model([
            self.top_tier_model_input_predictor,
            self.top_tier_model_input_from_mid_tier
        ], self.top_tier_model_predictor)

        def categorical_crossentropy(target, output):
            new_target_shape = [
                K.shape(output)[i] for i in xrange(K.ndim(output) - 1)
            ]
            output = K.reshape(output, (-1, self.q_levels))
            xdev = output - K.max(output, axis=1, keepdims=True)
            lsm = xdev - K.log(K.sum(K.exp(xdev), axis=1, keepdims=True))
            cost = -K.sum(lsm * K.reshape(target, (-1, self.q_levels)), axis=1)
            log2e = K.variable(np.float32(np.log2(np.e)))
            return K.reshape(cost, new_target_shape) * log2e

        self.srnn.compile(
            loss=categorical_crossentropy,
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
        self.srnn.save_weights(file_name)

    def load_weights(self, file_name):
        self.srnn.load_weights(file_name)
        self.predictor_mid_rnn.set_weights(self.mid_rnn.get_weights())

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
        labels_one_hot = labels_one_hot.reshape(labels_shape + (n_classes, ))
        return labels_one_hot.astype(labels_dtype)

    def _prep_batch(self, x, mask):
        x_slow = x[:, :-self.slow_fs]
        x_mid = x[:, self.slow_fs - self.mid_fs:-self.mid_fs]
        x_prev = x[:, self.slow_fs - self.mid_fs:-1]
        target = x[:, self.slow_fs:]
        target = self.numpy_one_hot(target, self.q_levels)
        if mask is None:
            mask = np.ones((x.shape[0], x.shape[1]))
        target_mask = mask[:, self.slow_fs:]
        return x_slow, x_mid, x_prev, target, target_mask

    def train_on_batch(self, x, mask, txt, txt_mask):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)

        return self.model().train_on_batch(
            [x_slow, x_mid, x_prev, txt, txt_mask], target, sample_weight=target_mask)

    def predict_on_batch(self, x, mask, txt, txt_mask):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)
        return self.model().predict_on_batch([x_slow, x_mid, x_prev, txt, txt_mask])

    def test_on_batch(self, x, mask, txt, txt_mask):
        x_slow, x_mid, x_prev, target, target_mask = self._prep_batch(x, mask)
        return self.model().test_on_batch(
            [x_slow, x_mid, x_prev, txt, txt_mask], target, sample_weight=target_mask)

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
            idxs = [
                np.argmax(random_state.multinomial(1, pvals=coeff[i]))
                for i in range(len(coeff))
            ]
            idx = np.array(idxs)
        return idx.astype(K.floatx())

    def numpy_sample_softmax(self, logits, random_state, debug=False):
        old_shape = logits.shape
        flattened_logits = logits.reshape((-1, logits.shape[logits.ndim - 1]))
        new_shape = list(old_shape)
        new_shape[-1] = 1
        samples = self.numpy_sample_softmax2d(flattened_logits, random_state,
                                              debug).reshape(new_shape)
        return samples

    def numpy_softmax(self, X, temperature=1.):
        # should work for both 2D and 3D
        dim = X.ndim
        X = X / temperature
        e_X = np.exp((X - X.max(axis=dim - 1, keepdims=True)))
        out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
        return out

    def window_plots(self, phis, windows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.title('Phis', fontsize=20)
        plt.xlabel("ascii #", fontsize=15)
        plt.ylabel("time steps", fontsize=15)
        plt.imshow(phis, interpolation='nearest', aspect='auto', cmap=cm.jet)
        plt.subplot(122)
        plt.title('Soft attention window', fontsize=20)
        plt.xlabel("one-hot vector", fontsize=15)
        plt.ylabel("time steps", fontsize=15)
        plt.imshow(windows, interpolation='nearest', aspect='auto', cmap=cm.jet)
        plt.savefig('maps.png')

    def sample(self, ts, random_state, debug, txt):
        samples = np.zeros((1, ts, 1), dtype='int32')
        Q_ZERO = self.q_levels // 2
        samples[:, :self.slow_fs] = Q_ZERO
        big_frame_level_outputs = None
        frame_level_outputs = None
        big_frame_pos_in_sentence = np.zeros((1, self.slow_rnn_cell.abet_size))
        big_frame_char_pvals = np.zeros((1, 1))
        self.set_h0_selector(False)
        txt_coded = filter_tokenize_ind(txt.lower(), get_code2char_char2code_maps(get_vocabulary())[1])
        txt_coded = np.expand_dims(txt_coded, axis=0)
        txt_mask = np.ones((1, txt_coded.shape[1]))


        for t in xrange(self.slow_fs, ts):
            if t % self.slow_fs == 0:
                big_frame_level_outputs, rnn_hid, k_offs = self.slow_tier_model_predictor. \
                    predict_on_batch([samples[:, t-self.slow_fs:t,:], txt_coded, txt_mask])
                big_frame_pos_in_sentence = np.vstack((big_frame_pos_in_sentence, k_offs))
                char_pvals = rnn_hid[0, self.slow_dim:]
                char_pvals = char_pvals.reshape((1, -1))
                big_frame_char_pvals = np.vstack((big_frame_char_pvals, char_pvals))
                print('max:', np.max(char_pvals), np.max(big_frame_pos_in_sentence))

            if t % self.mid_fs == 0:
                frame_level_outputs = self.mid_tier_model_predictor. \
                    predict_on_batch([samples[:, t-self.mid_fs:t],
                                      big_frame_level_outputs[:, (t / self.mid_fs) % (self.slow_fs / self.mid_fs)][:,np.newaxis,:]])

            sample_prob = self.top_tier_model_predictor. \
                predict_on_batch([samples[:, t-self.mid_fs:t],
                                  frame_level_outputs[:, t % self.mid_fs][:,np.newaxis,:]])
            sample_prob = self.numpy_softmax(sample_prob)
            samples[:, t] = self.numpy_sample_softmax(
                sample_prob, random_state, debug=debug > 0)
        pos_seq = big_frame_pos_in_sentence.reshape((-1, self.slow_rnn_cell.abet_size))
        char_pvals = big_frame_char_pvals.reshape((-1, self.slow_rnn_cell.abet_size))
        print ('k', np.average(pos_seq, axis=-1))
        self.window_plots(pos_seq, char_pvals)

        return samples[0].astype('float32')
