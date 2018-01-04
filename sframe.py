import numpy as np
from keras import initializers
from keras.layers import Input, Embedding
from keras.layers import Lambda, Reshape
from keras.models import Model
from keras import backend as K


def neib_tm1_mid_fs(x, fs_n, bs_n, ts_n, item_size_n=1):

    if K.backend() == 'theano':
        import theano
        x = K.reshape(x, (1, bs_n, 1, -1))
        x = theano.tensor.nnet.neighbours.images2neibs(
            x, (1, fs_n * item_size_n),
            neib_step=(1, item_size_n),
            mode='valid')
        return K.reshape(x, (bs_n, -1, fs_n * item_size_n))
    from neib import extract_image_patches
    x = K.reshape(x, (1, bs_n, 1, ts_n * item_size_n))
    n = extract_image_patches(
        x, (1, fs_n * item_size_n), (1, item_size_n),
        dim_ordering='th',
        border_mode='valid')
    n = K.permute_dimensions(n, [0, 3, 1, 2, 4, 5])

    return K.reshape(n, (bs_n, -1, fs_n * item_size_n))


class SFrame(object):
    def get_sframe_model_noembed(self,
                                 batch_shape,
                                 frame_size=2,
                                 q_levels=0,
                                 embed_size=0):
        self.model_input = Input(batch_shape=batch_shape)
        self.model = Reshape(
            (1, np.prod(batch_shape[1:])), name='sh_1d')(self.model_input)
        self.model = Lambda(
            lambda x: neib_tm1_mid_fs(x, frame_size, batch_shape[0], batch_shape[1]),
            output_shape=(batch_shape[1] - 1, frame_size),
            name='neib_1d')(self.model)
        self.framer = Model([self.model_input], self.model)

        return self.framer

    def get_embedding(self):
        return self.embedding

    def get_model(self):
        return self.framer

    def build_sframe_model(self,
                           batch_shape,
                           frame_size=2,
                           q_levels=256,
                           embed_size=64):
        # batch_shape: batch_size, prev_sample_seq_len==seq_len+1, 1
        self.model_input = Input(batch_shape=batch_shape)

        self.embedding = Embedding(
            q_levels,
            embed_size,
            embeddings_initializer=initializers.RandomNormal(stddev=1.),
            name='embedding_{}'.format(embed_size))
        self.model = self.embedding(self.model_input)
        self.model = Reshape(
            (1, np.prod(batch_shape[1:]) * embed_size),
            name='sh_1d')(self.model)
        self.model = Lambda(lambda x: neib_tm1_mid_fs(x, frame_size, batch_shape[0], batch_shape[1], item_size_n=embed_size),
                            output_shape=(batch_shape[1]-1, frame_size * embed_size),
                            name='neib_1b')(self.model)
        self.model = Reshape((batch_shape[1] - 1,
                              frame_size * embed_size))(self.model)
        self.framer = Model([self.model_input], self.model)
        return self.framer
