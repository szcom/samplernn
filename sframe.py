import numpy as np
import math
import keras.optimizers
from keras import initializers
from keras import layers
from keras.layers import merge
from keras.layers import Input, Dense, TimeDistributed, Embedding, Flatten
from keras.layers import Lambda, Layer, LSTM, GRU, SimpleRNN, Reshape, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives, activations

from numpy.random import choice
from neib import extract_image_patches


def neib_tm1_mid_fs(x, fs_n, bs_n, ts_n, item_size_n=1):

    if K.backend() == 'theano':
        import theano
        x = K.reshape(x, (1, bs_n, 1, -1))
        x = theano.tensor.nnet.neighbours.images2neibs(x, (1, fs_n*item_size_n), neib_step=(1, item_size_n), mode='valid')
        return K.reshape(x, (bs_n, -1, fs_n*item_size_n))
    #x = K.reshape(x, (1, K.shape(x)[0], 1, K.prod(K.shape(x)[1:])))
    x = K.reshape(x, (1, bs_n, 1, ts_n*item_size_n))
    n = extract_image_patches(x, (1, fs_n*item_size_n), (1, item_size_n), dim_ordering='th', border_mode='valid')
    n = K.permute_dimensions(n, [0, 3, 1, 2, 4, 5])

    return K.reshape(n, (bs_n, -1, fs_n*item_size_n))

class SFrame(object):
    def get_sframe_model_noembed(self, batch_shape, frame_size=2, q_levels=0, embed_size=0):
        self.model_input = Input(batch_shape=batch_shape)
        self.model = Reshape((1, np.prod(batch_shape[1:])), name='sh_1d')(self.model_input)
        self.model = Lambda(lambda x: neib_tm1_mid_fs(x, frame_size, batch_shape[0], batch_shape[1]),
                            output_shape=(batch_shape[1]-1, frame_size), name='neib_1d')(self.model)
        self.framer = Model([self.model_input], self.model)

        return self.framer

    def get_embedding(self):
        return self.embedding

    def get_model(self):
        return self.framer
    
    def build_sframe_model(self, batch_shape, frame_size=2, q_levels=256, embed_size=64):
        # batch_shape: batch_size, prev_sample_seq_len==seq_len+1, 1
        self.model_input = Input(batch_shape=batch_shape)

        np.random.seed(123) # TURN OFF ZZZ SEED randn
        #TURN OFF Embed init
        w = np.random.randn(q_levels, embed_size).astype('float32')

        self.embedding = Embedding(q_levels,
                                   embed_size,
                                   #TURN OFF Embed init embeddings_initializer=initializers.RandomNormal(stddev=1.),
                                   embeddings_initializer=initializers.constant(w),
                                   name='embedding_{}'.format(embed_size)) # , input_length=2*(batch_shape[1]-1))
        self.model = self.embedding(self.model_input)
        self.model = Reshape((1, np.prod(batch_shape[1:]) * embed_size), name='sh_1d')(self.model)
        self.model = Lambda(lambda x: neib_tm1_mid_fs(x, frame_size, batch_shape[0], batch_shape[1], item_size_n=embed_size),
                            output_shape=(batch_shape[1]-1, frame_size * embed_size),
                            name='neib_1b')(self.model)

#        self.model = Lambda(lambda x: K.reshape(x, (batch_shape[0], batch_shape[1]-1, frame_size*embed_size)),
#                            output_shape=(batch_shape[1]-1, frame_size*embed_size),
#                            name='sh_SLM1_512')(self.model)
#                            #)(self.model)
        self.model = Reshape((batch_shape[1]-1, frame_size*embed_size))(self.model)
        self.framer = Model([self.model_input], self.model)
        for w in self.embedding.weights:
            w.nonorm = True

        return self.framer
if __name__ == "__main__":
    import theano
    import theano.tensor as T
    from keras.backend import theano_backend as KTH

    np.random.seed(1337)
    X=np.arange(4*6*1).reshape(4,6,1)
    X[3,0] = 0
    xtf = K.variable(X, dtype='float32')
    rx1 = K.eval(neib_tm1_mid_fs(xtf, 2, X.shape[0], X.shape[1]))
    print(rx1)

    xtf_theano = KTH.variable(X, dtype='float32')
    xtf_theano = xtf_theano.reshape((1, 4, 1, -1))

    xtf_theano_nn = theano.tensor.nnet.neighbours.images2neibs(xtf_theano, (1, 2), neib_step=(1, 1), mode='valid')
    rx1_theano = KTH.eval(xtf_theano_nn).reshape((4,5,2))
    
    #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    #raise ValueError()
    sframe = SFrame()
    m = sframe.get_sframe_model_noembed((4,6,1))#X.shape)
    rx2 = m.predict_on_batch([X])
    print('rx2', rx2)
    assert np.all(rx2 == rx1_theano)
    assert np.all(rx1 == rx2)
    m = sframe.get_sframe_model((4,6,1), q_levels=256, embed_size=4)#X.shape)
    rx3 = m.predict_on_batch([X])
    print('rx3', rx3)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    raise ValueError()
    
    
