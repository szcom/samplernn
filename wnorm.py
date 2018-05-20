import warnings
import numpy as np
import keras.optimizers
import keras.regularizers
from keras import layers
from keras.layers import Dense
from keras.layers import GRU, GRUCell
from keras import backend as K

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
    g = layer.add_weight(
        name="{}_{}_g".format(layer.name,
                         weight.name.split(':')[-1]),
        shape=w_norm.shape,
        initializer=keras.initializers.Constant(w_norm))
    normed_weight = weight * (g / l2norm(weight))
    return normed_weight

class GRUCellWithWeightNorm(GRUCell):
    def __init__(self, *args, **kwargs):

        self.weight_norm = kwargs.pop('weight_norm', True)
        super(GRUCellWithWeightNorm, self).__init__(*args, **kwargs)


    def add_weight(self, *args, **kwargs):
        name = kwargs['name']
        print("parameters name", name)
        w = super(GRUCellWithWeightNorm, self).add_weight(*args, **kwargs)
        if self.weight_norm == False:
            return w
        if name == "kernel" or name == "recurrent_kernel":
            print("do weight norm", name)
            return weight_norm_regularizer(self, w)
        return w


class GruWithWeightNorm(GRU):
    def __init__(self,
                 *args,
                 **kwargs):
        self.state_selector = kwargs.pop('state_selector', None)
        self.weight_norm = kwargs.pop('weight_norm', True)
        super(GruWithWeightNorm, self).__init__(*args,
                                  **kwargs)
        kwargs.pop('stateful')
        kwargs.pop('return_sequences')
        kwargs['weight_norm'] = self.weight_norm
        self.cell = GRUCellWithWeightNorm(*args,
                       **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        initial_state = [
            K.switch(self.state_selector, init, state)
            for state, init in zip(self.states, initial_state)
        ]
        return super(GruWithWeightNorm, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)





class DenseWithWeightNorm(Dense):
    def __init__(self, dim, **kw):
        kernel_initializer = 'he_uniform'
        if 'kernel_initializer' in kw:
            kernel_initializer = kw.get('kernel_initializer')
            kw.pop('kernel_initializer')
        if 'weight_norm' in kw:
            self.weight_norm = kw.pop('weight_norm')
        else:
            self.weight_norm = True

        super(DenseWithWeightNorm, self).__init__(
            dim, kernel_initializer=kernel_initializer, **kw)

    def build(self, input_shape):
        super(DenseWithWeightNorm, self).build(input_shape)
        if self.weight_norm:
            self.kernel = weight_norm_regularizer(self, self.kernel)
