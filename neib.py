import tensorflow as tf
from keras import backend as K

def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid border mode:', border_mode)
    return padding

def extract_image_patches(X, ksizes, ssizes, border_mode="same", dim_ordering="tf"):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    ssizes : 2-d tuple with the strides size
    border_mode : 'same' or 'valid'
    dim_ordering : 'tf' or 'th'

    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_border_mode(border_mode)
    if dim_ordering == "th":
        X = K.permute_dimensions(X, [0, 2, 3, 1])
    bs_i, w_i, h_i, ch_i = K.int_shape(X)
    #print(bs_i, w_i, h_i, ch_i, 'sh')
    # 1=bs,1=w,6=h,4=chan
    # 1, 1, 6, 4,
    # org 4, 6, 1
    # 1, 1, 5, 8
    print(X)
    patches = tf.extract_image_patches(X, kernel, strides, [1, 1, 1, 1], padding)
    # Reshaping to fit Theano
    bs, w, h, ch = K.int_shape(patches)
    #print(bs, w, h, ch, 'sh patches')
    patches = tf.reshape(tf.transpose(tf.reshape(patches, [bs, w, h, -1, ch_i]), [0, 1, 2, 4, 3]),
                         [bs, w, h, ch_i, ksizes[0], ksizes[1]])
    if dim_ordering == "tf":
        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches

