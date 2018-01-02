from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.models import Model
import keras
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import soundsc
from kdllib import Blizzard_dataset, wav_to_qbins_frames
from kdllib import Blizzard_dataset_adapter
from keras import backend as K
from keras import objectives
import keras
from srnn import SRNN
import argparse
import time

#theano.config.compute_test_value = 'warn'
SLOW_DIM=1024
DIM=1024
SLOW_FS=8
parser = argparse.ArgumentParser(
    description='Train or sample the samplernn.')
parser.add_argument('--exp', help='Experiment size name (tiny|all)',
                    type=str, required=False, default='tiny')
parser.add_argument('--sample', help='path to weights file to load before sampling',
                    type=str, required=False, default=None)
parser.add_argument('--load', help='path to weights file to resume training',
                    type=str, required=False, default=None)
parser.add_argument('--slowdim', help='slow or 3rd tier layer size', type=int, required=False, default=1024)
parser.add_argument('--dim', help='1&2 tier layers size ', type=int, required=False, default=1024)
parser.add_argument('--nepochs', help='number of epochs to run ', type=int, required=False, default=None)
parser.add_argument('--batchsize', help='mini batch size ', type=int, required=False, default=None)
parser.add_argument('--trainstop', help='max index for traininging sequences ', type=int, required=False, default=None)
parser.add_argument('--validstop', help='max index for valid sequences ', type=int, required=False, default=None)
parser.add_argument('--svepoch', help='save weights every epoch', type=int, required=False, default=-1)
parser.add_argument('--cutlen', help='timesteps per subsequence', type=int, required=False, default=256)
parser.add_argument('--debug', help='dont sample softmax', type=int, required=False, default=0)
parser.add_argument('--samplerate', help='number of epochs to run ', type=int, required=False, default=16000)

args = parser.parse_args()
SLOW_DIM=args.slowdim
DIM=args.dim
cut_len=256
valid_stop_index= -1
train_stop_index=0.8

if args.exp == 'tiny':
   n_epochs         =10
   train_stop_index = 4
   minibatch_size   = 2
   valid_stop_index = 6

if args.exp == 'all':
   n_epochs         = 3
   minibatch_size   = 100
   valid_stop_index = 9000
   train_stop_index = 8000

cut_len = args.cutlen

if args.nepochs:
    n_epochs = args.nepochs

if args.batchsize:
    minibatch_size = args.batchsize

if args.trainstop:
   train_stop_index=args.trainstop
if args.validstop:
   valid_stop_index=args.validstop

    
frame_size = 1
bliz_train = Blizzard_dataset(minibatch_size=minibatch_size,
                              wav_folder_path='./blizzard/{}_parts/'.format(args.exp),
                              prompt_path='./blizzard/{}_parts/prompts.txt'.format(args.exp),
                              preproc_fn=wav_to_qbins_frames,
                              frame_size=frame_size,
                              fraction_range=[0, train_stop_index],
                              thread_cnt=1)
bliz_valid = Blizzard_dataset(minibatch_size=minibatch_size,
                              wav_folder_path='./blizzard/{}_parts/'.format(args.exp),
                              prompt_path='./blizzard/{}_parts/prompts.txt'.format(args.exp),
                              preproc_fn=wav_to_qbins_frames,
                              frame_size=frame_size,
                              fraction_range=[train_stop_index, valid_stop_index],
                              thread_cnt=1)

random_state = np.random.RandomState(1999)
np.random.seed(1337)



if args.sample:
    print('will do sampling')

    pred_srnn = SRNN(batch_size=1,
                 seq_len=SLOW_FS,
                 slow_fs=SLOW_FS,
                 slow_dim=SLOW_DIM,
                 dim=DIM,
                 mid_fs=2,
                 q_levels=256,
                 mlp_activation='relu')
    pred_srnn.load_weights(args.sample)
    print(pred_srnn.model().summary())

    try:
        from keras.utils import plot_model
        plot_model(pred_srnn.top_tier_model_predictor, to_file='top_model.png')
        plot_model(pred_srnn.mid_tier_model_predictor, to_file='mid_model.png')
        plot_model(pred_srnn.slow_tier_model_predictor, to_file='slow_model.png')
    except:
        print ('failed to plot models to png')
        pass
        
    w = pred_srnn.sample(4 * args.samplerate, random_state, args.debug)
    fs = args.samplerate
    wavfile.write("generated.wav", fs, soundsc(w))
    exit(-1)

srnn = SRNN(batch_size=minibatch_size,
            seq_len=cut_len,
            slow_fs=SLOW_FS,
            slow_dim=SLOW_DIM,
            dim=DIM,
            mid_fs=2,
            q_levels=256,
            mlp_activation='relu')

if args.load:
    print('will load weights from {}'.format(args.load))
    srnn.load_weights(args.load)

print(srnn.model().summary()) #.to_json(indent=4, separators=(',', ': ')))

try:
    from keras.utils import plot_model
    plot_model(srnn.model(), to_file='model.png')
except:
    print ('failed to plot models to png')
    pass

history_train_loss = []
history_valid_loss = []
total_iterations = 0
for epoch in range(n_epochs):
    t1 = time.time()
    
    epoch_train_loss = []
    epoch_valid_loss = []
    try:
        progbar = keras.utils.generic_utils.Progbar(train_stop_index)
        train_itr = iter(Blizzard_dataset_adapter(bliz_train, cut_len=cut_len, overlap=SLOW_FS))
        while True:
            total_iterations += 1
            x_part, x_mask_part, c_mb, c_mb_mask, reset = next(train_itr)
            srnn.set_h0_selector(reset)
            l = srnn.train_on_batch(x_part, x_mask_part)
            if reset:
                progbar.add(x_part.shape[0], values=[("train loss", l)])
            epoch_train_loss.append(l)

    except KeyboardInterrupt:
        bliz_train.reset()
        exit(-1)
    except StopIteration:
        pass

    print ("Epoch %s training loss in bits(%s) iters (%s)"
           %(epoch,  np.mean(epoch_train_loss),
             total_iterations))
    history_train_loss.append(np.mean(epoch_train_loss))

    if np.any(np.isnan(history_train_loss)):
        exit(-1)
    try:
        valid_itr = iter(Blizzard_dataset_adapter(bliz_valid, cut_len=cut_len, overlap=SLOW_FS))
        while True:
            x_part, x_mask_part, c_mb, c_mb_mask, reset = next(valid_itr)
            srnn.set_h0_selector(reset)
            l = srnn.test_on_batch(x_part, x_mask_part)
            epoch_valid_loss.append(l)
            # print("Validation cost:", l * np.log2(np.e), "This lh0.mean()", K.get_value(srnn.slow_lstm_h).mean())

    except KeyboardInterrupt:
        bliz_valid.reset()
        exit(-1)
    except StopIteration:
        pass
    print("Epoch %s valid loss in bits(%s)"
          %(epoch, np.mean(epoch_valid_loss)))
    history_valid_loss.append(np.mean(epoch_valid_loss))
    t2 = time.time()
    print("Epoch took %s seconds" %(t2-t1))
    if args.svepoch > 0 and (epoch % args.svepoch) == 0:
        srnn.save_weights('{}_srnn_sz{}_e{}_{}.h5'.format(args.exp, DIM, epoch, K.backend()))
        
    
    

plt.figure()

plt.plot(range(len(history_train_loss)), history_train_loss)
plt.plot(range(len(history_valid_loss)), history_valid_loss)
plt.savefig('costs.png')

srnn.save_weights('{}_srnn_sz{}_e{}.h5'.format(args.exp, DIM, epoch))
'''
THEANO_FLAGS='device=cpu' KERAS_BACKEND=tensorflow python train_srnn.py --exp=fruit --nepochs=20 --slowdim=5  --dim=5

Epoch 0 traning loss 5.433
Epoch 0 valid loss 5.17164
Epoch took 180.773623943 seconds
Epoch 1 traning loss 5.09957
Epoch 1 valid loss 4.77
Epoch took 168.445179939 seconds
Epoch 2 traning loss 4.91065
Epoch 2 valid loss 4.64064
Epoch took 168.261195183 seconds
Epoch 3 traning loss 4.84119
Epoch 3 valid loss 4.59733
Epoch took 167.280063868 seconds
Epoch 4 traning loss 4.79807
Epoch 4 valid loss 4.48115
Epoch took 168.029666901 seconds
Epoch 5 traning loss 4.72931
Epoch 5 valid loss 4.37493
Epoch took 167.541417837 seconds
Epoch 6 traning loss 4.66399
Epoch 6 valid loss 4.25709
Epoch took 166.553129911 seconds
Epoch 7 traning loss 4.61994
Epoch 7 valid loss 4.19107
Epoch took 168.856056213 seconds
Epoch 8 traning loss 4.58937
Epoch 8 valid loss 4.15351
Epoch took 167.383136988 seconds
Epoch 9 traning loss 4.56479
Epoch 9 valid loss 4.12446
Epoch took 166.6576159 seconds
Epoch 10 traning loss 4.54154
Epoch 10 valid loss 4.09872
Epoch took 169.539609909 seconds
Epoch 11 traning loss 4.519
Epoch 11 valid loss 4.07559
Epoch took 171.42190218 seconds
Epoch 12 traning loss 4.50057
Epoch 12 valid loss 4.12411
Epoch took 170.056602955 seconds
Epoch 13 traning loss 4.482
Epoch 13 valid loss 4.04243
Epoch took 170.239685059 seconds
Epoch 14 traning loss 4.46555
Epoch 14 valid loss 4.02697
Epoch took 168.914116144 seconds
Epoch 15 traning loss 4.4522
Epoch 15 valid loss 4.01533
Epoch took 167.499938011 seconds
Epoch 16 traning loss 4.44108
Epoch 16 valid loss 4.0065
Epoch took 169.092893124 seconds
Epoch 17 traning loss 4.41409
Epoch 17 valid loss 3.98921
Epoch took 168.400381088 seconds
Epoch 18 traning loss 4.39275
Epoch 18 valid loss 3.97922
Epoch took 169.911263943 seconds
Epoch 19 traning loss 4.38327
Epoch 19 valid loss 3.97043
Epoch took 167.843218803 seconds
'''

