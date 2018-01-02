# License: BSD 3-clause
# Authors: Kyle Kastner
from __future__ import print_function
import numpy as np
from scipy.io import wavfile
import os
import glob
import sys
import random

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib
import logging

sys.setrecursionlimit(40000)

"""
init logging
"""
logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)
"""
end logging
"""

def numpy_one_hot(labels_dense, n_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dtype = labels_dense.dtype
    labels_dense = labels_dense.ravel().astype("int32")
    n_labels = labels_dense.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot[np.arange(n_labels).astype("int32"),
                   labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(n_classes,))
    return labels_one_hot.astype(labels_dtype)


def tokenize_ind(phrase, vocabulary):
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = numpy_one_hot(phrase, vocabulary_size)
    return phrase

def filter_tokenize_ind(phrase, vocabulary):
    vocabulary_size = len(vocabulary.keys())
    filter_ = [char_ in vocabulary.keys() for char_ in phrase]
    phrase = [vocabulary[char_] for char_, cond in zip(phrase, filter_) if cond]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = numpy_one_hot(phrase, vocabulary_size)
    return phrase

def apply_quantize_preproc_to_normalized(X, n_bins=256):
    bins = np.linspace(0, 1, n_bins)
    def binn(x):
        shp = x.shape
        return (np.digitize(x.ravel(), bins) - 1).reshape(shp)
    X = [binn(Xi) for Xi in X]
    return X



def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-.5, .5) scaled version of X as float32, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    X -= 0.5
    X *= 0.95
    return X


class BlizzardThread(threading.Thread):
    cnt_finished = 0
    """Blizzard Thread"""
    def __init__(self, queue, out_queue, preproc_fn, char2code, frame_size):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue
        self.preproc_fn = preproc_fn
        self.char2code = char2code
        self.frame_size = frame_size

    def run(self):
        cnt = 0
        while True:
            # Grabs image path from queue
            cnt += 1
            wav_paths, texts = self.queue.get()
            if wav_paths[0] is None:
                BlizzardThread.cnt_finished += 1
                self.queue.task_done()
                return
            text_group = [filter_tokenize_ind(t.lower(), self.char2code) for t in texts]
            wav_group = [wavfile.read(wp) for wp in wav_paths]
            wav_group_samples = [w.astype('float64') for fs, w in wav_group]
            
            wav_group_samples = [(w - np.min(w))/(np.max(w) - np.min(w)) for w in wav_group_samples]

            wav_group_samples = [self.preproc_fn(wi, self.frame_size) for wi in wav_group_samples]

            self.out_queue.put((wav_group_samples, text_group))
            self.queue.task_done()

def wav_to_qbins_frames(x, frame_size, n_bins=256):

    x = apply_quantize_preproc_to_normalized([x], n_bins=n_bins)[0]
    x = x.astype('int32')
    if frame_size == 1:
        return x
    append = np.zeros((frame_size - len(x) % frame_size))
    x = np.hstack((x, apply_quantize_preproc([append], mn=x.min(), mx=x.max())[0]))
    return x.reshape(-1, frame_size)
            

class Blizzard_dataset(object):
    def __init__(self, minibatch_size=2,
                 wav_folder_path='wavn_fruit',
                 prompt_path='prompts_fruit.txt',
                 preproc_fn=lambda x: x,
                 thread_cnt=1,
                 frame_size=1,
                 fraction_range=[0., 1.]):
        self.wav_folder_path = wav_folder_path
        self.prompt_path = prompt_path
        self._pre = preproc_fn
        self.thread_cnt = thread_cnt
        self.frame_size = frame_size
        # extracted text

        with open(self.prompt_path, 'r') as f:
            tt = [t.strip().split('\t') for t in f.readlines()]
            tt=sorted(tt, key=lambda u: len(u[1]))
            if isinstance(fraction_range[1], float):
                fraction_range[0] = int(fraction_range[0]*len(tt))
                fraction_range[1] = int(fraction_range[1]*len(tt))
            tt = tt[fraction_range[0]:fraction_range[1]]

            wav_names = [t[0] for t in tt]
            raw_text = [t[1].strip().lower() for t in tt]
            all_symbols = set()
            for rt in raw_text:
                all_symbols = set(list(all_symbols) + list(set(rt)))
            self.wav_names = wav_names
            self.text = raw_text
            self.symbols = sorted(list(all_symbols))
        all_chars = ([chr(ord('a') + i) for i in range(26)] +
                     [',', '.', '!', '?', '<UNK>'])
        self.symbols = all_chars
        all_symbols = all_chars ###ZZZ override

        self.wav_paths = glob.glob(os.path.join(self.wav_folder_path, '*.wav'))
        self.minibatch_size = minibatch_size
        self._lens = np.array([float(len(t)) for t in self.text])
        self.code2char = dict(enumerate(all_symbols))
        self.char2code = {v: k for k, v in self.code2char.items()}
        self.vocabulary_size = len(self.char2code.keys())


        # Get only the smallest 50% of files for now
        _cut = np.percentile(self._lens, 5)
        _ind = np.where(self._lens <= _cut)[0]

        # ZZZ TODO: put procentile back self.text = [self.text[i] for i in _ind]
        # ZZZ self.wav_names = [self.wav_names[i] for i in _ind]
        assert len(self.text) == len(self.wav_names)
        final_wav_paths = []
        final_text = []
        final_wav_names = []
        for n, (w, t) in enumerate(zip(self.wav_names, self.text)):
            parts = w.split("chp")
                
            name = parts[0]
            if len(parts) == 1:
                chapter = 'wav'
            else:
                chapter = [pp for pp in parts[1].split("_") if pp != ''][0]
            for p in self.wav_paths:
                if name in p and chapter in p:
                    final_wav_paths.append(p)
                    final_wav_names.append(w)
                    final_text.append(t)
                    break

        random.seed(56)
        pack = list(zip(final_wav_paths, final_wav_names, final_text))
        random.shuffle(pack)
        ### TURN OFF SHUFFLE !!!!ZZZZ final_wav_paths, final_wav_names, final_text = zip(*pack)
        #print ('############################ wavs', final_wav_names)

        self.wav_names = final_wav_names
        self.wav_paths = final_wav_paths
        assert len(self.wav_names) == len(self.wav_paths)
        assert len(self.wav_paths) == len(self.text)

        self.n_per_epoch = len(self.wav_paths)
        self.n_samples_seen_ = 0

        self.buffer_size = 5
        self.minibatch_size = minibatch_size
        self.input_qsize = 5
        self.min_input_qsize = 2
        if len(self.wav_paths) % self.minibatch_size != 0:
            logger.info("WARNING: Sample size not an even multiple of minibatch size")
            logger.info("Truncating...")
            self.wav_paths = self.wav_paths[:-(
                len(self.wav_paths) % self.minibatch_size)]
            self.text = self.text[:-(
                len(self.text) % self.minibatch_size)]

        assert len(self.wav_paths) % self.minibatch_size == 0
        assert len(self.text) % self.minibatch_size == 0

        self.grouped_wav_paths = zip(*[iter(self.wav_paths)] *
                                      self.minibatch_size)
        self.grouped_text = zip(*[iter(self.text)] *
                                     self.minibatch_size)
        assert len(self.grouped_wav_paths) == len(self.grouped_text)

    def _init_queues(self):
        # Infinite...
        self.grouped_elements = itertools.cycle(zip(self.grouped_wav_paths,
                                                    self.grouped_text))
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self.it = []
        for i in range(self.thread_cnt):
            self.it.append(BlizzardThread(self.queue, self.out_queue,
                                     self._pre, self.char2code,
                                     self.frame_size))
            self.it[-1].start()

        # Populate queue with some paths to image data
        for n, _ in enumerate(range(self.input_qsize)):
            group = self.grouped_elements.next()
            self.queue.put(group)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def reset(self):
        self.n_samples_seen_ = 0
        try:
            with self.queue.mutex:
                self.queue.queue.clear()
            with self.out_queue.mutex:
                self.out_queue.queue.clear()
                self.out_queue.not_full.notifyAll()
            for _ in range(self.thread_cnt):
                exit_flag = ((None,), (None,))
                self.queue.put(exit_flag, True)
            for t in self.it:
                t.join()
            del self.queue
        except AttributeError:
            pass

    def _step(self):
        if self.n_samples_seen_ == 0:
            self._init_queues()

        if self.n_samples_seen_ >= self.n_per_epoch:
            self.reset()
            raise StopIteration("End of epoch")
        wav_group, text_group = self.out_queue.get()
        self.n_samples_seen_ += self.minibatch_size
        if self.queue.qsize() <= self.min_input_qsize:
            for i in range(self.input_qsize):
                group = self.grouped_elements.next()
                self.queue.put(group)
        li = list_iterator([wav_group, text_group], self.minibatch_size, axis=1, start_index=0,
                           stop_index=len(wav_group), make_mask=True)
        return next(li)

class Blizzard_dataset_adapter(object):
    def __init__(self, ds, cut_len, overlap=0, q_zero=128):
        self.ds = ds
        self.cut_len = cut_len
        self.overlap = overlap
        self.q_zero = q_zero
        self.ds.reset()

    def __iter__(self):
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(self.ds)
            batch_size = X_mb.shape[1]
            for part in xrange(X_mb.shape[0] // self.cut_len - 1):
                x_part = X_mb[self.cut_len*part : self.cut_len*(part+1)].transpose(1, 0, 2)
                x_part = np.concatenate([
                    np.full((batch_size, self.overlap, 1), self.q_zero, dtype='int32'),
                    x_part
                ], axis=1)

                x_mask_part = X_mb_mask[self.cut_len*part : self.cut_len*(part+1)].transpose(1, 0)
                x_mask_part = np.concatenate([
                    np.full((batch_size, self.overlap), 1, dtype='float32'),
                    x_mask_part
                ], axis=1)

                yield (x_part, x_mask_part.astype('float32'), c_mb, c_mb_mask, np.int32(part==0))
                
class Blizzard_dataset_adapter_nochars(object):
    def __init__(self, ds, cut_len, overlap=0, q_zero=128):
        self.ds = ds
        self.cut_len = cut_len
        self.overlap = overlap
        self.q_zero = q_zero
        self.ds.reset()

    def __iter__(self):
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(self.ds)
            batch_size = X_mb.shape[1]
            for part in xrange(X_mb.shape[0] // self.cut_len - 1):
                x_part = X_mb[self.cut_len*part : self.cut_len*(part+1)].transpose(1, 0, 2)
                x_part = np.concatenate([
                    np.full((batch_size, self.overlap, 1), self.q_zero, dtype='int32'),
                    x_part
                ], axis=1)

                x_mask_part = X_mb_mask[self.cut_len*part : self.cut_len*(part+1)].transpose(1, 0)
                x_mask_part = np.concatenate([
                    np.full((batch_size, self.overlap), 1, dtype='float32'),
                    x_mask_part
                ], axis=1)
                
                yield (x_part.reshape((batch_size, -1)), np.int32(part==0), x_mask_part.astype('float32'))
                



class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 randomize=False,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.randomize = randomize
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)
        self.one_hot_class_size = one_hot_class_size
        self.random_state = np.random.RandomState(2017)
        len0 = len(list_of_containers[0])
        assert all([len(ci) == len0 for ci in list_of_containers])
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)

    def reset(self):
        self.slice_start_ = self.start_index
        if self.randomize:
            start_ind = self.start_index
            stop_ind = min(len(self.list_of_containers[0]), self.stop_index)
            inds = np.arange(start_ind, stop_ind).astype("int32")
            # If start index is > 0 then pull some mad hackery to only shuffle
            # the end part - eg. validation set.
            self.random_state.shuffle(inds)
            if start_ind > 0:
                orig_inds = np.arange(0, start_ind).astype("int32")
                inds = np.concatenate((orig_inds, inds))
            new_list_of_containers = []
            for ci in self.list_of_containers:
                nci = [ci[i] for i in inds]
                if isinstance(ci, np.ndarray):
                    nci = np.array(nci)
                new_list_of_containers.append(nci)
            self.list_of_containers = new_list_of_containers

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = slice(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        sliced_c = []
        for c in self.list_of_containers:
            slc = c[ind]
            arr = np.asarray(slc)
            sliced_c.append(arr)
        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
            if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                maxlen = max([len(i) for i in sc])
                # Assume they at least have the same internal dtype
                if len(sc[0].shape) > 1:
                    total_shape = (maxlen, sc[0].shape[1])
                elif len(sc[0].shape) == 1:
                    total_shape = (maxlen, 1)
                else:
                    raise ValueError("Unhandled array size in list")
                if self.axis == 0:
                    raise ValueError("Unsupported axis of iteration")
                    new_sc = np.zeros((len(sc), total_shape[0],
                                       total_shape[1]))
                    new_sc = new_sc.squeeze().astype(sc[0].dtype)
                else:
                    new_sc = np.zeros((total_shape[0], len(sc),
                                       total_shape[1]))
                    new_sc = new_sc.astype(sc[0].dtype)
                    for m, sc_i in enumerate(sc):
                        if len(sc_i.shape) == 1:
                            new_sc[:len(sc_i), m, :] = sc_i.reshape(-1,1)
                        else:
                            new_sc[:len(sc_i), m, :] = sc_i
                sliced_c[n] = new_sc
            else:
                # Hit this case if all sequences are the same length
                #print('here', sc.shape)
                if self.axis == 1:
                    if len(sc.shape) == 2:
                        sliced_c[n] = sc[:, :, np.newaxis].transpose(1, 0, 2)
                    else:
                        sliced_c[n] = sc.transpose(1, 0, 2)
        return sliced_c
    
    def _slice_with_masks(self, ind):
        cs = self._slice_without_masks(ind)
        if self.axis == 0:
            ms = [np.ones_like(c[:, 0]) for c in cs]
            raise ValueError("NYI - see axis=0 case for ideas")
            sliced_c = []
            for n, c in enumerate(self.list_of_containers):
                slc = c[ind]
                for ii, si in enumerate(slc):
                    ms[n][ii, len(si):] = 0.
        elif self.axis == 1:
            ms = [np.ones_like(c[:, :, 0]) for c in cs]
            sliced_c = []
            for n, c in enumerate(self.list_of_containers):
                slc = c[ind]
                for ii, si in enumerate(slc):
                    # 2 0 64000 64000 0 (64000,) 20
                    #print(len(ms), n, len(ms[n]), len(si), ii, ms[n].shape, len(slc)) # ZZZ
                    ms[n][len(si):, ii] = 0.
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]


def numpy_one_hot(labels_dense, n_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dtype = labels_dense.dtype
    labels_dense = labels_dense.ravel().astype("int32")
    n_labels = labels_dense.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot[np.arange(n_labels).astype("int32"),
                   labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(n_classes,))
    return labels_one_hot.astype(labels_dtype)








