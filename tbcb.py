import warnings

import keras.backend as K
from keras.callbacks import Callback
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
class TensorBoardCallback(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardCallback, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        if K.backend() != 'tensorflow':
            if histogram_freq != 0:
                warnings.warn('You are not using the TensorFlow backend. '
                              'histogram_freq was set to 0')
                histogram_freq = 0
            if write_graph:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_graph was set to False')
                write_graph = False
            if write_images:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_images was set to False')
                write_images = False
            if embeddings_freq != 0:
                warnings.warn('You are not using the TensorFlow backend. '
                              'embeddings_freq was set to 0')
                embeddings_freq = 0

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output[0])
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        wait_time = 0.01
        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided.')
        if self.validation_data and self.histogram_freq:
            is_val_gen = (hasattr(self.validation_data, 'next') or
                          hasattr(self.validation_data, '__next__') or
                          isinstance(self.validation_data, Sequence))
            if epoch % self.histogram_freq == 0:
                if is_val_gen:
                    assert self.validation_steps is not None
                    is_sequence = isinstance(self.validation_data, Sequence)
                    if self.workers > 0:
                        if is_sequence:
                            enqueuer = OrderedEnqueuer(
                                self.validation_data,
                                use_multiprocessing=self.use_multiprocessing,
                                shuffle=self.shuffle)
                        else:
                            enqueuer = GeneratorEnqueuer(
                                self.validation_data,
                                use_multiprocessing=self.use_multiprocessing,
                                wait_time=wait_time)
                        enqueuer.start(workers=self.workers,
                                       max_queue_size=self.max_queue_size)
                        output_val_generator = enqueuer.get()
                    else:
                        if is_sequence:
                            output_val_generator = iter(self.validation_data)
                        else:
                            output_val_generator = self.validation_data
                    tensors = (self.model.inputs +
                               self.model.targets +
                               self.model.sample_weights)
                    if (self.model.uses_learning_phase and
                            not isinstance(K.learning_phase(), int)):
                        tensors += [K.learning_phase()]
                    i = 0
                    while i < self.validation_steps:
                        generator_output = next(output_val_generator)
                        if len(generator_output) == 2:
                            x, y = generator_output
                            sample_weight = None
                        elif len(generator_output) == 3:
                            x, y, sample_weight = generator_output

                        if isinstance(x, list):
                            batch_size = x[0].shape[0]
                        elif isinstance(x, dict):
                            batch_size = list(x.values())[0].shape[0]
                        else:
                            batch_size = x.shape[0]
                        if batch_size == 0:
                            raise ValueError('Received an empty batch. '
                                             'Batches should at least contain one '
                                             'item.')
                        x, y, sample_weight = self.model._standardize_user_data(
                            x, y, sample_weight=sample_weight,
                            batch_size=batch_size)
                        batch_val = x + y + sample_weight

                        if (self.model.uses_learning_phase and
                                not isinstance(K.learning_phase(), int)):
                            batch_val += [0.]
                        if not len(batch_val) == len(tensors):
                            raise ValueError('validation_data generator\'s output '
                                             'length ({0}) must be the same as '
                                             'the sum of the lengths of the '
                                             'model\'s inputs, targets and '
                                             'sample_weights ({1}).'
                                             .format(len(batch_val), len(tensors)))

                        feed_dict = dict(zip(tensors, batch_val))
                        result = self.sess.run([self.merged], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, epoch)
                        i += 1
                else:
                    val_data = self.validation_data
                    tensors = (self.model.inputs +
                               self.model.targets +
                               self.model.sample_weights)

                    if self.model.uses_learning_phase:
                        # fit_generator appended the learning phase to val_data,
                        # therfore it is appended here to tensors before verifying
                        # tensors and val_data are equal in size
                        tensors += [K.learning_phase()]
                    if not len(val_data) == len(tensors):
                        raise ValueError('validation_data length ({0}) must be '
                                         'the same as the sum of the lengths of '
                                         'the model\'s inputs, targets and '
                                         'sample_weights ({1}).'.format(
                                             len(val_data)), len(tensors))
                    val_size = val_data[0].shape[0]
                    i = 0
                    while i < val_size:
                        step = min(self.batch_size, val_size - i)
                        if self.model.uses_learning_phase:
                            # do not slice the learning phase
                            batch_val = [x[i:i + step] for x in val_data[:-1]]
                            batch_val.append(val_data[-1])
                        else:
                            batch_val = [x[i:i + step] for x in val_data]
                        assert len(batch_val) == len(tensors)
                        feed_dict = dict(zip(tensors, batch_val))
                        result = self.sess.run([self.merged], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, epoch)
                        i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
