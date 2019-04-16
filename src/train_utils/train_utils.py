import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        max_to_keep: indicates the maximum number of recent checkpoint files to keep.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 max_to_keep=3,
                 mode='auto',
                 period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.max_to_keep = max_to_keep
        self.period = period
        self.epochs_since_last_save = 0

        if os.path.isfile(filepath):
            raise ValueError(
                '%s is a file, should be a directory' % filepath)

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        print('train begin')
        self.load_model()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        self.save_model()
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.save_model()

    def load_model(self):
        current_subdirs = os.listdir(self.filepath)
        if len(current_subdirs) > 0:
            # restore model weights
            checkpoint_prefix = os.path.join(
                compat.as_text(self.filepath),
                compat.as_text(max(current_subdirs)),
                compat.as_text(constants.VARIABLES_DIRECTORY),
                compat.as_text(constants.VARIABLES_FILENAME))
            self.model.load_weights(checkpoint_prefix)
            print('\nLoaded latest model from %s' % checkpoint_prefix)

    def save_model(self):
        tf.contrib.saved_model.save_keras_model(self.model, self.filepath)
        while len(os.listdir(self.filepath)) > self.max_to_keep:
            delete_mode_path = os.path.join(self.filepath, min(os.listdir(self.filepath)))
            shutil.rmtree(delete_mode_path)
            print('\nDelete model %s' % delete_mode_path)
