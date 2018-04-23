import random

from keras.layers import Input, Dense, GaussianDropout, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
import numpy as np
import tensorflow as tf


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def config_keras():
    import tensorflow as tf
    from keras import backend
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    backend.tensorflow_backend.set_session(tf.Session(config=config))


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + K.epsilon())


def prepare_model(input_shape, hidden_sizes=[32, 64, 128], activation="relu", dropout_rate=0.1,
                  optimizer="rmsprop"):
    config_keras()

    input_a = Input(shape=(input_shape,), sparse=True, name="input_a")
    input_b = Input(shape=(input_shape,), sparse=True, name="input_b")

    emb_model = Sequential()
    emb_model.add(Dense(hidden_sizes[0], input_shape=(input_shape,)))
    emb_model.add(GaussianDropout(dropout_rate))
    emb_model.add(BatchNormalization())
    emb_model.add(Activation(activation))

    for size in hidden_sizes[1:]:
        emb_model.add(Dense(size))
        emb_model.add(GaussianDropout(dropout_rate))
        emb_model.add(BatchNormalization())
        emb_model.add(Activation(activation))

    emb_a = emb_model(input_a)
    emb_b = emb_model(input_b)

    dot_product = merge([emb_a, emb_b], mode='dot')
    output = Dense(1, activation="sigmoid", name="output")(dot_product)

    model = Model(inputs=[input_a, input_b], outputs=output)

    emb_model_a = Model(inputs=input_a, outputs=emb_a)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', precision,
                                                                            recall, f1score])
    return model, emb_model_a

