import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
import pickle

import os
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
import numpy as np

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
    
def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)

def treatment_auc(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return auc(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


def make_dragonnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def make_tarnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(inputs)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def make_ned(input_dim, reg_l2=0.01):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden1')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden2')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden3')(x)

    t_predictions = Dense(units=1, activation='sigmoid', name='ned_t_activation')(x)
    y_predictions = Dense(units=1, activation=None, name='ned_y_prediction')(x)

    concat_pred = Concatenate(1)([y_predictions, t_predictions])

    model = Model(inputs=inputs, outputs=concat_pred)
    return model


def post_cut(nednet, input_dim, reg_l2=0.01):
    for layer in nednet.layers:
        layer.trainable = False
    nednet.layers.pop()
    nednet.layers.pop()
    nednet.layers.pop()

    frozen = nednet

    x = frozen.layers[-1].output
    frozen.layers[-1].outbound_nodes = []
    input = frozen.input

    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_1')(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_1')(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_2')(
        y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_2')(
        y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions])

    model = Model(inputs=input, outputs=concat_pred)
    return model




def train_and_predict_dragons(x_train, x_test, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., 
                              dragon='', val_split=0.2, batch_size=512):
    feature = pickle.load(open("best_model/Multi_Treat_Model/" + "features.pkl", "rb+"))
    verbose = 1
#     y_scaler = StandardScaler().fit(y_unscaled)
#     y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []
    runs = 25
    for i in range(runs):
        if dragon == 'tarnet':
            dragonnet = make_tarnet(len(feature), 0.01)
        elif dragon == 'dragonnet':
            dragonnet = make_dragonnet(len(feature), 0.01)

        metrics = [regression_loss, binary_classification_loss, treatment_auc, track_epsilon]

        if targeted_regularization:
            loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss

        y_train, y_test = x_train[[LABEL_COL]], x_test[[LABEL_COL]]
        t_train, t_test = x_train[[TREATMENT_COL]], x_test[[TREATMENT_COL]]
        t_train[TREATMENT_COL] = np.where(t_train[TREATMENT_COL]=='control',0,1)
        t_test[TREATMENT_COL] = np.where(t_test[TREATMENT_COL]=='control',0,1)
        yt_train = np.concatenate([y_train, t_train], 1)
        yt_test = np.concatenate([y_test, t_test], 1)
        yt_train = np.array(yt_train,dtype=float)
        yt_test = np.array(yt_test,dtype=float)
#         print("yt_train:", yt_train.shape, yt_train)
            
        dragonnet.compile(
            optimizer=Adam(lr=1e-3),
            loss=loss, metrics=metrics)
        dragonnet.summary()

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_auc', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor=auc, factor=0.5, patience=5, 
                              verbose=verbose, mode='max',
                              min_delta=1e-8, cooldown=0, min_lr=0)
        ]

        print("dragonnet trainning begin")
        
        dragonnet.fit(x_train[feature].astype(float), yt_train, 
                      epochs=100,
                      batch_size=batch_size,
                      verbose=verbose,
                      validation_data=(x_test[feature].astype(float), yt_test),
                      callbacks=adam_callbacks
                     )
        

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_auc', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor=auc, factor=0.5, patience=5, verbose=verbose, 
                              mode='max',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        # should pick something better!
        sgd_lr = 1e-5
        momentum = 0.9
        dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                          metrics=metrics)
        dragonnet.fit(x_train[feature].astype(float), yt_train, callbacks=sgd_callbacks,
                      validation_data=(x_test[feature].astype(float), yt_test),
                      epochs=300,
                      batch_size=batch_size, verbose=verbose)

        yt_hat_test = dragonnet.predict(x_test[feature].astype(float))
        yt_hat_train = dragonnet.predict(x_train[feature].astype(float))
    
        K.clear_session()

    return yt_hat_test, yt_hat_train, dragonnet