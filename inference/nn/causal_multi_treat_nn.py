from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.optimizers import *
from keras.models import Model
from keras.layers import Lambda
from tensorflow import slice
from keras.engine.topology import Layer
 

multitreat_group = {}
for ti in sorted(set(X_train[TREATMENT_COL])):
    multitreat_group.update({ti: X_train[X_train[TREATMENT_COL]==ti].shape})

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
class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, traination_data, validation_data, include_on_batch=False, validation=True):
        super(RocAucMetricCallback, self).__init__()
        self.a = traination_data
        self.x = None
        self.y = None
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.include_on_batch = include_on_batch
        self.validation = validation
        self.batch_auc = []
        self.batch = None
        self.epoch_count = 0
        self.batch_count = 0

    def on_train_begin(self, logs={}):
        if not ('train_roc_auc' in self.params['metrics']):
            self.params['metrics'].append('train_roc_auc')
        if not ('val_roc_auc' in self.params['metrics']):
            self.params['metrics'].append('val_roc_auc')

    def on_train_end(self, logs={}):
        pass
    def on_batch_begin(self, batch, logs={}):
        if (self.include_on_batch):
            train = next(self.a)
            self.x = train[0]
            self.y = train[1]
            self.batch_count += 1
        pass

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            auc = roc_auc(self.y,  self.model.predict(self.x))
            self.batch_auc.append(auc)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_count += 1
        pass

    def on_epoch_end(self, epoch, logs={}):
        if (self.include_on_batch):
            train_auc = sum(self.batch_auc)/ self.batch_count
            logs['train_auc'] = train_auc
        if (self.validation):
            # self.model.save(model_path + 'epoch-%d.h5' %self.epoch_count)
            self.x = self.a[0]
            self.y = self.a[1]
            logs['train_roc_auc'] = roc_auc(self.y, self.model.predict(self.x))
            logs['val_roc_auc'] = roc_auc(self.y_val, self.model.predict(self.x_val))


class My_Multi_Treat_base_Net(Layer):

    def __init__(self, num_layers, kernel=None, dropout=0.2,activation='elu', **kwargs):
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        super(My_Multi_Treat_base_Net, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.weight = self.add_weight(name='weight',
    #                                    shape=[1, 1],
    #                                    initializer='RandomNormal',
    #                                    #  initializer='ones',
    #                                    trainable=True)
    #     super(My_Multi_Treat_base_Net, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        
        x = Dense(146, activation=self.activation, kernel_initializer='random_normal',
                  bias_initializer='zeros')(inputs)
        x = Dropout(self.dropout)(x)

        if(self.num_layers >= 1):
            for q in range(self.num_layers):
                x = Dense(146, activation=self.activation)(x)
                x = Dropout(self.dropout)(x)
        return x
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_layers': self.num_layers,
        'dropout': self.dropout,
        'activation': self.activation
        })
        return config


num_layers=8
len_feature = 147
batch_size = 1028
len_feature = 147
num_response = 2
output_dim=1

### keras multi treat model ###
inputs_c = Input(shape=len_feature,name="Input_c", dtype='float32')
inputs_t1 = Input(shape=len_feature,name="Input_t1", dtype='float32')
inputs_t2 = Input(shape=len_feature,name="Input_t2", dtype='float32')
inputs_t3 = Input(shape=len_feature,name="Input_t3", dtype='float32')
inputs_t4 = Input(shape=len_feature,name="Input_t4", dtype='float32')

shared_c = My_Multi_Treat_base_Net(num_layers=num_layers, name='shared_c')(inputs_c)
shared_t1 = My_Multi_Treat_base_Net(num_layers=num_layers, name='shared_t1')(inputs_t1)
shared_t2 = My_Multi_Treat_base_Net(num_layers=num_layers, name='shared_t2')(inputs_t2)
shared_t3 = My_Multi_Treat_base_Net(num_layers=num_layers, name='shared_t3')(inputs_t3)
shared_t4 = My_Multi_Treat_base_Net(num_layers=num_layers, name='shared_t4')(inputs_t4)

# for control output
control = Dense(units=140, activation='relu')(shared_c)
control_output = Dense(units=output_dim, activation='sigmoid', name='control')(control)

# for treat1 output
treat1 = Dense(units=140, activation='relu')(shared_t1)
treat1_output = Dense(units=output_dim, activation='sigmoid', name='treatment_06')(treat1)

# for treat2 output
treat2 = Dense(units=140, activation='relu')(shared_t2)
treat2_output = Dense(units=output_dim, activation='sigmoid', name='treatment_12')(treat2)

# for treat3 output
treat3 = Dense(units=140, activation='relu')(shared_t3)
treat3_output = Dense(units=output_dim, activation='sigmoid', name='treatment_15')(treat3)

# for treat4 output
treat4 = Dense(units=140, activation='relu')(shared_t4)
treat4_output = Dense(units=output_dim, activation='sigmoid', name='treatment_20')(treat4)

model = Model(inputs=[inputs_c, inputs_t1, inputs_t2, inputs_t3, inputs_t4], 
            outputs=[control_output, treat1_output, treat2_output, treat3_output, treat4_output])
model.compile(optimizer='Adam',
              loss={'control': 'binary_crossentropy', 'treatment_06': 'binary_crossentropy', 'treatment_12': 'binary_crossentropy', 
                    'treatment_15': 'binary_crossentropy', 'treatment_20': 'binary_crossentropy'},
              loss_weights={'control': 1, 'treatment_06': 2, 'treatment_12': 1.5, 
                    'treatment_15': 1.5, 'treatment_20': 1},
              metrics={'control': 'auc', 'treatment_06': 'auc', 'treatment_12': 'auc', 
                    'treatment_15': 'auc', 'treatment_20': 'auc'})
model.summary()



##### model trainning #####

## split data ##
train_control = X_train[X_train[TREATMENT_COL]=='control'].sample(n=94279, replace=False)
train_treat1 = X_train[X_train[TREATMENT_COL]=='treatment_06'].sample(n=94279, replace=False)
train_treat2 = X_train[X_train[TREATMENT_COL]=='treatment_12'].sample(n=94279, replace=False)
train_treat3 = X_train[X_train[TREATMENT_COL]=='treatment_15'].sample(n=94279, replace=False)
train_treat4 = X_train[X_train[TREATMENT_COL]=='treatment_20'].sample(n=94279, replace=False)
test_control = X_test[X_test[TREATMENT_COL]=='control'].sample(n=40373, replace=False)
test_treat1 = X_test[X_test[TREATMENT_COL]=='treatment_06'].sample(n=40373, replace=False)
test_treat2 = X_test[X_test[TREATMENT_COL]=='treatment_12'].sample(n=40373, replace=False)
test_treat3 = X_test[X_test[TREATMENT_COL]=='treatment_15'].sample(n=40373, replace=False)
test_treat4 = X_test[X_test[TREATMENT_COL]=='treatment_20'].sample(n=40373, replace=False)
print(train_control.shape)
print(train_treat1.shape)
print(train_treat2.shape)
print(train_treat3.shape)
print(train_treat4.shape)
print(test_control.shape)
print(test_treat1.shape)
print(test_treat2.shape)
print(test_treat3.shape)
print(test_treat4.shape)



tf.config.experimental_run_functions_eagerly(True)
model.fit(
    x=[train_control[feature],train_treat1[feature],train_treat2[feature],train_treat3[feature],train_treat4[feature]],
    y=[train_control[LABEL_COL],train_treat1[LABEL_COL],train_treat2[LABEL_COL],train_treat3[LABEL_COL],train_treat4[LABEL_COL]],
    batch_size=32,
    epochs=50,
    verbose=2,
    validation_data=([test_control[feature],test_treat1[feature],test_treat2[feature],test_treat3[feature],test_treat4[feature]],
                     [test_control[LABEL_COL],test_treat1[LABEL_COL],test_treat2[LABEL_COL],test_treat3[LABEL_COL],test_treat4[LABEL_COL]]),
    steps_per_epoch = 100,
    validation_steps=None,
    callbacks=[ keras.callbacks.ModelCheckpoint(
                'best_model/' + "weights.epoch-{epoch:02d}--tr_auc-{treatment_06_auc:.4f}-val_auc-{val_treatment_06_auc:.4f}.h5",
                monitor='val_treatment_06_auc', mode='max', save_best_only=True, verbose=1)
#                RocAucMetricCallback((train_X_array, train_Y_array), (valid_X_array, valid_Y_array)),
               # predict_test,
               # keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, verbose=1, mode='max'),
#                TensorBoard(log_dir=model_path + 'logs', write_graph=True, write_images=True,
#                            histogram_freq=1, update_freq='batch', write_grads=True)
              ]
)