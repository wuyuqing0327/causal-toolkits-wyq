from keras.layers import Input,LSTM,Bidirectional,Dense,Dropout,Concatenate,Embedding,GlobalMaxPool1D
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer
from keras.utils import plot_model

K.clear_session()
maxlen = 40
len_features = 147
n_class = 2
epochs = 20
steps_per_epoch= 5

class Causal_Multi_treat_NN(Layer):

    def __init__(self, control_name, **kwargs):

        self.control_name = control_name

        super(Causal_Multi_treat_NN, self).__init__(**kwargs)

    def build(selft, input_shape):
        ### 初始化权重&偏差
        self.weight = self.add_weight( (input_shape[-1],),
                                        initializer=self.init,
                                        name='{}_Weight'.format(self.name),
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint
                                    )
        if self.bias:
            self.bias = self.add_weight((input_shape[-1],),
                                        initializer='zero',
                                        name='{}_Bias'.format(self.name),
                                        regularizer=self.b_regularizer,
                                        constraint=self.b_constraint
                                        )
        else:
            self.bias = None
        super(Causal_Multi_treat_NN, self).build(input_shape)

    def call(self, x, treatment):
        inputs = Input(shape=(len_features,),name="Input")
        ###共享层
        share = Dense(128,activation='relu',
                        kernel_initializer=K.initializer.random_normal(mean=0,stddev=0.05,seed=None),
                        bias_initializer=K.initializers.constatn(value=0),
                        kernel_regularizer=K.regularizers.l1_l2(l2=0,l1=1e-6),
                        name = "Shared_Layer")(inputs)

        # 获得所有的treatment组的名字
        treatment_group_keys = list(set(treatment))
        # treatment_group_keys.remove(self.control_name)
        treatment_group_keys.sort()
        # output=
        for ti in treatment_group_keys:
            tmp_t = x[x[treatment]==ti]

            each_layer = Dense(64, activation='relu',
                            kernel_initializer=K.initializer.random_normal(mean=0,stddev=0.05,seed=None),
                            bias_initializer=K.initializers.constatn(value=0),
                            kernel_regularizer=K.regularizers.l1_l2(l2=0,l1=1e-6),
                            name = "layer1")(share)
            each_layer = Dropout(0.1)(each_layer)
            each_output = Dense(n_class, activation='sigmoid', name = "output")(each_layer)

            # weight=zero


        return



#
# ###模型有两个输出out1,crf_output
# model = Causal_Multi_treat_NN()
# model.compile(optimizer='adam',, loss='categorical_crossentropy', metric=['auc'], loss_weights=1)
#                 model.fit_generator(X_train[features], X_train[treatment], X_train[label]), batch_size=128,
#                 validation_data=[X_test[features], X_test[treatment], X_test[label],
#                 epochs=epochs, steps_per_epoch = steps_per_epoch, verbose=1, workers=-1)
#
# plot_model(model,to_file="model.png")