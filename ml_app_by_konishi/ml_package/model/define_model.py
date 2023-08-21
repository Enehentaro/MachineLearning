"""
TODO Make function of base class
"""
import pprint

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import ReLU

import ml_package

from ml_package.model import log_manager


# gpuの確認
# from tensorflow.python.client import device_lib
# pprint.pprint(device_lib.list_local_devices())


class MachineLearningModel(object):
    """Base machine learning model"""
    pass


class MLP(MachineLearningModel):
    """Definition of class that make MLP"""

    def __init__(self, params):
        self.params = params
        self.model = None
    
    def fit(self, tr_X, tr_y, va_X, va_y, verbose=0, callback_type="early_stopping"):
        input_dropout = self.params["input_dropout"]
        hidden_layers = self.params["hidden_layers"]
        hidden_units = self.params["hidden_units"]
        kernel_initializer = self.params["kernel_initializer"]
        hidden_activation = self.params["hidden_activation"]
        hidden_dropout = self.params["hidden_dropout"]
        batch_norm = self.params["batch_norm"]
        optimizer_type = self.params["optimizer_type"]
        optimizer_lr = self.params["optimizer_lr"]
        batch_size = self.params["batch_size"]
        
        #input shape must be tuple
        inputs = keras.Input(shape=(tr_X.shape[1],))
        hidden = Dropout(rate=input_dropout)(inputs)
        for _ in range(hidden_layers):
            hidden = Dense(units=hidden_units, kernel_initializer=kernel_initializer)(hidden)
            if batch_norm == "on":
                hidden = BatchNormalization()(hidden)
            if hidden_activation == "relu":
                hidden = ReLU()(hidden)
            elif hidden_activation == "leaky_relu":
                hidden = LeakyReLU(alpha=0.01)(hidden)
            elif hidden_activation == "prelu":
                hidden = PReLU()(hidden)
            else:
                raise NotImplementedError
            hidden = Dropout(rate=hidden_dropout)(hidden)
        
        outputs = Dense(1, kernel_initializer=kernel_initializer)(hidden)
        
        if optimizer_type == "sgd":
            optimizer = optimizers.SGD(learning_rate=optimizer_lr)
        elif optimizer_type == "adam":
            optimizer = optimizers.Adam(learning_rate=optimizer_lr)
        elif optimizer_type == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=optimizer_lr)
        else:
            raise NotImplementedError

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="mlp")
        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
        
        # 学習時の設定
        max_epoch = 400
        # callbackの作成．今のところearlystoppingとtensorboardのみ実装
        patience = 100
        if callback_type == "early_stopping":
            callbacks = [EarlyStopping(monitor="val_mae", patience=patience, verbose=verbose, restore_best_weights=False)]
        elif callback_type == "both":
            # 決定したMLPの形状表示
            self.model.summary()
            # tensorboard用のログディレクトリ作成
            log_dir_path, log_file_name = \
                log_manager.OptunaLogModel.decide_filename(what_log="../TensorBoardLogs")
            tb_log_dir = log_dir_path + log_file_name
            callbacks = [EarlyStopping(monitor="val_mae", patience=patience, verbose=verbose, restore_best_weights=True),
                         TensorBoard(log_dir=tb_log_dir, histogram_freq=1)]
            print("TensorBoardLogs path:", tb_log_dir)

        history = self.model.fit(
            tr_X, tr_y, epochs=max_epoch, batch_size=batch_size, verbose=verbose,
            validation_data=(va_X, va_y), callbacks=callbacks
        )
        
        return history
        
    def predict(self, X):
        # モデルを使用して予測するときにindexを元データと揃えておかないとmean_squared_errorを計算するときにNanとなりerrorが起きる
        y_pred = pd.DataFrame(self.model.predict(X), index=X.index)
        return y_pred