import pprint

import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import ReLU
from keras.models import Sequential

import ml_package

from ml_package.model import log_manager


# gpuの確認
pprint.pprint(device_lib.list_local_devices())

# MLPを定義するクラス
class MLP:
    def __init__(self, params):
        self.params = params
        self.model = None
    
    def fit(self, tr_x, tr_y, va_x, va_y, verbose=0, callback_type="early_stopping"):
        
        # パラメータの読み込み
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
        
        # モデルの定義
        self.model = keras.Sequential()
        
        # 入力層，kerasはSequentialモデルを作ったとき最初のlayerにinput_shapeまたはinput_dimで入力の形状をtupleで与える必要がある
        self.model.add(Dropout(rate=input_dropout, input_shape=(tr_x.shape[1],)))
        
        # 中間層
        for i in range(hidden_layers):
            # 全結合層
            self.model.add(Dense(units=hidden_units, kernel_initializer=kernel_initializer))
            # バッチ正規化の有無
            if batch_norm == "on":
                self.model.add(BatchNormalization())
            # 活性化関数の選択
            if hidden_activation == "relu":
                self.model.add(ReLU())
            elif hidden_activation == "leaky_relu":
                self.model.add(LeakyReLU(alpha=0.01))
            elif hidden_activation == "prelu":
                self.model.add(PReLU())
            # 指定のモノ以外が来たときには埋め込みエラーを吐く
            else:
                raise NotImplementedError
            # ドロップアウト
            self.model.add(Dropout(rate=hidden_dropout))
        
        # 出力層
        self.model.add(Dense(1, kernel_initializer=kernel_initializer))
        
        # optimizerの選択
        if optimizer_type == "sgd":
            optimizer = optimizers.SGD(learning_rate=optimizer_lr)
        elif optimizer_type == "adam":
            optimizer = optimizers.Adam(learning_rate=optimizer_lr)
        elif optimizer_type == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=optimizer_lr)
        # 指定のモノ以外が来たときには埋め込みエラーを吐く
        else:
            raise NotImplementedError
        
        # モデルのcompile
        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
        
        # 学習時の設定
        # エポック
        max_epoch = 400

        # callbackの作成．今のところearlystoppingとtensorboardのみ実装
        # 定めたパラメータの最小値更新が行われなければ打ち切り
        patience = 100
        if callback_type == "early_stopping":
            callbacks = [EarlyStopping(monitor="val_mae", patience=patience, verbose=verbose, restore_best_weights=False)]
        elif callback_type == "both":
            # 決定したMLPの形状表示
            self.model.summary()
            # tensorboard用のログディレクトリ作成
            control_log = log_manager.ControlLog()
            log_dir_path, log_file_name = control_log.decide_filename(what_log="TensorBoardLogs")
            tb_log_dir = log_dir_path + log_file_name
            callbacks = [EarlyStopping(monitor="val_mae", patience=patience, verbose=verbose, restore_best_weights=True),
                         TensorBoard(log_dir=tb_log_dir, histogram_freq=1)]
            print("TensorBoardLogs path:", tb_log_dir)

        history = self.model.fit(
            tr_x, tr_y, epochs=max_epoch, batch_size=batch_size, verbose=verbose,
            validation_data=(va_x, va_y), callbacks=callbacks
        )
        
        return history
        
    def predict(self, x):
        # モデルを使用して予測するときにindexを元データと揃えておかないとmean_squared_errorを計算するときにNanとなりerrorが起きる
        y_pred = pd.DataFrame(self.model.predict(x), index=x.index)
        return y_pred