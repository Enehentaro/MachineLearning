"""
TODO Make function of base class
"""
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Dropout
from keras.layers import ELU
from keras.layers import GlobalMaxPooling1D
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import ReLU
from keras.layers import Reshape

import ml_package

from ml_package.model import log_manager


# gpuの確認
# from tensorflow.python.client import device_lib
# pprint.pprint(device_lib.list_local_devices())


class MachineLearningModel(object):
    """Base machine learning model"""

    def __init__(self, params):
        self.params = params
        self.model = None
    pass


class MLP(MachineLearningModel):
    """Definition of class that make MLP"""

    def __init__(self, params):
        super().__init__(params=params)
    
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
    

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    PointNet consists of two core components. The primary MLP network, and the transformer
    net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
    network. The T-net is used twice. The first time to transform the input features (n, 3)
    into a canonical representation. The second is an affine transformation for alignment in
    feature space (n, 3). As per the original paper we constrain the transformation to be
    close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
    """

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


class PointNet(MachineLearningModel):
    """
    Definition of class that make PointNet.

    Title: Point cloud classification with PointNet
    Author: [David Griffiths](https://dgriffiths3.github.io)
    Date created: 2020/05/25
    Last modified: 2020/05/26
    Description: Implementation of PointNet for ModelNet10 classification.

    ## Introduction
    Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
    is a core problem in computer vision. This example implements the seminal point cloud
    deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
    detailed intoduction on PointNet see [this blog
    post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
    """

    def __init__(self, params):
        super().__init__(params=params)

    def conv_bn(self, x, filters):
        x = Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = BatchNormalization(momentum=0.0)(x)
        return ReLU()(x)

    def dense_bn(self, x, filters):
        x = Dense(filters)(x)
        x = BatchNormalization(momentum=0.0)(x)
        return ReLU()(x)
    
    def tnet(self, inputs, num_features):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = self.dense_bn(x, 128)
        x = Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return Dot(axes=(2, 1))([inputs, feat_T])

    # def tnet_tuned(self, inputs, num_features):
    #     """
    #     This is the T-Net obtained by Keras-tuner. From Ida.
    #     """

    #     # Initalise bias as the indentity matrix
    #     bias = keras.initializers.Constant(np.eye(num_features).flatten())
    #     reg = OrthogonalRegularizer(num_features)

    #     x = conv_bn(inputs, 96)
    #     x = conv_bn(x, 96)
    #     x = conv_bn(x, 256)
    #     x = layers.GlobalMaxPooling1D()(x)
    #     x = dense_bn(x, 512)
    #     x = dense_bn(x, 448)
    #     x = dense_bn(x, 224)
    #     x = layers.Dense(
    #         num_features * num_features,
    #         kernel_initializer="zeros",
    #         bias_initializer=bias,
    #         activity_regularizer=reg,
    #     )(x)
    #     feat_T = layers.Reshape((num_features, num_features))(x)
    #     # Apply affine transformation to input features
    #     return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def build_classification_pointnet(self, num_points, num_classes):
        """
        parameters
        ----------
        num_points : int

        num_classes : int
        
        """
        inputs = keras.Input(shape=(num_points, 3))

        x = self.tnet(inputs, 3)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 32)
        x = self.tnet(x, 32)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = Dropout(0.3)(x)
        x = self.dense_bn(x, 128)
        x = Dropout(0.3)(x)

        outputs = Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        # model.summary()

        """
        ### Train model

        Once the model is defined it can be trained like any other standard classification model
        using `.compile()` and `.fit()`.
        """

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["sparse_categorical_accuracy"],
        )

        return model
    
    def simplified_pointnet(self, input_point_cloud):
        """
        単純PointNet
        
        Parameters
        ----------
        trial
            OPTUNAのTrialオブジェクト
            
        input_point_cloud : keras.Input
            点群データ用のInputオブジェクト（Keras）
            
        Returns
        -------
        GlobalMaxPooling1D()
            全点のうちの最大値
            
        """
        conv_layers = self.params["conv_layers"]
        
        x = input_point_cloud
        for i in range(conv_layers):
            conv_filters = self.params[f"conv_filters_{i}"]
            x = self.conv_bn(x, conv_filters)
            
    #         x = keras.layers.Conv1D(hp_filters, kernel_size=1, padding="valid")(x)
    #         x = keras.layers.BatchNormalization(momentum=0.99)(x)       
    #         x = keras.layers.ReLU()(x)
        
        return GlobalMaxPooling1D()(x)
    
    def fit(self, tr_X, tr_y, va_X, va_y, verbose=0, callback_type="early_stopping"):
        """
        感染リスク分布予測モデル
        単純PointNetモデルに、メタデータ（空調条件）を合流させ、MLPで感染リスク予測

        parameters
        ----------
        tr_X : dict{
            "meta": metadata such as wind speed conditions,
            "point_cloud" : point cloud data
        }
            Explanatory variable for training.

        tr_y : 
            Objective variable for training.

        va_X : Same as tr_X

        va_y : Same as tr_y
        """
        inputs = []
        features = []
        for key, value in tr_X.items():
            if key == "meta":
                # MetaData
                input_meta = keras.Input(shape=(value.shape[1], ), name=key)
                # append metaData to inputs
                inputs.append(input_meta)
                # append metaData to features
                features.append(input_meta)
                
            elif key == "point_cloud":
                # point cloud
                input_point_cloud = keras.Input(shape=(value.shape[1], 3), name=key)
                # append point cloud to inputs
                inputs.append(input_point_cloud)
                # append fetures of point cloud to features
                features.append(
                    self.simplified_pointnet(input_point_cloud=input_point_cloud)
                )            
        
        x = Concatenate()(features)
        
        dense_layers = self.params["dense_layers"]
        activation = self.params["activation"]
        dropout = self.params["dropout"]
        optimizer_lr = self.params["optimizer_lr"]
        batch_size = self.params["batch_size"]
        
        for i in range(dense_layers):
            units = self.params[f"units_{i}"]
            x = keras.layers.Dense(units)(x)

            if activation == "relu":
                x = ReLU()(x)
            elif activation == "leaky_relu":
                x = LeakyReLU()(x)
            elif activation == "elu":
                x = ELU()(x)
            else:
                raise NotImplementedError
                
            x = Dropout(rate=dropout)(x)

        # x = keras.layers.Dense(y_sample.nunique())(x)
        # outputs = keras.layers.Softmax()(x)
        outputs = Dense(1)(x)

        optimizer = optimizers.Adam(learning_rate=optimizer_lr)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"],)

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