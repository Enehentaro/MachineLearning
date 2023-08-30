from keras.backend import clear_session
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm.autonotebook import tqdm
import xgboost as xgb

from ml_package.model import define_model
from ml_package.model.define_my_error import ModelNotFoundError


class Objective(object):
    """
    Base class for configuring Optuna
    """

    def __init__(self, model_name, split_method, X, y, n_trials):
        self.model_name = model_name
        self.split_method = split_method
        self.X = X
        self.y = y
        self.n_splits = 5
        # tqdm関連の設定
        self.bar = tqdm(total=n_trials*self.n_splits)
        self.bar.set_description("optimizing progress") 
        
    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if self.model_name.casefold() == "MLP".casefold():
            params = {
                "input_dropout" : trial.suggest_float("input_dropout", 0.0, 0.2, step=0.05),
                "hidden_layers" : trial.suggest_int("hidden_layers", 3, 10),
                "hidden_units" : trial.suggest_int("hidden_units", 32, 256, step=32),
                "kernel_initializer" : trial.suggest_categorical("kernel_initializer", ["he_normal", "he_uniform", "random_normal"]),
                "hidden_activation" : trial.suggest_categorical("hidden_activation", ["relu", "leaky_relu", "prelu"]),
                "hidden_dropout" : trial.suggest_float("hidden_dropout", 0.0, 0.3, step=0.05),
                "batch_norm" : trial.suggest_categorical("batch_norm", ["on", "off"]),
                "optimizer_type" : trial.suggest_categorical("optimizer_type", ["adam", "rmsprop"]),
                "optimizer_lr" : trial.suggest_float("optimizer_lr", 1e-4, 1e-2, log=True),
                "batch_size" : trial.suggest_int("batch_size", 32, 128, step=32)
            }
            model = define_model.MLP(params)

        elif self.model_name.casefold() == "XGB".casefold():
            params = {
                "max_depth" : trial.suggest_int("max_depth", 1, 10),
                "min_child_weight" : trial.suggest_int("min_child_weight", 1, 5),
                "gamma" : trial.suggest_uniform("gamma", 0, 1),
                "subsample" : trial.suggest_uniform("subsample", 0, 1),
                "colsample_bytree" : trial.suggest_uniform("colsample_bytree", 0, 1),
                "learning_rate" : trial.suggest_uniform("learning_rate", 0, 1),
                "reg_alpha" : trial.suggest_loguniform("reg_alpha", 0.0001, 10),
                "reg_lambda" : trial.suggest_loguniform("reg_lambda", 0.0001, 10),
                "n_estimators" : 1000,
                "booster" : "gbtree",
                "objective" : "reg:squarederror",
                "random_state" : 1,
                "eval_metric" : mean_squared_error
            }
            model = xgb.XGBRegressor(**params)

        elif self.model_name.casefold() == "PointNet".casefold():
            params = {
                "dense_layers" : trial.suggest_int('dense_layers', 0, 4, step=1),
                "activation" : trial.suggest_categorical('activation', ["relu", "leaky_relu", "elu"]), 
                "dropout" : trial.suggest_categorical('dropout', [0.0, 0.3, 0.5]),
                "conv_layers" : trial.suggest_int('conv_layers', 1, 3, step=1),
                "optimizer_lr" : trial.suggest_float('optimizer_lr', 1e-5, 1e-1, log=True),
                "batch_size" : trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128])
            }
            for i in range(params["dense_layers"]):
                units = {
                    f"units_{i}" : trial.suggest_int(f"units_{i}", 64, 512, step=64)
                }
                params.update(units)
            for i in range(params["conv_layers"]):
                conv_filters = {
                    f"conv_filters_{i}" : trial.suggest_int(f"conv_filters_{i}", 32, 256, step=32)
                }
                params.update(conv_filters)
            model = define_model.PointNet(params)

        else:
            raise ModelNotFoundError(f"No such model is defined: {self.model_name}")
        
        scores = []
        metrics = ["neg_mean_squared_error", "neg_mean_absolute_error"]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for _, (train_index, test_index) in enumerate(kf.split(X=self.X, y=self.y)):
            if self.model_name.casefold() == "MLP".casefold():
                history = model.fit(
                    tr_X=self.X.iloc[train_index], 
                    tr_y=self.y.iloc[train_index], 
                    va_X=self.X.iloc[test_index], 
                    va_y=self.y.iloc[test_index]
                )
                #履歴の最後の１０エポック
                val_loss_list = history.history['val_loss'][-10:]
                loss_max = np.max(val_loss_list) #終盤の誤差の最大値（振動抑制が目的）
                scores.append(loss_max)

            elif self.model_name.casefold() == "PointNet".casefold():
                history = model.fit(
                    tr_X=self.X.iloc[train_index], 
                    tr_y=self.y.iloc[train_index], 
                    va_X=self.X.iloc[test_index], 
                    va_y=self.y.iloc[test_index]
                )
                #履歴の最後の１０エポック
                val_loss_list = history.history['val_loss'][-10:]
                loss_max = np.max(val_loss_list) #終盤の誤差の最大値（振動抑制が目的）
                scores.append(loss_max)


            elif self.model_name == "XGB":
                model.fit(X=self.X.iloc[train_index], y=self.y.iloc[train_index])
                validate_pred = model.predict(self.X.iloc[test_index])
                scores.append(mean_squared_error(self.y.iloc[test_index], validate_pred))

            self.bar.update(1)
        return np.mean(scores)