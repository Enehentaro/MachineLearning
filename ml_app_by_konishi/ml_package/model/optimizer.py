import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.backend import clear_session
from ml_package.model import define_model
import xgboost as xgb


# optunaで最適化を行う値を計算するためのクラス
class Objective:
    def __init__(self, model_name, X, y, n_trials):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.n_splits = 5
        
        # tqdm関連の設定
        self.bar = tqdm(total=n_trials*self.n_splits)
        self.bar.set_description("optimizing progress") 
        
    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if self.model_name == "MLP":
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

        elif self.model_name == "XGB":
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

        else:
            print("***do not define such model***")
        
        # 最適化実行時の評価指標を格納するリスト
        scores = []
        
        # k分割交差検証の実装
        # 評価指標の決定
        metrics = ["neg_mean_squared_error", "neg_mean_absolute_error"]
        # 交差検証の分割方法を決定
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for i, (train_index, test_index) in enumerate(kf.split(X=self.X, y=self.y)):
            # 評価指標の決定，k分割交差検証の実装
            if self.model_name == "MLP":
                history = model.fit(tr_x=self.X.iloc[train_index], tr_y=self.y.iloc[train_index], 
                                    va_x=self.X.iloc[test_index], va_y=self.y.iloc[test_index])
            
                #履歴の最後の１０エポック
                val_loss_list = history.history['val_loss'][-10:] #List of loss
                loss_max = np.max(val_loss_list) #終盤の誤差の最大値（振動抑制が目的）
                
                #評価関数の計算
                scores.append(loss_max)

            elif self.model_name == "XGB":
                model.fit(X=self.X.iloc[train_index], y=self.y.iloc[train_index])
                validate_pred = model.predict(self.X.iloc[test_index])
                #評価指標の計算
                scores.append(mean_squared_error(self.y.iloc[test_index], validate_pred))


            self.bar.update(1)
        
        # オフィスまるごとを検証用データにするパターン
        # for validate_office_name in self.val_office_list:
        #     #リスト内包表記
        #     validate_data_index = [i for i in range(office_list.shape[0]) if any(office_list[i] == validate_office_name)]
        #     #validate_data_index以外をtrain_data_indexとする
        #     train_data_bool = np.ones(office_list.shape[0], dtype = bool)
        #     train_data_bool[validate_data_index] = False
        #     train_data_index = np.arange(office_list.shape[0])[train_data_bool]
            
        #     #トレーニングデータ、検証用データの振り分け
        #     train_explanatory_variable = self.X.iloc[train_data_index]
        #     validate_explanatory_variable = self.X.iloc[validate_data_index]
        #     train_objective_variable = self.y.iloc[train_data_index]
        #     validate_objective_variable = self.y.iloc[validate_data_index]
            
        #     #データをシャッフルする
        #     train_explanatory_variable = train_explanatory_variable.sample(frac=1, random_state=1)
        #     train_objective_variable = train_objective_variable.reindex(index=train_explanatory_variable.index)
        #     validate_explanatory_variable = validate_explanatory_variable.sample(frac=1, random_state=1)
        #     validate_objective_variable = validate_objective_variable.reindex(index=validate_explanatory_variable.index)

        #     #評価指標の決定，k分割交差検証の実装
        #     history = model.fit(tr_x=train_explanatory_variable, tr_y=train_objective_variable, 
        #                         va_x=validate_explanatory_variable, va_y=validate_objective_variable)
                    
        #     #履歴の最後の１０エポック
        #     val_loss_list = history.history['val_loss'][-10:] #List of loss
        #     loss_max = np.max(val_loss_list) #終盤の誤差の最大値（振動抑制が目的）
            
        #     #評価関数の計算
        #     scores.append(loss_max)
        
        return np.mean(scores)