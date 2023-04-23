# warningはpythonの標準ライブラリ．
# FutureWarnigが邪魔なので非表示にする．動作に支障が無ければ問題ない．また最適化によって解が収束しないときに出るConvergenceWarningも邪魔なので非表示にする．
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=(FutureWarning, ConvergenceWarning))#対象のwarningsクラスはタプルで渡す必要があるらしい
import pprint
import sys
sys.path.append("/mnt/MachineLearning")

# 各種モジュールのimport
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import statistics

from modules import show_mod
from modules.log_controler import ControlLog

from tqdm.notebook import tqdm

from sklearn import preprocessing

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.layers import BatchNormalization
from keras.layers import ReLU, LeakyReLU, PReLU
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.backend import clear_session

import optuna

def main():
    # 読み込むデータのパスの設定
    current_dir_path = os.getcwd()
    data_path = "/mnt/MachineLearning/data/"
    output_path = "/mnt/MachineLearning/MLTrial/images/"
    os.makedirs(output_path, exist_ok=True)

    # データの読み込み
    df_read = pd.read_csv(f"{data_path}summary_20230418.csv", index_col="case_name")
    office_array = df_read["office"].unique()

    RoI_name = "countTimeMean_onlyFloating"

    # オフィスのRoIをプロットして出力
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, title="RoI plot", xlabel=RoI_name, ylabel="case index")
    # カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8")
    colors = list(matplotlib.colors.CSS4_COLORS.values())#148色までなら対応可能

    for idx, target_office_name in enumerate(office_array):
        df = df_read[df_read["office"]==target_office_name]
        ax.scatter(df[RoI_name], df.index, 
                    s=80, c=colors[idx], marker=markers[2], edgecolor="white", label=target_office_name)
        
    ax.legend(loc="center right")
    ax.grid()
    fig.savefig(f"{output_path}office_RoI_plot.png")

    # 排気口位置a,b,offをダミー変数化
    df_read = pd.get_dummies(df_read, columns=['exhaust'])

    # 説明変数と目的変数の定義
    # 説明変数と目的変数にdtype=object型が含まれないように注意する。数字以外はTensor型に変換できないのでエラーの原因となる
    explanatory_variable =['aircon', 'ventilation', '1_x', '1_y', '1_angle', '2_x', '2_y', '2_angle', '3_x', '3_y', '3_angle', '4_x', '4_y', '4_angle', '5_x', '5_y', '5_angle', 
                        'size_x','size_y', 'exhaust_a', 'exhaust_b', 'exhaust_off']

    df_explanatory_variable = df_read[explanatory_variable]
    df_objective_variable = df_read[RoI_name]

    # 説明変数の標準化(only explanatory variable)
    stdscaler = preprocessing.StandardScaler()
    stdscaler.fit(df_explanatory_variable)
    np_explanatory_variable_std = stdscaler.transform(df_explanatory_variable)
    df_explanatory_variable_std = pd.DataFrame(np_explanatory_variable_std, index=df_explanatory_variable.index, columns=df_explanatory_variable.columns)

    # トレーニングデータとテストデータの分割
    train_explanatory_variable, test_explanatory_variable = train_test_split(df_explanatory_variable_std, test_size=0.3, random_state=0)
    train_data_index = train_explanatory_variable.index
    test_data_index = test_explanatory_variable.index
    train_objective_variable = df_objective_variable.loc[train_data_index]
    test_objective_variable = df_objective_variable.loc[test_data_index]

    # GPUの環境変数設定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # gpuの確認
    from tensorflow.python.client import device_lib
    pprint.pprint(device_lib.list_local_devices())

    # ハイパーパラメータサーチ
    hyperparameter_search(X=train_explanatory_variable, y=train_objective_variable)


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
            control_log = ControlLog()
            log_file_name, log_dir_path = control_log.decide_filename(what_log="TensorBoardLogs")
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

# optunaで最適化を行う値を計算するためのクラス
class Objective:
    def __init__(self, X, y, n_trials):
        self.X = X
        self.y = y
        
        # tqdm関連の設定
        # self.bar = tqdm(total = n_trials)
        # self.bar.set_description('Progress rate')
        
    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()
        print("*** clear_session() occured ! ***")
        
        # ハイパーパラメータの空間設定
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
        
        # MLP
        model = MLP(params)
        
        # 最適化実行時の評価指標を格納するリスト
        scores = []
        
        # k分割交差検証の実装
        # 評価指標の決定
        metrics = ["neg_mean_squared_error", "neg_mean_absolute_error"]
        # 交差検証の分割方法を決定
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        for i, (train_index, test_index) in enumerate(kf.split(X=self.X, y=self.y)):
            print(f"Fold{i}")
            # 評価指標の決定，k分割交差検証の実装
            history = model.fit(tr_x=self.X.iloc[train_index], tr_y=self.y.iloc[train_index], 
                                va_x=self.X.iloc[test_index], va_y=self.y.iloc[test_index])
            
            #履歴の最後の１０エポック
            val_loss_list = history.history['val_loss'][-10:] #List of loss
            loss_max = np.max(val_loss_list) #終盤の誤差の最大値（振動抑制が目的）
            
            #評価関数の計算
            scores.append(loss_max)
        
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
            
        # self.bar.update(1)
        
        return np.mean(scores)

def hyperparameter_search(X, y):
    model_name = "MLP"
    #前回の続きから最適化を開始するかのスイッチ．Trueでリスタートする．
    restart_switch = False

    if restart_switch:
        #前回の続きから最適化を開始してみる(sutdy_nameが残っていないとできない．study_nameが残っていないときはoptunaログから自分で調査して与えればok)
    #     study_name = 
        study = optuna.load_study(study_name=model_name+"_"+study_name[0], storage=sqlite_path)
        study.trials_dataframe()
        control_log = ControlLog()
        sqlite_path = control_log.set_log(*study_name)
        
    else:
        control_log = ControlLog()
        sqlite_path = control_log.set_log()
        study_name = control_log.decide_filename()

    #訓練時のパラメータ設定
    n_trials=10
    timeout=None

    """
    最後のcontrol_log.kill_handler()が回らないとログが不必要に上書きされるので例外処理で最後まで必ず回るようにする．
    exceptがtry内でエラーが生じたときの処理内容
    finallyはtry内でエラーが生じたとき，生じなかったときどちらも動く処理
    """

    try:
        #ハイパーパラメータの探索
        objective = Objective(X=X, y=y, n_trials=n_trials)

        #計算資源があるときはランダムサーチ，無ければTPESampler
        #storageのパスにすでにDBファイルがあれば，それを読み込むのがload_if_exists
    #     study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+study_name[0],
    #                                 sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),
    #                                 storage=sqlite_path, load_if_exists=True)
        study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+study_name[0],
                                    sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner(),
                                    storage=sqlite_path, load_if_exists=True)

        print(f"study name: {study_name[0]}")

        #最適化の実行．n_trialsは何回実行するか．指定しなければできるだけやる．他にもtimeoutは計算にかける時間の上限値を秒単位で指定できる
        #n_trialsまたはtimeoutのどちらかは指定したほうが良い．でないと永遠に計算し続け，pcが重くなる．
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

    except Exception as error:
        print(error)

    finally:
        #ハンドラの削除．これを行わないとログファイルが上書きされる．
        control_log.kill_handler()


def show_residual_plot(train_x, train_y, test_x, test_y, figsize=[10, 8]):
    xlim = [min(min(train_x), min(test_x))-5, max(max(train_x), max(test_x))+5]
    fig= plt.figure(figsize=figsize)
    plt.scatter(train_x, train_y, s=80, c="limegreen", marker="o", edgecolor="white", label="Training data")
    plt.scatter(test_x, test_y, s=80, c="steelblue", marker="s", edgecolor="white", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color="black", lw=2)
    plt.xlim(xlim)
    plt.tight_layout()
    plt.show()
    
def show_office_residual_plot(train_x, train_y, test_x, test_y, data_indices, office_list, figsize=[10, 8], xlim=None, ylim=None):
    # xlim = [min(min(train_x), min(test_x))-5, max(max(train_x), max(test_x))+5]
    fig= plt.figure(figsize=figsize)

    # カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8")
    colors = list(matplotlib.colors.CSS4_COLORS.values())

    for idx, target_office_name in enumerate(office_list):
        target_office_index = [i for i, data_index in enumerate(data_indices) if target_office_name + '_' in data_index]
        plt.scatter(train_x[target_office_index], train_y[target_office_index], 
                    s=80, c=colors[idx], marker=markers[2], edgecolor="white", label="Training:"+target_office_name)
        
    plt.scatter(test_x, test_y, s=80, c="steelblue", marker="x", edgecolor="white", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    # plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color="black", lw=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()