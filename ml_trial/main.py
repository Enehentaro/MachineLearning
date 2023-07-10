import argparse
import glob
import os
import pprint
import statistics
import sys
import traceback
import warnings

import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
import numpy as np
import optuna
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

import ml_package

from ml_package import log_controller
from ml_package import optimizer
from ml_package import show_mod

# FutureWarnigが邪魔なので非表示にする．動作に支障が無ければ問題ない．
# また最適化によって解が収束しないときに出るConvergenceWarningも邪魔なので非表示にする．
warnings.simplefilter("ignore", category=(FutureWarning, ConvergenceWarning))#対象のwarningsクラスはタプルで渡す必要があるらしい
sys.path.append("/mnt/MachineLearning")



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="MLP", help="select machine learning model.")
    args = parser.parse_args()
    return args

def main():
    args = argparser()

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
    hyperparameter_search(model_name=args.model, X=train_explanatory_variable, y=train_objective_variable)

def hyperparameter_search(model_name, X, y):
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
    n_trials=2
    timeout=None

    """
    最後のcontrol_log.kill_handler()が回らないとログが不必要に上書きされるので例外処理で最後まで必ず回るようにする．
    exceptがtry内でエラーが生じたときの処理内容
    finallyはtry内でエラーが生じたとき，生じなかったときどちらも動く処理
    """

    try:
        #ハイパーパラメータの探索
        objective = Objective(model_name=model_name, X=X, y=y ,n_trials=n_trials)

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
        t, v, tb = sys.exc_info()
        traceback.print_tb(tb)
        print(t, v)

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