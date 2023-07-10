"""
Define hyperparameter searcher
"""
import sys
import traceback

import optuna

import ml_package

from ml_package.model import log_controller
from ml_package.model import optimizer


class OptunaSearcher(object):
    """
    Base hyperparameter searcher by using optuna
    
    Parameters
    ----------
    data_path : str, default=DEFAULT_DATA_PATH
        Set data path where you want to use for ML.

    output_result_path : str, dfault=DEFAULT_OUTPUT_RESULT_PATH
        Set the path where result will be output.
    """

    def __init__(self, n_trials=10, timeout=None):
        self.n_trials = n_trials
        self.timeout = timeout



class OptunaHyperparameterSearcher(OptunaSearcher):
    """Definition of class that search hyperparameter by using optuna"""
    
    def __init__(self, n_trials=10, timeout=None):
        super().__init__(n_trials=n_trials, timeout=timeout)

    def control_log_decorator(self, func):
        def wrapper(self, *args, **kwargs):
            control_log = log_controller.ControlLog()
            sqlite_path = control_log.set_log_filehandler(logger_name=__name__)
            study_name = control_log.decide_filename()
            try:
                func(*args, **kwargs)
            except Exception as error:
                t, v, tb = sys.exc_info()
                traceback.print_tb(tb)
                print(t, v)
            finally:
                #ハンドラの削除．これを行わないとログファイルが上書きされる．
                control_log.kill_handler()
        return wrapper
    
    # @control_log_decorator
    # def hyperparameter_search(self, model_name, X, y, restart=False):
    #     #ハイパーパラメータの探索
    #     objective = optimizer.Objective(model_name=model_name, X=X, y=y ,n_trials=self.n_trials)
    #     #計算資源があるときはランダムサーチ，無ければTPESampler
    #     #storageのパスにすでにDBファイルがあれば，それを読み込むのがload_if_exists
    #     #     study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+study_name[0],
    #     #                                 sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),
    #     #                                 storage=sqlite_path, load_if_exists=True)
    #     study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+study_name[0],
    #                                 sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner(),
    #                                 storage=sqlite_path, load_if_exists=True)

    #     print(f"study name: {study_name[0]}")

    #     #最適化の実行．n_trialsは何回実行するか．指定しなければできるだけやる．他にもtimeoutは計算にかける時間の上限値を秒単位で指定できる
    #     #n_trialsまたはtimeoutのどちらかは指定したほうが良い．でないと永遠に計算し続け，pcが重くなる．
    #     study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)


def hyperparameter_search(model_name, X, y, restart=False):
    #前回の続きから最適化を開始するかのスイッチ．Trueでリスタートする．

    if restart:
        #前回の続きから最適化を開始してみる(sutdy_nameが残っていないとできない．study_nameが残っていないときはoptunaログから自分で調査して与えればok)
    #     study_name = 
        study = optuna.load_study(study_name=model_name+"_"+study_name[0], storage=sqlite_path)
        study.trials_dataframe()
        control_log = log_controller.ControlLog(*study_name)
        sqlite_path = control_log.set_log_filehandler(logger_name=__name__)
        
    else:
        control_log = log_controller.ControlLog()
        # sqlite_path = control_log.set_log_filehandler(logger_name=str(__name__))
        sqlite_path = control_log.set_log_filehandler()
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
        objective = optimizer.Objective(model_name=model_name, X=X, y=y ,n_trials=n_trials)

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