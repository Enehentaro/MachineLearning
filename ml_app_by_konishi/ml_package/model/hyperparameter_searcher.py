"""
Define hyperparameter searcher
TODO Make optimize setting like TPE or Random searcher
     Write docs of decorator
     Make restart for hyparasearch
"""
import sys
import traceback

import optuna

import ml_package

from ml_package.model import log_manager
from ml_package.model import optimizer


class OptunaSearcher(object):
    """
    Base hyperparameter searcher by using optuna
    
    Parameters
    ----------
    study_name : str
        optuna sutudy name
    
    sqlite_path : str
        sqlite file path where optuna results will be output
    """

    def __init__(self):
        self.study_name = None
        self.sqlite_path = None


class OptunaHyperparameterSearcher(OptunaSearcher):
    """Definition of class that search hyperparameter by using optuna"""
    
    def __init__(self):
        super().__init__()
    
    def control_log_decorator(func):
        def wrapper(self, *args, **kwargs):
            log_dir_path, log_file_name = \
                log_manager.OptunaLogModel.decide_filename(what_log="../OptunaLogs")
            self.study_name = log_file_name
            control_log = log_manager.OptunaLogModel(
                log_dir_path=log_dir_path, log_file_name=log_file_name
            )
            self.sqlite_path = control_log.set_sqlite_path()

            try:
                func(self, *args, **kwargs)
            except Exception as error:
                t, v, tb = sys.exc_info()
                traceback.print_tb(tb)
                print(t, v)
            finally:
                # If you do not delete handler, you access to previous logs.
                control_log.kill_handler()
        return wrapper

    @control_log_decorator
    def hyperparameter_search(self, model_name, split_method, X, y, 
                              restart=False, n_trials=10, timeout=None):
        """
        Search hyperparameter by using optuna

        parameter
        ---------
        model_name : str
            AI model name which you want to use.

        split_method : str
            Specify what you want to use train test split method.

        X : Input data. It could be pd.DataFrame

        y : Target data. Like the input data `X`

        restart : bool, default=False
            Restart switch to restart hyperparameter search by using optuna.

        n_trials : int, default=10
            Number of optimization iterations.

        timeout : int, default=None
            Time limit of optimization.
        """
        if restart:
        #前回の続きから最適化を開始してみる(sutdy_nameが残っていないとできない．study_nameが残っていないときはoptunaログから自分で調査して与えればok)
            study_name = None
            study = optuna.load_study(study_name=model_name+"_"+study_name[0], storage=sqlite_path)
            study.trials_dataframe()
            control_log = log_manager.OptunaLogModel(*study_name)
            sqlite_path = control_log.set_sqlite_path()

        objective = optimizer.Objective(
            model_name=model_name, split_method=split_method, 
            X=X, y=y,
            n_trials=n_trials
        )
        #計算資源があるときはランダムサーチ，無ければTPESampler
        #storageのパスにすでにDBファイルがあれば，それを読み込むのがload_if_exists
        #     study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+study_name[0],
        #                                 sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),
        #                                 storage=sqlite_path, load_if_exists=True)
        study = optuna.create_study(directions=["minimize"], study_name=model_name+"_"+self.study_name,
                                    sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner(),
                                    storage=self.sqlite_path, load_if_exists=True)
        print(f"study name: {self.study_name}")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)