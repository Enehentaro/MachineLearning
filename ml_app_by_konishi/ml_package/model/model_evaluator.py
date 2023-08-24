"""
Define model evaluator
"""
import re

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from ml_package.model import data_manager
from ml_package.model import define_model


class Evaluator(object):
    """
    Base machinelearning model evaluator.
    """

    def __init__(self, model_name, tr_X, tr_y, va_X, va_y):
        self.model_name = model_name
        self.tr_X = tr_X
        self.tr_y = tr_y
        self.va_X = va_X
        self.va_y = va_y


class OptunaEvaluator(Evaluator):
    """
    Machinelearning model evaluator by result of Optuna.
    """
    def __init__(self, model_name, tr_X, tr_y, va_X, va_y, optuna_logs_dir):
        super().__init__(model_name, tr_X, tr_y, va_X, va_y)
        self.optuna_logs_dir = optuna_logs_dir

    def evaluate(self):
        latest_study_path = \
            data_manager.FileModel.search_latest_file_in_dir(
            search_dir_path=self.optuna_logs_dir,
            glob_path="*/*.txt",
            num_latest_files=1
            )
        with latest_study_path[0][0].open(mode="r") as f:
            sqlite_path = f.readline().rstrip()
            study_name = re.search(r"[a-zA-Z]+_\d+$", f.readline()).group()
        study = optuna.load_study(study_name=study_name, storage=sqlite_path)

        print(f"best model in study {study_name}")
        print(f"best score: {study.best_value}")
        print(f"best params: {study.best_params}")

        #最適化結果の一覧表示
        study_value = []
        study_params = []
        for i in study.trials:
            study_value.append(i.value)
            study_params.append(i.params)
        df_study_value = pd.DataFrame(study_value)

        #pandasDataFrameのmin等は帰ってくる型がpandasSeries
        min_value = df_study_value.min(axis=0)[0]
        min_value_index = df_study_value.idxmin(axis=0)[0]#列の最小値のindexを取得
        use_params = study_params[min_value_index]
        best_trial = study.best_trial

        #決定したハイパーパラメータを使用して全訓練データで学習，評価
        #最適化結果から使ってみたいパラメータを選んでみた
        if self.model_name == "MLP":
            best_model = define_model.MLP(use_params)
            history = best_model.fit(
                tr_X=self.tr_X, tr_y=self.tr_y,
                va_X=self.va_X, va_y=self.va_y,
                verbose=1
            )

            train_pred = best_model.predict(self.tr_X)
            test_pred = best_model.predict(self.va_X)
            print("テストデータを用いた結果")
            print(f"loss train score:{mean_squared_error(self.tr_y, train_pred, squared=False)}")
            print(f"loss test score:{mean_squared_error(self.va_y, test_pred, squared=False)}")

            # test_dict = {}
            # test_dict[test_office] = {"best_trial":best_trial, "history":history}