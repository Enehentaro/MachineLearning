import ml_package

from ml_package.model import data_manager
from ml_package.model import model_evaluator
from ml_package.model import hyperparameter_searcher


DEFAULT_OPTUNALOGS_PATH = "../OptunaLogs"


def train_and_test_ml(model_name, split_method):
    RoI_preprocessor = data_manager.RoICsvDataModel()
    if model_name == "pointnet":
        train_explanatory_variable, test_explanatory_variable, \
        train_objective_variable, test_objective_variable = \
            RoI_preprocessor.preprocess_data(
                split_method=split_method,
                explanatory_variable_list=[
                    "office", "aircon", "ventilation", 
                    "exhaust_a", "exhaust_b", "exhaust_off"
                ]
            )
        point_cloud_preprocessor = data_manager.PointCloudDataModel()
        train_explanatory_variable = \
            point_cloud_preprocessor.get_office_dataset(
                df_core=train_explanatory_variable,
                explanatory_variable_list=[
                    "office", "aircon", "ventilation", 
                    "exhaust_a", "exhaust_b", "exhaust_off"
                ],
                std=True
            )
        test_explanatory_variable = \
            point_cloud_preprocessor.get_office_dataset(
                df_core=test_explanatory_variable,
                explanatory_variable_list=[
                    "office", "aircon", "ventilation", 
                    "exhaust_a", "exhaust_b", "exhaust_off"
                ],
                std=True
            )
    else:
        train_explanatory_variable, test_explanatory_variable, \
        train_objective_variable, test_objective_variable = \
            RoI_preprocessor.preprocess_data(split_method=split_method)
        
    searcher = hyperparameter_searcher.OptunaHyperparameterSearcher(
        optuna_logs_dir=DEFAULT_OPTUNALOGS_PATH
    )
    searcher.hyperparameter_search(
        model_name=model_name, 
        split_method=split_method,
        X=train_explanatory_variable, 
        y=train_objective_variable,
        n_trials=4
    )

    evaluator = model_evaluator.OptunaEvaluator(
        model_name=model_name,
        tr_X=train_explanatory_variable,
        tr_y=train_objective_variable,
        va_X=test_explanatory_variable,
        va_y=test_objective_variable,
        optuna_logs_dir=DEFAULT_OPTUNALOGS_PATH
    )
    evaluator.evaluate()

def train_ml(model_name, split_method):
    RoI_preprocessor = data_manager.RoICsvDataModel()
    train_explanatory_variable, _, train_objective_variable, _ = \
        RoI_preprocessor.preprocess_data(split_method=split_method)
    searcher = hyperparameter_searcher.OptunaHyperparameterSearcher(
        optuna_logs_dir=DEFAULT_OPTUNALOGS_PATH
    )
    searcher.hyperparameter_search(
        model_name=model_name, 
        split_method=split_method,
        X=train_explanatory_variable, 
        y=train_objective_variable,
        n_trials=2
    )

def test_ml(model_name, split_method):
    RoI_preprocessor = data_manager.RoICsvDataModel()
    if model_name == "PointNet".casefold():
        train_explanatory_variable, test_explanatory_variable, \
        train_objective_variable, test_objective_variable = \
            RoI_preprocessor.preprocess_data(
                split_method=split_method,
                explanatory_variable_list=[
                    "office", "aircon", "ventilation", 
                    "exhaust_a", "exhaust_b", "exhaust_off"
                ]
            )
        point_cloud_preprocessor = data_manager.PointCloudDataModel()
        train_explanatory_variable = \
            point_cloud_preprocessor.get_office_dataset(
                df_core=train_explanatory_variable,
                std=True
            )
        test_explanatory_variable = \
            point_cloud_preprocessor.get_office_dataset(
                df_core=test_explanatory_variable,
                std=True
            )
    else:
        train_explanatory_variable, test_explanatory_variable, \
        train_objective_variable, test_objective_variable = \
            RoI_preprocessor.preprocess_data(split_method=split_method)
        
    evaluator = model_evaluator.OptunaEvaluator(
        model_name=model_name,
        tr_X=train_explanatory_variable,
        tr_y=train_objective_variable,
        va_X=test_explanatory_variable,
        va_y=test_objective_variable,
        optuna_logs_dir=DEFAULT_OPTUNALOGS_PATH
    )
    evaluator.evaluate()


if __name__ == "__main__":
    test_ml()