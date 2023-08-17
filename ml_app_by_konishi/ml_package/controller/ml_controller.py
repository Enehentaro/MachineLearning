import ml_package

from ml_package.model import data_manager
from ml_package.model import hyperparameter_searcher


def train_ml_model(model_name, split_method):
    RoI_preprocessor = data_manager.RoICsvDataModel()
    train_explanatory_variable, test_explanatory_variable, \
    train_objective_variable, test_objective_variable = \
        RoI_preprocessor.preprocess_data(split_method=split_method)
    searcher = hyperparameter_searcher.OptunaHyperparameterSearcher()
    searcher.hyperparameter_search(
        model_name=model_name, 
        split_method=split_method,
        X=train_explanatory_variable, 
        y=train_objective_variable,
        n_trials=2
    )