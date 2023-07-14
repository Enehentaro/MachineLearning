"""
Define data manager model to process data in a variety of ways
TODO Separate standardization of training and test data
"""
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import ml_package

from ml_package.view import show_mod


DEFAULT_DATA_PATH = "/mnt/MachineLearning/data/summary_20230418.csv"
DEFAULT_OUTPUT_RESULT_PATH = \
    "/mnt/MachineLearning/ml_app_by_konishi/result/image/"


class DataModel(object):
    """
    Base data model
    
    Parameters
    ----------
    data_path : str, default=DEFAULT_DATA_PATH
        Data path where you want to use for ML.

    output_result_path : str, dfault=DEFAULT_OUTPUT_RESULT_PATH
        A path where result will be output.
    """

    def __init__(self, data_path=None, output_result_path=None):
        """Setup path of data to be loaded"""
        self.current_dir_path = os.getcwd()
        if not data_path:
            data_path = DEFAULT_DATA_PATH
        if not output_result_path:
            output_result_path = DEFAULT_OUTPUT_RESULT_PATH
        self.data_path = data_path
        self.output_result_path = output_result_path
        os.makedirs(output_result_path, exist_ok=True)


class RoICsvDataModel(DataModel):
    """Definition of class that manage csv data of RoI"""

    def __init__(self, data_path=None, output_result_path=None):
        super().__init__(data_path=data_path, output_result_path=output_result_path)
        self.df_loaded_data = self.load_data()
        
    def load_data(self):
        """
        Load csv data by using pandas.

        Returns
        -------
        self.df_data_loaded : pandas DataFrame
            Air conditioning and RoI are stored in the pandas DataFrame
        """
        self.df_loaded_data = pd.read_csv(f"{self.data_path}", 
                                          index_col="case_name")
        return self.df_loaded_data

    def preprocess_data(self, RoI_name="countTimeMean_onlyFloating", 
                        dummy_variable_list=None, explanatory_variable_list=None):
        """
        Preprocess data. e.g. make dummy variable, train test split...

        Parameters
        ----------
        RoI_name : str, default="countTimeMean_onlyFloating"
            RoI name to be used.

        dummy_variable_list : list[columns name to be got dummies],
            default=["exhaust"]
            Specify what you want to get dummy variable.

        explanatory_variable_list : list[columns name to be used],
            default=[
                "aircon", "ventilation", 
                "1_x", "1_y", "1_angle", 
                "2_x", "2_y", "2_angle", 
                "3_x", "3_y", "3_angle", 
                "4_x", "4_y", "4_angle", 
                "5_x", "5_y", "5_angle", 
                "size_x","size_y", 
                "exhaust_a", "exhaust_b", "exhaust_off"
            ]
            Specify what you want to use as variable.


        Returns
        -------
        rain_explanatory_variable, test_explanatory_variable,
        train_objective_variable, test_objective_variable : 
            tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
        """
        if dummy_variable_list is None:
            dummy_variable_list = ["exhaust"]

        if explanatory_variable_list is None:
            explanatory_variable_list = [
                "aircon", "ventilation", 
                "1_x", "1_y", "1_angle", 
                "2_x", "2_y", "2_angle", 
                "3_x", "3_y", "3_angle", 
                "4_x", "4_y", "4_angle", 
                "5_x", "5_y", "5_angle", 
                "size_x","size_y", 
                "exhaust_a", "exhaust_b", "exhaust_off"
            ]

        df_preprocessed_data = pd.get_dummies(self.df_loaded_data, 
                                              columns=dummy_variable_list)
        # Definition of explanatory and objective variable
        df_explanatory_variable = df_preprocessed_data[explanatory_variable_list]
        df_objective_variable = df_preprocessed_data[RoI_name]

        # Standardization of only explanatory variable
        stdscaler = preprocessing.StandardScaler()
        stdscaler.fit(df_explanatory_variable)
        np_explanatory_variable_std = stdscaler.transform(df_explanatory_variable)
        df_explanatory_variable_std = pd.DataFrame(
            np_explanatory_variable_std, 
            index=df_explanatory_variable.index, 
            columns=df_explanatory_variable.columns
        )

        # Split training and test data
        train_explanatory_variable, test_explanatory_variable = \
            train_test_split(df_explanatory_variable_std, test_size=0.3, random_state=0)
        train_objective_variable = \
            df_objective_variable.loc[train_explanatory_variable.index]
        test_objective_variable = \
            df_objective_variable.loc[test_explanatory_variable.index]

        return train_explanatory_variable, test_explanatory_variable, \
               train_objective_variable, test_objective_variable