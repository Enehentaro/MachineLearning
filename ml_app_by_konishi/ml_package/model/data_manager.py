"""
Define data manager model to process data in a variety of ways
TODO Separate standardization of training and test data
"""
import os
from pathlib import Path

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
        df_loaded_data : pandas DataFrame
            Air conditioning and RoI are stored in the pandas DataFrame
        """
        df_loaded_data = pd.read_csv(f"{self.data_path}", 
                                          index_col="case_name")
        return df_loaded_data
    
    def train_test_split_by_office_type(self, data, test_office_list):
        """
        train test split by office type

        Parameters
        ----------
        data : pd.DataFrame
            Data to be split. Data must have column["office"].
        
        test_office_list : list[test office name to be used]
            Specify what you want to use as test data.

        
        Returns
        -------
        train_bool_list, test_bool_list :
            list[bool list of training data index], list[bool list of test data index]
        """
        test_bool_list = np.zeros(data.shape[0], dtype=bool)
        for test_office in test_office_list:
            test_bool_list += data["office"]==test_office
        train_bool_list = [not x for x in test_bool_list]
        return train_bool_list, test_bool_list

    def preprocess_data(self, split_method, RoI_name="countTimeMean_onlyFloating",
                        dummy_variable_list=None, explanatory_variable_list=None, 
                        std=True,):
        """
        Preprocess data. e.g. make dummy variable, train test split...

        Parameters
        ----------
        split_method : str
            Specify what you want to use train test split method.

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
        train_explanatory_variable, test_explanatory_variable,
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
        if std:
            stdscaler = preprocessing.StandardScaler()
            stdscaler.fit(df_explanatory_variable)
            np_explanatory_variable_std = stdscaler.transform(df_explanatory_variable)
            df_explanatory_variable = pd.DataFrame(
                np_explanatory_variable_std, 
                index=df_explanatory_variable.index, 
                columns=df_explanatory_variable.columns
            )

        # Split training and test data
        if split_method == "random":
            train_explanatory_variable, test_explanatory_variable = \
                train_test_split(df_explanatory_variable, test_size=0.3, random_state=0)
            train_objective_variable = \
                df_objective_variable.loc[train_explanatory_variable.index]
            test_objective_variable = \
                df_objective_variable.loc[test_explanatory_variable.index]
        elif split_method == "office":
            train_bool_list, test_bool_list = \
                self.train_test_split_by_office_type(
                data=self.df_loaded_data, test_office_list=["office1"]
                )
            train_explanatory_variable = df_explanatory_variable[train_bool_list]
            test_explanatory_variable = df_explanatory_variable[test_bool_list]
            train_objective_variable = df_objective_variable[train_bool_list]
            test_objective_variable = df_objective_variable[test_bool_list]
            #shuffle data
            train_explanatory_variable = \
                train_explanatory_variable.sample(frac=1, random_state=1)
            train_objective_variable = \
                train_objective_variable.reindex(index=train_explanatory_variable.index)
            test_explanatory_variable = \
                test_explanatory_variable.sample(frac=1, random_state=1)
            test_objective_variable = \
                test_objective_variable.reindex(index=test_explanatory_variable.index)

        return train_explanatory_variable, test_explanatory_variable, \
               train_objective_variable, test_objective_variable


class FileModel(object):
    """
    Base file model
    """

    def __init__(self):
        pass

    @staticmethod
    def search_latest_file_in_dir(search_dir_path, glob_path, num_latest_files):
        """
        train test split by office type

        Parameters
        ----------
        search_dir_path : str
            Specify directory path to search.
        
        glob_path : str
            Specify glob path what you want to search.
            e.g. /*/*.txt

        num_latest_files : int
            The number of latest files you want to get.


        Returns
        -------
        latest_file_path : list[tuple(latest file path, file modified UNIX time)]
        """
        search_file_path = \
            [(p, p.stat().st_ctime) for p in Path(search_dir_path).glob(glob_path)]
        latest_file_path = \
            sorted(search_file_path, key=lambda x: x[1], reverse=True)[: num_latest_files]
        return latest_file_path


if __name__ == "__main__":
    m = RoICsvDataModel()
    # train_explanatory_variable, test_explanatory_variable, \
    # train_objective_variable, test_objective_variable = \
    #     m.preprocess_data()
    train_explanatory_variable, test_explanatory_variable, \
    train_objective_variable, test_objective_variable = \
        m.train_test_split_by_office_type()
    print(test_objective_variable)