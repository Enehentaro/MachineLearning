"""
Define data manager model to process data in a variety of ways
TODO Separate standardization of training and test data
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import ml_package

from ml_package.model.define_my_error import StandardizationError
from ml_package.model.define_my_error import CasenameSplitError
from ml_package.view import show_mod


DEFAULT_DATA_PATH = "/mnt/MachineLearning/data/summary_20230418.csv"
DEFAULT_POINT_CLOUD_DATA_PATH = \
    "/mnt/MachineLearning/Office3DModel/PointCloud_sampled/PointCloud_dict.npy"
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

    def standardization_test(self, name, X):
        """
        Test of standardization
        
        Parameters
        ----------
        name : str
            Name of array that you want to check standardization.

        X : np.ndarray
            A numpy array that you want to check standardization.
        """
        threshold = 1.e-5
        
        if abs(X.mean()) > threshold:
            raise StandardizationError(f"{name}_mean= {X.mean()}")
            # sys.stderr.write(f'StandardizationError: {name}_mean= {X.mean()}\n')
            
        if abs(X.std() - 1.) > threshold:
            raise StandardizationError(f"{name}_std= {X.std()}")
            # sys.stderr.write(f'StandardizationError: {name}_std= {X.std()}\n')
            

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
        df_loaded_data = pd.read_csv(f"{self.data_path}")
        self.test_DataFrame(df_loaded_data)
        df_loaded_data = df_loaded_data.set_index("case_name")
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
    
    def train_test_stratified_split_by_office(self, data):
        """
        train test stratified split by office

        Parameters
        ----------
        data : pd.DataFrame
            Data to be split. Data must have column["office"].

        
        Returns
        -------
        df_train, df_test : pd.DataFrame
            各オフィスから3つずつ層化抽出したもの
        """
        df_test = pd.DataFrame()
        for office in data["office"].unique():
            df = data[data["office"]==office].sample(n=3, random_state=0)
            df_test = pd.concat([df_test, df])
            
        df_train = data[~data.index.isin(df_test.index)]
            
        return df_train, df_test

    def preprocess_data(
        self, 
        split_method, 
        RoI_name="countTimeMean_onlyFloating",
        dummy_variable_list=None, 
        explanatory_variable_list=None, 
        std=True
    ):
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
            Specify what you want to use as variables.

        std : bool, default=True
            Specify whether to standardize or not.


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
        df_explanatory_variable = df_preprocessed_data[explanatory_variable_list].copy()
        df_objective_variable = df_preprocessed_data[RoI_name].copy()

        # Standardization of only explanatory variable
        if std:
            if "office" in explanatory_variable_list:
                df_explanatory_variable.drop(columns="office", inplace=True)
            stdscaler = preprocessing.StandardScaler()
            np_explanatory_variable_std = \
                stdscaler.fit_transform(df_explanatory_variable)
            df_explanatory_variable = pd.DataFrame(
                np_explanatory_variable_std, 
                index=df_explanatory_variable.index, 
                columns=df_explanatory_variable.columns
            )

        if "office" in explanatory_variable_list:
            df_explanatory_variable = df_explanatory_variable.join(
                pd.DataFrame(df_preprocessed_data["office"])
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
        elif split_method == "stratified":
            train_explanatory_variable, test_explanatory_variable = \
                self.train_test_stratified_split_by_office(
                    data=df_explanatory_variable
                )
            #shuffle data
            train_explanatory_variable = \
                train_explanatory_variable.sample(frac=1, random_state=1)
            test_explanatory_variable = \
                test_explanatory_variable.sample(frac=1, random_state=1)
            train_objective_variable = \
                df_objective_variable.loc[train_explanatory_variable.index]
            test_objective_variable = \
                df_objective_variable.loc[test_explanatory_variable.index]

        return train_explanatory_variable, test_explanatory_variable, \
               train_objective_variable, test_objective_variable
    
    def test_DataFrame(self, df):
        """
        ケース名と説明変数がマッチしているかをテスト
        ケース名に基づいたDataFrameを作成し、引数DataFrameと比較する

        parameters
        ----------
        df : pd.DataFrame
            Specify pandas DataFrame you want to test.
        """
        office_list, aircon_list, ventilation_list, exhaust_list = [], [], [], []
        
        for casename in df["case_name"]:
            casename_split = casename.split("_")
            
            # ケース名を分割して説明変数を抽出
            office, aircon, ventilation = casename_split[0], float(casename_split[1]), float(casename_split[2])
            
            if ventilation == 0:
                if len(casename_split) == 4: # 換気量がゼロなのに分割数が４の場合エラー
                    raise CasenameSplitError(casename_split)
                    sys.stderr.write(f'Error: {casename_split}\n')
                    return
                
                exhaust = 'off'
                
            else:
                if len(casename_split) == 3: # 換気量がゼロでないのに分割数が３の場合エラー
                    raise CasenameSplitError(casename_split)
                    sys.stderr.write(f'Error: {casename_split}\n')
                    return
                
                exhaust = casename_split[3][0] # "a" or "b"
                
            office_list.append(office)
            aircon_list.append(aircon)
            ventilation_list.append(ventilation)
            exhaust_list.append(exhaust)
            
        df_target = pd.DataFrame(
            {
                "office":office_list,
                "aircon":aircon_list,
                "ventilation":ventilation_list,
                "exhaust":exhaust_list
            },
            index=df.index
        )
                
    #     office_map = map(lambda casename: casename.split("_")[0], casename_list)
    #     df_target["office"] = list(office_map)
        
    #     office_map = map(lambda casename: casename.split("_")[0], casename_list)
    #     df_target["office"] = list(office_map)
        
    #     office_map = map(lambda casename: casename.split("_")[0], casename_list)
    #     df_target["office"] = list(office_map)
        
        # 引数DataFrameと、ケース名から作成したDataFrameが一致するかを比較
        pd.testing.assert_frame_equal(df[["office", "aircon", "ventilation", "exhaust"]], df_target)


class PointCloudDataModel(DataModel):
    """Definition of class that manage pointcloud data"""
    def __init__(self, data_path=DEFAULT_POINT_CLOUD_DATA_PATH, output_result_path=None):
        super().__init__(data_path, output_result_path)
        self.loaded_point_cloud = self.load_data()

    def load_data(self, check_point_cloud=False):
        """
        Load point cloud data.

        Returns
        -------
        point_cloud_dict : dict{office name: list[pointcloud]}
        """
        point_cloud = np.load(self.data_path, allow_pickle='TRUE')    # 0-D Array で返ってくる
        point_cloud_dict = point_cloud.item()    # 0-D Array をひとつのオブジェクトに変換
        if check_point_cloud:
            for office in point_cloud_dict.keys():
                show_mod.plot3d_points(
                    point_cloud=point_cloud_dict[office],
                    office=office, 
                    output_path=self.output_result_path
                )
        return point_cloud_dict
    
    def get_office_dataset(
        self,
        df_core,
        explanatory_variable_list=None, 
        std=True
    ):
        """
        全体データから、特定のオフィスおけるデータセットを返す。
        
        Parameters
        ----------
        df_core:pd.DataFrame
            Main pandas DataFrame.
            Office name column required to browse point cloud. 
            
        explanatory_variable_list : list[columns name to be used],
            default=[
                "aircon", "ventilation",
                "exhaust_a", "exhaust_b", "exhaust_off"
            ]
            Specify what you want to use as variables.
            
        std : bool, default=True
            Specify whether to standardize point cloud or not.
            
        Returns
        -------
        input_dict : dict{
            "meta": metadata such as wind speed conditions,
            "point_cloud" : point cloud data
        }
            機械学習に入力するデータの辞書。メタデータと点群データが詰まっている
        """
        if explanatory_variable_list is None:
            explanatory_variable_list = [
                "aircon", "ventilation", 
                "exhaust_a", "exhaust_b", "exhaust_off"
            ]
        df_main = df_core.copy()
            
        # make Dictionary of point cloud list in order of MainDataFrame
        point_cloud_list = []
        for office in df_main["office"]:
            point_cloud_list.append(self.loaded_point_cloud[office])
            
        np_point_cloud = np.array(point_cloud_list)
        del point_cloud_list
        
        if std:
            temp_array = np_point_cloud.reshape(-1,3)
            stdscaler = preprocessing.StandardScaler()
            np_point_cloud = \
                stdscaler.fit_transform(temp_array).reshape(np_point_cloud.shape)
                
        # make input-Dictionary and to ND-Array
        input_dict = {
            "meta":df_main[explanatory_variable_list],
            "point_cloud":np_point_cloud,
        }

        return input_dict


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
        latest_file_path : list[tuple(latest file path, UNIX time the file was modified)]
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