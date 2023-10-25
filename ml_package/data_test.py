import sys
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

def test_DataFrame(df:pd.DataFrame):
    """
    ケース名と説明変数がマッチしているかをテスト
    ケース名に基づいたDataFrameを作成し、引数DataFrameと比較する
    """
    
    office_list, aircon_list, ventilation_list, exhaust_list = [], [], [], []
    
    for casename in df["case_name"]:
        casename_split = casename.split("_")
        
        # ケース名を分割して説明変数を抽出
        office, aircon, ventilation = casename_split[0], float(casename_split[1]), float(casename_split[2])
        
        if ventilation == 0:
            if len(casename_split) == 4: # 換気量がゼロなのに分割数が４の場合エラー
                sys.stderr.write(f'Error: {casename_split}\n')
                return
            
            exhaust = 'off'
            
        else:
            if len(casename_split) == 3: # 換気量がゼロでないのに分割数が３の場合エラー
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
    assert_frame_equal(df[["office", "aircon", "ventilation", "exhaust"]], df_target)

def standardization_test(name:str, X:np.ndarray):
    """
    標準化されているかをテスト
    """
    threshold = 1.e-5
    
    if abs(X.mean()) > threshold:
        sys.stderr.write(f'StandardizationError: {name}_mean= {X.mean()}\n')
        
    if abs(X.std() - 1.) > threshold:
        sys.stderr.write(f'StandardizationError: {name}_std= {X.std()}\n')
        