import os
import random
import numpy as np
import json
import glob

def sampling_pointCloudnd(NUM_POINTS:int):
    import pyvista as pv
    """
    データの前処理

    オフィスモデル（STLファイル）を読み込み、点群情報のみ取り出し、
    ランダムサンプリングで指定個の点群(ndarray)にする。
    """
    
    stl_path = '/content/drive/MyDrive/PointNetTrial/data/shape/'

    output_path = '/content/drive/MyDrive/PointNetTrial/data/pointCloud_sampled/'

    case_list = os.listdir(stl_path)

    print(case_list)

    case_label_dict = {}

    for index, case_name in enumerate(case_list):
        filename = stl_path + case_name +'/shape.stl'
        mesh_origin = pv.read(filename)
        points_origin = mesh_origin.points

        rand_list = random.sample(range(points_origin.shape[0]), k=NUM_POINTS)
        points_sampled = points_origin[rand_list, :]
        
        np.save(output_path + '/' + case_name, points_sampled)
        case_label_dict[case_name] = index

    with open(output_path + '/' + 'labels.json', mode="w") as file:
        json.dump(case_label_dict, file, indent=2, ensure_ascii=False)


def read_sampledPointCloud(path:str):
    """
    ランダムサンプリング結果フォルダへアクセスし、
    2048個の点群(ndarray)を取得
    """
    path_list = glob.glob(path + '/*.npy')
    pcArray_dict = {}

    # with open(path + '/' + 'labels.json', mode="r") as file:
    #     case_label_dict = json.load(file)

    for file_path in path_list:
        points = np.load(file_path)
        case_name = os.path.splitext(os.path.basename(file_path))[0]

        pcArray_dict[case_name] = points

    return pcArray_dict

    

if __name__ == '__main__':
    sampling_pointCloudnd()
