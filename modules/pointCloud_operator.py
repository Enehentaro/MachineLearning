import os
import random
import numpy as np
import json
import glob
from stl import mesh
from enum import Enum, auto

# 列挙型の定義
class OfficePart(Enum):
    aircon = auto()
    airvent = auto()
    body = auto()
    room = auto()
    desks = auto()


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

def readSTL_and_makePointCloudDict(officemodel_path, case_list):
    """
    データの前処理

    オフィスモデル（STLファイル）を読み込み、点群情報のみ取り出し、
    ランダムサンプリングで指定個の点群(ndarray)にする。


    aircon: air conditioner
    airvent: hole of Air vent
    body: body of human
    layout: deskd and walls
    """


    # 列挙型のメンバーの列挙
    for part in OfficePart:
        print(part)

    NUM_POINTS = 2048

    pointCloud_dict = {}
    for casename in case_list:
        officename = casename.replace('case', 'office') 

        #まずは単純に読み込み、辞書に格納
        pc_dict = {}
        for part in OfficePart:
            path = officemodel_path + '/' + casename + '/' + part.name
            # print(path)
            stl_list = glob.glob(path + "/*.stl")
            # print(stl_list)
            pc_list = []
            for stlfname in stl_list:
                for mesh_read in mesh.Mesh.from_multi_file(stlfname):
                    points = mesh_read.points.reshape([-1, 3])
                    pc_list.append(points)

            points_concat = np.concatenate(pc_list)
            points_concat = np.unique(points_concat, axis=0)
            pc_dict[part] = points_concat.copy()


        """
        ここから都合合わせのため調整
        """
        pc_dict_modified = {}

        pc_dict_modified["aircon"] = pc_dict[OfficePart.aircon]
        pc_dict_modified["airvent"] = pc_dict[OfficePart.airvent]

        pc_body = pc_dict[OfficePart.body]
        rand_list = random.sample(range(len(pc_body)), k=2048)
        pc_dict_modified["body"] = pc_body[rand_list, :]

        pc_room = pc_dict[OfficePart.room]
        num_sampling_desks = 1024 - len(pc_room)
        print("num_sampling_desks = ", num_sampling_desks)
        pc_desks = pc_dict[OfficePart.desks]
        rand_list = random.sample(range(len(pc_desks)), k=num_sampling_desks)
        pc_sampled_desks = pc_desks[rand_list, :]
        pc_dict_modified["layout"] = np.concatenate([pc_room, pc_sampled_desks])


        pointCloud_dict[officename] = pc_dict_modified


    

if __name__ == '__main__':
    sampling_pointCloudnd()
