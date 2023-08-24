import argparse
import logging
import os
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning
from tensorflow import config

import ml_package

import ml_package.controller.ml_controller


logging.basicConfig(level=logging.INFO)

# FutureWarnigが邪魔なので非表示にする．動作に支障が無ければ問題ない．
# また最適化によって解が収束しないときに出るConvergenceWarningも邪魔なので非表示にする．
warnings.simplefilter("ignore", category=(FutureWarning, ConvergenceWarning))#対象のwarningsクラスはタプルで渡す必要があるらしい
sys.path.append("/mnt/MachineLearning/ml_app_by_konishi")

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--execute_type", type=int, default=0,
        help="Select execute type by int. 0:training and test, 1:only training, 2:only test"
    )
    parser.add_argument(
        "-c", "--cpu", type=str, default="0",
        help="If you want to use oneDNN, set to 1."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, 
        help="Select machine learning model."
    )
    parser.add_argument(
        "-s", "--split_method", type=str, default="random",
        help="Select train test split method. \"random\" or \"office\""
    )
    args = parser.parse_args()
    return args

def main():
    args = argparser()

    # GPUの環境変数設定
    gpus = config.list_physical_devices(device_type = 'GPU')
    if gpus:
        print(f">> GPU detected. {gpus[0].name}")
        print("Set GPU number if you want to use GPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = input(">>")

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = args.cpu

    if args.execute_type == 0:
        ml_package.controller.ml_controller.train_and_test_ml(
            model_name=args.model, split_method=args.split_method
        )
    elif args.execute_type == 1:
        ml_package.controller.ml_controller.train_ml(
            model_name=args.model, split_method=args.split_method
        )
    elif args.execute_type == 2:
        ml_package.controller.ml_controller.test_ml(
            model_name=args.model, split_method=args.split_method
        )


if __name__ == "__main__":
    main()