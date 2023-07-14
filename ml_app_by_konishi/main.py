import argparse
import logging
import os
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

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
        "-m", "--model", type=str, required=True, 
        help="Select machine learning model."
    )
    args = parser.parse_args()
    return args

def main():
    args = argparser()

    # GPUの環境変数設定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    ml_package.controller.ml_controller.train_ml_model(model_name=args.model)
    
if __name__ == "__main__":
    main()