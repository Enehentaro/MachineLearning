import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_scatter(x, y):
    if type(x) == pd.DataFrame:
        x = x.to_numpy()
    if type(y) == pd.DataFrame:
        y = y.to_numpy().ravel()

    #カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0],
                    y=x[y==cl, 1],
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor="black")
    

def plot_decision_regions(x, y, classifier, resolution=0.02):
    if type(x) == pd.DataFrame:
        x = x.to_numpy()
    if type(y) == pd.DataFrame:
        y = y.to_numpy().ravel()
        
    #カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #決定領域のプロット
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    #各特徴量を1次元配列に変換して予測
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)#ravelはflattenと同じ働きをするが，コピーを返さない
    #予測結果をグラフを描画するためのグリッド形状に変換
    z = z.reshape(xx1.shape)
    #等高線の描画
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0],
                    y=x[y==cl, 1],
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor="black")