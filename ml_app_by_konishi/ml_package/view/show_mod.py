import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

colors = list(mpl.colors.cnames.keys())

def make_colormap(colors):
    #可視化用カラーマップの作成
    nmax = float(len(colors) - 1)
    color_list = []
    for n, c in enumerate(colors):
        color_list.append((n/nmax, c))

    return mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list)

def show_image(file_name, title_name="", fontsize=15, cmap="bwr",origin="lower", cbar=False, position="right", size="5%",\
               pad=0.1, vmin=-1, vmax=1):
    #選んだ1個をプロット
    fig, ax = plt.subplots()
    
    if len(title_name)!=0:
        plt.title(title_name, fontsize=fontsize)

    if cbar:
        im = ax.imshow(file_name, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position=position, size=size, pad=pad)
        #rightはカラーバーの位置sizeはカラーバーの幅が画像に対してどれくらいの大きさか、padは画像とカラーバーの隙間
        cbar_fig = fig.colorbar(im, cax=cax)
    
    else:
        im = ax.imshow(file_name, cmap=cmap, origin=origin)

    plt.tight_layout()
    plt.show()

def show_images(file_name, row, column, title_name="", fontsize=15, cmap="bwr",origin="lower", cbar=False,\
                position="right", size="5%", pad=0.1, vmin=-1, vmax=1):
    #複数プロット
    num_image = row * column
    fig = plt.figure(figsize=(19.2, 10.8))
    for i in range(num_image):
        ax = fig.add_subplot(row, column, i + 1)
        
        if len(title_name)!=0:
            plt.title(title_name[i], fontsize=fontsize)

        if cbar:
            im = ax.imshow(file_name[i], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(position=position, size=size, pad=pad)
            cbar_fig = fig.colorbar(im, cax=cax)
        
        else:
            im = ax.imshow(file_name[i], cmap=cmap, origin=origin)
            
    plt.tight_layout()
    plt.show()
    
def show_residual_plot(train_x, train_y, test_x, test_y, figsize=[10, 8]):
    xlim = [min(min(train_x), min(test_x))-5, max(max(train_x), max(test_x))+5]
    fig= plt.figure(figsize=figsize)
    plt.scatter(train_x, train_y, s=80, c="limegreen", marker="o", edgecolor="white", label="Training data")
    plt.scatter(test_x, test_y, s=80, c="steelblue", marker="s", edgecolor="white", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color="black", lw=2)
    plt.xlim(xlim)
    plt.tight_layout()
    plt.show()
    
def show_office_residual_plot(train_x, train_y, test_x, test_y, data_indices, office_list, figsize=[10, 8], xlim=None, ylim=None):
#     xlim = [min(min(train_x), min(test_x))-5, max(max(train_x), max(test_x))+5]
    fig= plt.figure(figsize=figsize)

    #カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8")
    # colors = ("red", "blue", "limegreen", "gray", "cyan", "black", "purple", "green",
    #           "orange", "yellow", "crimson", "goldenrod", "orchid", "khaki", "darkgray")

    for idx, target_office_name in enumerate(office_list):
        target_office_index = [i for i, data_index in enumerate(data_indices) if target_office_name + '_' in data_index]
        plt.scatter(train_x[target_office_index], train_y[target_office_index], 
                    s=80, c=colors[idx], marker=markers[2], edgecolor="white", label="Training:"+target_office_name)
        
    plt.scatter(test_x, test_y, s=80, c="steelblue", marker="x", edgecolor="white", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    # plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color="black", lw=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
    
def plot3d_points(point_cloud, office, output_path):
    """
    点群データ（２次元配列）をプロット
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], alpha=0.4, s=2)
    # ax.set_axis_off()
    fig.savefig(f"{output_path}{office}_point_cloud.png")

    plt.clf()
    plt.close()

    


    # # オフィスのRoIをプロットして出力
    # fig = plt.figure(figsize=[10, 8])
    # ax = fig.add_subplot(111, title="RoI plot", xlabel=RoI_name, ylabel="case index")
    # # カラーマップ等の準備
    # markers = ("s", "x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8")
    # colors = list(matplotlib.colors.CSS4_COLORS.values())#148色までなら対応可能

    # for idx, target_office_name in enumerate(office_array):
    #     df = df_read[df_read["office"]==target_office_name]
    #     ax.scatter(df[RoI_name], df.index, 
    #                 s=80, c=colors[idx], marker=markers[2], edgecolor="white", label=target_office_name)
        
    # ax.legend(loc="center right")
    # ax.grid()
    # fig.savefig(f"{output_path}office_RoI_plot.png")