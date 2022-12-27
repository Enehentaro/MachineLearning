import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

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
    
def show_office_residual_plot(train_x, train_y, test_x, test_y, office_list, figsize=[10, 8], xlim=[50, 250], ylim=[-60,120]):
#     xlim = [min(min(train_x), min(test_x))-5, max(max(train_x), max(test_x))+5]
    fig= plt.figure(figsize=figsize)

    #カラーマップ等の準備
    markers = ("s", "x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8")
    colors = ("red", "blue", "limegreen", "gray", "cyan", "black", "purple", "green",
              "orange", "yellow", "crimson", "goldenrod", "orchid", "khaki", "darkgray")

    for idx, target_office_name in enumerate(np.unique(office_list)):
        target_office_index = [i for i in range(office_list.shape[0]) if any(office_list[i] == target_office_name)]
        plt.scatter(train_x[target_office_index], train_y[target_office_index], 
                    s=80, c=colors[idx], marker=markers[2], edgecolor="white", label="Training:"+target_office_name)
        
    plt.scatter(test_x, test_y, s=80, c="steelblue", marker="x", edgecolor="white", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color="black", lw=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()