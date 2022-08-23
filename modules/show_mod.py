import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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