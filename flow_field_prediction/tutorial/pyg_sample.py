import torch
print("PyTorch ==", torch.__version__)
print("CUDA available", torch.cuda.is_available())
print("CUDA ==", torch.version.cuda)

import torch.nn.functional as F
 
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
 
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

def main():
    # データセットをダウンロード
    dataset = KarateClub()
    
    print("グラフ数:", len(dataset))  # グラフ数: 1
    print("クラス数:",dataset.num_classes)  # クラス数: 2
    
    data = dataset[0]  # 1つめのグラフ
    check_graph(data)

    # networkxのグラフに変換
    nxg = to_networkx(data)

    # 可視化のためのページランク計算
    pr = nx.pagerank(nxg)
    pr_max = np.array(list(pr.values())).max()

    # 可視化する際のノード位置
    draw_pos = nx.spring_layout(nxg, seed=0) 

    # ノードの色設定
    cmap = plt.get_cmap('tab10')
    labels = data.y.numpy()
    colors = [cmap(l) for l in labels]

    # 図のサイズ
    fig = plt.figure(figsize=(10, 10))

    # 描画
    nx.draw_networkx_nodes(nxg, 
                        draw_pos,
                        node_size=[v / pr_max * 1000 for v in pr.values()],
                        node_color=colors, alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

    plt.title('KarateClub')
    fig.savefig("graph.png")

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            hidden_size = 5
            self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
            self.conv2 = GCNConv(hidden_size, dataset.num_classes)
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            
            return F.log_softmax(x, dim=1)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルのインスタンス生成
    model = Net()
    # print(model)

    # モデルを訓練モードに設定
    model.train()

    #input data
    data = dataset[0]
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # learnig loop
    for epoch in range(10000):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    # モデルを評価モードに設定
    model.eval()
    
    # 推論
    _, pred = model(data).max(dim=1)
    
    print("結果：", pred)
    print("真値：", data["y"])


    test_dataset = KarateClub()
    test_data = test_dataset[0]
    
    x = test_data["x"]
    edge_index = test_data['edge_index']
    
    # change link of ID:0, ID:2, ID:8, ID:9 and ID:33
    edge_index[1][61] = 0  # 「ノード9⇒ノード2」から「ノード9⇒ノード0」にリンク変更
    edge_index[1][7] = 9   # 「ノード0⇒ノード8」から「ノード0⇒ノード9」にリンク変更
    edge_index[1][56] = 9   # 「ノード8⇒ノード0」から「ノード8⇒ノード9」にリンク変更
    edge_index[1][62] = 8   # 「ノード9⇒ノード33」から「ノード9⇒ノード8」にリンク変更
    edge_index[1][140] = 2  # 「ノード33⇒ノード9」から「ノード33⇒ノード2」にリンク変更
    edge_index[1][30] = 33  # 「ノード2⇒ノード9」から「ノード2⇒ノード33」にリンク変更
    
    t_data = Data(x=x, edge_index=edge_index)
    check_graph(t_data)

    model.eval() #モデルを評価モードにする。
    _, pred = model(t_data).max(dim=1)
    
    print(" ======== リンク変更前のラベル ======== ")
    print(data["y"])
    print(" ==== リンク変更後のラベル予測結果 ==== ")
    print(pred)

def check_graph(data):
    '''グラフ情報を表示'''
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("====== ノードのクラス:y ======")
    print(data['y'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])

if __name__ == "__main__":
    main()