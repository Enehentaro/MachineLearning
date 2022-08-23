import random

def get_k_cross_ID_list(data_size:int, k:int, shuffle:bool=True):
    """
    k分割交差検証用のリストを返す関数。

    Parameters
    ----------
    data_size : int
        全データサイズ
    k : int
        分割数
    shuffle : bool
        分割前にIDをシャッフルするか否か

    Returns
    -------
    list_splitted : list[(trainID : list[int], testID : list[int])]
        リストの要素はk個のタプルであり、タプル内にtrain用IDとtest用IDのセットが入っている。

    """

    ID_list = [i for i in range(data_size)]
    if shuffle :
        random.shuffle(ID_list)

    test_size = data_size // k  #割り算の商（整数）

    list_splitted = []
    for i in range(k):
        if i == k-1:
            test_ID = ID_list[test_size*i : ]
        else:
            test_ID = ID_list[test_size*i : test_size*(i+1)]

        train_ID = [i for i in ID_list if i not in test_ID]
        list_splitted.append((train_ID, test_ID))

    return list_splitted


if __name__ == '__main__':
    print(get_k_cross_ID_list(10, 5))
    print(get_k_cross_ID_list(10, 3))
    print(get_k_cross_ID_list(10, 7))