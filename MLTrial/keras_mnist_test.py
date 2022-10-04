## keras mnist サンプル
##
## Shin.Onda 2021/06/24 all rights reserved.
## 

## keras をインポート
import tensorflow as tf
from tensorflow import keras as kr
## 結果の評価　数値計算用
import numpy as np

## trainX 学習用データ、trainY 正解データ、testX テスト用データ、testY 正解データ
(trainX, trainY), (testX, testY) = kr.datasets.mnist.load_data()

## 形式変換
# テストデータを変換
trainX = trainX.reshape(60000, 784)
trainX = trainX/255.
testX = testX.reshape(10000, 784)
testX = testX/255.
# 正解データを変換
trainY = kr.utils.to_categorical(trainY, 10)
testY = kr.utils.to_categorical(testY, 10)

## モデル構築
# 中間層１層　全結合型
model = kr.models.Sequential()
model.add(kr.layers.Dense(units=256, input_shape=(784,), activation='relu'))
model.add(kr.layers.Dropout(0.5))

# 出力層
model.add(kr.layers.Dense(units=10, activation='softmax'))

## 学習
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=100, epochs=20, validation_split=0.1)

## テスト
result = model.predict(testX)

## 結果を表示
pred = np.array(result).argmax(axis=1)
print(pred)
label=testY.argmax(axis=1)
print(label)
accuracy = np.mean(pred == label, axis=0)
print(accuracy)

