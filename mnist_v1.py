from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def mnist_2():
    np.random.seed(1671)  # for reproducibility

    # network and training
    NB_EPOCH = 100
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10  # number of outputs = number of digits
    # OPTIMIZER = SGD()  # SGD optimizer, explained later in this chapter
    OPTIMIZER = Adadelta()
    LOSS = categorical_crossentropy
    N_HIDDEN = 128
    VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

    # x_が画像データ, y_が0~9のラベル
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
    RESHAPED = 784
    #
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    #
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    model = Sequential()

    image_rows = 28  # 画像縦の画素数
    image_cols = 28  # 画像横の画素数
    image_color = 1  # 画素の色数1　グレースケール[0,255]
    image_shape = (image_rows, image_cols, image_color)  # 画像のデータ形式
    image_size = image_rows * image_cols * image_color  # 画像のニューロン数

    input_size = image_size  # 変換する画像のデータ形式(入力層のニューロン数)
    output_size = 10  # 画像10分類が目標(出力層のニューロン数)
    # ①②「MNIST」のデータをロード/データを訓練用とテスト用に分ける
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    # データの前処理(1)(2)(3)
    # (1)「MNIST」のデータを三次元配列に整形
    train_X = train_X.reshape(-1, image_rows, image_cols, image_color)
    test_X = test_X.reshape(-1, image_rows, image_cols, image_color)
    # (2)「MNIST」の画素データを正規化
    train_X = train_X.astype('float32') / 255
    test_X = test_X.astype('float32') / 255
    # (3)「MNIST」の10分類のラベルデータをone-hotベクトルに直す
    train_Y = to_categorical(train_Y.astype('int32'), output_size)
    test_Y = to_categorical(test_Y.astype('int32'), output_size)

    # before
    x_train_be = train_X[train_X <= 1]
    y_train_be = train_Y[train_Y <= 1]
    x_test_be = test_X[test_X <= 1]
    y_test_be = test_Y[test_Y <= 1]

    x_train_be += train_X[2 < train_X <= 3]
    y_train_be += train_Y[2 < train_Y <= 3]
    x_test_be += test_X[2 < test_X <= 3]
    y_test_be += test_Y[2 < test_Y <= 3]

    x_train_be += train_X[5 < train_X <= 6]
    y_train_be += train_Y[5 < train_Y <= 6]
    x_test_be += test_X[5 < test_X <= 6]
    y_test_be += test_Y[5 < test_Y <= 6]

    x_train_be += train_X[7 < train_X <= 8]
    y_train_be += train_Y[7 < train_Y <= 8]
    x_test_be += test_X[7 < test_X <= 8]
    y_test_be += test_Y[7 < test_Y <= 8]

    # after
    x_train_af = train_X[1 < train_X <= 2]
    y_train_af = train_Y[1 < train_Y <= 2]
    x_test_af = train_X[1 < test_X <= 2]
    y_test_af = train_Y[1 < test_Y <= 2]

    x_train_af += train_X[3 < train_X <= 5]
    y_train_af += train_Y[3 < train_Y <= 5]
    x_test_af += test_X[6 < test_X <= 2]
    y_test_af += test_Y[3 < test_Y <= 5]

    x_train_af += train_X[6 < train_X <= 7]
    y_train_af += train_Y[6 < train_Y <= 7]
    x_test_af += test_X[6 < test_X <= 7]
    y_test_af += test_Y[6 < test_Y <= 7]

    x_train_af += train_X[8 < train_X <= 9]
    y_train_af += train_Y[8 < train_Y <= 9]
    x_test_af += test_X[8 < test_X <= 9]
    y_test_af += test_Y[8 < test_Y <= 9]

    # ③分類器「畳み込み層のあるCNN-LeNet」を設計
    # ③-1 分類器の定義
    model = Sequential()  # 分類器のインスタンスを生成し，目的に応じたレイヤ(層)を追加する

    # (1)畳み込み層とMaxPooling層を利用した画像の特徴分析
    # (入力28X28)->畳み込み層->(出力32)
    model.add(Conv2D(20, input_shape=image_shape, kernel_size=5, padding="same", activation='relu'))  # 畳み込みで入力層を追加

    # (入力20)->MaxPooling層->(出力20)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling層を追加

    # (入力20)->畳み込み層->(出力50)
    model.add(Conv2D(50, kernel_size=5, padding="same", activation='relu'))  # 畳み込み層を追加

    # (入力50)->MaxPooling層->(出力50)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling層を追加

    # (2) 全結合層を使用した画像分類
    # (入力50)-> Flatten層->(出力2450)
    model.add(Flatten())  # Flatten層　入力を平滑化

    # (入力9216)-> 全結合層->(出力128)
    model.add(Dense(500, activation='relu'))  # 全結合層を追加

    # (入力500)->全結合層Dense->(出力10)
    model.add(Dense(output_size, activation='softmax'))  # 出力層を追加

    # model.add(Conv2D(6, kernel_size=5, strides=(1, 1), padding='same', activation='tanh', input_shape=(RESHAPED,)))
    # model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    # model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh'))
    # model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(120, activation='tanh'))
    # model.add(Dense(84, activation='tanh'))
    # model.add(Dense(NB_CLASSES, activation='softmax'))

    # model.compile(
    #     loss=LOSS,
    #     optimizer=OPTIMIZER,
    #     metrics=['accuracy']
    # )
    #
    # print(model.summary())
    #
    # model.fit(X_train, Y_train,
    #           batch_size=BATCH_SIZE,
    #           epochs=NB_EPOCH,
    #           verbose=VERBOSE,
    #           validation_split=VALIDATION_SPLIT)
    # score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    # print("\nTest score:", score[0])
    # print('Test accuracy:', score[1])

    # ③-2 分類器「畳み込み層のあるCNN」をコンパイル
    model.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    # ④分類器「畳み込み層のあるCNN」でミニバッチを繰り返し訓練を実行
    hist = model.fit(train_X, train_Y, batch_size=128, epochs=12, verbose=1, validation_data=(test_X, test_Y))
    # ⑤分類器「畳み込み層のあるCNN」でテストを実行
    score = model.evaluate(test_X, test_Y, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    # ⑥結果の出力
    # ⑥-時間をかけた訓練結果のモデル保存
    # 場所はカレントディレクトリに保存
    model.save('keras_mnist-CNN-LeNet_model.h5')# モデル構造と重みパラメータを含む


    # ⑥-2 学習経過をグラフに記録
    # 正解率の推移をプロット
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # ロスの推移をプロット
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def mnist_1():
    np.random.seed(1671)  # for reproducibility

    # network and training
    NB_EPOCH = 100
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10  # number of outputs = number of digits
    # OPTIMIZER = SGD()  # SGD optimizer, explained later in this chapter
    OPTIMIZER = Adam()
    N_HIDDEN = 128
    VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

    # x_が画像データ, y_が0~9のラベル
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
    RESHAPED = 784
    #
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    #
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    # 10 outputs
    # final stage is softmax

    model = Sequential()
    model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])

    # callbacks = [make_tensorboard(set_dir_name='keras_MINST_V1')]

    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=NB_EPOCH,
              verbose=VERBOSE,
              validation_split=VALIDATION_SPLIT)

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])
