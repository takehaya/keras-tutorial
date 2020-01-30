from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

import tools.loadcsv as nd


def mnist():
    np.random.seed(1671)  # for reproducibility
    nnc = nd.NncData()
    x_train, x_test, y_train, y_test = nnc.load_nnc_image('./tools/mnist_test.csv', width=28, height=28)


    image_rows = 28  # 画像縦の画素数
    image_cols = 28  # 画像横の画素数
    image_color = 1  # 画素の色数1　グレースケール[0,255]
    image_shape = (image_rows, image_cols, image_color)  # 画像のデータ形式

    output_size = 10  # 画像10分類が目標(出力層のニューロン数)

    train_X = x_train
    test_X = x_test
    train_Y = y_train
    test_Y = y_test


    # 畳み込み層のあるCNN-LeNet
    model = Sequential()

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


    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    hist = model.fit(
        train_X,
        train_Y,
        batch_size=128,
        epochs=12,
        verbose=1,
        validation_data=(test_X, test_Y)
    )

    score = model.evaluate(test_X, test_Y, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])

    model.save('keras_mnist-CNN-LeNet_model.h5')


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
    print(X_train.shape, 'train samples')
    print(X_test.shape, 'test samples')
    #
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
