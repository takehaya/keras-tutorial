from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import tools.loadcsv as nd


def mnist():
    np.random.seed(1671)  # for reproducibility
    nnc = nd.NncData()
    NB_CLASSES = 1
    # x_train, x_test, y_train, y_test = nnc.load_nnc_image('./tools/mnist_test.csv', width=28, height=28)
    x_train, x_test, y_train, y_test = nnc.load_nnc_image(
        './tools/small_mnist_4or9_training.csv',
        width=28,
        height=28,
        category=NB_CLASSES,
    )
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    train_X = x_train
    test_X = x_test
    train_Y = y_train
    test_Y = y_test

    print('x_train shape:', x_train.shape)
    print(train_X.shape[0], 'train samples')
    print(train_X.shape[1:], 'train samples')
    print('hogehoge samples')

    print(train_X.shape, 'test samples')
    print(test_X.shape, 'train samples')
    print(train_Y.shape, 'train samples')

    print(test_Y.shape, 'test samples')

    # RNN-Elman
    dim_in = 28
    dim_out = 1
    length = 28
    n_hidden = 128
    batch_size = 64
    epochs = 100

    model = Sequential()
    model.add(SimpleRNN(n_hidden, input_shape=(dim_in, length), return_sequences=True))
    model.add(TimeDistributed(Dense(dim_out)))
    # model.add(Flatten())
    model.add(Activation('softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        metrics=['accuracy']
    )
    model.summary()

    hist = model.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_X, test_Y),
    )

    score = model.evaluate(test_X, test_Y, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])

    model.save('keras_mnist-RNN-ElmanNet_model.h5')# モデル構造と重みパラメータを含む


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
