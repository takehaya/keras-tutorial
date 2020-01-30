import mnist_RNN as rnn
import mnist_CNN as cnn
import csv
from keras.datasets import mnist
from keras.utils import np_utils

def main():
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # nb_classes = 10
    # img_rows, img_cols = 28, 28
    #
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    #
    # # one-hot encoding:
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    # print()
    # print('MNIST data loaded: train:', len(X_train), 'test:', len(X_test))
    # print('X_train:', X_train.shape)
    # print('X_test:', X_test.shape)
    # print('Y_train:', Y_train.shape)
    # print('Y_test:', Y_test.shape)

    rnn.mnist_bi()
    # rnn.mnist_cat()

    # cnn.mnist()
    # outputlist = []
    # with open('./tools/mnist_test.csv') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)  # ヘッダーを読み飛ばしたい時
    #     outputlist.append(header)
    #     for r in reader:  # for文を用いて一行ずつ読み込む
    #         s = r[0].split("/")
    #         if s[2] == "1" or s[2] == "9":
    #             print(r)
    #             outputlist.append(r)
    # with open("mnist_test1-9.csv", "w", encoding="utf-8") as f:  # 文字コードutf8に指定
    #     writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
    #     writer.writerows(outputlist)  # csvファイルに書き込み


if __name__ == '__main__':
    main()
