import mnist_RNN as rnn # Elman
import mnist_CNN as cnn # Lenet, NN
import csv
import sys


# 自分でファイルを抽出して改造する時に利用してください
def make_csv(readfile='./tools/mnist_test.csv', writefile="mnist_test1-9.csv"):
    outputlist = []
    with open(readfile) as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時
        outputlist.append(header)
        for r in reader:
            s = r[0].split("/")
            if s[2] == "1" or s[2] == "9":
                print(r)
                outputlist.append(r)
    with open(writefile, "w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # 改行記号で行を区切る
        writer.writerows(outputlist)


def main():
    for args in sys.argv[1:]:
        if args == "rnn_bi":
            rnn.mnist_bi()
        elif args == "rnn_cat":
            rnn.mnist_cat()
        elif args == "cnn_cat":
            cnn.mnist()
        elif args == "make_csv":
            make_csv()
        else:
            print("not selected")


if __name__ == '__main__':
    main()
