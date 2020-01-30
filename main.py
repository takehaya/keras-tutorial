import mnist_RNN as rnn
import mnist_CNN as cnn
import csv

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
    rnn.mnist_bi()
    # rnn.mnist_cat()
    # cnn.mnist()


if __name__ == '__main__':
    main()
