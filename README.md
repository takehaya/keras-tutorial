# NNC based keras turorial

## description
これは、Sony Neural Network Console(NNC)に実装されてる, 

* RNN: elmannet mnist 
 * tutorial.recurrent_neural_networks.elman_net
* CNN: LeNet mnist
 * tutorial.binary_networks.binary_net_mnist_LeNet

をkerasで実装した直したものである。

ここで得られることとしては、
* NNCのデータセットの転用方法
* NNC側で体験したものを再実装する力
* kerasにおいての実装から計測方法
* CSVを書き換えることで4or9だけではなく自分で1or9などを実装できる

NNCを利用して雰囲気を掴み、それを実装することで理解を深めるのに利用できると思います。

## require
* pipenv installed
* python3 installed
* keras + backend plaidml

## prepare

```bash
pipenv install

# exec subshell
pipenv shell 

# keras backend setup
plaidml-setup

# make dataset
cd tools
python create_mnist_csv.py
```

## run example

### RNN
```bash
# 4 or 9 binary Category 
python main.py rnn_bi

# 0-9 ten Category 
python main.py rnn_cat
```

### CNN
```bash
# 0-9 ten Category 
python main.py rnn_cat
```

## 参考資料と謝辞
* https://github.com/keras-team/keras/issues/5838
* http://people.ischool.berkeley.edu/~dbamman/anlp19_slides/20_sequence_labeling.pdf
* https://ahstat.github.io/RNN-Keras-understanding-computations/
* https://www.slideshare.net/Sony_Neural_Network_Console_Libraries/20180227recurrentneuralnetworks

最後ではあるが、NNCを通じてデータを公開してくれたソニー株式会社様、まとめる機会をくれた東北学院大の担当教員たちに感謝。
