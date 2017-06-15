
# Sequence Kernel Networks

This example trains a multi-layer string kernel network on Penn Treebank (PTB) language modeling benchmark.

The model obtains the state-of-the-art result on PTB, getting a test perplexity of ~64.

The following techniques have been adopted for STOA results: 
- [Varational dropout](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
- [Highway connections](https://arxiv.org/abs/1505.00387)
- [Weight tying](https://arxiv.org/abs/1608.05859) between word vectors and softmax output embeddings

## Data

The PTB data is the processed version from [(Mikolov et al, 2010)](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf).

It can be downloaded from [https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb](https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb)


## Usage

Code requires [Theano](http://deeplearning.net/software/theano/), and has been tested on Theano 0.9.0

Example runs and the results:

```
export PYTHONPATH=/path_to_repo/icml17_knn                           # Set python lib path

python main.py -d 355 --lr_decay 0.9 --dropout 0.5 --rnn_dropout 0   # Test ppl of 69.3,  5m parameters
python main.py -d 950 --lr_decay 0.95 --lr_decay_epoch 30            # Test ppl of 65.5,  20m parameters
python main.py -d 860 --lr_decay 0.98 --depth 4 --max_epoch 200      # Test ppl of 63.8,  20m parameters
```
(Note: you need to specify training, validation and test files via arguments `--train` `--dev` `--test`)

<br>

`python main.py --help` gives the following arguments:
```
optional arguments:
  --train            training set
  --dev              validation set
  --test             test set
  --hidden_dim, -d   hidden dimension
  --learning_rate    learning rate
  --activation       type of activation (none, tanh, sigmoid etc.)
  --batch_size       mini batch size
  --unroll_size      bptt unroll size
  --depth            number of stacking recurrent layers
  --dropout          dropout rate on word vectors and the last output layer
  --rnn_dropout      variational dropout within RNN cells
  --highway          whether to use highway connections (0 or 1)
  --lr_decay_epoch   decrease learning rate after this epoch
  --lr_decay         decrease learning rate by this factor after one epoch
  --max_epoch        maxmimum number of training epochs
```
