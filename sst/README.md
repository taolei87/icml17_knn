# Sequence Kernel Networks

This example trains a multi-layer string kernel network on Stanford Sentiment Treebank (SST).

Data can be found [here](https://github.com/taolei87/text_convnet/tree/master/data)

<br>

### Results

Fine-grained classification  |  Dev acc. |  Test acc. 
:--- |:--- |:---
d=200, dropout 0.35, rnn dropout 0.2, lr decay 0.95  |  53.7 (±0.5)  |  52.4 (±0.5)
**Binary classification**  |  |  
d=200, dropout 0.35, rnn dropout 0.1, lr decay 0.95  |  90.1 (±0.5)  |  89.6 (±0.3)
| |

We use a 3-layer network with around 540k parameters. Glove word embeddings are normalized to unit vectors and fixed during training and testing.

<br>

### Usage

Code requires [Theano](http://deeplearning.net/software/theano/), and has been tested on Theano 0.9.0

`python main.py --help` gives the following arguments:
```
optional arguments:
  --train            training set
  --dev              validation set
  --test             test set
  --hidden_dim, -d   hidden dimension
  --learning_rate    learning rate
  --activation       type of activation (none, relu, tanh etc.)
  --batch_size       mini batch size
  --depth            number of stacking recurrent layers
  --dropout          dropout rate between layers
  --rnn_dropout      variational dropout within RNN cells
  --highway          whether to use highway connections (0 or 1)
  --lr_decay         decrease learning rate by this factor after each epoch
  --multiplicative   whether to use multiplicative KNN or additive KNN (0 or 1)
  --max_epoch        maxmimum number of training epochs
```
