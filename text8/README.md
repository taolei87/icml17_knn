### Setup
  - Download `text8.zip` from http://mattmahoney.net/dc/text8.zip
  - `train_knn.py` needs `pycuda`. Recommended installation via `pip install pycuda` using `virtualenv`.
  - Make sure `cuda/bin` is in system env `PATH`.
  
### How to run

Example: `python train_lstm.py --cuda --path text8.zip --depth 3 --d 1024`

<br>

`python main.py --help` gives the following arguments:
```
optional arguments:
  --path             dataset path
  --d                hidden dimension
  --learning_rate    learning rate
  --activation       type of activation (none, tanh)
  --batch_size       mini batch size
  --unroll_size      bptt unroll size
  --depth            number of stacking recurrent layers
  --drop_x           dropout rate on word vectors
  --drop_o           dropout rate on the output of each layer
  --rnn_dropout      variational dropout within RNN cells
  --lr_decay_epoch   decrease learning rate after this epoch
  --lr_decay         decrease learning rate by this factor after one epoch
  --clip_norm        clip gradient norm to this value
  --max_epoch        maxmimum number of training epochs
```
