import sys
import os
import argparse
import time
import random
import math
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cuda_functional as MF


def read_data(path, num_test_chars=5000000):
    data = zipfile.ZipFile(path).read('text8')
    train_data = data[:-2*num_test_chars]
    dev_data = data[-2*num_test_chars:-num_test_chars]
    test_data = data[-num_test_chars:]
    return train_data, dev_data, test_data


def create_batches(data_text, map_to_ids, batch_size, cuda):
    data_ids = map_to_ids(data_text)
    N = len(data_ids)
    L = ((N-1)/batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size,-1).T)
    y = np.copy(data_ids[1:L+1].reshape(batch_size,-1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    if cuda:
        x, y = x.cuda(), y.cuda()
    return x, y


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, words, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.embedding = nn.Embedding(self.n_V, n_d)

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text],
                 dtype='uint8'
        )

class FastKNNLayer(nn.Module):
    def __init__(self, n_in, n_out, activation=F.tanh, rnn_dropout=0.1, highway=1):
        super(FastKNNLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.highway = highway
        self.activation = activation

        self.input_op = nn.Linear(n_in, n_out, bias=False)
        self.lambda_op = nn.Linear(n_in, n_out)
        if highway:
            assert n_in == n_out
            self.highway_op = nn.Linear(n_in, n_out)

    def forward(self, input, hidden):
        assert input.dim() == 3 # (len, batch, n_in)
        assert isinstance(hidden, tuple) or isinstance(hidden, list)
        assert len(hidden) == 2
        c0, h0 = hidden # (batch, n_out)
        assert c0.size() == h0.size()
        #assert c0.dim() == 2
        #assert c0.size(0) == input.size(1) and c0.size(1) == self.n_out

        length, bs = input.size(0), input.size(1)
        n_in, n_out = self.n_in, self.n_out

        if self.training and (self.rnn_dropout>0):
            mask_x = self.get_dropout_mask_((bs,n_in), self.rnn_dropout)
            mask_x = mask_x.expand_as(input)
            x = input*mask_x
            #mask_h = self.get_dropout_mask_(c0.size(), self.rnn_dropout)
        else:
            x = input
            #mask_h = None

        x_2d = x.view(-1, n_in)
        wx = self.input_op(x_2d).view(length, bs, n_out)
        decay = self.lambda_op(x_2d).view(length, bs, n_out)
        decay = F.sigmoid(decay)
        wx = wx*(1-decay)
        c = MF.weighted_cumsum(wx, decay, c0)
        h = self.activation(c)
        if self.highway:
            #if mask_h:
            #    h = h*mask_h.unsqueeze(0).expand_as(h)
            transform = self.highway_op(x_2d).view(length, bs, n_out)
            transform = F.sigmoid(transform)
            h = h*transform + input*(1-transform)

        return c, h

    def get_dropout_mask_(self, size, p):
        w = self.input_op.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))

class FastKNN(nn.Module):
    def __init__(self, n_in, n_out, depth, activation=F.tanh,
                drop_o=0.0, rnn_dropout=0.1):
        super(FastKNN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.depth = depth
        self.activation = activation
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = []
        self.seq = nn.Sequential()
        self.drop_o = nn.Dropout(args.drop_o)

        for i in range(depth):
            l = FastKNNLayer(
                n_in = n_in if i==0 else n_out,
                n_out = n_out,
                activation = activation,
                rnn_dropout = rnn_dropout
            )
            self.rnn_lst.append(l)
            self.seq.add_module(str(i), l)

    def forward(self, input, hidden):
        assert input.dim() == 3 # (len, batch, n_in)
        assert isinstance(hidden, tuple) or isinstance(hidden, list)
        assert len(hidden) == 2
        prevc, prevh = hidden  # (depth, batch, n_out)
        assert prevc.dim() == 3 and prevh.dim() == 3
        prevc = prevc.chunk(self.depth, 0)
        prevh = prevh.chunk(self.depth, 0)

        prevx = input
        lstc, lsth = [], []
        for i, rnn in enumerate(self.rnn_lst):
            c, h = rnn(prevx, (prevc[i], prevh[i]))
            prevx = self.drop_o(h)
            lstc.append(c[-1])
            lsth.append(h[-1])

        return prevx, (torch.stack(lstc), torch.stack(lsth))


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.d
        self.depth = args.depth
        self.drop_x = nn.Dropout(args.drop_x)
        self.embedding_layer = EmbeddingLayer(
            self.n_d,
            [ chr(97+i) for i in range(26) ] + [' ']
        )
        if args.activation == 'tanh':
            activation = F.tanh
        else:
            activation = lambda x: x
        self.n_V = self.embedding_layer.n_V
        #self.rnn = nn.LSTM(self.n_d, self.n_d, self.depth)
        self.rnn = FastKNN(self.n_d, self.n_d, self.depth,
            activation = activation,
            drop_o = args.drop_o,
            rnn_dropout = args.rnn_dropout
        )
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        # tie weights
        #self.output_layer.weight = self.embedding_layer.embedding.weight

        #self.init_weights()

    def init_weights(self):
        val_range = (3.0/self.n_d)**0.5
        for p in self.parameters():
            if p.dim() == 2:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, x, hidden):
        emb = self.drop_x(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        #output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.depth, batch_size, self.n_d).zero_()),
                Variable(weight.new(self.depth, batch_size, self.n_d).zero_())
        )

    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

def train_model(epoch, model, train):
    model.train()
    args = model.args

    clip_norm = args.clip_norm
    weight_decay = args.weight_decay
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = (len(train[0])-1)/unroll_size + 1

    start_time = time.time()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(batch_size)
    for i in range(N):
        x = train[0][i*unroll_size:(i+1)*unroll_size].long()
        y = train[1][i*unroll_size:(i+1)*unroll_size].long().view(-1)
        x, y =  Variable(x), Variable(y)
        hidden = tuple(Variable(h.data) for h in hidden)

        model.zero_grad()
        output, hidden = model(x, hidden)
        assert x.size(1) == batch_size
        loss = criterion(output, y) / x.size(1)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip_norm)
        for p in model.parameters():
            if p.requires_grad:
                if weight_decay > 0:
                    p.grad.data.add_(weight_decay, p.data)
                p.data.add_(-args.lr, p.grad.data)

        total_loss += loss.data[0] / x.size(0)
        if i%100 == 0:
            sys.stdout.write("\r{} ppl={:.2f} mpe={:.1f}".format(
                i, np.exp(total_loss/(i+1)),
                N*(time.time()-start_time)/(i+1)/60.0
            )+" "*10)
            sys.stdout.flush()

    train_ppl = np.exp(total_loss/N)
    train_bpc = np.log2(train_ppl)
    sys.stdout.write("\rEpoch={}  lr={:.4f} train_loss={:.3f} train_ppl={:.3f}"
            " train_bpc={:.3f}\t[{:.2f}m]\n".format(
        epoch,
        args.lr,
        total_loss/N,
        train_ppl,
        train_bpc,
        (time.time()-start_time)/60.0
    ))
    sys.stdout.flush()

def eval_model(prefix, model, valid):
    model.eval()
    total_loss = 0.0
    unroll_size = model.args.unroll_size
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(1)
    N = (len(valid[0])-1)/unroll_size + 1
    for i in range(N):
        x = valid[0][i*unroll_size:(i+1)*unroll_size].long()
        y = valid[1][i*unroll_size:(i+1)*unroll_size].long().view(-1)
        x, y = Variable(x, volatile=True), Variable(y)
        hidden = tuple(Variable(h.data) for h in hidden)
        output, hidden = model(x, hidden)
        loss = criterion(output, y)
        total_loss += loss.data[0]
    avg_loss = total_loss / valid[1].numel()
    ppl = np.exp(avg_loss)
    bpc = np.log2(ppl)
    sys.stdout.write("\t[{}] {}_loss={:.3f} {}_ppl={:.1f} {}_bpc={:.3f}\n".format(
        prefix,
        prefix, avg_loss,
        prefix, ppl,
        prefix, bpc
    ))
    sys.stdout.flush()
    return bpc

def main(args):
    train, dev, test = read_data(args.path)

    model = Model(train, args)
    if args.cuda:
        model.cuda()
    model.init_weights()
    sys.stdout.write("vocab size: {}\n".format(
        model.embedding_layer.n_V
    ))
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    model.print_pnorm()
    sys.stdout.write("\n")

    map_to_ids = model.embedding_layer.map_to_ids
    train = create_batches(train, map_to_ids, args.batch_size, args.cuda)
    dev = create_batches(dev, map_to_ids, 1, args.cuda)
    test = create_batches(test, map_to_ids, 1, args.cuda)

    unchanged = 0
    best_dev = 1e+8
    for epoch in range(args.max_epoch):
        if args.lr_decay_epoch>0 and epoch>=args.lr_decay_epoch:
            args.lr *= args.lr_decay
        train_model(epoch, model, train)
        model.print_pnorm()
        dev_ppl = eval_model('dev', model, dev)
        if dev_ppl < best_dev:
            unchanged = 0
            best_dev = dev_ppl
            eval_model('test', model, test)
        else:
            unchanged += 1
        if unchanged >= 5: break
        sys.stdout.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--path", type=str, required=True, help="path to text8 file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=128)
    argparser.add_argument("--unroll_size", type=int, default=100)
    argparser.add_argument("--max_epoch", type=int, default=500)
    argparser.add_argument("--d", type=int, default=500)
    argparser.add_argument("--drop_x", type=float, default=0.1)
    argparser.add_argument("--drop_o", type=float, default=0.1)
    argparser.add_argument("--rnn_dropout", type=float, default=0.0)
    argparser.add_argument("--depth", type=int, default=3)
    argparser.add_argument("--lr", type=float, default=0.2)
    argparser.add_argument("--lr_decay", type=float, default=1/1.03)
    argparser.add_argument("--lr_decay_epoch", type=int, default=5)
    argparser.add_argument("--weight_decay", type=float, default=1e-7)
    argparser.add_argument("--clip_norm", type=float, default=10)
    argparser.add_argument("--activation", type=str, default='tanh')

    args = argparser.parse_args()
    print args
    main(args)
