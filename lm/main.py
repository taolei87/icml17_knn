
import sys
import os
import argparse
import time
import random
import math

import numpy as np
import theano
import theano.tensor as T

import nn
from nn import Dropout, EmbeddingLayer, Layer, apply_dropout
from nn import get_activation_by_name, create_optimization_updates
from nn.evaluation import evaluate_average

from utils import say
from misc import read_corpus, create_batches, HighwayLayer, KernelNN
from options import load_arguments

class Model(object):
    def __init__(self):
        return

    def ready(self, args, train):
        # len * batch
        depth = args["depth"]
        self.args = args
        self.idxs = T.imatrix()
        self.idys = T.imatrix()
        self.init_state = [ T.matrix(dtype=theano.config.floatX) for i in xrange(depth*2) ]

        dropout_prob = np.float64(args["dropout"]).astype(theano.config.floatX)
        self.dropout = theano.shared(dropout_prob)
        rnn_dropout_prob = np.float64(args["rnn_dropout"]).astype(theano.config.floatX)
        self.rnn_dropout = theano.shared(rnn_dropout_prob)

        self.n_d = args["hidden_dim"]

        embedding_layer = EmbeddingLayer(
                n_d = self.n_d,
                vocab = set(w for w in train)
            )
        self.n_V = embedding_layer.n_V

        say("Vocab size: {}\tHidden dim: {}\n".format(
                self.n_V, self.n_d
            ))

        activation = get_activation_by_name(args["activation"])

        layers = self.layers = [ ]
        for i in xrange(depth):
            rnn_layer = KernelNN(
                    n_in = self.n_d,
                    n_out = self.n_d,
                    activation = activation,
                    highway = args["highway"],
                    dropout = self.rnn_dropout
                )
            layers.append(rnn_layer)


        output_layer = Layer(
                n_in = self.n_d,
                n_out = self.n_V,
                activation = T.nnet.softmax,
            )
        output_layer.W = embedding_layer.embeddings.T

        # (len*batch) * n_d
        x_flat = embedding_layer.forward(self.idxs.ravel())

        # len * batch * n_d
        x = apply_dropout(x_flat, self.dropout)
        #x = x_flat
        x = x.reshape( (self.idxs.shape[0], self.idxs.shape[1], self.n_d) )

        # len * batch * (n_d+n_d)
        self.last_state = []
        prev_h = x
        for i in xrange(depth):
            hidden = self.init_state[i*2:i*2+2]
            c, h = layers[i].forward_all(prev_h, hidden, return_c=True)
            self.last_state += [ c[-1], h[-1] ]
            prev_h = h


        prev_h = apply_dropout(prev_h, self.dropout)
        self.p_y_given_x = output_layer.forward(prev_h.reshape(x_flat.shape))

        idys = self.idys.ravel()
        self.nll = T.nnet.categorical_crossentropy(
                        self.p_y_given_x,
                        idys
                    )

        self.params = [ x for l in layers for x in l.params ]
        self.params += [ embedding_layer.embeddings, output_layer.b ]
        self.num_params = sum(len(x.get_value(borrow=True).ravel())
                                for x in self.params)
        say("# of params in total: {}\n".format(self.num_params))
        layers += [ embedding_layer, output_layer ]

    def train(self, args, train, dev, test=None):
        embedding_layer = self.layers[-2]

        dropout_prob = np.float64(args["dropout"]).astype(theano.config.floatX)
        rnn_dropout_prob = np.float64(args["rnn_dropout"]).astype(theano.config.floatX)
        batch_size = args["batch_size"]
        unroll_size = args["unroll_size"]

        train = create_batches(train, embedding_layer.map_to_ids, batch_size)

        dev = create_batches(dev, embedding_layer.map_to_ids, 1)

        if test is not None:
            test = create_batches(test, embedding_layer.map_to_ids, 1)

        cost = T.sum(self.nll) / self.idxs.shape[1]
        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args["learning_rate"],
                eps = args["eps"],
                method = args["learning"]
            )[:3]

        train_func = theano.function(
                inputs = [ self.idxs, self.idys ] + self.init_state,
                outputs = [cost, gnorm] + self.last_state,
                updates = updates
            )
        eval_func = theano.function(
                inputs = [ self.idxs, self.idys ] + self.init_state,
                outputs = [self.nll] + self.last_state,
            )

        N = (len(train[0])-1)/unroll_size + 1
        say(" train: {} tokens, {} mini-batches\n".format(
                len(train[0].ravel()), N
            ))
        say(" dev: {} tokens\n".format(len(dev[0].ravel())))

        say("\tp_norm: {}\n".format(
                self.get_pnorm_stat()
            ))

        decay_epoch = args["lr_decay_epoch"]
        decay_rate = args["lr_decay"]
        lr_0 = args["learning_rate"]
        iter_cnt = 0

        depth = args["depth"]
        unchanged = 0
        best_dev = 1e+10
        start_time = 0
        max_epoch = args["max_epoch"]
        for epoch in xrange(max_epoch):
            unchanged += 1
            if unchanged > 20: break

            if decay_epoch > 0 and epoch >= decay_epoch:
                lr.set_value(np.float32(lr.get_value()*decay_rate))

            start_time = time.time()

            prev_state = [ np.zeros((batch_size, self.n_d),
                    dtype=theano.config.floatX) for i in xrange(depth*2) ]

            train_loss = 0.0
            for i in xrange(N):
                # get current batch
                x = train[0][i*unroll_size:(i+1)*unroll_size]
                y = train[1][i*unroll_size:(i+1)*unroll_size]

                iter_cnt += 1
                ret = train_func(x, y, *prev_state)
                cur_loss, grad_norm, prev_state = ret[0], ret[1], ret[2:]
                train_loss += cur_loss/len(x)

                if i % 10 == 0:
                    say("\r{}".format(i))

                if i == N-1:
                    self.dropout.set_value(0.0)
                    self.rnn_dropout.set_value(0.0)
                    dev_preds = self.evaluate(eval_func, dev, 1, unroll_size)
                    dev_loss = evaluate_average(
                            predictions = dev_preds,
                            masks = None
                        )
                    dev_ppl = np.exp(dev_loss)
                    self.dropout.set_value(dropout_prob)
                    self.rnn_dropout.set_value(rnn_dropout_prob)

                    say("\r\n")
                    say( ( "Epoch={}  lr={:.4f}  train_loss={:.3f}  train_ppl={:.1f}  " \
                        +"dev_loss={:.3f}  dev_ppl={:.1f}\t|g|={:.3f}\t[{:.1f}m]\n" ).format(
                            epoch,
                            float(lr.get_value(borrow=True)),
                            train_loss/N,
                            np.exp(train_loss/N),
                            dev_loss,
                            dev_ppl,
                            float(grad_norm),
                            (time.time()-start_time)/60.0
                        ))
                    say("\tp_norm: {}\n".format(
                            self.get_pnorm_stat()
                        ))

                    if dev_ppl < best_dev:
                        best_dev = dev_ppl
                        if test is None: continue
                        self.dropout.set_value(0.0)
                        self.rnn_dropout.set_value(0.0)
                        test_preds = self.evaluate(eval_func, test, 1, unroll_size)
                        test_loss = evaluate_average(
                                predictions = test_preds,
                                masks = None
                            )
                        test_ppl = np.exp(test_loss)
                        self.dropout.set_value(dropout_prob)
                        self.rnn_dropout.set_value(rnn_dropout_prob)
                        say("\tbest_dev={:.1f}  test_loss={:.3f}  test_ppl={:.1f}\n".format(
                                best_dev, test_loss, test_ppl))
                        if best_dev < 200: unchanged=0

        say("\n")

    def evaluate(self, eval_func, dev, batch_size, unroll_size):
        depth = self.args["depth"]
        predictions = [ ]
        init_state = [ np.zeros((batch_size, self.n_d),
                dtype=theano.config.floatX) for i in xrange(depth*2) ]
        N = (len(dev[0])-1)/unroll_size + 1
        for i in xrange(N):
            x = dev[0][i*unroll_size:(i+1)*unroll_size]
            y = dev[1][i*unroll_size:(i+1)*unroll_size]
            ret = eval_func(x, y, *init_state)
            pred, init_state = ret[0], ret[1:]
            predictions.append(pred)
        return predictions

    def get_pnorm_stat(self):
        lst_norms = [ ]
        for p in self.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.1f}".format(l2))
        return lst_norms

def main(args):
    if args["train"]:
        assert args["dev"]
        train = read_corpus(args["train"])
        dev = read_corpus(args["dev"])
        test = read_corpus(args["test"]) if args["test"] else None
        model = Model()
        model.ready(args, train)
        model.train(args, train, dev, test)

if __name__ == "__main__":
    args = load_arguments()
    print args
    main(args)
