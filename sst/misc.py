import random

import numpy as np
import theano
import theano.tensor as T

from nn import RecurrentLayer, Layer, sigmoid, linear
from nn import get_dropout_mask



def read_corpus(path):
    with open(path) as fin:
        lines = fin.readlines()
    lines = [ x.strip().split() for x in lines ]
    lines = [ x for x in lines if x ]
    corpus_x = [ x[1:] for x in lines ]
    corpus_y = [ int(x[0]) for x in lines ]
    return corpus_x, corpus_y


def create_one_batch(ids, x, y):
    batch_x = np.column_stack( [ x[i] for i in ids ] )
    batch_y = np.array( [ y[i] for i in ids ] )
    return batch_x, batch_y


# shuffle training examples and create mini-batches
def create_batches(perm, x, y, batch_size):

    # sort sequences based on their length
    # permutation is necessary if we want different batches every epoch
    lst = sorted(perm, key=lambda i: len(x[i]))

    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by = create_one_batch(ids, x, y)
            batches_x.append(bx)
            batches_y.append(by)
            ids = [ i ]
    bx, by = create_one_batch(ids, x, y)
    batches_x.append(bx)
    batches_y.append(by)

    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    return batches_x, batches_y


class HighwayLayer(object):

    def __init__(self, n_d):
        self.n_d = n_d
        self.gate = Layer(n_d, n_d, sigmoid)

    def forward(self, x, h):
        t = self.gate.forward(x)
        return h*t + x*(1-t)

    @property
    def params(self):
        return self.gate.params

    @params.setter
    def params(self, param_list):
        self.gate.params = param_list



class KernelNN(object):
    '''
    Recurrent network derived from sequence kernel (i.e. string kernel)

    The variant that works better for WSJ language modeling is implemented here
    using the following hyper-parameter configuration:

        1) each layer has n-gram aggregation of n=1

        2) decay factor lambda is controlled using a neural gate:
                lambda[t] = sigmoid_gate(x[t], h[t-1])
                c[t] = lambda[t]*c[t-1] + (1-lambda[t])*(W*x[t])
    '''
    def __init__(self, n_in, n_out, activation, highway=True, dropout=None):
        self.n_in, self.n_out = n_in, n_out
        self.highway = highway
        self.activation = activation
        self.dropout = dropout

        self.lambda_gate = RecurrentLayer(n_in, n_out, sigmoid)
        self.input_layer = Layer(n_in, n_out, linear, has_bias=False)
        if highway:
            self.highway_layer = HighwayLayer(n_out)

    def forward(self, x, c_tm1, h_tm1, mask_h):
        assert x.ndim == 2
        assert c_tm1.ndim == 2
        assert h_tm1.ndim == 2
        n_in, n_out = self.n_in, self.n_out
        activation = self.activation
        lambda_gate, input_layer = self.lambda_gate, self.input_layer

        forget_t = lambda_gate.forward(x, h_tm1*mask_h)
        in_t = 1-forget_t
        wx_t = input_layer.forward(x)
        c_t = c_tm1*forget_t + wx_t*in_t
        h_t = activation(c_t)
        if self.highway:
            h_t = self.highway_layer.forward(x, h_t*mask_h)

        return [c_t, h_t]

    def forward_all(self, x, hc0=None, return_c=False):
        assert x.ndim == 3 # size (len, batch, d)
        if hc0 is None:
            c0 = h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
        else:
            assert len(hc0) == 2
            c0, h0 = hc0

        if self.dropout is None:
            mask_h = T.ones((x.shape[1], self.n_out), dtype=theano.config.floatX)
        else:
            mask_h = get_dropout_mask((x.shape[1], self.n_out), self.dropout)
            mask_x = get_dropout_mask((x.shape[1], x.shape[2]), self.dropout)
            mask_x = mask_x.dimshuffle(('x',0,1))
            x = x*mask_x

        c, h = theano.scan(
                fn = self.forward,
                sequences = x,
                outputs_info = [ c0, h0 ],
                non_sequences = mask_h
            )[0]
        if return_c:
            return [c, h]
        else:
            return h

    @property
    def params(self):
        lst = self.input_layer.params + self.lambda_gate.params
        if self.highway:
            lst += self.highway_layer.params
        return lst

    @params.setter
    def params(self, param_list):
        k1 = len(self.input_layer.params)
        k2 = len(self.lambda_gate.params)
        self.input_layer.params = param_list[:k1]
        self.lambda_gate.params = param_list[k1:k1+k2]
        if self.highway:
            self.highway_layer.params = param_list[k1+k2:]


class MKernelNN(object):
    '''
    Recurrent network derived from sequence kernel (i.e. string kernel)

    '''
    def __init__(self, n_in, n_out, activation, highway=True, dropout=None, combine_c=True):
        self.n_in, self.n_out = n_in, n_out
        self.highway = highway
        self.activation = activation
        self.dropout = dropout
        self.combine_c = combine_c

        self.lambda_gate = RecurrentLayer(n_in, n_out, sigmoid)
        self.input_layer_1 = Layer(n_in, n_out, linear, has_bias=False)
        self.input_layer_2 = Layer(n_in, n_out, linear, has_bias=False)
        if highway:
            self.highway_layer = HighwayLayer(n_out)

    def forward(self, x, ci_tm1, cii_tm1, h_tm1, mask_h):
        assert x.ndim == 2
        assert ci_tm1.ndim == 2
        assert cii_tm1.ndim == 2
        assert h_tm1.ndim == 2
        n_in, n_out = self.n_in, self.n_out
        activation = self.activation
        lambda_gate = self.lambda_gate
        input_layer_1, input_layer_2 = self.input_layer_1, self.input_layer_2

        forget_t = lambda_gate.forward(x, h_tm1*mask_h)
        in_t = 1-forget_t
        xi_t  = input_layer_1.forward(x)
        xii_t = input_layer_2.forward(x)
        ci_t  = ci_tm1*forget_t + xi_t*in_t
        cii_t = cii_tm1*forget_t + (ci_tm1*xii_t)*in_t
        h_t = activation(ci_t+cii_t) if self.combine_c else activation(cii_t)

        if self.highway:
            h_t = self.highway_layer.forward(x, h_t*mask_h)

        return [ci_t, cii_t, h_t]

    def forward_all(self, x, hc0=None, return_c=False):
        assert x.ndim == 3 # size (len, batch, d)
        if hc0 is None:
            ci0 = cii0 = h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
        else:
            assert len(hc0) == 3
            ci0, cii0, h0 = hc0

        if self.dropout is None:
            mask_h = T.ones((x.shape[1], self.n_out), dtype=theano.config.floatX)
        else:
            mask_h = get_dropout_mask((x.shape[1], self.n_out), self.dropout)
            mask_x = get_dropout_mask((x.shape[1], x.shape[2]), self.dropout)
            mask_x = mask_x.dimshuffle(('x',0,1))
            x = x*mask_x

        ci, cii, h = theano.scan(
                fn = self.forward,
                sequences = x,
                outputs_info = [ ci0, cii0, h0 ],
                non_sequences = mask_h
            )[0]
        if return_c:
            return [ci, cii, h]
        else:
            return h

    @property
    def params(self):
        lst = self.input_layer_1.params + self.input_layer_2.params + self.lambda_gate.params
        if self.highway:
            lst += self.highway_layer.params
        return lst

    @params.setter
    def params(self, param_list):
        params = self.params
        assert len(params) == len(param_list)
        for p, q in zip(params, param_list):
            p.set_value(q.get_value())





