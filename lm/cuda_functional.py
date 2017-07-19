import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.autograd import gradcheck
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


tmp_ = torch.rand(1,1).cuda()

KNN_CODE = """
extern "C" {

    __forceinline__ __device__ float sigmoidf(float in)
    {
        return 1.f / (1.f + expf(-in));
    }

    __global__ void knn_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            int len, int batch, int d,
                            float * __restrict__ h, float * __restrict__ c,
                            int use_tanh)
    {
        int ncols = batch*d;
        int ncols3 = ncols*3;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        float cur = *(init + col);
        const float *up = u + (col*3);
        const float *xp = x + col;
        float *cp = c + col;
        float *hp = h + col;
        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = use_tanh ? tanh(cur) : cur;
            *hp = (val-(*xp))*g2 + (*xp);
            up += ncols3;
            xp += ncols;
            cp += ncols;
            hp += ncols;
        }
    }

    __global__ void knn_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_c,
                            int len, int batch, int d,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int use_tanh)
    {
        int ncols = batch*d;
        int ncols3 = ncols*3;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = 0;

        const float *up = u + (col*3) + (len-1)*ncols3;
        const float *xp = x + col + (len-1)*ncols;
        const float *cp = c + col + (len-1)*ncols;

        const float *ghp = grad_h + col + (len-1)*ncols;
        const float *gcp = grad_c + col + (len-1)*ncols;
        float *gup = grad_u + (col*3) + (len-1)*ncols3;
        float *gxp = grad_x + col + (len-1)*ncols;

        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = use_tanh ? tanh(*cp) : (*cp);
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));

            const float gh_val = *ghp;
            //float gc_val = *gcp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val-x_val)*g2*(1-g2);
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = use_tanh ? (g2*(1-c_val*c_val)) : g2;
            const float gc = gh_val*tmp + cur + (*gcp);

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*g1*(1-g1);
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols3;
            xp -= ncols;
            cp -= ncols;
            gup -= ncols3;
            gxp -= ncols;
            ghp -= ncols;
            gcp -= ncols;
        }

        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }

}
"""

KNN_PROG = Program(KNN_CODE.encode('utf-8'), 'knn_prog.cu'.encode('utf-8'))
KNN_PTX = KNN_PROG.compile()
KNN_MOD = function.Module()
KNN_MOD.load(bytes(KNN_PTX.encode()))
KNN_FWD_FUNC = KNN_MOD.get_function('knn_fwd')
KNN_BWD_FUNC = KNN_MOD.get_function('knn_bwd')

Stream = namedtuple('Stream', ['ptr'])
KNN_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)


class KNN_Compute(Function):

    def __init__(self, use_tanh):
        super(KNN_Compute, self).__init__()
        self.use_tanh = use_tanh

    def forward(self, u, x, bias, init=None):
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = x.size(-1)
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        c = x.new(*x.size())
        h = x.new(*x.size())

        KNN_FWD_FUNC(args=[
            u.data_ptr(),
            x.data_ptr(),
            bias.data_ptr(),
            init_.data_ptr(),
            length,
            batch,
            d,
            h.data_ptr(),
            c.data_ptr(),
            self.use_tanh],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=KNN_STREAM
        )

        self.save_for_backward(u, x, bias, init, c)

        return h, c

    def backward(self, grad_h, grad_c):
        u, x, bias, init, c = self.saved_tensors
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = x.size(-1)
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_x = x.new(*x.size())
        grad_bias = x.new(2, batch, d)
        grad_init = x.new(batch, d)

        KNN_BWD_FUNC(args=[
            u.data_ptr(),
            x.data_ptr(),
            bias.data_ptr(),
            init_.data_ptr(),
            c.data_ptr(),
            grad_h.data_ptr(),
            grad_c.data_ptr(),
            length,
            batch,
            d,
            grad_u.data_ptr(),
            grad_x.data_ptr(),
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.use_tanh],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=KNN_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init



class FastKNNCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.0, rnn_dropout=0.0, out_dropout=0.0, use_tanh=0):
        super(FastKNNCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout or dropout
        self.out_dropout = out_dropout or dropout
        self.use_tanh = use_tanh

        self.weight = nn.Parameter(torch.Tensor(n_in, n_out, 3))
        self.bias = nn.Parameter(torch.Tensor(n_out*2))
        if n_in != n_out:
            self.transform_op = nn.Linear(n_in, n_out)

    def forward(self, input, c0):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.view(-1, n_in)
        u = x_2d.mm(self.weight.view(n_in, -1))
        if n_in == n_out:
            h, c = KNN_Compute(self.use_tanh)(u, input, self.bias, c0)
        else:
            x_transformed = self.transform_op(x_2d)
            if input.dim() == 3:
                x_transformed = x_transformed.view(-1, batch, n_out)
            h, c = KNN_Compute(self.use_tanh)(u, x_transformed, self.bias, c0)

        if self.training and (self.out_dropout>0):
            mask_h = self.get_dropout_mask_((batch, n_out), self.out_dropout)
            h = h * mask_h.expand_as(h)

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class FastKNN(nn.Module):
    def __init__(self, n_in, n_out, depth, dropout=0.0,
                    out_dropout=0.0, rnn_dropout=0.0, use_tanh=0):
        super(FastKNN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.depth = depth
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = []
        self.seq = nn.Sequential()

        for i in range(depth):
            l = FastKNNCell(
                n_in = n_in if i==0 else n_out,
                n_out = n_out,
                dropout = dropout,
                rnn_dropout = rnn_dropout,
                out_dropout = out_dropout,
                use_tanh = use_tanh
            )
            self.rnn_lst.append(l)
            self.seq.add_module(str(i), l)

    def forward(self, input, c0):
        assert input.dim() == 3 # (len, batch, n_in)
        assert c0.dim() == 3    # (depth, batch, n_out)
        c0 = c0.chunk(self.depth, 0)

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c[-1])

        return prevx, torch.stack(lstc)


def test1():
    a = Variable(torch.FloatTensor(20, 80, 1024).zero_().add(0.5).cuda())
    h = Variable(torch.FloatTensor(80, 1024).zero_().add(0.5).cuda())
    cell = FastKNNCell(1024,1024).cuda()
    start = time.time()
    for i in range(1000):
        out = cell(a, h)
        out[0].sum().backward()
    print ("test1: {:.6f}".format(
        (time.time()-start)/1000
    ))

    L = 5
    M = 20
    D = 20
    input_pair = (
        Variable(torch.randn(L,M,D*3).float().cuda(), requires_grad=True),
        Variable(torch.randn(L,M,D).float().cuda(), requires_grad=True),
        Variable(torch.randn(D*2).float().cuda(), requires_grad=True),
        Variable(torch.randn(M,D).float().cuda(), requires_grad=True)
    )
    test_grad = gradcheck(KNN_Compute(1), input_pair, eps=1e-3, atol=1e-3)
    print (test_grad)

#def test2():
#    a = Variable(torch.FloatTensor(100, 32, 1024).zero_().add(0.5).cuda())
#    h = Variable(torch.FloatTensor(32, 1024).zero_().add(0.5).cuda())
#    cell = FastKNNLayer(1024, 1024, activation=lambda x:x, rnn_dropout=0).cuda()
#    start = time.time()
#    for i in range(10000):
#        out = cell(a, (h,h))
#        out[1].sum().backward()
#    print "test2: {:.6f}".format(
#        (time.time()-start)/10000
#    )
#
def test3():
    a = Variable(torch.FloatTensor(20, 80, 1024).zero_().add(0.5).cuda())
    h = Variable(torch.FloatTensor(4, 80, 1024).zero_().add(0.5).cuda())
    cell = nn.LSTM(1024, 1024, 4, dropout=0.0).cuda()
    start = time.time()
    for i in range(1000):
        out = cell(a, (h,h))
        out[0].sum().backward()
        #out[0][0,0,0].backward()
    print ("test3: {:.6f}".format(
        (time.time()-start)/1000
    ))


if __name__=="__main__":
    test1()
#    test2()
    test3()


