import time

import numpy as np
import torch
from torch.autograd import Function, Variable
from torch.autograd import gradcheck

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()


class WeightedCumsum(Function):

    MOD_TEMPLATE_ = """
    __global__ void wcumsumd(float *dest, float *weight, float *source, float *init,
                                    int nrows, int ncols, int reverse)
    {{
        if (reverse) {{
            for (unsigned col = blockIdx.x*{THREAD_PER_BLK}+threadIdx.x; col < ncols; col += {NUM_BLK}*{THREAD_PER_BLK}) {{
                float *d = dest + ((nrows-1)*ncols + col);
                float *w = weight + ((nrows-1)*ncols + col);
                float *s = source + ((nrows-1)*ncols + col);
                float cur = *s;
                *d = cur;
                d -= ncols;
                s -= ncols;
                for (int row = nrows-2; row >= 0; --row) {{
                    cur = cur*(*w) + (*s);
                    *d =  cur;
                    d -= ncols;
                    w -= ncols;
                    s -= ncols;
                }}
            }}
        }} else {{
            for (unsigned col = blockIdx.x*{THREAD_PER_BLK}+threadIdx.x; col < ncols; col += {NUM_BLK}*{THREAD_PER_BLK}) {{
                float *d = dest + col;
                float *w = weight + col;
                float *s = source + col;
                float cur = *(init + col);
                for (int row = 0; row < nrows; ++row) {{
                    cur = cur*(*w) + (*s);
                    *d =  cur;
                    d += ncols;
                    w += ncols;
                    s += ncols;
                }}
            }}
        }}
    }}
    """

    MOD_DICT_ = {}

    def get_kernel_function_(self, thread_per_block, num_block):
        fn = self.MOD_DICT_.get((thread_per_block, num_block), None)
        if fn is None:
            code = self.MOD_TEMPLATE_.format(
                THREAD_PER_BLK = thread_per_block,
                NUM_BLK = num_block
            )
            module = SourceModule(code)
            fn = module.get_function("wcumsumd")
            self.MOD_DICT_[(thread_per_block, num_block)] = fn
        return fn


    def forward(self, input, weight, init=None):
        nrows = input.size(0)
        ncols = input.numel() / nrows
        thread_per_block = min(512, ncols)
        num_block = min(4096, (ncols-1)/thread_per_block+1)

        init_ = input.new(ncols).zero_() if init is None else init
        assert init_.numel() == ncols

        wcumsumd = self.get_kernel_function_(
            thread_per_block,
            num_block
        )
        output = input.new(*input.size())
        wcumsumd(
            Holder(output),
            Holder(weight),
            Holder(input),
            Holder(init_),
            np.int32(nrows),
            np.int32(ncols),
            np.int32(0),
            block = (thread_per_block,1,1), grid = (num_block,1)
        )

        self.save_for_backward(input, weight, output, init)

        return output

    def backward(self, grad_output):
        input, weight, output, init = self.saved_tensors
        grad_input = grad_weight = grad_init = None

        nrows = input.size(0)
        ncols = input.numel() / nrows
        thread_per_block = min(512, ncols)
        num_block = min(1024, (ncols-1)/thread_per_block+1)

        wcumsumd = self.get_kernel_function_(
            thread_per_block,
            num_block
        )
        grad_input = input.new(*input.size())
        wcumsumd(
            Holder(grad_input),
            Holder(weight),
            Holder(grad_output),
            Holder(input), # not used
            np.int32(nrows),
            np.int32(ncols),
            np.int32(1),
            block = (thread_per_block,1,1), grid = (num_block,1)
        )
        grad_weight = grad_input.clone()
        if input.size(0)>1:
            grad_weight[1:].mul_(output[:-1])
        if init is None:
            grad_weight[0:1].zero_()
        else:
            grad_weight[0:1].mul_(init)

        return grad_input, grad_weight, grad_init


def weighted_cumsum(input, weight, init=None):
    return WeightedCumsum()(input, weight, init)



def test_forward():
    func = WeightedCumsum()
    init = Variable(torch.FloatTensor(32, 1000).zero_().cuda())
    a = Variable(torch.FloatTensor(35, 32, 1000).zero_().add(0.5).cuda())
    b = Variable(torch.FloatTensor(35, 32, 1000).zero_().add(1.0).cuda())
    start_time = time.time()
    for i in range(10000):
        func(b,a,init)
    print (time.time()-start_time)/10000
    print (func(a,b) - a.cumsum(0)).min()

def test_backward():
    init = Variable(torch.FloatTensor(32, 1000).zero_().cuda())
    a = Variable(torch.FloatTensor(35, 32, 1000).zero_().add(0.5).cuda())
    b = Variable(torch.FloatTensor(35, 32, 1000).zero_().add(1.0).cuda())
    a.requires_grad = True
    b.requires_grad = True
    start_time = time.time()
    for i in range(10000):
        out = WeightedCumsum()(b,a,init)
        out.sum().backward()
    print (time.time()-start_time)/10000

    L = 1
    M = 20
    input_pair = (
        Variable(torch.randn(L,M).float().cuda(), requires_grad=True),
        Variable(torch.FloatTensor(L,M).uniform_(0,1).cuda(), requires_grad=True),
        Variable(torch.randn(M).float().cuda(), requires_grad=False)
    )
    test1 = gradcheck(WeightedCumsum(), input_pair, eps=1e-3, atol=1e-3)

    print test1

if __name__=="__main__":
    test_forward()
    test_backward()
    torch.cuda.synchronize()

