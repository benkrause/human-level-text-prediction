#!/usr/bin/env python
"""Sample script of recurrent neural network language model.
This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm
"""


import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import training
import sys
import cupy as cp
import os
from mLSTMWN_ch3 import mLSTM
from WN import WN
from chainer import serializers

def get_char(fname):
    fid = open(fname,'rb')
    byte_array = fid.read()
    text = [0]*len(byte_array)
    for i in range(0,len(byte_array)):
        text[i] = int(byte_array[i])
    unique = list(set(text))
    unique.sort()

    mapping = dict(zip(unique,list(range(0,len(unique)))))
    for i in range(0,len(text)):
        text[i] = mapping[text[i]]
    return text, mapping
def get_char2(fname,mapping):
    fid = open(fname,'rb')
    byte_array = fid.read()
    text = [0]*len(byte_array)
    for i in range(0,len(byte_array)):
        text[i] = int(byte_array[i])

    for i in range(0,len(text)):
        text[i] = mapping[text[i]]
    return text

def ortho_init(shape):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120


    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    a=a.astype(dtype=np.float32)



    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape

    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return q
    #print(np.max(q))

# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, nvocab, nunits, train=True):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed=L.EmbedID(nvocab, 400)
            self.WhxWN = WN(400,nunits*4)
            self.WmxWN = WN(400,nunits)
            self.WmhWN = WN(nunits,nunits)
            self.WhmWN = WN(nunits,nunits*4)

            self.l1=mLSTM(out_size=nunits)
            self.l2=L.Linear(nunits, nvocab)


        for param in self.params():
            print(param.data.shape)
            #param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        if False:
            self.l1.b.data[2::4] = 3
            Wembd = np.random.uniform(-1, 1, self.embed.W.data.shape)
            Wembd =Wembd.astype(dtype=np.float32)

            self.embed.W.data = Wembd

            self.WhxWN.W.data = ortho_init(self.WhxWN.W.data.shape)
            norm = np.linalg.norm(self.WhxWN.W.data, axis=1)
            self.WhxWN.g.data = norm

            self.WmxWN.W.data = ortho_init(self.WmxWN.W.data.shape)
            norm = np.linalg.norm(self.WmxWN.W.data, axis=1)
            self.WmxWN.g.data = norm

            self.WmhWN.W.data = ortho_init(self.WmhWN.W.data.shape)
            norm = np.linalg.norm(self.WmhWN.W.data, axis=1)
            self.WmhWN.g.data = norm

            self.WhmWN.W.data = ortho_init(self.WhmWN.W.data.shape)
            norm = np.linalg.norm(self.WhmWN.W.data, axis=1)
            self.WhmWN.g.data = norm

            self.l2.W.data= ortho_init(self.l2.W.data.shape)

        self.train = train

    def reset_state(self):
        self.l1.reset_state()

    def applyWN(self):
        self.Whx = self.WhxWN()
        self.Wmx = self.WmxWN()
        self.Wmh = self.WmhWN()
        self.Whm = self.WhmWN()



    def __call__(self, x):


        h0 = self.embed(x)

        h1 = self.l1(h0,self.Whx,self.Wmx,self.Wmh,self.Whm)


        y = self.l2(h1)

        return y


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.

def test(model,inputs,targets):
    inputs = Variable(inputs)
    targets = Variable(targets)

    targets.to_gpu()
    inputs.to_gpu()
    model.applyWN()
    loss=0
    for j in range(inputs.shape[1]):
        output = model(inputs[:,j])
        loss = loss+ F.softmax_cross_entropy(output,targets[:,j])
        loss.unchain_backward()

    finalloss = loss.data/inputs.shape[1]
    return finalloss







def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=20,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')

    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--seq', default="text8",
                        help='path to text file for training')

    parser.add_argument('--unit', '-u', type=int, default=2800,
                        help='Number of LSTM units')
    args = parser.parse_args()


    #number of training iterations per model save, log write, and validation set evaluation
    interval = 100

    #size of vocab, all tokens less frequent than the maxl th most frequent token are mapped to a single token
    maxl=20000

    #first ntrain words of dataset will be used for training
    ntrain = 90000000

    #the nval words immediately after the first ntrain words will be used for validation
    nval = 5000000

    #ntest not currently used
    ntest = 5000000


    seqlen = args.bproplen
    tseqlen = seqlen
    nbatch = args.batchsize



    filename= args.seq

    text,mapping = get_char(filename)
    print(len(mapping))
    sequence = np.array(text).astype(np.int32)

    text2 = get_char2('jefferson.txt',mapping)

    sequence2 = np.array(text2).astype(np.int32)



    tstart = 0
    ntrain = 1000000

    itrain = sequence[tstart:tstart+ntrain]
    ttrain = sequence[tstart+1:tstart+1+ntrain]

    itrain = itrain.reshape(nbatch,int(ntrain/nbatch))
    ttrain = ttrain.reshape(nbatch,int(ntrain/nbatch))


    start1 = 90*(10**6)-1
    nval= 5000000




    ival = sequence2[:-1]
    tval = sequence2[1:]
    ival= np.expand_dims(ival,axis=0)
    tval= np.expand_dims(tval,axis=0)

   # itrain = itrain[0:4,:]
   # ttrain = ttrain[0:4,:]

    #test = sequence[ntrain+nval:ntrain+nval+ntest]




    nvocab = 27# train is just an array of integers
    print('#vocab =', nvocab)

    modelname = 'model'
    # Prepare an RNNLM model
    rnn = RNNForLM(nvocab, args.unit)
    serializers.load_npz(modelname, rnn)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
#    optimizer = RMSprop(lr = .0002, gamma = .01)
#    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    # Set up a trainer



    print('starting')







    start = 0
    loss_sum = 0
    i=0
    it_num=0
    rnn.train = False

    rnn.applyWN()





    print(ival.shape)
    for param in rnn.params():
        param.gsum =0*param.data
        param.mom =0*param.data
    while True:
        # Get the result of the forward pass.
        fin = start+tseqlen

        if fin>(itrain.shape[1]):
            break

        inputs = itrain[:,start:fin]
        targets = ttrain[:,start:fin]
        start = fin


        inputs = Variable(inputs)
        targets = Variable(targets)

        targets.to_gpu()
        inputs.to_gpu()
        it_num+=1
        loss = 0

        for j in range(tseqlen):
                    # Get the next batch (a list of tuples of two word IDs)

            output = rnn(inputs[:,j])
            loss = loss+ F.softmax_cross_entropy(output,targets[:,j])


        loss = loss/(tseqlen)

        # Zero all gradients before updating them.
        rnn.zerograds()

        loss.backward()
        for param in rnn.params():
            param.gsum += param.grad*param.grad
        loss.unchain_backward()
    print('done stat')

    gsumsum = 0
    len1 = 0
    lr = .0003

    alpha = .01
    beta = .001
    for param in rnn.params():
        param.gsum = cp.sqrt(param.gsum)
        gsumsum = gsumsum+cp.mean(param.gsum)

        len1+=1



        param.d0 = 1*param.data

    gsumavg = gsumsum/len1

    for param in rnn.params():
        param.decrate = param.gsum/gsumavg
        f = cp.nonzero(param.decrate>(.5/alpha))
        param.decrate[f] = (.5/alpha)

    start = 0
    loss_sum = 0
    i=0
    it_num=0
    rnn.train = False


    rnn.reset_state()



    losses = list()
    done = False
    while True:

        fin = start+seqlen

        if fin>(ival.shape[1]):
            if not done:
                done = True
                seqlen = (ival.shape[1])-start
                fin = start+seqlen
            else:

                break

        inputs = ival[:,start:fin]
        targets = tval[:,start:fin]
        start = fin



        inputs = Variable(inputs)
        targets = Variable(targets)

        targets.to_gpu()
        inputs.to_gpu()
        it_num+=1
        loss = 0
        rnn.applyWN()
        for j in range(seqlen):
                    # Get the next batch (a list of tuples of two word IDs)

            output = rnn(inputs[:,j])
            loss1 = F.softmax_cross_entropy(output,targets[:,j])
            losses.append(1.4427*cp.asnumpy(loss1.data))
            loss = loss+loss1



        loss = loss/(seqlen)

        # Zero all gradients before updating them.
        rnn.zerograds()
        loss_sum += loss.data
        if (i+1)%100 == 0:
            print(i+1)
            print(1.4427*loss_sum/(i+1))

        # Calculate and update all gradients.
        loss.backward()
        s = 0;


        # Use the optmizer to move all parameters of the network
        # to values which will reduce the loss.

        for param in rnn.params():


            dW = -lr*param.grad/(param.gsum+beta)

            dW +=  alpha*(param.decrate)*(param.d0-param.data)
            param.data +=dW


        loss.unchain_backward()

        i+=1
    loss_sum = (1.4427*loss_sum/i)
    print('total loss:')
    print(loss_sum)
    np.save('losses.npy',losses)
    print('cross-entropy on final 75 characters (bits/char):')
    print(np.array(losses[-75:]).mean())





main()
