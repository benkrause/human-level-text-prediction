import numpy
import six

import chainer
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.activation import lstm
from chainer import initializers
from chainer import link
from chainer import Function
import chainer.functions as F
from chainer.links.connection import linear
from chainer import functions as f
from chainer import variable


class LSTMBase(link.Chain):

    def __init__(self, out_size, bias_init=0, forget_bias_init=1):
        super(LSTMBase, self).__init__()

        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init
        self.out_size = out_size
        self.state_size = out_size
    #    if out_size is not None:
        self._initialize_params()

    def _initialize_params(self):

        bias_initializer = initializers.Zero()
        self.add_param('b', self.state_size*4, initializer=bias_initializer)
    #    a, i, f, o = lstm._extract_gates(self.b.data.reshape(1, 4 * self.state_size, 1))
    #    initializers.init_weight(a, self.bias_init)
    #    initializers.init_weight(i, self.bias_init)
    #    initializers.init_weight(f, self.forget_bias_init)
    #    initializers.init_weight(o, self.bias_init)


class StatelessLSTM(LSTMBase):

    """Stateless LSTM layer.
    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.
    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.
    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
    """

    def __call__(self, c, h, x):
        """Returns new cell state and updated output of LSTM.
        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.
        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.
        """
        if self.upward.has_uninitialized_params:
            in_size = x.size // x.shape[0]
            self.upward._initialize_params(in_size)
            self._initialize_params()
        if self.upward2.has_uninitialized_params:
            in_size = x.size // x.shape[0]
            self.upward2._initialize_params(in_size)
            self._initialize_params()




        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((x.shape[0], self.state_size), dtype=x.dtype),
                volatile='auto')
        return lstm.lstm(c, lstm_in)



class mLSTM(LSTMBase):

    """Fully-connected LSTM layer.
    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.
    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.
    This link supports variable length inputs. The mini-batch size of the
    current input must be equal to or smaller than that of the previous one.
    The mini-batch size of ``c`` and ``h`` is determined as that of the first
    input ``x``.
    When mini-batch size of ``i``-th input is smaller than that of the previous
    input, this link only updates ``c[0:len(x)]`` and ``h[0:len(x)]`` and
    doesn't change the rest of ``c`` and ``h``.
    So, please sort input sequences in descending order of lengths before
    applying the function.
    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            Maybe be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            Maybe be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.
    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.
    """

    def __init__(self,out_size, **kwargs):
        super(mLSTM, self).__init__(out_size, **kwargs)
        self.reset_state()

    def to_cpu(self):
        super(mLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(mLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, c, h):
        """Sets the internal state.
        It sets the :attr:`c` and :attr:`h` attributes.
        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.
        """
        assert isinstance(c, chainer.Variable)
        assert isinstance(h, chainer.Variable)
        c_ = c
        h_ = h
        if self.xp == numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu()
            h_.to_gpu()
        self.c = c_
        self.h = h_

    def reset_state(self):
        """Resets the internal state.
        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.
        """
        self.c = self.h = None

    def __call__(self, x,Whx,Wmx,Wmh,Whm):
        """Updates the internal state and returns the LSTM outputs.
        Args:
            x (~chainer.Variable): A new batch from the input sequence.
        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.
        """
    #    if self.upward.has_uninitialized_params:
    #        in_size = x.size // x.shape[0]
    #        self.upward._initialize_params(in_size)
    #        self._initialize_params()
    #    if self.upward2.has_uninitialized_params:
    #        in_size = x.size // x.shape[0]
    #        self.upward2._initialize_params(in_size)
    #        self._initialize_params()

        batch = x.shape[0]
    #    Whx = self.upward()

    #    Wmx = self.upward2()

        factor_in = F.linear(x,Wmx)
        lstm_in = F.linear(x,Whx,self.b)

        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than the '
                       'size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
    #            Wmh = self.lateral1()

                mult_in = F.linear(h_update,Wmh)

                mult_out = mult_in*factor_in
        #        Whm = self.lateral2()
                lstm_in += F.linear(mult_out,Whm)

            else:
    #            Wmh = self.lateral1()

                mult_in = F.linear(self.h,Wmh)

                mult_out = mult_in*factor_in
        #        Whm = self.lateral2()
                lstm_in += F.linear(mult_out,Whm)

        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = lstm.lstm(self.c, lstm_in)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
