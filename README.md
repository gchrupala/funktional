# funktional 

A minimalistic toolkit for functionally composable neural network
layers with Theano.

Rationale
---------

Conceptually, neural network layers are functions on tensors.  In
funktional, layers are implemented as callable Python objects which
means that they can be easily mixed and matched with regular Python
functions operating on Theano tensors. Function-like layers are also
easy to compose into more complex networks using function application
and function composition idioms familiar from general purpose
programming.

For example, suppose we need to implement a recurrent encoder-decoder
network. As the encoder we use a GRU layer which takes one argument
(the input sequence) and returns a sequence of hidden states. The
decoder is another GRU which takes two arguments (an initial state,
and a sequence of outputs), and returns a sequence of hidden
states. From these hidden states we then predict the next output
element:

```
                y1  y2  y3  y.
                ^   ^   ^   ^
                |   |   |   |            
h0->h1->h2->h3->g1->g2->g3->g4
    ^   ^   ^   ^   ^   ^   ^
    |   |   |   |   |   |   |
    x1  x2  x.  y0  y1  y2  y3
```

With funktional, you would create the layers and then compose them
using function application like so:

```python

def last(x):
    """Returns the last time step of all sequences in x."""
    return x.dimshuffle((1,0,2))[-1]

class EncoderDecoder(Layer):
    def __init__(self, size_in, size, size_out):
        self.Encode = GRUH0(size_in=size_in, size=size)
        self.Decode = GRU(size_in=size_out, size=size)
        self.Output = Dense(size_in=size, size=size_out)
        self.params = self.Encode.params + self.Decode.params + self.Output.params

    def __call__(self, inp, out_prev):
        return self.Output(self.Decode(last(self.Encode(inp)), out_prev))
```

Note that in the definition of ` __call__` we specify the network
connectivity using function application syntax, and mix `Layer`
objects together with regular Python functions such as `last`. The
`EncoderDecoder` we defined can in turn be composed with other layers
or functions:

```python
Encdec = EncoderDecoder(size_in, size, size_out)
output = softmax3d(Encdec(input, output_prev))
```

See `layer.py` for more examples of layer compositions.

The file `example.py` uses funktional to implement a recurrent
auto-encoder, which learns to map sentences to a hidden state vector
and to decode the vector into the original sentence.  A GPU is
recommended to run this example.

Limitations
-----------

Funktional is not meant to be a full-fledged, self-contained toolkit,
but only to provide a convenient interface. Functionality such as
preprocessing, activation functions or implementations of various
variants of stochastic gradient descent are available via some of the
existing toolkits such as
[Passage](https://github.com/IndicoDataSolutions/Passage) or
[Keras](https://github.com/fchollet/keras). These toolkits can be a
awkward if we need to implement a complicated custom architecture.
Funktional is intended to make this use case easy.

