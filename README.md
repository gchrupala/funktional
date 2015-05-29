# funktional
A minimalistic toolkit for composing neural networks with Theano. 

Mathematically, neural network layers are functions mapping tensors to tensors. 

In funktional, layers are implemented as callable Python objects which means that they can be 
easily mixed and matched with regular Python functions operating on 
Theano tensors. Function-like layers are also easy to compose into more complex networks using function 
applicationa and function composition idioms familiar from general purpose programming.
