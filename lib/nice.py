# From: https://github.com/paultsw/nice_pytorch/tree/master

"""
Implementation of models from paper.
"""
import torch
import torch.nn as nn
import torch.nn.init as init

from einops import rearrange


def _build_relu_network(latent_dim, hidden_dim, num_layers):
    """Helper function to construct a ReLU network of varying number of layers."""
    _modules = [ nn.Linear(latent_dim, hidden_dim) ]
    for i in range(num_layers):
        if i > 0:
            _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
        _modules.append( nn.BatchNorm1d(hidden_dim) )
    _modules.append( nn.Linear(hidden_dim, latent_dim) )
    return nn.Sequential( *_modules )
    

class NICEModel(nn.Module):
    """
    Replication of model from the paper:
      "Nonlinear Independent Components Estimation",
      Laurent Dinh, David Krueger, Yoshua Bengio (2014)
      https://arxiv.org/abs/1410.8516

    Contains the following components:
    * four additive coupling layers with nonlinearity functions consisting of
      five-layer RELUs
    * a diagonal scaling matrix output layer
    """
    def __init__(self, input_dim, num_layers, nonlin_hidden_dim, nonlin_num_layers):
        super(NICEModel, self).__init__()
        assert (input_dim % 2 == 0), "[NICEModel] only even input dimensions supported for now"
        # assert (num_layers > 2), "[NICEModel] num_layers must be at least 3"
        self.input_dim = input_dim
        half_dim = int(input_dim / 2)

        nonlin_build = lambda: _build_relu_network(half_dim, nonlin_hidden_dim, nonlin_num_layers)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            parity = 'even' if (i+1)%2==0 else 'odd' # Match original
            self.layers.append(AdditiveCouplingLayer(input_dim, parity, nonlin_build()))
        self.scaling_diag = nn.Parameter(torch.ones(input_dim))

        # randomly initialize weights:
        for layer in self.layers:
            for p in layer.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)


    def forward(self, xs):
        """
        Forward pass through all invertible coupling layers.
        
        Args:
        * xs: float tensor of shape (B,dim).

        Returns:
        * ys: float tensor of shape (B,dim).
        """
        for layer in self.layers:
            xs = layer(xs)
        xs = torch.matmul(xs, torch.diag(torch.exp(self.scaling_diag)))
        return xs


    def inverse(self, ys):
        """Invert a set of draws from gaussians"""
        xs = torch.matmul(ys, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
        for layer in reversed(self.layers):
            xs = layer.inverse(xs)
        return xs


"""
Implementation of NICE bijective triangular-jacobian layers.
"""

# ===== ===== Coupling Layer Implementations ===== =====

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _interleave(first, second, order):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.
    
    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, repsectively.
    """
    # cols = []
    # if order == 'even':
    #     for k in range(second.shape[1]):
    #         cols.append(first[:,k])
    #         cols.append(second[:,k])
    #     if first.shape[1] > second.shape[1]:
    #         print('uneven shape size')
    #         cols.append(first[:,-1])
    # else:
    #     for k in range(first.shape[1]):
    #         cols.append(second[:,k])
    #         cols.append(first[:,k])
    #     if second.shape[1] > first.shape[1]:
    #         print('uneven shape size')
    #         cols.append(second[:,-1])
    # return torch.stack(cols, dim=1)
    
    # Much more efficient!
    if order == 'even':
        return rearrange([first, second], 't b d -> b (d t)')
    else:
        return rearrange([second, first], 't b d -> b (d t)')
        


class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, nonlinearity):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))
        
        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(_BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in ['even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(x), self.nonlinearity(self._first(x))),
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(y), self.nonlinearity(self._first(y))),
            self.partition
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""
    def coupling_law(self, a, b):
        return (a + b)
    def anticoupling_law(self, a, b):
        return (a - b)


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""
    def coupling_law(self, a, b):
        return torch.mul(a,b)
    def anticoupling_law(self, a, b):
        return torch.mul(a, torch.reciprocal(b))


class AffineCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b1 + b2, where (b1,b2) is a partition of b."""
    def coupling_law(self, a, b):
        return torch.mul(a, self._first(b)) + self._second(b)
    def anticoupling_law(self, a, b):
        # TODO
        raise NotImplementedError("TODO: AffineCouplingLayer (sorry!)")