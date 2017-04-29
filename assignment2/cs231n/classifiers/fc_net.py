from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        shapes = ((input_dim, hidden_dim), (hidden_dim, num_classes))
        
        self.params["W1"] = weight_scale * np.random.randn(shapes[0][0], shapes[0][1])
        self.params["b1"] = np.zeros(shapes[0][1])
        self.params["W2"] = weight_scale * np.random.randn(shapes[1][0], shapes[1][1])
        self.params["b2"] = np.zeros(shapes[1][1])
                       
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        N = X.shape[0]
        Xflt = X.reshape(N, -1)
        D = Xflt.shape[1]
        
        # Layer 1
        out1a, cache1a = affine_forward(Xflt, self.params["W1"], self.params["b1"])
        out1b, cache1b = relu_forward(out1a)
        
        # Layer 2
        out2a, cache2a = affine_forward(out1b, self.params["W2"], self.params["b2"])
        
        scores = out2a
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dout2a = softmax_loss(out2a, y)
        #scores -= np.max(scores, axis = 1).reshape(-1, 1)
        #eScores = np.exp(scores)
        #lineloss = (-scores[range(N), y] + np.log(np.sum(eScores, axis = 1)))
        #loss = np.sum(lineloss) / N
        loss += 0.5 * self.reg * np.sum(self.params["W1"] * self.params["W1"])
        loss += 0.5 * self.reg * np.sum(self.params["W2"] * self.params["W2"])
        
        ### Calculate grad ###
        # dloss = np.exp(scores) / np.sum(eScores, axis = 1).reshape(-1, 1)
        dout1b, grads["W2"], grads["b2"] = affine_backward(dout2a, cache2a)
        grads["W2"] += self.reg * self.params["W2"]
        
        dout1a = relu_backward(dout1b, cache1b)
        dx, grads["W1"], grads["b1"] = affine_backward(dout1a, cache1a)
        grads["W1"] += self.reg * self.params["W1"]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        shapes = [] # ((input_dim, hidden_dim), (hidden_dim, num_classes))
        hidden_dims.append(num_classes)
        
        cuript = input_dim
        for i in range(0, self.num_layers):
            curshape = (cuript, hidden_dims[i])
            curwname = "W{0}".format(i + 1)
            curbname = "b{0}".format(i + 1)
            self.params[curwname] = weight_scale * np.random.randn(curshape[0], curshape[1])
            self.params[curbname] = np.zeros(curshape[1])
            cuript = curshape[1]
            if self.use_batchnorm and i < self.num_layers - 1:
                curgammaname = "gamma{0}".format(i + 1)
                curbetaname = "beta{0}".format(i + 1)
                self.params[curgammaname] = np.ones(curshape[1])
                self.params[curbetaname] = np.zeros(curshape[1])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        N = X.shape[0]
        Xflt = X.reshape(N, -1)
        
        caches = {}
        caches["affine"] = []
        caches["batchnorm"] = []
        caches["relu"] = []
        
        curinput = Xflt
        for i in range(0, self.num_layers - 1):
            curwname = "W{0}".format(i + 1)
            curbname = "b{0}".format(i + 1)
            curgammaname = "gamma{0}".format(i + 1)
            curbetaname = "beta{0}".format(i + 1)
            
            # Step 1: Affine
            o1, c1 = affine_forward(curinput, self.params[curwname], self.params[curbname])
            caches["affine"].append(c1)
            
            # Step 2: Batch Normalization.
            if self.use_batchnorm:
                o2, c2 = batchnorm_forward(o1, self.params[curgammaname], self.params[curbetaname], self.bn_params[i])
                caches["batchnorm"].append(c2)
            else:
                o2 = o1
            
            # Step 3: Relu
            o3, c3 = relu_forward(o2)
            caches["relu"].append(c3)
            
            # Step 4: Dropout
            # o4, c4 = dropout()
            
            curinput = o3
        
        lastwname = "W{0}".format(self.num_layers)
        lastbname = "b{0}".format(self.num_layers)
        op, cp = affine_forward(curinput, self.params[lastwname], self.params[lastbname])
        scores = op
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dop = softmax_loss(op, y)
        dout1b, grads[lastwname], grads[lastbname] = affine_backward(dop, cp)
        
        curdipt = dout1b
        for i in range(self.num_layers - 2, -1, -1):
            curwname = "W{0}".format(i + 1)
            curbname = "b{0}".format(i + 1)
            curgammaname = "gamma{0}".format(i + 1)
            curbetaname = "beta{0}".format(i + 1)
            
            # Step 1: DD Dropout
            # o4, c4 = dropout()
            
            # Step 2: DD Relu
            do3 = relu_backward(curdipt, caches["relu"][i])
            
            # Step 3: Batch Normalization.
            if self.use_batchnorm:
                do2, grads[curgammaname], grads[curbetaname] = batchnorm_backward(do3, caches["batchnorm"][i])
            else:
                do2 = do3
            
            # Step 4: Affine
            do1, grads[curwname], grads[curbname] = affine_backward(do2, caches["affine"][i])
            
            curdipt = do1
        
        for i in range(0, self.num_layers):
            curwname = "W{0}".format(i + 1)
            curbname = "b{0}".format(i + 1)
            loss += 0.5 * self.reg * np.sum(self.params[curwname] * self.params[curwname])
            grads[curwname] += self.reg * self.params[curwname]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
