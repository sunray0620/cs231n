import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  for i in range(N):
    # Calculate loss
    scores = X[i].dot(W)  # (1, C)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    cur_loss = -correct_class_score + np.log(np.sum(np.exp(scores)))
    loss += cur_loss
    # Calculate dW
    cur_dW = np.zeros_like(W)
    cur_dW[:, y[i]] = -X[i]
    cur_dW += np.dot(X[i].reshape(-1, 1), np.exp(scores).reshape(1, -1)) / np.sum(np.exp(scores)) # (D, C)
    dW += cur_dW

  loss = (loss / N) + 0.5 * reg * np.sum(W * W)
  dW = (dW / N) + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  scores = X.dot(W) # (N, C)
  scores -= np.max(scores, axis = 1).reshape(-1, 1)
  correct_class_scores = scores[range(N), y]  # (N, 1)
  scores_exp = np.exp(scores)
  scores_expsum = np.sum(np.exp(scores), axis = 1)

  # loss value.
  loss = -correct_class_scores + np.log(scores_expsum)
  loss = np.sum(loss) / N + 0.5 * reg * np.sum(W * W)

  yy = np.zeros((N, C))
  yy[range(N), y] = 1
  dW_term1 = np.dot(X.T, yy)

  ss = scores_exp / scores_expsum.reshape(N, -1)
  dW_term2 = np.dot(X.T, ss)

  dW = -dW_term1 + dW_term2
  dW = (dW / N) + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

