import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    likelihood = np.exp(scores)/np.sum(np.exp(scores))
    loss += -np.log(likelihood[y[i]])
    dW[:,y[i]] -= X[i,:]
    for j in xrange(num_classes):
      dW[:,j] += X[i,:]*likelihood[j]

  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  loss = 0.0

  scores = X.dot(W)
  scores -= np.amax(scores,axis=1)[:,None]
  exp_scores = np.exp(scores)
  likelihood = exp_scores/np.sum(exp_scores,axis=1)[:,None]
  loss = np.sum(-np.log(likelihood[range(num_train),y]))
    
  coeffs = likelihood.copy()
  coeffs[range(num_train),y] -= 1
  dW = (X.T).dot(coeffs)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg*np.sum(W * W)
  dW += 2*reg*W

  return loss, dW

