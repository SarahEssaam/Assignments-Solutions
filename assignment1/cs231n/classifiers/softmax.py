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
  dW = np.zeros_like(W).T
  N = X.shape[0]
  D = X.shape[1]
  K = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  f = X.dot(W)
  for i in range(N):
        f[i,:] -= np.max(f[i,:])
        sum_fi = np.sum(np.exp(f[i,:]))
        p_yi = np.exp(f[i,y[i]])/sum_fi
        L_i = -np.log(p_yi)
        loss += L_i
    
        for k in range(K):
            P = np.exp(f[i,k])/sum_fi
            for j in range(D):
                dW[k,j] += (P - (k == y[i])) * X[i,j]
  dW = dW.T / N
  loss /= N
  #Regularization term
  dW[:-1,:] += 2*reg*W[:-1,:]
  loss += reg*np.sum(np.square(W[:-1,:]))    
  
  #-------The Gradient--------------------------------
  #Gradient Of Softmax = (p-y).xT
  #see this https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
  #and this https://math.stackexchange.com/questions/2060944/gradient-of-a-softmax-applied-on-a-linear-function?rq=1
  #and this https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  #for explanation
  '''
  #integrated the gradient with the above for loop that computes loss
  for n in range(N):
    #x[n] now of dimension D
    sum_fi = np.sum(np.exp(f[n,:]))
    for k in range(K):
        P = np.exp(f[n,k])/sum_fi
        for j in range(D):
            dW[k,j] += (P - (k == y[n])) * X[n,j]
  dW = dW.T / N
  #Regularization term
  dW[:-2,:] += 2*reg*W[:-2,:]
  '''
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
  N = X.shape[0]
  K = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #1) Loss

  f = X.dot(W)
  f -= np.max(f,axis=1)[np.newaxis, :].T
  P = np.exp(f)/(np.sum(np.exp(f),axis=1)[np.newaxis, :].T)
  #P = np.exp(f[np.arange(N),y]) / np.sum(np.exp(f),axis=1)
  L_i = -np.log(P[np.arange(N),y])
  loss = np.sum(L_i)/N + reg*np.sum(np.square(W[:-1,:]))
    
  #2)Gradient
  y_ext = np.zeros_like(f,dtype=np.int8)
  y_ext[np.arange(N),y]=1
  dW = np.dot((P - y_ext).T,X).T
  dW = dW/N 
  dW[:-1,:] += 2*reg*W[:-1,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

