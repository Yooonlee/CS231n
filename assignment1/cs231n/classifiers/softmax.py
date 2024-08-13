from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ### numeric instability => 최댓값을 빼서 scaling 해줌 
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
          scores = X[i].dot(W)
          scores -= max(scores)
          # loss
          loss_i = -scores[y[i]] + np.log(sum(np.exp(scores)))
          loss += loss_i
          # grad
          for j in range(num_classes):
                softmax_output = np.exp(scores[j]) / sum(np.exp(scores))
                ##### W, dW 는 열이 의미가 있음 / 열 = class 
                dW[:, j] += softmax_output * X[i]
          #### 정답 클래스의 경우 y_i = 1 이기에 추가해줌 다른 클래스의 y_i = 0
          dW[:, y[i]] -= X[i]
    
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    
    # Softmax Loss
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    #### 모든 데이터에 대해 softmax loss를 더함 
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]))
    
    # Weight Gradient
    ####### 정답 class에 대해 (p_i - 1) 해줌  / 1 = y_i
    softmax_matrix[np.arange(num_train), y] -= 1
    ####### 정답 class에 대해 x_i(p_i - 1) / 1 = y_i
    dW = X.T.dot(softmax_matrix)
    
    # Average 
    loss /= num_train
    dW /= num_train
    
    # Regularization
    loss += reg * np.sum(W*W)
    dW += reg * 2 * W 
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
