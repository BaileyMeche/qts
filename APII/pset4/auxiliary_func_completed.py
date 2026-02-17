#### auxiliary functions


import numpy as np
import pandas as pd

### Fuctions ###
def fw2(x):
    '''
    Input: 
    x: a 2-dimension matrix, size:(N,T)
    
    Output:
    [i,j]: x[i,j] should be the largest value in x
    '''

    
    pos = np.unravel_index(np.argmax(x),x.shape)
    
    return pos

def fw2_min(x):
    '''
    Input: 
    x: a 2-dimension matrix, size:(N,T)
    
    Output:
    [i,j]: x[i,j] should be the smallest value in x
    '''
    pos = np.unravel_index(np.argmin(x),x.shape)
    return pos


def fw1(x):
    '''
    Input: 
    x: a 1-dimension vector, size:N
    
    Output:
    i: x[i] should be the largest value in x
    '''
    return x.argmax()

def fw1_min(x):
    '''
    Input: 
    x: a 1-dimension vector, size:N
    
    Output:
    i: x[i] should be the smallest value in x
    '''
    x = np.asarray(x)
    return x.argmin()


    
    
def sq(a,b,step):
    '''
    Input: 
    a: min of the list
    b: max of the list
    step: for each iteration, we add 'step' to the previous element
    
    Output:
    r: the list from a to b and the difference between each element is step
    
    There's a difference between this function and np.arange(), we should include b in the
    output, while np.arange() will not include the end value.
    '''

    r = np.arange(a,b+1e-10,step)
    
    return r

def vip(b,xtrain,ytrain,mtrain):
    '''
    vip means Variable Importance 
    Input:
    b: coefficients matrix (p x 1)
    xtrain: covariates matrix (N x p)
    ytrain: reponse vector(demeaned)
    mtrain: the mean of ytrain
    
    Output:
    v: variable importance, should be the same size as b
    
    Follow these steps to implement the function:
    First, compute the aproximation of y by xtrain(b) and mtrain
    Then, compute the original R^2
    for each variable in b, we set it to zero and compute a new R^2. 
    We define the difference between the original R^2 and new one as the 
    importance of a certain variable.
    
    '''
    
    v        = np.zeros(len(b))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ### HINT: use np.copy() to make sure b does not change during the function.  ###
    ##############################################################################
    b = np.asarray(b).reshape(-1, 1)
    y_true = ytrain
    if np.abs(np.mean(ytrain)) < 1e-8:
        y_true = ytrain + mtrain
    yhat = xtrain.dot(b) + mtrain
    r2_orig = 1 - np.sum(np.power(yhat - y_true, 2)) / np.sum(np.power(y_true - mtrain, 2))
    for i in range(len(b)):
        b1 = np.copy(b)
        b1[i, 0] = 0
        yhat1 = xtrain.dot(b1) + mtrain
        r2_new = 1 - np.sum(np.power(yhat1 - y_true, 2)) / np.sum(np.power(y_true - mtrain, 2))
        v[i] = r2_orig - r2_new
    v = v.reshape(1, -1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return v

    
def vip_model(clf,xtrain,ytrain,mtrain):
    '''
    vip means Variable Importance 
    Input:
    clf: the pretrained model
    xtrain: covariates matrix (N x p)
    ytrain: reponse vector(demeaned)
    mtrain: the mean of ytrain
    Output:
    v: variable importance, should be the same size with b
    
    Follow these steps to implement the function:
    ############First, compute N as a vector from the matrix xtrain
    Then, compute the original R^2
    for each variable in b, we set it to zero and compute a new R^2. 
    We define the difference between the original R^2 and new one as the 
    importance of certain variable.
    
    '''
    
    yhatbig1 = clf.predict(xtrain)[:,0]
    v = np.zeros(xtrain.shape[1])
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    y_true = ytrain
    if np.abs(np.mean(ytrain)) < 1e-8:
        y_true = ytrain + mtrain
    yhat = yhatbig1 + mtrain
    r2_orig = 1 - np.sum(np.power(yhat - y_true, 2)) / np.sum(np.power(y_true - mtrain, 2))
    for i in range(xtrain.shape[1]):
        x1 = np.copy(xtrain)
        x1[:, i] = 0
        yhat1 = clf.predict(x1)[:, 0] + mtrain
        r2_new = 1 - np.sum(np.power(yhat1 - y_true, 2)) / np.sum(np.power(y_true - mtrain, 2))
        v[i] = r2_orig - r2_new
    v = v.reshape(1, -1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return v

    
def ys(x,n=100):
    '''
    Input:
    x: vector, (N,)
    n: output size
    
    Output:
    b: vector, (n,)
    
    We divide the input x into n groups, if it cannot be fully divided, we just ignore the last
    few elements of x. We output the sum of each group calculated in abstraction form.
    '''
    l = len(x)
    k = int(l/n)
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.sum(np.abs(x[(i*k):(i*k+k)]))
    return b

def g1(x):
    '''
    Input: x should be a 2d matrix, (N,p)
    For each column in x, we divide it by its sum of certain column.
    Output: the modified x.
    Be careful that doesn't change the original x.
    '''
    
    p  = x.shape[1]
    x1 = np.copy(x)
    x1 = x1 / np.sum(x1,axis = 0)
    return x1


def bd(x):
    '''
    Input x is a vector, (N,)
    Set all non-zero elements in x as 1.
    
    '''
    x = np.asarray(x)
    y = np.ones_like(x)
    y *= (x != 0)

    return y


def f(x,ind):
    '''
    Input:
    x: a vector has the size of (N,)
    ind: index of x, (n,), n < N
    
    Output:
    r: Size of (n+1,)
    The first n elements of r should be the certain index of x which is stored in ind.
    The last element of r should be the mean of rest of x.
    '''

    n      = len(ind)
    r      = np.zeros(n+1)
    r[0:n] = x[ind]
    ind1   = np.delete(np.arange(len(x)),ind,0)
    r[n]   = np.mean(x[ind1])
    return r


def ip(x):
    '''
    In this function, we will modify x.
    Input x should be a vector has the size of (N,), if all elements in x is negative, 
    then we return a one-value (N,) vector devided by the length of input x. 
    Otherwise we set the negative elements in x to be zero, and output the
    abstraction of modified vector devided by the sum of the abstraction of modified vector.
    '''

    a = np.array(x, copy=True)
    a[a<0] = 0
    if np.sum(a) == 0:
        return np.ones(len(x))/1.0/len(x)
    else:
        return np.abs(a)/np.sum(np.abs(a))
    
