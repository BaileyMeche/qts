### Simulation Regression Models
### All Regressions 
### OLS, OLS+H, PCR, PLS, Lasso, Lasso+H, Ridge, Ridge+H, ENet, ENet+H and Group Lasso, Group Lasso+H.
### Including the Oracle Regression Model  

'''
### Server-Run Codes (Run 1 MCMC Simu on each node)
import argparse
args = argparse.ArgumentParser()
args.add_argument("Symbol", help="MCMC")
arg = args.parse_args()
number = arg.Symbol
MC=int(number) 
'''
MC = 1


import numpy as np
import pandas as pd
from scipy import linalg, optimize
from sklearn import linear_model
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import random 
random.seed(MC*123)
import timeit
from auxiliary_func import *

nc = 3
groups = np.random.randint(0,nc,8)


datanum   = '100'   
#datanum   = '200'
path      = './Simu/'
dirstock  = path+'SimuData_'+datanum+'/'



def cut_knots_degree2(x,n,th):
    '''
    In this function you should implement a vector of basic functions, you can find the specific
    definition in the paper. Remember we use a spline series of order two. There's a little 
    difference between paper and this function, here, the functions are 1,(z-c1)^2,(z-c2)^2,...,
    (z-cn)^2, and from the third element, we only care about the situations when z>=ci, otherwise, 
    we set it to zero.
    
    Input: 
    x: input data, it can have size of both (N,) and (N, T)
    n: the number of knots
    th: knots in the functions, have the size of (n, T)
    
    
    Output:
    resultfinal: should have the size of (N,T*(n+1))
    
    For each x[:,i], we compute a series of p(x[:,i]) which result should 
    have size of N*(n+1), we loop i from 1 to T, then the output should be N by T*(n+1)
    '''
    a = x.shape[0]
    if len(x.shape) == 1:
        b = 1
    else:
        b = x.shape[1]
    
    resultfinal = np.zeros((a,b*(n+1)))
    
    for i in range(b):
        xcut                         = x[:,i]
        xcutnona                     = np.copy(xcut)
        xcutnona[np.isnan(xcutnona)] = 0
        
        ##############################################################################
        ### TODO:    implement the function                                        ###
        ###          you should store demeaned value to the output                 ###
        ##############################################################################
        xcutnona = xcutnona.reshape(-1)
        basis = np.zeros((a, n + 1))
        basis[:, 0] = xcutnona
        basis[:, 1] = np.power(xcutnona - th[0, i], 2)
        for j in range(2, n + 1):
            basis[:, j] = np.power(np.maximum(xcutnona - th[j - 1, i], 0), 2)
        basis = basis - np.mean(basis, axis=0)
        resultfinal[:, (i * (n + 1)):((i + 1) * (n + 1))] = basis

        ##############################################################################
        ##############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    return resultfinal


def loss(y,yhat):
    '''
    Ordinary least square loss
    Input: 
    y: the real y
    yhat: the prediction of y
    
    Output: 
    the ordinary least square loss which is the mean of the squared difference
    it should be a scalar return
    
    '''
    
    m = np.zeros(len(y))
    
    ##############################################################################
    ### TODO: return the ordinary least square loss of the prediction          ###
    ### the code should be very simple, 1 line is enough                       ###
    ##############################################################################
    m = np.mean(np.power(yhat - y, 2))
    ##############################################################################
    ##############################################################################
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return m
    
    
def lossw(y,yhat,w):
    '''
    Ordinary least square loss with weight
    Input: 
    y: the real y
    yhat: the prediction of y
    w: weight matrix, have the same size with y
    
    Output: 
    the sum of weight matrix elementwise multiplies by (yhat-y)^2 and 
    then divided by the sum of weight matrix
    it should be a scalar return
    
    '''
    
    m = np.zeros(len(y))
    ##############################################################################
    ### TODO: return the ordinary least square loss of the prediction          ###
    ### the code should be very simple, 1 line is enough                       ###
    ##############################################################################
    m = np.sum(w * np.power(yhat - y, 2)) / np.sum(w)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return m


def losshw(y,yhat,w,mu):
    '''
    OLS + H with weight matrix
    
    Input: 
    y: the real y
    yhat: the prediction of y
    w: weight matrix, same size with y
    mu: the tuning hyperparameter of Huber robust objective function
    
    Output:
    the OLS+H with w loss, should be a scalar
    '''
    
    m = np.zeros(len(y))
    ##############################################################################
    ### TODO: implement the loss function                                      ###
    ### HINT: you just need to modify the return value the previous function   ###
    ###       lossh(), it's like the difference between loss() and lossw()     ###
    ##############################################################################
    diff = yhat - y
    l = np.where(np.abs(diff) <= mu, np.power(diff, 2), 2 * mu * np.abs(diff) - mu ** 2)
    m = np.sum(w * l) / np.sum(w)

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ############################################################################## 
    return m
    
def lossh(y,yhat,mu):
    '''
    OLS + H
    Heavy tails are a well-known attribute of financial returns and stock-level 
    predictor variables. Convexity of the least squares loss places extreme emphasis 
    on large errors, thus outliers can undermine the stablity of OLS-based prediction.
    In the machine learning literature, a common choice for counteracting the deleterious 
    effect of heavy-tailed observations is the Huber robust objective function.
    
    Input: 
    y: the real y
    yhat: the prediction of y
    mu: the tuning hyperparameter of Huber robust objective function
    
    Output:
    the OLS+H loss, should be a scalar
    '''
    
    l      = np.zeros(len(y))
    ##############################################################################
    ### TODO: implement the loss function of OLS+H, the output should          ###
    ###       be the mean of a scalar loss.                                    ###
    ##############################################################################
    diff = yhat - y
    l = np.where(np.abs(diff) <= mu, np.power(diff, 2), 2 * mu * np.abs(diff) - mu ** 2)

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ############################################################################## 
    
    return np.mean(l)
    

def f_grad(XX,XY,w):
    '''
    To update coefficient matrix, we should do some gradient descent on it. This function
    is to generate grad for regressions. 
    Input:
    XX: NT*NT matrix of X'X  
    w: NT*1 vector of weight 
    XY: NT*1 vector of X'y 
    '''
    
    m = np.zeros(XY.shape)
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    m = XX.dot(w) - XY

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ############################################################################## 
    return m

def f_gradh(w,X,y,mu):
    '''
    Compute gradients for regressions with huber function, we strict backward pass returns
    as [-mu,mu], if returns are out of this area, then we use mu instead of returns.
    
    Input: 
    w: coefficients matrix (p,1)
    X: matrix, (N,p)
    y: response vector （N,1）
    mu: tuning parameters of Huber robust objective function
    
    Output: 
    g: gradients for w, same shape with w
    '''
    
    m = np.zeros((len(w),1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    res = X.dot(w) - y
    res = np.clip(res, -mu, mu)
    m = X.T.dot(res)

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return m
    
    
def soft_thresholdl(w,mu):
    '''
    soft_threshold for lasso
    deactivate some knots in w, we only care about the value whose abstractions
    are greater than mu.
    '''
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))    


def soft_thresholdr(w,mu):
    '''
    soft_threshold for ridge
    shrink w by 1+mu
    '''
    return w/(1+mu)


def soft_thresholde(w,mu):
    '''
    soft_threshold for elastic net
    deactivate some knots in w and shrink it by 1+0.5*mu 
    '''
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- 0.5*mu,0)) /(1+0.5*mu)


def soft_thresholda(w,alpha,mu):
    '''
    deactivate some knots in w and shrink it by 1+alpha*mu
    '''
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- alpha*mu,0)) /(1+alpha*mu)


def soft_thresholdg(w,mu):
    '''
    if any group of weights is close to zero, then we set them directly to zero.
    '''
    w1 = np.copy(w)
    
    for i in range(nc):
        ind   = groups==i
        wg    = w1[ind,:]
        nn    = wg.shape[0]
        n2    = np.sqrt(np.sum(np.power(wg,2)))
        if n2 <= mu:
            w1[ind,:] = np.zeros((nn,1))
        else:
            w1[ind,:] = wg-mu*wg/n2
    return w1


def proximal(XX,XY,tol,L,l1,func):
    '''
    accelarated proximal gradient algorithm for simple regression model.  We use a kind 
    of gradient algorithm which can quickly decrease dimemsions of weights and estimate them.
    
    First, we set weight matrix v to zero at the same size as XX.shape[0].
    
    Then, we loop t in range of max_iter, we update w using f_grad(): w = v -  grad/L
    and use func to either deactivate or shrink w.
    
    Lastly, update v: v = w + t/(t+3) * (w - w_prev). 
    If v and prev_v are equal or nearly equal, end the loop.
    
    Output is all non-zero values in v
    
    Input:
    XX: NT * NT
    XY: NT * 1
    tol: scalar value
    L: scalar value
    l1: scalar value
    func: function to handle w
    ''' 
    
    dim      = XX.shape[0]
    max_iter = 30000
    gamma    = 1/L
    
    m = np.zeros(dim).T
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    w = np.zeros((dim, 1))
    v = np.zeros((dim, 1))
    v_prev = np.copy(v)
    w_prev = np.copy(w)

    for t in range(max_iter):
        grad = f_grad(XX, XY, v)
        w = v - gamma * grad
        w = func(w, gamma * l1)
        v = w + t / (t + 3) * (w - w_prev)
        if np.linalg.norm(v - v_prev) < tol:
            break
        v_prev = np.copy(v)
        w_prev = np.copy(w)

    m = v.reshape(-1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return m


def proximalH(w,X,y,mu,tol,L,l1,func):
    '''
    accelarated proximal gradient algorithm for regression with huber function, it should be 
    nearly the same as proximal() except for the gradient function. 
    You can directly copy proximal() and just modify it a bit.
    
    Input:
    w: have the size of (P,) which is a list, not matrix( be care about the result dimension)
    X: size of (NT,P)
    y: size of (NT,1)
    mu: tuning parameter of Huber function
    rest are the same as proximal()
    
    '''
    
    
    max_iter = 30000
    gamma    = 1/L
    P        = np.max(w.shape)
    
    res = np.zeros((P,1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    w = np.asarray(w).reshape(P, 1)
    v = np.copy(w)
    v_prev = np.copy(v)
    w_prev = np.copy(w)

    for t in range(max_iter):
        grad = f_gradh(v, X, y, mu)
        w = v - gamma * grad
        w = func(w, gamma * l1)
        v = w + t / (t + 3) * (w - w_prev)
        if np.linalg.norm(v - v_prev) < tol:
            break
        v_prev = np.copy(v)
        w_prev = np.copy(w)

    res = v.reshape(-1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return res
    
    
def PCR(X,y,A):
    '''
    Principal component regression
    Input: 
    X: covariates matrix (N * p)
    y: response vector(demeaned), (N * 1)
    A: the number of steps
    Output:
    B: coefficients matrix (p * A)
    
    First, implement pca to do dimensional reduction, then you will get a (p,A) matrix of
    weights. Make original predictions on X using these weights.
    Second, we loop A-1 times, for the ith step, we use the first i+1 elements P_i+1 in 
    the original predictions.
    Compute the matrix multiplication of (P_i+1'*P_i+1)^(-1), P_i+1', y and 
    first i+1 element of weights. This should output a (p,1) matrix, then we store it in B[,i].
    Hint: use np.linalg.pinv()
    '''
    
    res = np.zeros((X.shape[1], A))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    weights = vt.T[:, :A]
    res[:, 0] = 0
    for i in range(1, A):
        scores = X.dot(weights[:, :i])
        theta = np.linalg.pinv(scores.T.dot(scores)).dot(scores.T).dot(y)
        res[:, i] = (weights[:, :i].dot(theta)).reshape(-1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return res
    

def pls(X,y,A):
    '''
    Partial least square:
    Input:
    X: covariates matrix (N x p)
    y: response vector (should be demeaned) (N,1)
    A: the number of steps.
    Output:
    B: coefficients matrix (p x A)
    
    Follow the steps to implement the function:
    First, compute s = X'y, (p*1)
    Then, we loop A steps, for each step, q =s^2, it shoud be a scalar value
    r = s*q, r has the same size as s
    t = X*r (N*1)
    normarlization t by t = (t-E(t)）/sqrt(t^2)
    divide r by sqrt(t^2)
    p = X't, (p*1)
    q = y't, it should be a scalar value
    u = y*q
    v = p
    For non-first steps, we update v and u by:
    v -= all previous v multiplied (all previous v multiplied by p)
    u -= all previous t multiplied (all previous t multiplied by u)
    then, divide v by its norm, update s :
    s -= v*v's
    Finally, we store coefficients in B, for i =1:A,
    B[:,i] = the first i elements of r multiplied by the trans of the first i elements of q
    '''
    
    res = np.zeros((X.shape[1], A))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    n, p = X.shape
    s = X.T.dot(y)
    V = np.zeros((p, A))
    T = np.zeros((n, A))
    R = np.zeros((p, A))
    Q = np.zeros((A, 1))

    for i in range(A):
        q_scalar = float(s.T.dot(s))
        r = s * q_scalar
        t = X.dot(r)
        t = t - np.mean(t)
        t_norm = np.sqrt(np.sum(np.power(t, 2)))
        if t_norm != 0:
            t = t / t_norm
            r = r / t_norm

        p_vec = X.T.dot(t)
        q_scalar = float(y.T.dot(t))
        u = y * q_scalar
        v = p_vec

        if i > 0:
            v = v - V[:, :i].dot(V[:, :i].T.dot(p_vec))
            u = u - T[:, :i].dot(T[:, :i].T.dot(u))

        v_norm = np.sqrt(np.sum(np.power(v, 2)))
        if v_norm != 0:
            v = v / v_norm

        s = s - v * (v.T.dot(s))

        V[:, i] = v.reshape(-1)
        T[:, i] = t.reshape(-1)
        R[:, i] = r.reshape(-1)
        Q[i, 0] = q_scalar

    res[:, 0] = 0
    for i in range(1, A):
        res[:, i] = (R[:, :i].dot(Q[:i])).reshape(-1)

    #                               END OF YOUR CODE                             #
    ##############################################################################
    return res
    

def write_dir(hh = [1]):
    for h in hh:
        title = path+'/Simu_'+datanum+'/Reg%d'%h
        if not os.path.exists(title) and MC==1:
            os.makedirs(title)
        if not os.path.exists(title+'/B') and MC==1:
            os.makedirs(title+'/B')
        if not os.path.exists(title+'/VIP') and MC==1:
            os.makedirs(title+'/VIP')
    return
            
def OLS(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos):
    '''
    This function is the simple OLS, implement it and you will get coefficient matrix
    and variable importance matrix.
    Input:
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix
    
    Follow the steps to implement code here:
    First, you can use linear_model.LinearRegression to create a linear model and fit the model
    with xtrain and ytrain_demean
    Second, predict y of xoos with the model(don't forget to add mean of y).
    Then compute OOS R^2 with the previous prediction.
    Third, predict y of xtrain with the model, then compute IS R^2. 
    At last, output b of the model and compute variable importance of each coefficient (you 
    can use vip() to compute importance)
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(xtrain, ytrain_demean)

    yhat = clf.predict(xoos) + mtrain
    r2_oos = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))

    yhat_is = clf.predict(xtrain) + mtrain
    r2_is = 1 - np.sum(np.power(yhat_is - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

    b = np.asarray(clf.coef_)
    v = vip(b.T, xtrain, ytrain, mtrain)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return r2_oos,r2_is,b,v


def OLSH(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,mu,tol,L,func):
    '''
    This function is the simple OLS and Huber function. Because OLS+H is based on OLS, 
    we have to first call the OLS function. The steps to compute R^2 is nearly the same as
    OLS function.
    Input:
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix
    
    Follow the steps to implement code here:
    First, call OLS to get the coefficient matrix.
    Then, use proximalH to compute the coefficient matrix for OLS+H
    Second, compute r2_oos and r2_is as in the OLS function
    Last, compute importance of coefficients.
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ### HINT: be careful about dimensions, a little carefulness may make mistake##
    ##############################################################################
    r2_oos_ols, r2_is_ols, b_ols, _ = OLS(xtrain, ytrain, ytrain_demean, mtrain, xoos, yoos)
    b_start = np.asarray(b_ols).reshape(-1, 1)

    b_h = proximalH(b_start, xtrain, ytrain_demean, mu, tol, L, 0, func)
    b_h = b_h.reshape(-1, 1)

    yhat = xoos.dot(b_h) + mtrain
    r2_oos = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))

    yhat_is = xtrain.dot(b_h) + mtrain
    r2_is = 1 - np.sum(np.power(yhat_is - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

    b = b_h.reshape(1, -1)
    v = vip(b_h, xtrain, ytrain, mtrain)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return r2_oos,r2_is,b,v


def PCAR(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,ne):
    
    '''
    PCA Regression
    Input: 
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    xtest: test sample of characteristics
    ytest: test sample of returns
    ne: tuning parameter of PCR
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix
    
    Follow the steps to implement code here:
    First, call PCR function to compute xtrain.shape[1]*ne matrix, each column can be a set of coefficients. 
    Second, for each column in the coefficients matrix, we compute R^2 by xtest, 
    xtrain, and xoos respectively, we define ind as the index which has the maximum R^2 of xtest.
    Then, r2_oos = the ind element R^2 of xoos 
    r2_is = the ind element R^2 of xtrain
    b = the ind column of coefficients matrix
    
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    B = PCR(xtrain, ytrain_demean, ne)
    r = np.zeros((3, ne))
    for i in range(ne):
        b_i = B[:, i].reshape(-1, 1)
        yhat = xtest.dot(b_i) + mtrain
        r[0, i] = 1 - np.sum(np.power(yhat - ytest, 2)) / np.sum(np.power(ytest - mtrain, 2))
        yhat = xoos.dot(b_i) + mtrain
        r[1, i] = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))
        yhat = xtrain.dot(b_i) + mtrain
        r[2, i] = 1 - np.sum(np.power(yhat - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

    ind = int(fw1(r[0, :]))
    r2_oos = r[1, ind]
    r2_is = r[2, ind]
    b = B[:, ind].reshape(1, -1)
    v = vip(B[:, ind].reshape(-1, 1), xtrain, ytrain, mtrain)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return r2_oos,r2_is,b,v

def PLSR(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,ne):
    '''
    PLS Regression
    Input: 
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    xtest: test sample of characteristics
    ytest: test sample of returns
    ne: tuning parameter of pls
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix
    
    Call PLS to output an xtrain.shape[1]*ne matrix, and the rest of the function should
    be the same as PCAR
    
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    B = pls(xtrain, ytrain_demean, ne)
    r = np.zeros((3, ne))
    for i in range(ne):
        b_i = B[:, i].reshape(-1, 1)
        yhat = xtest.dot(b_i) + mtrain
        r[0, i] = 1 - np.sum(np.power(yhat - ytest, 2)) / np.sum(np.power(ytest - mtrain, 2))
        yhat = xoos.dot(b_i) + mtrain
        r[1, i] = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))
        yhat = xtrain.dot(b_i) + mtrain
        r[2, i] = 1 - np.sum(np.power(yhat - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

    ind = int(fw1(r[0, :]))
    r2_oos = r[1, ind]
    r2_is = r[2, ind]
    b = B[:, ind].reshape(1, -1)
    v = vip(B[:, ind].reshape(-1, 1), xtrain, ytrain, mtrain)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return r2_oos,r2_is,b,v

def Lasso(XX,XY,xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,mu,tol,L,alpha,lamv):
    '''
    Lasso Regression and Lasso Regression with Huber function
    Tuning parameter: the L1 penalty lambda
    
    Input: 
    XX: xtrain' * xtrain
    XY: xtrain' * ytrain_demean
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    xtest: test sample of characteristics
    ytest: test sample of returns
    lamv: tuning parameter of L1 lenalty lambda
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix for both lasso and lassoH
    
    
    Follow the steps to implement code here:
    First, for each l in lamv, l2 = 10^l, call proximal() to compute coefficient matrix 
    Second, we compute R^2 by xtest, xtrain and xoos respectively, we define ind as the index 
    of the maximum R^2 of xtest.
    Then, r2_oos = the ind element R^2 of xoos 
    r2_is = the ind element R^2 of xtrain
    and, we can find the best tuning parameter l2 = 10^lamv[ind], so we can compute b and v for
    both Lasso and LassoH
    
    b equals certain coefficients.
    
    b_H is computed using proximalH() and b
    Then, compute r2 and v as before.
    
    
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    r2_oos_H = 0
    r2_is_H  = 0
    b_H      = np.zeros((xtrain.shape[1], 1))
    v_H      = np.zeros((xtrain.shape[1], 1))
    
    # each row in r stores R^2 for xtest, xoos and xtrain respectively
    r = np.zeros((3,len(lamv)))
    
    for j in range(len(lamv)):
        l2       = 10**lamv[j]
        ##############################################################################
        ### TODO: implement the function                                           ###
        ##############################################################################
        b        = proximal(XX,XY,tol,L,l2,soft_thresholdl)
        b        = b.reshape(len(b),1)
        yhatbig1 = xtest.dot(b)+mtrain
        r[0,j]   = 1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
        yhatbig1 = xoos.dot(b)+mtrain
        r[1,j]   = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
        yhatbig1 = xtrain.dot(b)+mtrain
        r[2,j]   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

    r2_oos   = r[1,int(fw1(r[0,:]))]    
    r2_is    = r[2,int(fw1(r[0,:]))]    
    l2       = 10**lamv[int(fw1(r[0,:]))]
    print('Lasso',l2,'[-2,4]')
    b        = proximal(XX,XY,tol,L,l2,soft_thresholdl)
    bT       = b.reshape(len(b),1)
    v        = vip(bT,xtrain,ytrain,mtrain)
    
    # LassoH
    b_H      = proximalH(bT,xtrain,ytrain_demean,mu,tol,L,l2,soft_thresholdl)
    bT       = b_H.reshape(len(b_H),1)

    yhatbig1 = xoos.dot(bT)+mtrain
    r2_oos_H = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
    yhatbig1 = xtrain.dot(bT)+mtrain
    r2_is_H  = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
    v_H      = vip(bT,xtrain,ytrain,mtrain)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H


def ridge(XX,XY,xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,mu,tol,L,alpha,lamv):
    '''
    Ridge Regression and Ridge Regression with Huber function
    Tuning parameter: the L2 penalty lambda
    
    Input: 
    XX: xtrain' * xtrain
    XY: xtrain' * ytrain_demean
    xtrain: training sample of characteristics
    ytrain: training sample of returns
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    xtest: test sample of characteristics
    ytest: test sample of returns
    lamv: tuning parameter of L2 lenalty lambda
    
    Output:
    r2_oos,r2_is,coefficient matrix and importance matrix for both ridge and ridgeH
    
    
    The steps in ridge regression is the same as the lasso regression, except for one function.
    We use soft_thresholdl() in the lasso regression, while we use soft_thresholdr() in the ridge regression.
    
    
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    r2_oos_H = 0
    r2_is_H  = 0
    b_H      = np.zeros((xtrain.shape[1], 1))
    v_H      = np.zeros((xtrain.shape[1], 1))
                
    r = np.zeros((3,len(lamv)))
    

    for j in range(len(lamv)):
        
        l2       = 10**lamv[j]
        ##############################################################################
        ### TODO: implement the function                                           ###
        ##############################################################################
        
        b        = proximal(XX,XY,tol,L,l2,soft_thresholdr)
        b        = b.reshape(len(b),1)
        yhatbig1 = xtest.dot(b)+mtrain
        r[0,j]   = 1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
        yhatbig1 = xoos.dot(b)+mtrain
        r[1,j]   = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
        yhatbig1 = xtrain.dot(b)+mtrain
        r[2,j]   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

    r2_oos    = r[1,int(fw1(r[0,:]))]    
    r2_is     = r[2,int(fw1(r[0,:]))]    
    l2        = 10**lamv[int(fw1(r[0,:]))]
    print('Ridge',l2,'[0,6]')
    b         = proximal(XX,XY,tol,L,l2,soft_thresholdr)
    bT        = b.reshape(len(b),1)
    v         = vip(bT,xtrain,ytrain,mtrain)
     
    # ridge + H

    b_H       = proximalH(bT,xtrain,ytrain_demean,mu,tol,L,l2,soft_thresholdr)
    # reshape bH
    bh        = b_H.reshape(len(b_H),1)

    yhatbig1  = xoos.dot(bh)+mtrain
    r2_oos_H  = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
    yhatbig1  = xtrain.dot(bh)+mtrain
    r2_is_H   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
    v_H       = vip(bh,xtrain,ytrain,mtrain)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H


def Enet(XX,XY,xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,mu,tol,L,alpha,lamv):
    '''
    ### Elastic Net  and Elastic Net + H
    ### Tuning parameter: the L1+L2 penalty lambda
    Everything is the same as lasso and ridge, except this functions uses the soft_threshold function
    You only need to modify a bit from previous functions.
    '''
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    r2_oos_H = 0
    r2_is_H  = 0
    b_H      = np.zeros((xtrain.shape[1], 1))
    v_H      = np.zeros((xtrain.shape[1], 1))
    
    r = np.zeros((3,len(lamv)))

    for j in range(len(lamv)):
        l2       = 10**lamv[j]
        ##############################################################################
        ### TODO: implement the function                                           ###
        ##############################################################################
        b        = proximal(XX,XY,tol,L,l2,soft_thresholde)
        b        = b.reshape(len(b),1)
        yhatbig1 = xtest.dot(b)+mtrain
        r[0,j]   = 1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
        yhatbig1 = xoos.dot(b)+mtrain
        r[1,j]   = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
        yhatbig1 = xtrain.dot(b)+mtrain
        r[2,j]   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

    r2_oos    = r[1,int(fw1(r[0,:]))]    
    r2_is     = r[2,int(fw1(r[0,:]))]    
    l2        = 10**lamv[int(fw1(r[0,:]))]
    print('Ridge',l2,'[0,6]')
    b         = proximal(XX,XY,tol,L,l2,soft_thresholde)
    bT        = b.reshape(len(b),1)
    v         = vip(bT,xtrain,ytrain,mtrain)
    
    # Elastic Net + H

    b_H       = proximalH(bT,xtrain,ytrain_demean,mu,tol,L,l2,soft_thresholde)
    # reshape bH
    bh        = b_H.reshape(len(b_H),1)

    yhatbig1  = xoos.dot(bh)+mtrain
    r2_oos_H  = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
    yhatbig1  = xtrain.dot(bh)+mtrain
    r2_is_H   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
    v_H       = vip(bh,xtrain,ytrain,mtrain)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H

def Oracle(mo,nump,xtrain,ytrain,mtrain,xoos,yoos):
    '''
    Oracle Linear Regression
    The prediction process is similar to OLS, however, we will first change the input data.
    For mo = 1 (linear model)
    Set x to only 3 columns, the first and second are the corresponding columns in x. 
    The third column equals the nump+3 column of x.
    For mo = 2 (nonlinear model)
    Also set x to only 3 columns, the first column is the square of the 1st column in x.
    The second column is the elementwise product of the 1st and 2nd columns of x.
    The third column is the sign of the nump+3 column in x.
    
    We only output r2_oos and r2_is in this function.
    '''
    
    r2_oos,r2_is = 0,0
    
    if mo == 1:
        x        = np.zeros((xtrain.shape[0],3))
        x[:,0]   = xtrain[:,0]
        x[:,1]   = xtrain[:,1]
        x[:,2]   = xtrain[:,nump+2]
        x1       = np.zeros((xoos.shape[0],3))
        x1[:,0]  = xoos[:,0]
        x1[:,1]  = xoos[:,1]
        x1[:,2]  = xoos[:,nump+2]
        
        ##############################################################################
        ### TODO: implement the function                                           ###
        ##############################################################################
        clf = linear_model.LinearRegression(fit_intercept=False)
        ytrain_demean = ytrain - mtrain
        clf.fit(x, ytrain_demean)
        yhat = clf.predict(x1) + mtrain
        r2_oos = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))
        yhat_is = clf.predict(x) + mtrain
        r2_is = 1 - np.sum(np.power(yhat_is - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    if mo == 2:
        x        = np.zeros((xtrain.shape[0],3))
        x[:,0]   = np.power(xtrain[:,0],2)
        x[:,1]   = xtrain[:,1]*xtrain[:,0]
        x[:,2]   = np.sign(xtrain[:,nump+2])
        x1       = np.zeros((xoos.shape[0],3))
        x1[:,0]  = np.power(xoos[:,0],2)
        x1[:,1]  = xoos[:,1]*xoos[:,0]
        x1[:,2]  = np.sign(xoos[:,nump+2])
        
        ##############################################################################
        ### TODO: implement the function                                           ###
        ##############################################################################
        clf = linear_model.LinearRegression(fit_intercept=False)
        ytrain_demean = ytrain - mtrain
        clf.fit(x, ytrain_demean)
        yhat = clf.predict(x1) + mtrain
        r2_oos = 1 - np.sum(np.power(yhat - yoos, 2)) / np.sum(np.power(yoos - mtrain, 2))
        yhat_is = clf.predict(x) + mtrain
        r2_is = 1 - np.sum(np.power(yhat_is - ytrain, 2)) / np.sum(np.power(ytrain - mtrain, 2))

        #                               END OF YOUR CODE                             #
        ##############################################################################
            
  
    return r2_oos,r2_is 


def Group_lasso(kn,xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,xtest,ytest,mu,tol,lamv):
    '''
    ### Group Lasso  and Group
    ### Tuning parameter: the group lasso penalty lambda
    Input:
    kn: the number of knots
    xtrain: training sample of characteristics, (N,p)
    ytrain: training sample of returns 
    ytrain_demean: demeaned training sample of returns
    mtrain: the mean of ytrain
    xoos: out-of-sample characteristics used to compute OOS R^2
    yoos: out-of-sample returns used to compute OOS R^2
    xtest: test sample of characteristics
    ytest: test sample of returns
    lamv: the group lasso penalty lambda
    In this function, we will use cut_knots_degree2 to transfer characteristics.
    
    Follow the steps to implement the function below:
    1. For each column in xtrain, xtest and xoos, divide by the standard deviation
    (if std > 0) of the corresponding column in xtrain.
    2. compute XX = xtrain' * xtrain, XY = xtrain' * ytrain_demean
    3. let L equal the maximum singular value of XX, hint: np.linalg.svd
    4. nc=p/(kn+1), groups should repeat from 0 to nc, each kn+1 times(notice: both nc and groups
    are global variable)
    5. Now, compute r2_oos, r2_is, b and v using 
    soft_thresholdg() for both group lasso and group lasso+H
    '''
    
    r2_oos = 0
    r2_is  = 0
    b      = np.zeros((xtrain.shape[1], 1))
    v      = np.zeros((xtrain.shape[1], 1))
    r2_oos_H = 0
    r2_is_H  = 0
    b_H      = np.zeros((xtrain.shape[1], 1))
    v_H      = np.zeros((xtrain.shape[1], 1))
    
    th      = np.zeros((kn,xtrain.shape[1]))
    th[2,:] = 0
    for i in range(xtrain.shape[1]):
        th[:,i] = np.percentile(xtrain[:,i],np.arange(kn)*100.0/kn)

    xtrain  = cut_knots_degree2(xtrain,kn,th)
    xtest   = cut_knots_degree2(xtest,kn,th)
    xoos    = cut_knots_degree2(xoos,kn,th)
    
    ##############################################################################
    ### TODO: implement the function                                           ###
    ##############################################################################
    for i in range(xtrain.shape[1]):
        s   = np.std(xtrain[:,i])
        if s > 0:
            xtrain[:,i] = xtrain[:,i] / s
            xtest[:,i]  = xtest[:,i]  / s
            xoos[:,i]   = xoos[:,i]   / s

    Y       = ytrain_demean
    XX      = xtrain.T.dot(xtrain)
    
    b       = np.linalg.svd(XX)
    print('L=',b[1][0])
    L       = b[1][0]         # L equals the maximum singular value of XX
    XY      = xtrain.T.dot(Y)
    

        
    global nc
    nc      = (XX.shape[1])//(kn+1)
    global groups
    groups  = np.repeat(np.arange(nc),int(kn+1))
    
    
    r       = np.zeros((3,len(lamv)))

    for j in range(len(lamv)):
        l2       = 10**lamv[j]
        b        = proximal(XX,XY,tol,L,l2,soft_thresholdg)
        b        = b.reshape(len(b),1)
        yhatbig1 = xtest.dot(b)+mtrain
        r[0,j]   = 1-sum(np.power(yhatbig1-ytest,2)) /sum(np.power(ytest-mtrain,2))
        yhatbig1 = xoos.dot(b)+mtrain
        r[1,j]   = 1-sum(np.power(yhatbig1-yoos,2))  /sum(np.power(yoos-mtrain,2))
        yhatbig1 = xtrain.dot(b)+mtrain
        r[2,j]   = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))


    r2_oos   = r[1,int(fw1(r[0,:]))]    
    r2_is    = r[2,int(fw1(r[0,:]))]    
    l2       = 10**lamv[int(fw1(r[0,:]))]
    print('GLasso',l2,'[0.5,3]')
    b        = proximal(XX,XY,tol,L,l2,soft_thresholdg)
    bT       = b.reshape(len(b),1)
    
    

    v        = vip(bT,xtrain,ytrain,mtrain)
    

    b_H      = proximalH(bT,xtrain,ytrain_demean,mu,tol,L,l2,soft_thresholdg)
    bh       = b_H.reshape(len(b_H),1)

    yhatbig1 = xoos.dot(bh)+mtrain
    r2_oos_H = 1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
    yhatbig1 = xtrain.dot(bh)+mtrain
    r2_is_H  = 1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
                

    v_H      = vip(bh,xtrain,ytrain,mtrain)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
                
    return r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H



def MC_simulation(MC = [1], datanum = 100, horizon = [1],model = [1]):
    
    ### hh is the horizon parameter, e.g. hh=1 means using monthly return as response variable. 
    ### hh=3 is quarterly, hh=6 is Half-year and hh=12 is annually.

    start = timeit.default_timer()
    write_dir()
    
    for hh in horizon:

        title = path+'/Simu_'+str(datanum)+'/Reg%d'%hh
        if datanum == 100:
            nump   = 50
        if datanum == 200:
            nump   = 100

        mu  = 0.2*np.sqrt(hh)
        tol = 1e-10

    
        ### Start to MC ###

        for M in MC:
            for mo in model:

                N       = 200      ### Number of CS tickers
                m       = nump*2   ### Number of Characteristics
                T       = 180      ### Number of Time Periods

                per     = np.tile(np.arange(N)+1,T)
                time    = np.repeat(np.arange(T)+1,N)
                stdv    = 0.05
                theta_w = 0.005

                c       = pd.read_csv(dirstock+'c%d.csv'%M,delimiter=',').values
                r1      = pd.read_csv(dirstock+'r%d_%d_%d.csv'%(mo,M,hh),
                                      delimiter=',').iloc[:,0].values

                ### Add Some Elements ###
                daylen       = np.repeat(N,T/3)
                daylen_test  = daylen
                ind          = range(0,(N*T//3))
                xtrain       = c[ind,:]
                ytrain       = r1[ind]
                ytrain       = ytrain.reshape(len(ytrain),1)
                trainper     = per[ind]
                ind          = range((N*T//3),(N*(T*2//3-hh+1)))
                xtest        = c[ind,:]
                ytest        = r1[ind]
                ytest        = ytest.reshape(len(ytest),1)
                testper      = per[ind]

                l1           = c.shape[0]
                l2           = len(r1)
                l3           = l2-np.sum(np.isnan(r1))
                print(l1,l2,l3)
                ind          = range((N*T*2//3),min(l1,l2,l3))
                xoos         = c[ind,:]
                yoos         = r1[ind]
                yoos         = yoos.reshape(len(yoos),1)
                del c
                del r1

                ### Demean Returns ### 
                ytrain_demean= ytrain-np.mean(ytrain)
                ytest_demean = ytest-np.mean(ytest)
                mtrain       = np.mean(ytrain)
                mtest        = np.mean(ytest)


                ### Calculate Sufficient Stats ###
                sd           = np.zeros(xtrain.shape[1])
                for i in range(xtrain.shape[1]):
                    s = np.std(xtrain[:,i])
                    if s > 0:
                        xtrain[:,i] = xtrain[:,i] /s
                        xtest[:,i]  = xtest[:,i]  /s
                        xoos[:,i]   = xoos[:,i]   /s
                        sd[i]       = s

                XX           = xtrain.T.dot(xtrain)
                b            = np.linalg.svd(XX)
                L            = b[1][0]
                print('Lasso L=',L)
                Y            = np.matrix(ytrain_demean)
                XY           = xtrain.T.dot(Y)


                ### Start to Train ###

                R2_oos       = np.zeros(13)  ### OOS R2
                R2_is        = np.zeros(13)   ### IS R2
             
                ### OLS ###
           
                modeln           = 0

                r2_oos,r2_is,b,v = OLS(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos)
                R2_oos[modeln]   = r2_oos
                R2_is[modeln]    = r2_is
                
                df               = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                df               = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                

                print('###Simple OLS OK!###')
                
                ### OLS+H


                modeln += 1
                r2_oos,r2_is,b,v = OLSH(xtrain,ytrain,ytrain_demean,mtrain,xoos,yoos,\
                                        mu,tol,L,soft_thresholdl)
                
                R2_oos[modeln]   = r2_oos
                R2_is[modeln]    = r2_is

                df               = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df               = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)


                print('###Simple OLS+H OK!###')
                


                ### PCA Regression ###
                ### Tuning parameter: the number of PCs

                modeln+=1
                ne               = 30
                
                r2_oos,r2_is,b,v = PCAR(xtrain,ytrain,ytrain_demean,\
                                        mtrain,xoos,yoos,xtest,ytest,ne)
                
                R2_oos[modeln]   = r2_oos    
                R2_is[modeln]    = r2_is

                
                df               = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df               = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###PCA Regression Good!###')
                
                

                ### PLS Regression ###
                ### Tuning parameter: the number of components

                modeln+=1
                ne               = 30
                r2_oos,r2_is,b,v = PLSR(xtrain,ytrain,ytrain_demean,\
                                        mtrain,xoos,yoos,xtest,ytest,ne)

                R2_oos[modeln]   = r2_oos    
                R2_is[modeln]    = r2_is  

                
                df               = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df               = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###PLS Regression Good!###')
                



                ### Lasso ###
                ### Tuning parameter: the L1 penalty lambda

                modeln+=1
                lamv           = sq(-2,4,0.1)
                alpha          = 1
                
                r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H = Lasso(XX,XY,xtrain,ytrain,\
                                                      ytrain_demean,mtrain,\
                                                      xoos,yoos,xtest,ytest,\
                                                      mu,tol,L,alpha,lamv)
                

                R2_oos[modeln] = r2_oos    
                R2_is[modeln]  = r2_is    
                
                df             = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df             = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###Lasso Good!###')

                

                
                modeln+=1
                R2_oos[modeln] = r2_oos_H
                
                R2_is[modeln]  = r2_is_H
                
                df             = pd.DataFrame(b_H)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df             = pd.DataFrame(v_H)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                print('###Lasso+H Good!###')




                
                ### Ridge ### 
                ### Tuning parameter: the L2 penalty lambda
                modeln+=1
                lamv            = sq(0,6,0.1)
                alpha           = 1

                r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H = ridge(XX,XY,xtrain,ytrain,\
                                                      ytrain_demean,mtrain,\
                                                      xoos,yoos,xtest,ytest,\
                                                      mu,tol,L,alpha,lamv)
                

                R2_oos[modeln]  = r2_oos    
                R2_is[modeln]   = r2_is

                df              = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df              = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###Ridge Good!###')

                modeln+=1

                R2_oos[modeln]  = r2_oos_H
                
                R2_is[modeln]   = r2_is_H

                df              = pd.DataFrame(b_H)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                df              = pd.DataFrame(v_H)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                print('###Ridge+H Good!###')




                
                ### Elastic Net ###
                ### Tuning parameter: the L1+L2 penalty lambda
                modeln+=1
                lamv           = sq(-2,4,0.1)
                alpha          = 0.5

                r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H = ridge(XX,XY,xtrain,ytrain,\
                                                      ytrain_demean,mtrain,\
                                                      xoos,yoos,xtest,ytest,\
                                                      mu,tol,L,alpha,lamv)

                R2_oos[modeln] = r2_oos    
                R2_is[modeln]  = r2_is 
                                
                df             = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df             = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###Enet Good!###')

                modeln+=1
                R2_oos[modeln] = r2_oos_H
                
                R2_is[modeln]  = r2_is_H
                                
                df             = pd.DataFrame(b_H)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                
                df             = pd.DataFrame(v_H)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
                print('###Enet+H Good!###')

                
                ### Oracle Models ###
                modeln+=1
                
                r2_oos,r2_is   = Oracle(mo,nump,xtrain,ytrain,mtrain,xoos,yoos)
                
                R2_oos[modeln] = r2_oos    
                R2_is[modeln]  = r2_is 

                
                print('###Oracle OLS!###')
                
                
                ### Group Lasso ###
                
                
                ### Tuning parameter: the group lasso penalty lambda

                kn             = 4 # the number of knots
                modeln        += 1
                lamv           = sq(0.5,3,0.1)
                r2_oos,r2_is,b,v,r2_oos_H,r2_is_H,b_H,v_H = Group_lasso(kn,xtrain,ytrain,\
                                                        ytrain_demean,mtrain,\
                                                        xoos,yoos,xtest,ytest,mu,tol,lamv)
                
                R2_oos[modeln] = r2_oos    
                R2_is[modeln]  = r2_is 
                                
                df             = pd.DataFrame(b)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                df             = pd.DataFrame(v)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###Group Lasso Good!###')

                modeln+=1
                
                R2_oos[modeln] = r2_oos_H
                
                R2_is[modeln]  = r2_is_H
                                
                df             = pd.DataFrame(b_H)
                df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                df             = pd.DataFrame(v_H)
                df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

                print('###Group Lasso+H Good!###')




                
                print(R2_oos)
                df   = pd.DataFrame(R2_oos)
                df.to_csv(title+'/roos_%d_%d.csv'%(mo,M),header=False, index=False)

                print(R2_is)
                df   = pd.DataFrame(R2_is)
                df.to_csv(title+'/ris_%d_%d.csv'%(mo,M),header=False, index=False)
                
                stop = timeit.default_timer()
                print('Time: ', stop - start)
