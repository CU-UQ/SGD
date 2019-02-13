#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file uses a linear regression example to show the use of  StochasticGradientDescent module.
Available classes:
    (1) Stochastic gradient descent
    (2) SGD with momentum
    (3) Nesterov accelerated SGD
    (4) AdaGrad
    (5) RMSprop
    (6) Adam
    (7) Adamax
    (8) Adadelta
    (9) Nadam
    (10) Stochastic average gradient
    (11) Mini-batch stochastic gradient descent
    (12) SVRG
    
Copyright (C) 2019  Subhayan De

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
Created on Mon Jul  9 21:19:43 2018
@author: Subhayan De (email: Subhayan.De@colorado.edu)
"""
# import matplotlib and numpy packages
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import the algorithm classes from the SGD module
import SGD as sgd

def main():
    # Generate data
    np.random.seed(0)
    n = 1000
    X = 2.0*np.random.rand(n,1)
    
    # parameters
    w1 = 3.0
    w2 = 4.5
    # noisy data
    y = w1 + w2 * X + np.random.randn(n,1)
    
    X_b = np.c_[np.ones((n,1)), X] # add 1 to each instance
    # save data and x to files to be used later to calculate objectives and gradients
    np.savetxt('test1_data.txt',y)
    np.savetxt('test1_x.txt',X_b)
    
    # select the algorithm to run
    # acceptable terms: SGD, SGDmomentum, SGDnesterov, AdaGrad, RMSprop, Adam, Adamax, Adadelta, Nadam, minibatchSGD, SAG, SVRG
    alg = 'Adam'
    
    # initial parameter
    w10 = 2.0
    w20 = 0.5
    theta = np.array([w10, w20])
    R = objFun(theta) # initial objective
    it = 0 # set iteration counter to 0
    maxIt = 2500 # maximum iteration
    dR = gradFun(theta) # initial gradient
    if alg == 'SGD':
        # Stochastic Gradient Descent
        eta = 0.0025 # learning rate
        opt = sgd.SGD(obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'SGDmomentum':
        # Stochastic Gradient Descent with momentum
        eta = 0.001 # learning rate
        opt = sgd.SGD(obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun, momentum = 0.9) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()    
    elif alg == 'SGDnesterov':
        # Stochastic Gradient Descent with Nesterov momentum
        eta = 0.001 # learning rate
        opt = sgd.SGD(obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun, momentum = 0.9,nesterov = True) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'AdaGrad':
        # AdaGrad
        eta = 0.25 # learning rate
        opt = sgd.AdaGrad(gradHist=0.0,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'RMSprop':
        # RMSprop
        eta = 0.9 # learning rate
        opt = sgd.RMSprop(gradHist=0.0,rho=0.1,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'Adam':
        # Adam
        eta = 0.025 # learning rate
        opt = sgd.Adam(m = 0.0,v = 0.0,beta1 = 0.9,beta2 = 0.999,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'Adamax':
        # Adamax
        eta = 0.025 # learning rate
        opt = sgd.Adamax(m = 0.0,u = 0.0,beta1 = 0.9,beta2 = 0.999,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'Adadelta':
        # Adadelta
        eta = 1.0 # learning rate
        opt = sgd.Adadelta(gradHist=0.0,updateHist=0.0,rho=0.99,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'Nadam':
        # Nadam
        eta = 0.01# learning rate
        opt = sgd.Nadam(m = 0.0,v = 0.0,beta1 = 0.9,beta2 = 0.999,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=gradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'minibatchSGD':
        # mini batch stochastic gradient descent
        eta = 0.025 # learning rate
        opt = sgd.minibatchSGD(nSamples = 10,nTotSamples = n,newGrad = 0.0,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=batchGradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'SAG':
        # stochastic average gradient descent
        eta = 0.0025 # learning rate
        opt = sgd.SAG(nSamples = 20,nTotSamples= n, obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=batchGradFun) # initialize
        opt.performIter() # perform iterations
        thetaHist = opt.getParamHist()
    elif alg == 'SVRG':
        # stochastic variance reduced gradient descent
        eta = 0.004
        opt = sgd.SVRG(nTotSamples = n, innerIter = 10, outerIter = 200, option = 1,obj = R, grad = dR, eta = eta, param = theta, iter = it, maxiter=maxIt, objFun=objFun, gradFun=batchGradFun)
        opt.performOuterIter()
        thetaHist = opt.getParamHist()
    else:
        raise ValueError('No such algorithm is in the module.\n Please use one of the following options:\nSGD, SGDmomentum, SGDnesterov, AdaGrad, RMSprop, Adam, Adamax, Adadelta, Nadam, minibatchSGD, SAG, SVRG')
        
    
    # Plot the results
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    delta = 0.025
    w1 = np.arange(-2.0, 10.0, delta)
    w2 = np.arange(-2.0, 10.0, delta)
    Xx, Yy = np.meshgrid(w1, w2)
    nx = np.shape(Xx)
    Z = np.zeros(nx)
    for i in range(nx[0]):
        for j in range(nx[1]):
            Z[i,j] = (np.linalg.norm(y - Xx[i,j]-Yy[i,j]*X,2))**2/n
            
    plt.figure()
    levels = np.arange(0, 40, 4)
    CS = plt.contour(Xx, Yy, Z, levels,origin='lower',
                 linewidths=2,
                 extent=(-2, 10, -2, 10))
    #plt.clabel(CS, inline=1, fontsize=10)
    # Thicken the zero contour.
    zc = CS.collections[6]
    plt.setp(zc, linewidth=4)

    plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=10)
    im = plt.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.Wistia, extent=(-2, 10, -2, 10))

    # make a colorbar
    plt.colorbar(im, shrink=0.8, extend='both')
    plt.plot(thetaHist[0,:], thetaHist[1,:],'r.',linewidth = 6)
    titl = opt.alg+' with a learning rate '+str(eta)
    plt.title(titl)
    return opt

def objFun(param):
    # objective function
    y = np.loadtxt('test1_data.txt')
    X_b = np.loadtxt('test1_x.txt')
    n = np.size(y)
    yprime = X_b.dot(param)
    obj = np.sum(np.multiply(y-yprime,y-yprime))/n
    return obj

def gradFun(param):
    # gradient function
    y = np.loadtxt('test1_data.txt')
    X_b = np.loadtxt('test1_x.txt')
    n = np.size(y)
    nprime = np.random.randint(n)
    xi = X_b[nprime:nprime+1]
    yi = y[nprime:nprime+1]
    grad = 2.0 * xi.T.dot(xi.dot(param) - yi)
    return grad

def batchGradFun(param,nBatch):
    # batch gradient function
    y = np.loadtxt('test1_data.txt')
    X_b = np.loadtxt('test1_x.txt')
    n = np.size(y)
    nParam = np.size(param)
    batchGrad = np.zeros((nParam,nBatch))
    nprime = np.random.choice(range(n), nBatch, replace = False)
    for i in range(nBatch):
        xi = X_b[nprime[i]:nprime[i]+1]
        yi = y[nprime[i]:nprime[i]+1]
        batchGrad[:,i] = 2.0 * xi.T.dot(xi.dot(param) - yi)
    return batchGrad,nprime

if __name__ == "__main__":
    opt = main()
