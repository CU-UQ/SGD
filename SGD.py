#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the class file that implements: 
(i) Stochastic Gradient Descent, 
(ii) SGD with Momentum,
(iii) NAG,
(iv) AdaGrad, 
(iv) RMSprop,
(vi) Adam, 
(vii) Adamax,
(viii) Adadelta,
(ix) Nadam, 
(x) SAG, 
(xi) minibatch SGD, 
(xii) SVRG.

NOTE: Currently, the stopping conditions are maximum number of iteration and 2nd norm of gradient vector.
Time-delay and exponential learnong schedules are implemented.

Created on Sat Jun 30 01:04:28 2018
@author: Subhayan De

No parts of this can be reproduced or reused without prior written permission from the author.

Author's note:  add kSGD, 2nd order methods
"""

import numpy as np
import time

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
        Call in a loop to create terminal progress bar
        parameters:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class SGD(object):
    """ 
    ==============================================================================
    |                     Stochastic Gradient Descent class                      |
    ==============================================================================
    Initialization:
        sgd = SGD(obj, grad, eta, param, iter, maxIter, objFun, gradFun, 
                  lowerBound, upperBound, stopGrad, momentum, nesterov, 
                  learnSched, lrParam)
        
    NOTE: To perform just one iteration provide either grad or gradFn. 
          obj  or objFn are optional.
    ==============================================================================
    Attributes:
        obj:        objective (optional input)
        grad:       Gradient information 
                    (array of dimension nParam-by-1, optional input)
        eta:        learning rate ( = 1.0, default)
        param:      the parameter vector (array of dimension nParam-by-1)
        nParam:     number of parameters
        iter:       iteration number
        maxIter:    maximum iteration number (optional, default = 1)
        objFun:     function handle to evaluate the objective 
                    (not required for maxit = 1 )
        gradFun:    function handle to evaluate the gradient 
                    (not required for maxit = 1 )
        lowerBound: lower bound for the parameters (optional input)
        upperBound: upper bound for the parameters (optional input)
        paramHist:  parameter evolution history
        stopGrad:   stopping criterion based on 2-norm of gradient vector
        momentum:   momentum parameter (default = 0)
        nesterov:   set to True if Nesterov momentum equation to be used 
                    (default = False)
        learnSched: learning schedule (constant, exponential or time-based, 
                                       default = constant)
        lrParam:    learning schedule parameter (default =0.1)
        alg:        algorithm used
        __version__:version of the code
    ==============================================================================
    Methods:
     Public:
        getParam:       returns the parameter values
        getObj:         returns the current objective value
        getGrad:        returns the current gradient information
        update:         perform a single iteration
        performIter:    perform maxIter number of iterations
        getParamHist:   returns parameter update history
     Private:
        __init___:          initialization
        evaluateObjFn:      evaluates the objective function
        evaluateGradFn:     evaluates the gradients
        satisfyBounds:      satisfies the parameter bounds
        learningSchedule:   learning schedule
        stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Bottou, Léon, Frank E. Curtis, and Jorge Nocedal. 
    "Optimization methods for large-scale machine learning." 
    SIAM Review 60.2 (2018): 223-311.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """ 
    def __init__(self,**kwargs):
        allowed_kwargs = {'obj', 'grad', 'param', 'eta', 'iter', 'maxiter', 'objFun', 'gradFun', 'lowerBound', 'upperBound', 'oldGrad', 'stopGrad', 'momentum', 'nesterov','learnSched', 'lrParam'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed to optimizer at: ' + str(k))

        self.__dict__.update(kwargs)
        self.nParam = np.size(self.param)
        # Checks and setting default values
        # Iteration numbers
        if hasattr(self,'iter') == False:
            self.iter = 0 # set the iteration number
        self.currentIter = self.iter
        # stopping criteria
        # max iteration no.
        if hasattr(self,'maxiter') == False:
            self.maxiter = 1 # set the default max iteration number
        # minimum gradient
        if hasattr(self,'stopGrad') == False:
            self.stopGrad = 1e-6
        # Parameter values
        if hasattr(self,'param') == False:
            raise ValueError('Parameter vector is missing')
        # Gradient information
        if hasattr(self,'grad') == False:
            print('No gradient information provided at iteration: 1')
            if hasattr(self,'gradFun') == False:
                raise ValueError('Please provide the gradient function')
        elif np.size(self.grad) != self.nParam:
            raise ValueError('Gradient dimension mismatch')
        if self.maxiter > 1 and hasattr(self,'gradFun') == False:
            raise ValueError('Please provide the gradient function')
        # Objective values
        if hasattr(self,'objFun') == False and self.maxiter > 1:
            raise ValueError('Please provide the objective function')
        if hasattr(self,'obj') == False:
            self.obj = np.array([])
            if hasattr(self,'objFun'):
                self.evaluateObjFn(self)
        else:
            self.obj = np.array([self.obj])
        # Learning rate
        if hasattr(self,'eta') == False:
            self.eta = 1.0
            print('*NOTE: No learning rate provided, assumed as 1.0')
        else:
            print('Learning rate = ',self.eta,'\n')    
        if hasattr(self,'lowerBound') == False:
            self.lowerBound = -np.inf*np.ones(self.nParam)
        elif np.size(self.lowerBound) == 1:
            self.lowerBound = self.lowerBound*np.ones(self.nParam)
        else:
            raise ValueError('parameter lower bound dimension mismatch')
        # Set the upper bounds
        if hasattr(self,'upperBound') == False:
            self.upperBound = np.inf*np.ones(self.nParam)
        elif np.size(self.upperBound) == 1:
            self.upperBound = self.upperBound*np.ones(self.nParam)
        else:
            raise ValueError('parameter upper bound dimension mismatch')
        # Momentum
        #self.alg = 'SGD with Momentum'
        if hasattr(self,'alg') == False:
            self.alg = 'SGD+momentum'
            if hasattr(self,'momentum') == False:
                self.alg = 'SGD'
                self.momentum = 0.0;
        self.paramHist = np.reshape(self.param,(2,1))
        self.__version__ = '0.0.1'
        self.stop = False
        self.updateParam = np.zeros(self.nParam)
        # Nesterov momentum
        if hasattr(self, 'nesterov'):
            if self.nesterov == True:
                self.alg = 'SGD+Nesterov momentum'
                if hasattr(self,'gradFun') == False:
                    raise ValueError('provide gradient function information with Nesterov')
        else:
            self.nesterov = False
        # learning schedule
        if hasattr(self,'learnSched') == False:
            self.learnSched = 'constant'
        elif self.learnSched != 'exponential' and self.learnSched != 'time-based':
            print('no such learning schedule in this module\nSet to constant')
            self.learnSched = 'constant'
        elif hasattr(self,'lrParam') == False:
            self.lrParam = 0.1
        print('Learning schedule: ',self.learnSched)
            
        
    def __version__(self):
        """
        version of the code
        """
        print(self.__version__)
        
    def getParam(self):
        """
        To get the next parameter values
        """
        print(self.nParam,'parameters have been updated!\n')
        return self.param
    
    def getObj(self):
        """
        To get the current objective (if possible)
        """
        self.evaluateObjFn()
        return self.obj
    
    def getGrad(self):
        """
        To get the gradients
        """
        return self.grad
    
    def getParamHist(self):
        """
        To get parameter history
        """
        return self.paramHist
    
    def evaluateObjFn(self):
        """
        This evalutes the objective function
        objFun should be a function handle with input: param, output: objective
        """
        if not self.obj.any():
            print('No objective information provided to SGD')
        else:
            self.obj = np.append(self.obj,self.objFun(self.param))
            #print('Current objective value: ', self.obj[self.currentIter],'\n')
    
    def evaluateGradFn(self):
        """
        This evalutes the gradient function for i-th data point, where i in [0, n]
        gradFun should be a function handle with input: param, output: gradient
        """
        self.grad = self.gradFun(self.param)
        
    def satisfyBounds(self):
        """
        This satisfies the parameter bounds (if any)
        """
        # Set the lower bounds
        #print(self.lowerBound)
        
        # Satisfy the bounds
        for i in range(self.nParam):
            if self.param[i] > self.upperBound[i]:
                self.param[i] = self.upperBound[i]
            elif self.param[i] < self.lowerBound[i]:
                self.param[i] = self.lowerBound[i]
                
    def update(self):
        """
        Perform one iteration of SGD
        """
        # Perform one iteration of SGD
        SGD.learningSchedule(self)
        if self.nesterov == True:
            grdnt = self.gradFun(self.param - self.momentum*self.updateParam)
            self.updateParam = self.updateParam*self.momentum + self.etaCurrent*grdnt
        else:
            self.updateParam = self.updateParam*self.momentum + self.etaCurrent*self.grad
        self.param=self.param - self.updateParam
        #self.param=self.param - self.eta*self.grad
        # satisfy the parameter bounds
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of Stochatsic Gradient Descent has been performed successfully!\n')
        
    def performIter(self):
        """
        Performs all the iterations of SGD
        """
        SGD.printAlg(self)
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
        #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
    
    def stopCrit(self):
        """
        Checks stopping criteria
        """
        if self.grad.ndim >1:
            self.avgGrad = np.mean(self.grad,axis =1)
            if np.linalg.norm(self.avgGrad)<self.stopGrad:
                self.stop = True
        elif np.linalg.norm(self.grad)<self.stopGrad:
            self.stop = True
            
    def learningSchedule(self):
        """
        creates a learning schedule for SGD
        """
        if self.learnSched == 'constant':
            self.etaCurrent =self.eta # no change        
        elif self.learnSched == 'exponential':
            self.etaCurrent = self.eta*np.exp(-self.lrParam*self.currentIter)
            print(self.etaCurrent)
        elif self.learnSched == 'time-based':
            self.etaCurrent = self.eta/(1.0+self.lrParam*self.currentIter)
    
    def printAlg(self):
        """
        prints algorithm
        """
        print('\nAlgorithm: ',self.alg,'\n')
        
    def printProgress(self):
        # Update Progress Bar
        if hasattr(self,'outerIter'):
            printProgressBar(self.currentIter, self.outerIter, prefix = self.alg, suffix = ('Complete: Time Elapsed = '+str(np.around(time.clock()-self.t,decimals=2))+'s'+', Objective = '+str(np.around(self.obj[self.currentIter-1],decimals=6))+'    '), length = 25)
        else:
            printProgressBar(self.currentIter, self.maxiter, prefix = self.alg, suffix = ('Complete: Time Elapsed = '+str(np.around(time.clock()-self.t,decimals=2))+'s'+', Objective = '+str(np.around(self.obj[self.currentIter-1],decimals=6))+'    '), length = 25)


class AdaGrad(SGD):
    """
    ==============================================================================
    |                Adaptive Subgradient Method (AdaGrad) class                 |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        adg = AdaGrad(gradHist, obj, grad, eta, param, 
                      iter, maxIter, objFun, gradFun, lowerBound, upperBound)
        
    NOTE: gradHist:     historical information of gradients 
                        (array of dimension nparam-by-1).
                        This should equal to zero for 1st iteration
    ==============================================================================
    Attributes: 
        obj:            Initial objective value (optional input)
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate ( = 1.0, default)
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        gradHist:       sum of gradient history (see the algorithm)
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        iter:           iteration number (optional input)
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:performs all the iterations inside a for loop
        getGradHist:returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history

     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of AdaGrad
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Duchi, John, Elad Hazan, and Yoram Singer. 
    "Adaptive subgradient methods for online learning and stochastic optimization." 
    Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,gradHist=0.0,**kwargs):
    #def __init__(self,grad,learningRate,param,nParam,gradHist):
        """ Initialize the AdaGrad class object. 
            This can be used to perform one iteration of AdaGrad. 
        """
        self.alg = 'AdaGrad'
        SGD.printAlg(self)
        #SGD.__init__(self,grad,learningRate,param,nParam)
        SGD.__init__(self,**kwargs)
        self.epsilon=np.finfo(float).eps # The machine precision
        if np.sum(gradHist) != 0.0:
            self.gradHist=np.reshape(gradHist,(self.nParam))
        else:
            self.gradHist = np.zeros(self.nParam)
        
    def update(self):
        """
        Perform one iteration of AdaGrad
        """
        SGD.learningSchedule(self)
        self.gradHist += np.multiply(self.grad,self.grad); # Sum of gradient history
        # Perform one iteration of AdaGrad
        self.param=self.param - np.divide((self.etaCurrent*self.grad),(np.sqrt(self.gradHist)+self.epsilon))
        # satisfy the parameter bounds
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of AdaGrad has been performed successfully!\n')
        
    def performIter(self):
        """
        Performs all the iterations of AdaGrad
        """
        
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
                
    def getGradHist(self):
        """
        Returns accumulated gradient history
        """
        return self.gradHist
    
class RMSprop(SGD):
    """
    ==============================================================================
    |                               RMSprop class                                |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        rp = RMSprop(gradHist, updatehist, rho, obj, grad, eta, param, 
                       iter, maxIter, objFun, gradFun, lowerBound, upperBound)
        NOTE: gradHist: historical information of gradients 
                        (array of dimension nparam-by-1)
                        this should equal to zero for 1st iteration
    ==============================================================================
    Attributes: 
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate = 1 by default
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        gradHist:       gradient history accumulator (see the algorithm)
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        rho:            exponential decay rate (0.95 may be a good choice)
        iter:           iteration number (optional)
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:performs all the iterations inside a for loop
        getGradHist:returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adadelta
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Geoffrey 	Hinton 
    "rmsprop: Divide the gradient by a running average of its recent magnitude." 
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,gradHist=0.0,rho=0.9,**kwargs):
        """ Initialize the Adadelta class object. 
            This can be used to perform one iteration of Adadelta. 
        """
        self.alg = 'RMSprop'
        SGD.printAlg(self)
        SGD.__init__(self,**kwargs)
        self.epsilon=np.finfo(float).eps # The machine precision
        # Initialize gradient history
        if np.sum(gradHist) != 0.0:
            if np.size(gradHist) != self.nParam:
                raise ValueError('Gradient history dimension mismatch')
            else:
                self.gradHist=np.reshape(gradHist,(self.nParam))
        else:
            self.gradHist = np.zeros(self.nParam)
        # Initialize rho
        self.rho = rho
        
    def update(self):
        """
        Perform one iteration of RMSprop
        """
        # update gradient history acccumulator
        SGD.learningSchedule(self)
        self.gradHist+=self.rho*self.gradHist+(1.0-self.rho)*np.multiply(self.grad,self.grad); # Sum of gradient history
        # Perform one iteration of RMSprop
        RMSg = np.sqrt(self.gradHist)+self.epsilon
        updateParam = ((np.divide(self.grad,RMSg)))
        self.param=self.param-self.etaCurrent*updateParam
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of RMSprop has been performed successfully!\n')
        
    def performIter(self):
        """
        Performs all the iterations of RMSprop
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
        
    def getGradHist(self):
        """
        This returns the gradient history
        """
        return self.gradHist
    
class Adam(SGD):
    """
    ==============================================================================
    |                   Adaptive moment estimation (Adam) class                  |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        adm = Adam(m, v, beta1, beta2, obj, grad, eta, param, 
                   iter, maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes: 
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        beta1, beta2:   exponential decay rates in [0,1) 
                        (default beta1 = 0.9, beta2 = 0.999)
        m:              First moment (array of dimension nParam-by-1)
        v:              Second raw moment (array of dimension nParam-by-1)
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        iter:           iteration number
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:    performs all the iterations inside a for loop
        getGradHist:    returns gradient history (default is zero)
        getMoments:     returns history of moments
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adam
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Kingma, Diederik P., and Jimmy Ba. 
    "Adam: A method for stochastic optimization." 
    arXiv preprint arXiv:1412.6980 (2014).
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,m = 0.0,v = 0.0,beta1 = 0.9,beta2 = 0.99,**kwargs):
#    def __init__(self,grad,learningRate,parameters,numParam,gradHist,beta1,beta2):
        """ Initialize the adagrad class object. 
        This can be used to perform one iteration of Adam. 
        """
        self.alg = 'Adam'
        SGD.printAlg(self)
        self.beta1 = beta1 # decay rate (beta1 = 0.9 is a good suggestion)
        self.beta2 = beta2 # decay rate (beta2 = 0.999 is a good suggetion)
        self.epsilon=np.finfo(float).eps # The machine precision
        SGD.__init__(self,**kwargs)
        # Initialize first moment
        if np.sum(m) != 0.0:
            if np.size(m) != self.nParam:
                raise ValueError('First moment dimension mismatch')
            else:
                self.m=np.reshape(m,(self.nParam))
        else:
            self.m = np.zeros(self.nParam)
        # Initialize second raw moment
        if np.sum(v) != 0.0:
            if np.size(v) != self.nParam:
                raise ValueError('Second raw moment dimension mismatch')
            else:
                self.v=np.reshape(v,(self.nParam))
        else:
            self.v = np.zeros(self.nParam)
        
    def update(self):
        """ Perform one iteration of Adam
        """
        SGD.learningSchedule(self)
        # Moment updates
        self.m = self.beta1*self.m + (1.0-self.beta1)*self.grad # Update biased first moment estimate
        self.mHat = self.m/(1.0-self.beta1**(self.currentIter+1)) # Compute bias-corrected first moment estimate
        #print(self.mHat)
        self.v = self.beta2*self.v + (1.0-self.beta2)*np.multiply(self.grad,self.grad) # Update biased second moment estimate
        self.vHat = self.v/(1.0-self.beta2**(self.currentIter+1)) # Compute bias-corrected second moment estimate
        # Parameter updates
        self.param = self.param - np.divide((self.etaCurrent*self.mHat),(np.sqrt(self.vHat))+self.epsilon)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of Adam has been performed successfully!\n')   
        
    def performIter(self):
        """
        Performs all the iterations of Adam
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
        
    def getMoments(self):
        """
        This returns the updated moments
        """
        return self.m, self.v
    
class Adamax(SGD):
    """
    ==============================================================================
    |                  Adaptive moment estimation (Adamax) class                 |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        admx = Adamax(m, v, beta1, beta2, obj, grad, eta, param, 
                   iter, maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes: (all private)
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        beta1, beta2:   exponential decay rates in [0,1) 
                        (default beta1 = 0.9, beta2 = 0.999)
        m:              First moment (array of dimension nParam-by-1)
        u:              infinity norm constrained second moment 
                        (array of dimension nParam-by-1)
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        iter:           iteration number
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:    performs all the iterations inside a for loop
        getGradHist:    returns gradient history (default is zero)
        getMoments:     returns history of moments
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adam
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Kingma, Diederik P., and Jimmy Ba. 
    "Adam: A method for stochastic optimization." 
    arXiv preprint arXiv:1412.6980 (2014).
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,m = 0.0,u = 0.0,beta1 = 0.9,beta2 = 0.99,**kwargs):
#    def __init__(self,grad,learningRate,parameters,numParam,gradHist,beta1,beta2):
        """ Initialize the adagrad class object. 
        This can be used to perform one iteration of Adamax. 
        """
        self.alg = 'Adamax'
        SGD.printAlg(self)
        self.beta1 = beta1 # decay rate (beta1 = 0.9 is a good suggestion)
        self.beta2 = beta2 # decay rate (beta2 = 0.999 is a good suggetion)
        self.epsilon=np.finfo(float).eps # The machine precision
        SGD.__init__(self,**kwargs)
        # Initialize first moment
        if np.sum(m) != 0.0:
            if np.size(m) != self.nParam:
                raise ValueError('First moment dimension mismatch')
            else:
                self.m=np.reshape(m,(self.nParam))
        else:
            self.m = np.zeros(self.nParam)
        # Initialize second raw moment
        if np.sum(u) != 0.0:
            if np.size(u) != self.nParam:
                raise ValueError('Second raw moment dimension mismatch')
            else:
                self.u=np.reshape(u,(self.nParam))
        else:
            self.u = np.zeros(self.nParam)
        
    def update(self):
        """ Perform one iteration of Adamax
        """
        SGD.learningSchedule(self)
        # Moment updates
        self.m = self.beta1*self.m + (1.0-self.beta1)*self.grad # Update biased first moment estimate
        self.mHat = self.m/(1.0-self.beta1**(self.currentIter+1)) # Compute bias-corrected first moment estimate
        self.u = np.maximum(self.beta2*self.u,np.abs(self.grad))
#        self.v = self.beta2*self.v + (1.0-self.beta2)*np.multiply(self.grad,self.grad) # Update biased second moment estimate
        # Parameter updates
        self.param = self.param - np.divide((self.etaCurrent*self.mHat),self.u)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of Adamax has been performed successfully!\n')   
        
    def performIter(self):
        """
        Performs all the iterations of Adamax
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
        
    def getMoments(self):
        """
        This returns the updated moments
        """
        return self.m, self.v
        
class Adadelta(SGD):
    """
    ==============================================================================
    |                               ADADELTA class                               |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        add = Adadelta(gradHist, updatehist, rho, obj, grad, eta, param, 
                       iter, maxIter, objFun, gradFun, lowerBound, upperBound)
        NOTE: gradHist: historical information of gradients 
                        (array of dimension nparam-by-1)
                        this should equal to zero for 1st iteration
    ==============================================================================
    Attributes: (all private)
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate = 1 by default
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        gradHist:       gradient history accumulator (see the algorithm)
        updateHist:     parameter update history accumulator
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        rho:            exponential decay rate (0.95 may be a good choice)
        iter:           iteration number (optional)
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:performs all the iterations inside a for loop
        getGradHist:returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adadelta
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Zeiler, Matthew D. 
    "Adadelta: an adaptive learning rate method." 
    arXiv preprint arXiv:1212.5701 (2012).
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,gradHist=0.0,updateHist=0.0,rho=0.95,**kwargs):
        """ Initialize the Adadelta class object. 
            This can be used to perform one iteration of Adadelta. 
        """
        self.alg = 'Adadelta'
        SGD.printAlg(self)
        SGD.__init__(self,**kwargs)
        self.epsilon=np.finfo(float).eps # The machine precision
        # Initialize gradient history
        if np.sum(gradHist) != 0.0:
            if np.size(gradHist) != self.nParam:
                raise ValueError('Gradient history dimension mismatch')
            else:
                self.gradHist=np.reshape(gradHist,(self.nParam))
        else:
            self.gradHist = np.zeros(self.nParam)
        # Initialize parameter history
        if np.sum(updateHist) != 0.0:
            if np.size(updateHist) != self.nParam:
                raise ValueError('Gradient history dimension mismatch')
            else:
                self.updateHist=np.reshape(updateHist,(self.nParam))
        else:
            self.updateHist = np.zeros(self.nParam)
        # Initialize rho
        self.rho = rho
        # Set eta to 1.0
        if self.eta!=1.0:
            print('Learning rate = ',self.eta,'!= 1.0\nSo, the learning rate is set to 1.0\n')
        self.eta = 1.0
        
    def update(self):
        """
        Perform one iteration of Adadelta
        """
        self.epsilon = 1e-6
        if self.currentIter<200:
            self.epsilon = 0.1
        else:
            self.epsilon = 1e-6
        SGD.learningSchedule(self)
        # update gradient history acccumulator
        self.gradHist+=self.rho*self.gradHist+(1.0-self.rho)*np.multiply(self.grad,self.grad); # Sum of gradient history
        # Perform one iteration of Adadelta
        RMSdx = np.sqrt(self.updateHist)+self.epsilon
        RMSg = np.sqrt(self.gradHist)+self.epsilon
        updateParam = np.multiply((np.divide(RMSdx,RMSg)),self.grad)
        self.param=self.param-self.etaCurrent*updateParam
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of Adadelta has been performed successfully!\n')
        # update parameter history accumulator
        self.updateHist = self.rho*self.updateHist+(1.0-self.rho)*np.multiply(updateParam,updateParam)
        
    def performIter(self):
        """
        Performs all the iterations of Adadelta
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
        
    def getGradHist(self):
        """
        This returns the gradient history
        """
        return self.gradHist
    
    def getUpdateHist(self):
        """
        This returns the parameter update history
        """
        self.updateHist
        
class Nadam(SGD):
    """
    ==============================================================================
    |         Nesterov-accelerated Adaptive moment estimation (Nadam) class      |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        nadm = Nadam(m, v, beta1, beta2, obj, grad, eta, param, iter, 
                     maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes: (all private)
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        beta1, beta2:   exponential decay rates in [0,1) 
                        (default beta1 = 0.9, beta2 = 0.999)
        m:              First moment (array of dimension nParam-by-1)
        v:              Second raw moment (array of dimension nParam-by-1)
        epsilon:        square-root of machine-precision 
                        (required to avoid division by zero)
        iter:           iteration number
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:    performs all the iterations inside a for loop
        getGradHist:    returns gradient history (default is zero)
        getMoments:     returns history of moments
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adam
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Timothy Dozat. 
      "Incorporating Nesterov Momentum into Adam". 
       ICLR Workshop, (1):2013–2016, 2016.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,m = 0.0,v = 0.0,beta1 = 0.9,beta2 = 0.99,**kwargs):
#    def __init__(self,grad,learningRate,parameters,numParam,gradHist,beta1,beta2):
        """ Initialize the adagrad class object. 
        This can be used to perform one iteration of Adam. 
        """
        self.alg = 'Nadam'
        SGD.printAlg(self)
        self.beta1 = beta1 # decay rate (beta1 = 0.9 is a good suggestion)
        self.beta2 = beta2 # decay rate (beta2 = 0.999 is a good suggetion)
        self.epsilon=np.finfo(float).eps # The machine precision
        SGD.__init__(self,**kwargs)
        # Initialize first moment
        if np.sum(m) != 0.0:
            if np.size(m) != self.nParam:
                raise ValueError('First moment dimension mismatch')
            else:
                self.m=np.reshape(m,(self.nParam))
        else:
            self.m = np.zeros(self.nParam)
        # Initialize second raw moment
        if np.sum(v) != 0.0:
            if np.size(v) != self.nParam:
                raise ValueError('Second raw moment dimension mismatch')
            else:
                self.v=np.reshape(v,(self.nParam))
        else:
            self.v = np.zeros(self.nParam)
        
        
    def update(self):
        """ 
        Perform one iteration of Nadam
        """
        SGD.learningSchedule(self)
        # Moment updates
        self.m = self.beta1*self.m + (1.0-self.beta1)*self.grad # Update biased first moment estimate
        self.mHat = self.m/(1.0-self.beta1**(self.currentIter+1)) # Compute bias-corrected first moment estimate
        self.v = self.beta2*self.v + (1.0-self.beta2)*np.multiply(self.grad,self.grad) # Update biased second moment estimate
        self.vHat = self.v/(1.0-self.beta2**(self.currentIter+1)) # Compute bias-corrected second moment estimate
        # Parameter updates
        mHat2 = self.beta1*self.mHat+(1.0-self.beta1)*self.grad/(1.0-self.beta1**(self.currentIter+1))
        self.param = self.param - np.divide((self.etaCurrent*mHat2),(np.sqrt(self.vHat))+self.epsilon)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of Nadam has been performed successfully!\n')   
        
    def performIter(self):
        """
        Performs all the iterations of Nadam
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)
        
    def getMoments(self):
        """
        This returns the updated moments
        """
        return self.m, self.v
        
class SAG(SGD):
    """
    ==============================================================================
    |                   Stochastic Average Gradient (SAG) class                  |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        sag = SAG(nSamples, nTotSamples, fullGrad = 0.0, obj, grad, eta, param, 
                  iter, maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes: (all private)
        fullGrad:           Full gradient information 
                        (array of dimension nParam-by-nTotSamples)
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        nTotSamples:    total number of samples
        nSamples:       number of gradients updated at each iteration
        iter:           iteration number (optional)
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        learnSched:     learning schedule (constant, exponential or time-based, 
                                       default = constant)
        lrParam:        learning schedule parameter (default =0.1)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:performs all the iterations inside a for loop
        getGradHist:returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of SAG
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Roux, Nicolas L., Mark Schmidt, and Francis R. Bach. 
    "A stochastic gradient method with an exponential convergence rate 
     for finite training sets." 
    Advances in neural information processing systems. 2012.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,nSamples,nTotSamples,fullGrad =0.0,**kwargs):
        """ Initialize the SAG class object. 
            This can be used to perform one iteration of SAG. 
        """
        self.alg = 'SAG'
        SGD.printAlg(self)
        grad = fullGrad
        SGD.__init__(self,**kwargs)
        # Assign total number of samples
        if type(nTotSamples) != int:
            raise TypeError('nSamples not an integer value')
        else:
            self.nTotSamples = nTotSamples
        # Assign number of samples to be replaced at each iteration
        if type(nSamples) != int:
            raise TypeError('nSamples not an integer value')
        else:
            self.nSamples = nSamples
        # Initialize gradients
        if np.sum(fullGrad) != 0:
            if np.size(fullGrad)/nTotSamples != self.nParam:
                raise ValueError('Full gradient dimension mismatch')
            else:
                fullGrad = np.reshape(fullGrad,(self.nParam,nTotSamples))
        else:
            self.fullGrad = np.zeros((self.nParam,self.nTotSamples))
            try: 
                self.gradFun
            except NameError: 
                print('Please provide gradient function name')
            self.fullGrad, nprime = self.gradFun(self.param,self.nTotSamples)
        self.grad = self.fullGrad

    def update(self):
        """
        Perform one iteration of SAG
        """
        if hasattr(self,'gradFun'):
            batchGrad,nprime = self.gradFun(self.param,self.nSamples)
        else:
            nprime = np.random.choice(range(self.nTotSamples), self.nSamples, replace = False)
            batchGrad = self.fullGrad[:,nprime]
        # Perform one iteration of SAG
        for i in range(self.nSamples):
            #self.evaluateGradFn()
            self.fullGrad[:,nprime[i]] = batchGrad[:,i]
        
        SGD.learningSchedule(self)
        self.param=self.param-self.etaCurrent*np.mean(self.fullGrad,1)
        #print(np.mean(self.fullGrad,1),self.param)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of SAG has been performed successfully!\n')
        
    def performIter(self):
        """
        Performs all the iterations of SAG
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.stopCrit(self)
                
                
class minibatchSGD(SGD):
    """
    ==============================================================================
    |                           minibatch SGD class                              |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization: 
        mbsgd = minibatchSGD(nSamples, nTotSamples,newGrad = 0.0,
                              obj, grad, eta, param, iter, maxiter, 
                              objFun, gradFun, lowerBound, upperBound)
        
    ==============================================================================
    Attributes:
        alg:            minibatchSGD
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        newGrad:        gradient information 
                        (array of dimension nParam-by-nSamples)
        nSamples:       number of gradients updated at each iteration
        iter:           iteration number (optional)
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        lowerBound:     lower bound for the parameters (optionalinput)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        learnSched:     learning schedule (constant, exponential or time-based, 
                                       default = constant)
        lrParam:        learning schedule parameter (default =0.1)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:        performs all the iterations inside a for loop
        getGradHist:        returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of minibatch SGD
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: 
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,nSamples,nTotSamples = np.inf,newGrad = 0.0,**kwargs):
        """ Initialize the minibatch SGD class object. 
            This can be used to perform one iteration of minibatch SGD. 
        """
        self.alg = 'minibatchSGD'
        SGD.printAlg(self)
        self.grad = newGrad
        SGD.__init__(self,**kwargs)
        # Assign number of samples used at each iteration
        if type(nSamples) != int:
            raise TypeError('nSamples not an integer value')
        else:
            self.nSamples = nSamples
        # Total number of samples
        if type(nTotSamples) != int:
            raise TypeError('nTotSamples not an integer value')
        else:
            self.nTotSamples = nTotSamples
        # Check for total number of samples
        if nTotSamples < nSamples:
            print('nTotSamples can not be smaller that nSamples\n')
            print('nTotSamples = nSamples is set\n')
            print('NOTE: performing a batch gradient descent')
        elif nTotSamples == nSamples:
            print('NOTE: performing a batch gradient descent')
        elif nTotSamples < np.inf:
            print('NOTE: performing a minibatch SGD with ', nSamples/nTotSamples*100, '% of total samples')
        else:
            print('NOTE: performing a minibatch SGD with ', nSamples, ' samples')
        # Initialize new gradients
        if np.sum(newGrad) != 0.0:
            if np.size(newGrad)/nSamples != self.nParam:
                raise ValueError('New gradient dimension mismatch')
            else:
                self.newGrad=np.reshape(newGrad,(self.nParam))
        else:
            self.newGrad = np.zeros((self.nParam,self.nSamples))
            try: 
                self.gradFun
            except NameError: 
                print('Please provide gradient function name')
            self.newGrad, nprime = self.gradFun(self.param,self.nSamples)

    def update(self):
        """
        Perform one iteration of minibatch SGD
        """
        SGD.learningSchedule(self)
        if self.maxiter>1:
            self.newGrad,nprime = self.gradFun(self.param,self.nSamples)
        # Perform one iteration of minibatch SGD
        self.param=self.param-self.etaCurrent*np.mean(self.newGrad,1)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        #print('One iteration of minibatch SGD has been performed successfully!\n')
        
    def performIter(self):
        """
        Performs all the iterations of minibatch SGD
        """
        # initialize progress bar
        printProgressBar(0, self.maxiter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        for i in range(self.iter,self.maxiter,1):
            if self.stop == True:
                break
            #print('iteration', i+1, 'out of', self.maxiter)
            self.update()
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)       
                SGD.stopCrit(self)
        
class SVRG(SGD):
    """
    ==============================================================================
    |              Stochastic variance reduced gradient (SVRG) class             |
    |               derived class from Stochastic Gradient Descent               |
    ==============================================================================
    Initialization:
        opt = SVRG(nTotSamples, innerIter = 10, outerIter = 200, option = 1,obj, 
        grad, eta, param, iter, maxiter, objFun, gradFun)
        
    NOTE: option = 1 or 2 as suggested in the reference paper.
    ==============================================================================
    Attributes:
        alg:            SVRG
        eta:            learning rate 
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        fullGrad:       Full gradient information 
                        (array of dimension nParam-by-nTotSamples)
        nTotSamples:    total number of samples
        innerIter:      inner iteration
        outerIter:      outer iteration
        iter:           iteration number (optional input)
        maxIter:        maximum iteration number 
                        (optional, default = innerIter*outerIter)
        objFun:         function handle to evaluate the objective 
                        (not required for maxit = 1 )
        gradFun:        function handle to evaluate the gradient 
                        (not required for maxit = 1 )
        mu:             average gradient in the outer iteration
        paramBest:      best estimate of the param in the oter iteration
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performOuterIter:   performs all the iterations inside a for loop
        getGradHist:        returns gradient history (default is zero)
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:           initialization
        innerUpdate:        performs inner iterations of SVRG
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    Reference: Johnson, Rie, and Tong Zhang. 
    "Accelerating stochastic gradient descent using predictive variance reduction." 
    Advances in neural information processing systems. 2013.
    ==============================================================================
    written by Subhayan De (email: Subhayan.De@colorado.edu), July, 2018.
    ==============================================================================
    """
    def __init__(self,nTotSamples, innerIter = 10, outerIter = 200, option = 1, **kwargs):
        """ Initialize the SVRG class object. 
            This can be used to perform one iteration of SVRG. 
        """
        self.alg = 'SVRG'
        SGD.printAlg(self)
        SGD.__init__(self,**kwargs)
        self.nTotSamples = nTotSamples
        # Check inner iteration and outer iteration values
        if innerIter*outerIter > self.maxiter:
            self.maxiter = innerIter*outerIter
            print('Maximum iteration number is set to ',self.maxiter)
        self.innerIter = innerIter
        self.outerIter = outerIter
        self.paramBest = self.param
        # Initialize gradients
        try: 
            self.gradFun
        except NameError: 
            print('Please provide gradient function name')
        self.fullGrad, nprime = self.gradFun(self.param,self.nTotSamples)
        self.grad = self.fullGrad        
        self.mu = np.mean(self.grad,1)
        self.option = option
        
    def innerUpdate(self):
        """
        Perform inner iterations of SVRG
        """
        for i in range(self.innerIter):
            SGD.learningSchedule(self)
            it = np.random.randint(self.nTotSamples)
            bestParamGrad, notNeeded = self.gradFun(self.paramBest,1)
            bestParamGrad = np.reshape(bestParamGrad,(2))
            self.param = self.param - self.etaCurrent*(self.grad[:,it]-bestParamGrad+self.mu)
            SGD.satisfyBounds(self)
            self.paramHist = np.append(self.paramHist,np.reshape(self.param,(2,1)), axis = 1)
        if self.option == 1:
            
            self.paramBest = self.param
        else:
            ind = np.random.randint(low = self.totIter, high = self.totIter+self.innerIter)
            self.paramBest = self.paramHist[:,ind]
        
    def performOuterIter(self):
        """
        Performs all the iterations of SVRG
        """
        # initialize progress bar
        printProgressBar(0, self.outerIter, prefix = self.alg, suffix = 'Complete', length = 25)
        self.t = time.clock()
        self.totIter = 0
        for i in range(self.iter,self.outerIter,1):
            if self.stop == True:
                break
            #print('Outer iteration', i+1, ' of', self.outerIter, ' (inner iteration = ', self.innerIter,')')
            self.innerUpdate()
            self.totIter = self.totIter + (i+1)*self.innerIter
            self.currentIter = i+1
            # print progress bar
            SGD.printProgress(self)
            self.grad, notNeeded = self.gradFun(self.paramBest,self.nTotSamples)
            self.mu = np.mean(self.grad,1)
            # Update the objective and gradient
            if self.maxiter > 1: # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)  
                SGD.stopCrit(self)