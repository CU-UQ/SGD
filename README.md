# SGD
Implementation of Stochastic Gradient Descent algorithms in Python (GNU GPLv3)

Download the SGD module from https://github.com/CU-UQ/SGD.  
See the demo https://github.com/CU-UQ/SGD/blob/master/sgd_demo.py for an example of the implementation.  
For a description of the algorithms see Ruder (2016) (https://arxiv.org/abs/1609.04747).  
Required packages: numpy, time  

This module implements:  
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

*NOTE*: Currently, the stopping conditions are maximum number of iteration and 2nd norm of gradient vector is smaller than a tolerance value. Only, time-delay and exponential learning schedules are implemented.

Download this file and use *import SGD as sgd* to use the algorithms.  
See *sgd_demo.py* for an example.
