# SGD
Implementation of Stochastic Gradient Descent algorithms in Python (GNU GPLv3)  
If you find this code useful please cite the article:  
### Topology Optimization under Uncertainty using a Stochastic Gradient-based Approach ###  
Subhayan De, Jerrad Hampton, Kurt Maute, and Alireza Doostan (2019)  
https://arxiv.org/pdf/1902.04562.pdf  

Download the SGD module from https://github.com/CU-UQ/SGD.  
See the demo https://github.com/CU-UQ/SGD/blob/master/sgd_demo.py for an example of the implementation.  
For a description of the algorithms see De et al (2019) (https://arxiv.org/pdf/1902.04562.pdf) and Ruder (2016) (https://arxiv.org/abs/1609.04747).  
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
