# SGD
Implementation of Stochastic Gradient Descent algorithms in Python (GNU GPLv3)  
If you find this code useful please cite the article:  
### Topology Optimization under Uncertainty using a Stochastic Gradient-based Approach ###  
Subhayan De, Jerrad Hampton, Kurt Maute, and Alireza Doostan (2020)  
Structural and Multidisciplinary Optimization, 62(5), 2255-2278.
https://doi.org/10.1007/s00158-020-02599-z  

#BibTeX entry:# 
@article{de2020topology, 
  title={Topology optimization under uncertainty using a stochastic gradient-based approach}, 
  author={De, Subhayan and Hampton, Jerrad and Maute, Kurt and Doostan, Alireza}, 
  journal={Structural and Multidisciplinary Optimization}, 
  volume={62}, 
  number={5}, 
  pages={2255--2278}, 
  year={2020}, 
  publisher={Springer} 
}

Download the SGD module from https://github.com/CU-UQ/SGD.  
See the demo https://github.com/CU-UQ/SGD/blob/master/sgd_demo.py for an example of the implementation.  
For a description of the algorithms, see De et al (2020) (https://doi.org/10.1007/s00158-020-02599-z) and Ruder (2016) (https://arxiv.org/abs/1609.04747).  
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
