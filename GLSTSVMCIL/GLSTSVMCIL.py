from math import acos, pi
from sys import path
from os.path import isfile
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import cg

from pandas import read_csv
#from modin.pandas import read_csv
path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class GLSTSVMCIL(BaseTSVMClassifier):

    def __init__(self, g=1.0, c1=1.0, c2=1.0, c3=1.0, r=0.1, mu= 1.0, kernel_name ='rbf', 
                 iteration=100, tolerance=1e-05, label_name=None,
                 compute_errors_and_angles = True) -> None:
        """

            c1,c2,c3: hyperplane parameters
            
            r: specified radius to enumerate neighborhoods of a sample

            mu: the parameter of kernel when 'RBF' or 'sigmoid' kernel has choosen

            kernel_name: 'rbf', 'linear', simoid'

            iteration: to iterate numerical method

            telorance: to control solution of numerical method

            label_name: class label name
        """

        super().__init__(kernel_name= kernel_name, mu=mu)
        # Hyperplane parameters
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        # radius to enumerate neighborhoods of each sample
        self.r  = r
        # iteration for cunjogate gradiant method
        self.iteration = iteration
        # telorance for cunjogate gradiant method
        self.tolerance = tolerance
        # train data path is used when 'memory_less_trick' is True
        self.label_name = label_name
        # to avoid from deviding by zero
        self.delta = 0.00001
        # is uset to calculate the errors and angles in the fit method.
        self.compute_errors_and_angles = compute_errors_and_angles
        
    def solve(self,A,b, tolerance=1e-05, maxiter=100):
      solved = cg(A,b, rtol=tolerance, maxiter=maxiter)[0]            
      solved = solved.reshape(solved.shape[0],1)
      return solved

    def predict(self,X_test):

        X_test = np.asarray(X_test)

        predicted = []
        
        for x in X_test:

            x = np.asarray(x).reshape(1,self.n_features)
            
            if self.kernel_name == None or self.kernel_name == '':
                x = np.hstack((x,[[1]]))[0]
                f1 = self.u1.T @ x
                f2 = self.u2.T @ x

            else:
                KxA = self.Kernel(x,self.A)
                KxB = self.Kernel(x,self.B)
                
                f1 = float( np.hstack((KxA + np.ones((KxA.shape[0])), KxB + np.ones((KxB.shape[0])))) @ self.alpha/ self.c1 )
                f2 = float( np.hstack((KxB + np.ones((KxB.shape[0])), KxA + np.ones((KxA.shape[0])),[[f1]])) @ self.beta / self.c4)
            
            predicted.append(self.classes[0] if abs(f1)< abs(f2) else self.classes[1])
        
        return np.asarray(predicted)
