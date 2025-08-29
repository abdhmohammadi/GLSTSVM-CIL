from math import acos, pi
from sys import path
from os.path import isfile
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import cg

from pandas import read_csv

path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class GLSTSVMCIL(BaseTSVMClassifier):

    def __init__(self, g=1.0, c1=1.0, c2=1.0, c3=1.0, r=0.1, mu= 1.0, 
                 kernel_name ='rbf', 
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
      return cg(A,b, rtol=tolerance, maxiter=maxiter)[0].reshape(solved.shape[0],1)            
      

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

    def fit(self, X_train, y_train):
        
        if self.kernel_name == '' or self.kernel_name == None:
            self.linear_fit(X_train=X_train,y_train=y_train)
        else:
            self.nonlinear_fit(X_train=X_train,y_train=y_train)
    
    def nonlinear_fit(self,X_train,y_train):
        
        self.A, m1, self.B, m2, self.classes, self.n_features = self.split(X_train,y_train)
        # delete consuming memory by raw data
        del X_train, y_train

        #start weight computing time
        start = perf_counter()
        # inverted weight matrix for the Positive class,
        # S1 is the shape of (m1,m1)        
        S1 = np.diagflat(self.get_weights(label= 'positive'))
        # computing time for weights of the Positive class 
        self.wc1 = perf_counter() - start
        
        #start weight computing time
        start = perf_counter()
        # inverted weight matrix for the Negative class
        S2 = np.diagflat(self.get_weights(label= 'negative'))
        # S1 is the shape of (m2,m2)
        # computing time for weights of the Positive class 
        self.wc2 = perf_counter() - start

        # wc_time is weight computaional time for weighting both class
        self.wc_time = self.wc1 + self.wc2
        
        # start of training time
        start = perf_counter()
        # computing the kernel matrix of A,
        # KAA is the shape of (m1,m1)
        KAA = self.Kernel(self.A,self.A)
        # computing the kernel matrix of BB,
        # KBB is the shape of (m2,m2)
        KBB = self.Kernel(self.B,self.B)
        # computing the kernel matrix of A and B,
        # KAB is the shape of (m1,m2)
        KAB = self.Kernel(self.A,self.B)
        M1  = np.ones((m1,m1))
        M2  = np.ones((m2,m2))
        M12 = np.ones((m1,m2))

        # Create symetric and positive definite matrix
        c1 = self.c1/self.c2
        col1 = np.vstack((c1*S1 + KAA + M1, KAB.T + M12.T))
        c2 = self.c1/self.c3 
        col2 = np.vstack((KAB + M12, c2*S2 + KBB + M2))
        # Symetric positive definite matrix
        # Dimention of Q is (m1 + m2)*(m1 + m2)
        Q  = np.hstack((col1,col2))
        # This eq has not the minus sign in the source paper(LSFLSTSVM-CIL) 
        b = -self.c1 * np.vstack((np.zeros((m1,1)),np.ones((m2,1))))
        
        del col1,col2

        # The alpha is the lagrangian vector of the problem 
        alpha = self.solve(Q,b,tolerance=self.tolerance,maxiter=self.iteration)
        
        del Q, b

        KAAM1 = KAA + M1
        KBBM2 = KBB + M2
        KABM  = KAB + M12
        
        del KAA, KBB, KAB, M1, M2, M12
        
        # Row 1 Column 3 is of shape (m2,1)
        RC13 = ((np.hstack((KABM.T,KBBM2)) @ alpha).reshape(m2,1))/self.c1
        
        # Row 2 Column 3 is of shape (m1,1)
        RC23 = ((np.hstack((KAAM1,KABM)) @ alpha).reshape(m1,1))/self.c1
        
        col1 = np.vstack(( KAAM1, KABM.T))
        col2 = np.vstack(( KABM, KBBM2))
        # a float number
        RC33 = float((alpha.T @ np.hstack((col1,col2)) @ alpha)/(self.c1**2))
        
        c45 = self.c4/self.c5
        c46 = self.c4/self.c6
        # shape of (m1 + m2 + 1, m2)
        col1 = np.vstack((KBBM2 + c45*S2, KABM, RC13.T))
        col2 = np.vstack((KABM.T,KAAM1 + c46*S1, RC23.T))
        col3 = np.vstack((RC13,RC23,[self.c4*RC33]))

        del KAAM1, KABM, KBBM2, RC23, RC13

        # shape of (m1 + m2 + 1, m1 + m2 + 1)
        Q = np.hstack((col1,col2,col3))

        b = self.c4 * np.vstack((np.zeros((m2,1)),np.ones((m1,1)),[[0]]))
        
        del col1, col2, col3

        # The beta is lagrangian vector of second problem
        beta = self.solve(Q,b,tolerance=self.tolerance,maxiter=self.iteration)
        
        self.alpha = alpha
        self.beta  = beta
        
        self.training_time = perf_counter() - start

        self.SSD = 'Not avilable'
        self.norm_angle = 'Not avilable'
        self.planes_angle = 'Not avilable'
        #self.S1 = S1
        #self.S2 = S2

        del alpha, beta, Q, b

    def linear_fit(self,X_train,y_train):
                
        self.A, m1, self.B, m2, self.classes, self.n_features = self.split(X_train,y_train)
        # delete consuming memory by raw data
        del X_train, y_train

        #start weight computing time
        start = perf_counter()
        # inverted weight matrix for the Positive class        
        S1 = np.diagflat(self.get_weights(label= 'positive'))
        # computing time for weights of the Positive class 
        self.wc1 = perf_counter() - start
        
        #start weight computing time
        start = perf_counter()
        # inverted weight matrix for the Negative class
        S2 = np.diagflat(self.get_weights(label= 'negative'))
        # computing time for weights of the Positive class 
        self.wc2 = perf_counter() - start

        # wc_time is weight computaional time for weighting both class
        self.wc_time = self.wc1 + self.wc2
        
        # start of training time
        # ============ First Hyper-plane ===================
        start = perf_counter()

        AAT = self.A @ self.A.T
        ABT = self.A @ self.B.T
        BBT = self.B @ self.B.T

        M1  = np.ones((m1,m1))
        M2  = np.ones((m2,m2))
        M12 = np.ones((m1,m2))
        
        c = self.c1/self.c2
        col1 = np.vstack((c*S1 + AAT + M1, ABT.T + M12.T))
        
        c = self.c1/self.c3 
        col2 = np.vstack((ABT + M12, c*S2 + BBT + M2))

        Q = np.hstack((col1,col2))

        b = -self.c1 * np.vstack((np.zeros((m1,1)),np.ones((m2,1))))
        
        del col1,col2

        #alpha is lagrange vector of alpha1, alpha2        
        alpha = self.solve(Q,b,tolerance=self.tolerance,maxiter=self.iteration)

        del Q, b
        #shape of alpha is (m1+m2,)
        # compute [w1,b1]
        Ae = np.hstack((self.A,np.ones((m1,1))))
        Be = np.hstack((self.B,np.ones((m2,1))))
        # Now, u1 is [w1,b1]
        self.u1 = np.hstack((Ae.T,Be.T)) @ alpha/self.c1
        self.alpha = alpha
        
        del alpha
        
        # ========= Second hyper-plane ===================

        c45 = self.c4/self.c5
        c46 = self.c4/self.c6
        BS = Be @ self.u1
        AS = Ae @ self.u1

        col1 = np.vstack((c45*S2 + BBT + M2 , ABT + M12, BS.T))
        col2 = np.vstack((ABT.T + M12.T , c46 * S1 + AAT + M1, AS.T))
        col3 = np.vstack((BS.reshape((m2,1)) , AS.reshape((m1,1)), self.u1.T @ self.u1 + self.c4))

        Q = np.hstack((col1,col2,col3))
        #b = self.c4 * np.vstack((np.zeros((m1,1)),np.ones((m2,1)),[[0]]))
        b = self.c4 * np.vstack((np.zeros((m2,1)),np.ones((m1,1)),[[0]]))
        del col1, col2, col3

        #beta is lagrange vector of beta1, beta2,beta3
        # Here we solve Qx=b
        beta = self.solve(Q,b,tolerance=self.tolerance,maxiter=self.iteration)

        del Q, b

        # u2 = [w2,b2]
        self.u2 = np.hstack((Be.T,Ae.T,self.u1.reshape((len(self.u1),1)))) @ beta/ self.c4

        # store lagrangian multipliers
        self.beta  = beta

        # end of training task
        self.training_time = perf_counter() - start

        if self.compute_errors_and_angles == True:
           
           if self.kernel_name == '' or self.kernel_name == None:
              # calculate angle
              v1 = self.u1[:-1]
              v2 = self.u2[:-1]

              length = np.linalg.norm(v1) * np.linalg.norm(v2)
              
              self.angle = acos((v1.T @ v2)/length)*180/pi 
              
              eta1 = - S1 @ self.alpha[:m1]/self.c2
              ssd1 = float(eta1.T @ eta1)
              eta2 = - S2 @ self.beta[:m2]/self.c5
              ssd2 = float(eta2.T @ eta2)
              self.SSD = (ssd1 + ssd2)**0.5

        else: 
            self.angle = 'The angles are avilable in linear mode.'
            self.SSD   = 'The SSD is avilable in linear mode.'
        
        # cleanup  memory
        del BS,AS, beta, M1, M2, M12,AAT,BBT,ABT#,S1,S2